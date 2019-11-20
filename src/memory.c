/* ----------------------------------------------------------------------------
Copyright (c) 2019, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/

/* ----------------------------------------------------------------------------
This implements a layer between the raw OS memory (VirtualAlloc/mmap/sbrk/..)
and the segment and huge object allocation by mimalloc. There may be multiple
implementations of this (one could be the identity going directly to the OS,
another could be a simple cache etc), but the current one uses large "regions".
In contrast to the rest of mimalloc, the "regions" are shared between threads and
need to be accessed using atomic operations.
We need this memory layer between the raw OS calls because of:
1. on `sbrk` like systems (like WebAssembly) we need our own memory maps in order
   to reuse memory effectively.
2. It turns out that for large objects, between 1MiB and 32MiB (?), the cost of
   an OS allocation/free is still (much) too expensive relative to the accesses 
   in that object :-( (`malloc-large` tests this). This means we need a cheaper 
   way to reuse memory.
3. This layer allows for NUMA aware allocation.

Possible issues:
- (2) can potentially be addressed too with a small cache per thread which is much
  simpler. Generally though that requires shrinking of huge pages, and may overuse
  memory per thread. (and is not compatible with `sbrk`).
- Since the current regions are per-process, we need atomic operations to
  claim blocks which may be contended
- In the worst case, we need to search the whole region map (16KiB for 256GiB)
  linearly. At what point will direct OS calls be faster? Is there a way to
  do this better without adding too much complexity?
-----------------------------------------------------------------------------*/
#include "mimalloc.h"
#include "mimalloc-internal.h"
#include "mimalloc-atomic.h"

#include <string.h>  // memset

#include "bitmap.inc.c"

// Internal raw OS interface
size_t  _mi_os_large_page_size();
bool    _mi_os_protect(void* addr, size_t size);
bool    _mi_os_unprotect(void* addr, size_t size);
bool    _mi_os_commit(void* p, size_t size, bool* is_zero, mi_stats_t* stats);
bool    _mi_os_decommit(void* p, size_t size, mi_stats_t* stats);
bool    _mi_os_reset(void* p, size_t size, mi_stats_t* stats);
bool    _mi_os_unreset(void* p, size_t size, bool* is_zero, mi_stats_t* stats);

// arena.c
void    _mi_arena_free(void* p, size_t size, size_t memid, mi_stats_t* stats);
void*   _mi_arena_alloc(size_t size, bool* commit, bool* large, bool* is_zero, size_t* memid, mi_os_tld_t* tld);
void*   _mi_arena_alloc_aligned(size_t size, size_t alignment, bool* commit, bool* large, bool* is_zero, size_t* memid, mi_os_tld_t* tld);



// Constants
#if (MI_INTPTR_SIZE==8)
#define MI_HEAP_REGION_MAX_SIZE    (256 * GiB)  // 48KiB for the region map 
#elif (MI_INTPTR_SIZE==4)
#define MI_HEAP_REGION_MAX_SIZE    (3 * GiB)    // ~ KiB for the region map
#else
#error "define the maximum heap space allowed for regions on this platform"
#endif

#define MI_SEGMENT_ALIGN          MI_SEGMENT_SIZE

#define MI_REGION_MAX_BLOCKS      MI_BITMAP_FIELD_BITS
#define MI_REGION_SIZE            (MI_SEGMENT_SIZE * MI_BITMAP_FIELD_BITS)    // 256MiB  (64MiB on 32 bits)
#define MI_REGION_MAX             (MI_HEAP_REGION_MAX_SIZE / MI_REGION_SIZE)  // 1024  (48 on 32 bits)
#define MI_REGION_MAX_OBJ_BLOCKS  (MI_REGION_MAX_BLOCKS/4)                    // 64MiB
#define MI_REGION_MAX_OBJ_SIZE    (MI_REGION_MAX_OBJ_BLOCKS*MI_SEGMENT_SIZE)  

// Region info is a pointer to the memory region and two bits for 
// its flags: is_large, and is_committed.
typedef union mi_region_info_u {
  uintptr_t value;
  struct {
    bool  valid;
    bool  is_large;
    int   numa_node;
  };
} mi_region_info_t;


// A region owns a chunk of REGION_SIZE (256MiB) (virtual) memory with
// a bit map with one bit per MI_SEGMENT_SIZE (4MiB) block.
typedef struct mem_region_s {
  volatile _Atomic(uintptr_t)        info;        // is_large, and associated numa node + 1 (so 0 is no association)
  volatile _Atomic(void*)            start;       // start of the memory area (and flags)
  mi_bitmap_field_t                  in_use;      // bit per in-use block
  mi_bitmap_field_t                  dirty;       // track if non-zero per block
  mi_bitmap_field_t                  commit;      // track if committed per block (if `!info.is_committed))
  mi_bitmap_field_t                  reset;       // track reset per block
  volatile _Atomic(uintptr_t)        arena_memid; // if allocated from a (huge page) arena-
} mem_region_t;

// The region map
static mem_region_t regions[MI_REGION_MAX];

// Allocated regions
static volatile _Atomic(uintptr_t) regions_count; // = 0;        


// local
static bool mi_region_pop_segment_resets(int numa_node, size_t need_blocks, bool allow_large, mem_region_t** pregion, mi_bitmap_index_t* pbit_idx, mi_os_tld_t* tld);
// static void mi_region_flush_segment_resets(mi_os_tld_t* tld);
static void mi_region_segment_reset(mem_region_t* region, size_t blocks, mi_bitmap_index_t bit_idx, mi_os_tld_t* tld);


/* ----------------------------------------------------------------------------
Utility functions
-----------------------------------------------------------------------------*/

// Blocks (of 4MiB) needed for the given size.
static size_t mi_region_block_count(size_t size) {
  return _mi_divide_up(size, MI_SEGMENT_SIZE);
}

/*
// Return a rounded commit/reset size such that we don't fragment large OS pages into small ones.
static size_t mi_good_commit_size(size_t size) {
  if (size > (SIZE_MAX - _mi_os_large_page_size())) return size;
  return _mi_align_up(size, _mi_os_large_page_size());
}
*/

// Return if a pointer points into a region reserved by us.
bool mi_is_in_heap_region(const void* p) mi_attr_noexcept {
  if (p==NULL) return false;
  size_t count = mi_atomic_read_relaxed(&regions_count);
  for (size_t i = 0; i < count; i++) {
    uint8_t* start = (uint8_t*)mi_atomic_read_ptr_relaxed(&regions[i].start);
    if (start != NULL && (uint8_t*)p >= start && (uint8_t*)p < start + MI_REGION_SIZE) return true;
  }
  return false;
}


static void* mi_region_blocks_start(const mem_region_t* region, mi_bitmap_index_t bit_idx) {
  void* start = mi_atomic_read_ptr(&region->start);
  mi_assert_internal(start != NULL);
  return ((uint8_t*)start + (bit_idx * MI_SEGMENT_SIZE));  
}

static size_t mi_memid_create(mem_region_t* region, mi_bitmap_index_t bit_idx) {
  mi_assert_internal(bit_idx < MI_BITMAP_FIELD_BITS);
  size_t idx = region - regions;
  mi_assert_internal(&regions[idx] == region);
  return (idx*MI_BITMAP_FIELD_BITS + bit_idx)<<1;
}

static size_t mi_memid_create_from_arena(size_t arena_memid) {
  return (arena_memid << 1) | 1;
}


static bool mi_memid_is_arena(size_t id, mem_region_t** region, mi_bitmap_index_t* bit_idx, size_t* arena_memid) {
  if ((id&1)==1) {
    if (arena_memid != NULL) *arena_memid = (id>>1);
    return true;
  }
  else {
    size_t idx = (id >> 1) / MI_BITMAP_FIELD_BITS;
    *bit_idx   = (mi_bitmap_index_t)(id>>1) % MI_BITMAP_FIELD_BITS;
    *region    = &regions[idx];
    return false;
  }
}


/* ----------------------------------------------------------------------------
  Allocate a region is allocated from the OS (or an arena)
-----------------------------------------------------------------------------*/

static bool mi_region_try_alloc_os(size_t blocks, bool commit, bool allow_large, mem_region_t** region, mi_bitmap_index_t* bit_idx, mi_os_tld_t* tld)
{
  // not out of regions yet?
  if (mi_atomic_read_relaxed(&regions_count) >= MI_REGION_MAX - 1) return false;

  // try to allocate a fresh region from the OS
  bool region_commit = (commit && mi_option_is_enabled(mi_option_eager_region_commit));
  bool region_large = (commit && allow_large);
  bool is_zero = false;
  size_t arena_memid = 0;
  void* const start = _mi_arena_alloc_aligned(MI_REGION_SIZE, MI_SEGMENT_ALIGN, &region_commit, &region_large, &is_zero, &arena_memid, tld);
  if (start == NULL) return false;
  mi_assert_internal(!(region_large && !allow_large));
  mi_assert_internal(!region_large || region_commit);

  // claim a fresh slot
  const uintptr_t idx = mi_atomic_increment(&regions_count);
  if (idx >= MI_REGION_MAX) {
    mi_atomic_decrement(&regions_count);
    _mi_arena_free(start, MI_REGION_SIZE, arena_memid, tld->stats);
    return false;
  }

  // allocated, initialize and claim the initial blocks
  mem_region_t* r = &regions[idx];
  r->arena_memid  = arena_memid;
  mi_atomic_write(&r->in_use, 0);
  mi_atomic_write(&r->dirty, (is_zero ? 0 : MI_BITMAP_FIELD_FULL));
  mi_atomic_write(&r->commit, (region_commit ? MI_BITMAP_FIELD_FULL : 0));
  mi_atomic_write(&r->reset, 0);
  *bit_idx = 0;
  mi_bitmap_claim(&r->in_use, 1, blocks, *bit_idx, NULL);
  mi_atomic_write_ptr(&r->start, start);

  // and share it 
  mi_region_info_t info;
  info.valid = true;
  info.is_large = region_large;
  info.numa_node = _mi_os_numa_node(tld);
  mi_atomic_write(&r->info, info.value); // now make it available to others
  *region = r;
  return true;
}

/* ----------------------------------------------------------------------------
  Try to claim blocks in suitable regions
-----------------------------------------------------------------------------*/

static bool mi_region_is_suitable(const mem_region_t* region, int numa_node, bool allow_large ) {
  // initialized at all?
  mi_region_info_t info;
  info.value = mi_atomic_read_relaxed(&region->info);
  if (info.value==0) return false;

  // numa correct
  if (numa_node >= 0) {  // use negative numa node to always succeed
    int rnode = info.numa_node;
    if (rnode >= 0 && rnode != numa_node) return false;
  }

  // check allow-large
  if (!allow_large && info.is_large) return false;

  return true;
}


static bool mi_region_try_claim(int numa_node, size_t blocks, bool allow_large, mem_region_t** region, mi_bitmap_index_t* bit_idx, mi_os_tld_t* tld)
{
  // try all regions for a free slot  
  const size_t count = mi_atomic_read(&regions_count);
  size_t idx = tld->region_idx; // Or start at 0 to reuse low addresses? 
  for (size_t visited = 0; visited < count; visited++, idx++) {
    if (idx >= count) idx = 0;  // wrap around
    mem_region_t* r = &regions[idx];
    if (mi_region_is_suitable(r, numa_node, allow_large)) {
      if (mi_bitmap_try_find_claim_field(&r->in_use, 0, blocks, bit_idx)) {
        tld->region_idx = idx;    // remember the last found position
        *region = r;
        return true;
      }
    }
  }
  return false;
}


static void* mi_region_try_alloc(size_t blocks, bool* commit, bool* is_large, bool* is_zero, size_t* memid, mi_os_tld_t* tld)
{
  mi_assert_internal(blocks <= MI_BITMAP_FIELD_BITS);
  mem_region_t* region;
  mi_bitmap_index_t bit_idx;
  const int numa_node = (_mi_os_numa_node_count() <= 1 ? -1 : _mi_os_numa_node(tld));
  // first try to claim from delayed resets
  if (!mi_region_pop_segment_resets(numa_node, blocks, *is_large, &region, &bit_idx, tld)) {
    // then try to claim in existing regions
    if (!mi_region_try_claim(numa_node, blocks, *is_large, &region, &bit_idx, tld)) {
      // otherwise try to allocate a fresh region
      if (!mi_region_try_alloc_os(blocks, *commit, *is_large, &region, &bit_idx, tld)) {
        // out of regions or memory
        return NULL;
      }
    }
  }
  
  // found a region and claimed `blocks` at `bit_idx`
  mi_assert_internal(region != NULL);
  mi_assert_internal(mi_bitmap_is_claimed(&region->in_use, 1, blocks, bit_idx));

  mi_region_info_t info;
  info.value = mi_atomic_read(&region->info);
  void* start = mi_atomic_read_ptr(&region->start);
  mi_assert_internal(!(info.is_large && !*is_large));
  mi_assert_internal(start != NULL);

  *is_zero = mi_bitmap_unclaim(&region->dirty, 1, blocks, bit_idx);  
  *is_large = info.is_large;
  *memid = mi_memid_create(region, bit_idx);
  void* p = (uint8_t*)start + (mi_bitmap_index_bit_in_field(bit_idx) * MI_SEGMENT_SIZE);

  // commit
  if (*commit) {
    // ensure commit
    bool any_uncommitted;
    mi_bitmap_claim(&region->commit, 1, blocks, bit_idx, &any_uncommitted);
    if (any_uncommitted) {
      mi_assert_internal(!info.is_large);
      bool commit_zero;
      _mi_mem_commit(p, blocks * MI_SEGMENT_SIZE, &commit_zero, tld);
      if (commit_zero) *is_zero = true;
    }
  }
  else {
    // no need to commit, but check if already fully committed
    *commit = mi_bitmap_is_claimed(&region->commit, 1, blocks, bit_idx);
  }  
  mi_assert_internal(mi_bitmap_is_claimed(&region->commit, 1, blocks, bit_idx));

  // unreset reset blocks
  if (mi_bitmap_is_any_claimed(&region->reset, 1, blocks, bit_idx)) {
    mi_assert_internal(!info.is_large);
    mi_assert_internal(!mi_option_is_enabled(mi_option_eager_commit) || *commit); 
    mi_bitmap_unclaim(&region->reset, 1, blocks, bit_idx);
    bool reset_zero;
    _mi_mem_unreset(p, blocks * MI_SEGMENT_SIZE, &reset_zero, tld);
    if (reset_zero) *is_zero = true;
  }
  mi_assert_internal(!mi_bitmap_is_any_claimed(&region->reset, 1, blocks, bit_idx));

  #if (MI_DEBUG>=2)
  if (*commit) { ((uint8_t*)p)[0] = 0; }
  #endif
  
  // and return the allocation  
  mi_assert_internal(p != NULL);  
  return p;
}


/* ----------------------------------------------------------------------------
 Allocation
-----------------------------------------------------------------------------*/

// Allocate `size` memory aligned at `alignment`. Return non NULL on success, with a given memory `id`.
// (`id` is abstract, but `id = idx*MI_REGION_MAP_BITS + bitidx`)
void* _mi_mem_alloc_aligned(size_t size, size_t alignment, bool* commit, bool* large, bool* is_zero, size_t* memid, mi_os_tld_t* tld)
{
  mi_assert_internal(memid != NULL && tld != NULL);
  mi_assert_internal(size > 0);
  *memid = 0;
  *is_zero = false;
  bool default_large = false;
  if (large==NULL) large = &default_large;  // ensure `large != NULL`  
  if (size == 0) return NULL;
  size = _mi_align_up(size, _mi_os_page_size());

  // allocate from regions if possible
  size_t arena_memid;
  const size_t blocks = mi_region_block_count(size);
  if (blocks <= MI_REGION_MAX_OBJ_BLOCKS && alignment <= MI_SEGMENT_ALIGN) {
    void* p = mi_region_try_alloc(blocks, commit, large, is_zero, memid, tld);
    mi_assert_internal(p == NULL || (uintptr_t)p % alignment == 0);    
    if (p != NULL) {
      #if (MI_DEBUG>=2)
      if (*commit) { ((uint8_t*)p)[0] = 0; }
      #endif
      return p;
    }
    _mi_warning_message("unable to allocate from region: size %zu\n", size);
  }

  // and otherwise fall back to the OS
  void* p = _mi_arena_alloc_aligned(size, alignment, commit, large, is_zero, &arena_memid, tld);
  *memid = mi_memid_create_from_arena(arena_memid);
  mi_assert_internal( p == NULL || (uintptr_t)p % alignment == 0);
  if (p != NULL && *commit) { ((uint8_t*)p)[0] = 0; }
  return p;
}



/* ----------------------------------------------------------------------------
Free
-----------------------------------------------------------------------------*/

// Free previously allocated memory with a given id.
void _mi_mem_free(void* p, size_t size, size_t id, bool full_commit, bool any_reset, mi_os_tld_t* tld) {
  mi_assert_internal(size > 0 && tld != NULL);
  if (p==NULL) return;
  if (size==0) return;
  size = _mi_align_up(size, _mi_os_page_size());
  
  size_t arena_memid = 0;
  mi_bitmap_index_t bit_idx;
  mem_region_t* region;
  if (mi_memid_is_arena(id,&region,&bit_idx,&arena_memid)) {
   // was a direct arena allocation, pass through
    _mi_arena_free(p, size, arena_memid, tld->stats);
  }
  else {
    // allocated in a region
    mi_assert_internal(size <= MI_REGION_MAX_OBJ_SIZE); if (size > MI_REGION_MAX_OBJ_SIZE) return;
    const size_t blocks = mi_region_block_count(size);
    mi_assert_internal(blocks + bit_idx <= MI_BITMAP_FIELD_BITS);
    mi_region_info_t info;
    info.value = mi_atomic_read(&region->info);
    mi_assert_internal(info.value != 0);
    void* blocks_start = mi_region_blocks_start(region, bit_idx);
    mi_assert_internal(blocks_start == p); // not a pointer in our area?
    mi_assert_internal(bit_idx + blocks <= MI_BITMAP_FIELD_BITS);
    if (blocks_start != p || bit_idx + blocks > MI_BITMAP_FIELD_BITS) return; // or `abort`?

    // committed?
    if (full_commit && (size % MI_SEGMENT_SIZE) == 0) {
      mi_bitmap_claim(&region->commit, 1, blocks, bit_idx, NULL);
    }

    if (any_reset) {
      // set the is_reset bits if any pages were reset
      mi_bitmap_claim(&region->reset, 1, blocks, bit_idx, NULL);
    }

    // reset the blocks to reduce the working set.
    if (!info.is_large && mi_option_is_enabled(mi_option_segment_reset) &&
        mi_option_is_enabled(mi_option_eager_commit))  // cannot reset halfway committed segments, use only `option_page_reset` instead            
    {
      mi_region_segment_reset(region, blocks, bit_idx, tld);
    }    

    // and unclaim
    bool all_unclaimed = mi_bitmap_unclaim(&region->in_use, 1, blocks, bit_idx);
    mi_assert_internal(all_unclaimed);
  }
}


/* ----------------------------------------------------------------------------
  collection
-----------------------------------------------------------------------------*/
void _mi_mem_collect(mi_os_tld_t* tld) {
  // free every region that has no segments in use.
  uintptr_t rcount = mi_atomic_read_relaxed(&regions_count);
  for (size_t i = 0; i < rcount; i++) {
    mem_region_t* region = &regions[i];
    if (mi_atomic_read_relaxed(&region->info) != 0) {
      // if no segments used, try to claim the whole region
      uintptr_t m;
      do {
        m = mi_atomic_read_relaxed(&region->in_use);
      } while(m == 0 && !mi_atomic_cas_weak(&region->in_use, MI_BITMAP_FIELD_FULL, 0 ));
      if (m == 0) {
        // on success, free the whole region
        void* start = mi_atomic_read_ptr(&regions[i].start);
        size_t arena_memid = mi_atomic_read_relaxed(&regions[i].arena_memid);
        memset(&regions[i], 0, sizeof(mem_region_t));
        // and release the whole region
        mi_atomic_write(&region->info, 0);
        if (start != NULL) { // && !_mi_os_is_huge_reserved(start)) {          
          _mi_arena_free(start, MI_REGION_SIZE, arena_memid, tld->stats);
        }
      }
    }
  }
}



/*--------------------------------------------------------
  Free Queue's
--------------------------------------------------------*/

#define MI_RESET_MAX_FLUSH  (8)
#define MI_FQUEUE_COUNT     (2*1024)

typedef struct mi_finfo_s {
  mi_msecs_t  expire;
  size_t      memid;
  size_t      blocks;
} mi_finfo_t;


/*--------------------------------------------------------
 Atomic segment reset queue
--------------------------------------------------------*/

#if (MI_INTPTR_SIZE < 8)
typedef uint8_t       mi_index_t;
#define MI_INDEX_MAX  (0xFF)
#else 
typedef uint16_t      mi_index_t;
#define MI_INDEX_MAX  (0xFFFF)
#endif

typedef union mi_frange_s {
  uintptr_t value;
  struct {
    mi_index_t rtop;
    mi_index_t top;
    mi_index_t bot;
    mi_index_t epoch;
  };
} mi_frange_t;

typedef struct mi_fqueue_s {
  volatile _Atomic(uintptr_t) range;
  mi_finfo_t elems[MI_FQUEUE_COUNT];
} mi_fqueue_t;


static mi_fqueue_t  segment_resets;

/*--------------------------------------------------------
Dequeue
--------------------------------------------------------*/
#if (MI_DEBUG>=2)
static bool mi_frange_is_valid(mi_frange_t r) {
  mi_assert_internal(r.rtop == r.top || r.rtop == r.top+1);
  mi_assert_internal(r.rtop <= MI_INDEX_MAX);
  mi_assert_internal(r.top >= r.bot);
  mi_assert_internal(r.rtop - r.bot <= MI_FQUEUE_COUNT);
  return true;
}
#endif

static bool mi_frange_is_empty(mi_frange_t r) {
  mi_assert_internal(mi_frange_is_valid(r));
  return (r.bot == r.top);
}

static mi_frange_t mi_frange_deq(mi_frange_t r) {
  mi_assert_internal(mi_frange_is_valid(r));
  mi_assert_internal(r.top > r.bot&& r.top <= MI_INDEX_MAX);
  r.bot++;
  return r;
}

static bool mi_fqueue_deq(mi_fqueue_t* fq, const mi_msecs_t only_expired, mi_finfo_t* finfo) {
  mi_frange_t r = { 0 };
  // note: assumes that a loop iteration finishes before the epoch 
  // wraps around (i.e. all other threads queue/dequeue N times
  // such that the queue wraps around epoch times; (32-bit count on 64-bits, 16-bit count on 32-bits)
  do {
    r.value = mi_atomic_read_relaxed(&fq->range);
    if (mi_frange_is_empty(r)) return false;
    *finfo = fq->elems[r.bot % MI_FQUEUE_COUNT]; // racy read but ok    
    if (only_expired != 0 && finfo->expire >= only_expired) {
      // not expired yet
      return false;
    }
  } while (!mi_atomic_cas_weak(&fq->range, mi_frange_deq(r).value, r.value));
  return true;
}


/*--------------------------------------------------------
Enqueue
--------------------------------------------------------*/

static bool mi_frange_is_full(mi_frange_t r) {
  mi_assert_internal(mi_frange_is_valid(r));
  return ((r.rtop - r.bot) >= MI_FQUEUE_COUNT);
}

static bool mi_frange_can_reserve(mi_frange_t r) {
  mi_assert_internal(mi_frange_is_valid(r));
  return (r.rtop == r.top);
}

static mi_frange_t mi_frange_reserve_enq(mi_frange_t r) {
  mi_assert_internal(!mi_frange_is_full(r));
  if (r.rtop >= (MI_INDEX_MAX/2)) {
    // re-normalize
    r.rtop %= MI_FQUEUE_COUNT;
    r.top %= MI_FQUEUE_COUNT;
    r.bot %= MI_FQUEUE_COUNT;
    r.epoch++;
    mi_assert_internal(!mi_frange_is_full(r));
  }
  mi_assert_internal(r.rtop == r.top);
  r.rtop = r.top + 1;
  return r;
}

static mi_frange_t mi_frange_finish_enq(mi_frange_t r) {
  mi_assert_internal(mi_frange_is_valid(r));
  mi_assert_internal(r.rtop == r.top+1);
  r.top = r.rtop;
  return r;
}

static bool mi_fqueue_enq(mi_fqueue_t* fq, const mi_finfo_t* finfo) {
  // try to reserve
  mi_frange_t r;
  do {
    r.value = mi_atomic_read_relaxed(&fq->range);
    if (mi_frange_is_full(r)) return false;    
  } while (!mi_frange_can_reserve(r) ||
    !mi_atomic_cas_weak(&fq->range, mi_frange_reserve_enq(r).value, r.value));
  // now write in our reserved spot
  fq->elems[r.top % MI_FQUEUE_COUNT] = *finfo;
  // and finish the enqueue
#if (MI_DEBUG>=2)
  const mi_frange_t r_enq = r;
#endif
  do {
    r.value = mi_atomic_read_relaxed(&fq->range); // dequeues may have happened
    mi_assert_internal((r.rtop % MI_FQUEUE_COUNT) == (r_enq.rtop + 1) % MI_FQUEUE_COUNT);
    mi_assert_internal((r.top % MI_FQUEUE_COUNT) == (r_enq.top % MI_FQUEUE_COUNT));
  } while (!mi_atomic_cas_weak(&fq->range, mi_frange_finish_enq(r).value, r.value));
  return true;
}


/*--------------------------------------------------------
 Try alloc/free into an fqueue
--------------------------------------------------------*/

static void mi_region_reset_now(mem_region_t* region, size_t blocks, mi_bitmap_index_t bit_idx, mi_os_tld_t* tld) {  
  if (!mi_bitmap_claim(&region->reset, 1, blocks, bit_idx, NULL))  // note: the reset may be on a subset of the blocks only
  {
    // at least some blocks were not yet reset
    void* p = mi_region_blocks_start(region, bit_idx);
    _mi_mem_reset(p, blocks*MI_SEGMENT_SIZE, tld);
  }  
}

static bool mi_region_pop_segment_resets(int numa_node, size_t need_blocks, bool allow_large, mem_region_t** pregion, mi_bitmap_index_t* pbit_idx, mi_os_tld_t* tld) 
{
  // flush expired entries
  const mi_msecs_t now = _mi_clock_now();
  mi_finfo_t expired;
  size_t pop_count = 0;
  while (mi_fqueue_deq(&segment_resets, now, &expired)) {
    mi_assert_internal(now >= expired.expire);
    // pop one
    mem_region_t* region;
    mi_bitmap_index_t bit_idx;
    bool is_arena = mi_memid_is_arena(expired.memid, &region, &bit_idx, NULL);
    mi_assert_internal(!is_arena);
    if (is_arena) continue;

    mi_region_info_t info;
    info.value = mi_atomic_read_relaxed(&region->info);
    mi_assert_internal(info.value!=0);
    if (info.value == 0) continue;
    
    // try to claim this block instead of resetting
    if (pregion != NULL && need_blocks == expired.blocks) {
      if (mi_region_is_suitable(region, numa_node, allow_large)) {
        if (mi_bitmap_try_claim_field(&region->in_use, 1, need_blocks, bit_idx)) {
          // success, claimed to `bit_idx`.
          // todo: reset any left-over blocks if `expired.blocks > need_blocks`
          *pregion = region;
          *pbit_idx = bit_idx;
          pregion = NULL;// stop looking
        }
      }
    }
   
    // try to reset it, but abandon the effort if some blocks are in use already
    // TODO: handle partially claimed? i.e. reset per block instead of per range of blocks?
    if (mi_bitmap_try_claim_field(&region->in_use, 1, expired.blocks, bit_idx)) {
      // we got it
      mi_region_reset_now(region, expired.blocks, bit_idx, tld);
      mi_bitmap_unclaim(&region->in_use, 1, expired.blocks, bit_idx);
    }
    else {
      // already claimed; skip the reset :-)
    }

    // limit 
    pop_count++;
    if (pop_count >= MI_RESET_MAX_FLUSH) break;
  }
  return (pregion == NULL);  // claimed?
}

static void mi_region_flush_segment_resets(mi_os_tld_t* tld) {
  mi_region_pop_segment_resets(-1, 0, false, NULL, NULL, tld);
}


static void mi_region_segment_reset(mem_region_t* region, size_t blocks, mi_bitmap_index_t bit_idx, mi_os_tld_t* tld) {
  mi_assert_internal(mi_bitmap_is_claimed(&region->in_use, 1, blocks, bit_idx));
  // first flush
  mi_region_flush_segment_resets(tld);

  // then try to enqueue
  mi_finfo_t finfo;
  finfo.expire = _mi_clock_now() + mi_option_get(mi_option_reset_delay);
  finfo.blocks = blocks;
  finfo.memid = mi_memid_create(region, bit_idx);
  if (!mi_fqueue_enq(&segment_resets, &finfo)) {
    // if failed, reset it now
    mi_region_reset_now(region, blocks, bit_idx, tld);
  }
}


/* ----------------------------------------------------------------------------
  Other
-----------------------------------------------------------------------------*/

bool _mi_mem_reset(void* p, size_t size, mi_os_tld_t* tld) {
  return _mi_os_reset(p, size, tld->stats);
}

bool _mi_mem_unreset(void* p, size_t size, bool* is_zero, mi_os_tld_t* tld) {
  return _mi_os_unreset(p, size, is_zero, tld->stats);
}

bool _mi_mem_commit(void* p, size_t size, bool* is_zero, mi_os_tld_t* tld) {
  return _mi_os_commit(p, size, is_zero, tld->stats);
}

bool _mi_mem_decommit(void* p, size_t size, mi_os_tld_t* tld) {
  return _mi_os_decommit(p, size, tld->stats);
}

bool _mi_mem_protect(void* p, size_t size) {
  return _mi_os_protect(p, size);
}

bool _mi_mem_unprotect(void* p, size_t size) {
  return _mi_os_unprotect(p, size);
}
