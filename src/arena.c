/* ----------------------------------------------------------------------------
Copyright (c) 2019, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/

/* ----------------------------------------------------------------------------
"Arenas" are fixed area's of OS memory from which we can allocate
large blocks (>= MI_ARENA_BLOCK_SIZE, 32MiB). 
In contrast to the rest of mimalloc, the arenas are shared between 
threads and need to be accessed using atomic operations.

Currently arenas are only used to for huge OS page (1GiB) reservations,
otherwise it delegates to direct allocation from the OS.
In the future, we can expose an API to manually add more kinds of arenas 
which is sometimes needed for embedded devices or shared memory for example.
(We can also employ this with WASI or `sbrk` systems to reserve large arenas
 on demand and be able to reuse them efficiently).

The arena allocation needs to be thread safe and we use an atomic
bitmap to allocate. The current implementation of the bitmap can
only do this within a field (`uintptr_t`) so we can allocate at most
blocks of 2GiB (64*32MiB) and no object can cross the boundary. This
can lead to fragmentation but fortunately most objects will be regions
of 256MiB in practice.
-----------------------------------------------------------------------------*/
#include "mimalloc.h"
#include "mimalloc-internal.h"
#include "mimalloc-atomic.h"

#include <string.h>  // memset

#include "bitmap.inc.c"  // atomic bitmap

// os.c
void* _mi_os_alloc_aligned(size_t size, size_t alignment, bool commit, bool* large, mi_os_tld_t* tld);
void  _mi_os_free(void* p, size_t size, mi_stats_t* stats);
bool  _mi_os_commit(void* p, size_t size, bool* is_zero, mi_stats_t* stats);

void* _mi_os_alloc_huge_os_pages(size_t pages, int numa_node, mi_msecs_t max_secs, size_t* pages_reserved, size_t* psize);
void  _mi_os_free_huge_pages(void* p, size_t size, mi_stats_t* stats);

int   _mi_os_numa_node_count(void);

/* -----------------------------------------------------------
  Arena allocation
----------------------------------------------------------- */

#define MI_SEGMENT_ALIGN      MI_SEGMENT_SIZE
#define MI_ARENA_BLOCK_SIZE   MI_SEGMENT_SIZE
#define MI_ARENA_BLOCK_COUNT  MI_BITMAP_FIELD_BITS

#define MI_ARENA_SIZE         (MI_ARENA_BLOCK_COUNT*MI_ARENA_BLOCK_SIZE)  // 256MiB (64MiB on 32-bit)

#if MI_INTPTR_SIZE >= 8
#define MI_MAX_ARENAS         ((256 * GiB) / MI_ARENA_SIZE)               // 1024 (at most 256GiB in default arenas)
#else
#define MI_MAX_ARENAS         ((3 * GiB) / MI_ARENA_SIZE)                 // 48 (at most 3GiB in default arenas)
#endif
#define MI_MAX_STATIC_ARENAS  (256)


#define MI_ARENA_MAX_OBJ_SIZE (MI_ARENA_SIZE / 2)                         // 128MiB (32MiB on 32-bit)


// Define `mi_arena_t` in a packed way as we statically allocate MAX_ARENAS + MAX_STATIC_ARENAS for 3 lists:
// 3*(1024+256)*sizeof(mi_arena_t) = 120KiB on 64 bit


// Use either a direct bitmap field or a pointer to the fields for large arena's 
typedef union mi_arena_bitmap_u {
  mi_bitmap_field_t  field;
  mi_bitmap_field_t* bitmap;
} mi_arena_bitmap_t;


// A memory arena descriptor
typedef struct mi_arena_s {
  void*      start;                       // the start of the memory area
  volatile _Atomic(uintptr_t) field_count;// number of bitmap fields (always 1 for default arenas)
  int16_t    numa_node;                   // associated NUMA node + 1
  uint16_t   block_count;                 // size of the area in arena blocks (of `MI_ARENA_BLOCK_SIZE`)
  uint8_t    is_zero_init : 1;            // is the arena zero initialized?
  uint8_t    is_fixed : 1;                // fixed memory (cannot be decommitted or reset) (like large OS pages)
  uint8_t    is_committed : 1;            // is the memory committed
  mi_arena_bitmap_t  blocks_map;          // bitmap of in-use blocks
  mi_arena_bitmap_t  blocks_com;          // if (is_committed) then: are the blocks potentially non-zero? else: are the blocks committed (and potentially non-zero)?
} mi_arena_t;


// There are 3 arena lists: reserved, committed, and fixed (large OS pages) memory
#define MI_ARENA_KINDS       (3)
typedef enum mi_arena_kind_e {
  mi_arena_fixed,
  mi_arena_committed,
  mi_arena_reserved
} mi_arena_kind_t;


#define MI_MAX_ARENAS_COUNT  (MI_ARENA_KINDS*(MI_MAX_ARENAS+MI_MAX_STATIC_ARENAS))  // must be less than 0xFFFF (16 bits)

static mi_arena_t mi_arenas[MI_MAX_ARENAS_COUNT];
static _Atomic(uintptr_t) mi_arenas_count[MI_ARENA_KINDS];         // default (dynamic) arenas
static _Atomic(uintptr_t) mi_arenas_static_count[MI_ARENA_KINDS];  // static arenas of various sizes (like reserved huge OS page arenas)


/* -----------------------------------------------------------
  Arena allocations get a memory id where the lower 16 bits are
  the arena index +1, and the upper bits the bitmap index.
----------------------------------------------------------- */

// Use `0` as a special id for direct OS allocated memory.
#define MI_MEMID_OS   0

static size_t mi_memid_create(size_t arena_index, mi_bitmap_index_t bitmap_index) {
  mi_assert_internal(arena_index < 0xFFFF);
  mi_assert_internal(((bitmap_index << 16) >> 16) == bitmap_index); // no overflow?
  return ((bitmap_index << 16) | ((arena_index+1) & 0xFFFF));
}

static void mi_memid_indices(size_t memid, size_t* arena_index, mi_bitmap_index_t* bitmap_index) {
  mi_assert_internal(memid != MI_MEMID_OS);
  *arena_index = (memid & 0xFFFF) - 1;
  *bitmap_index = (memid >> 16);
}

static size_t mi_block_count_of_size(size_t size) {
  return _mi_divide_up(size, MI_ARENA_BLOCK_SIZE);
}

static mi_bitmap_t mi_arena_bitmap_map(mi_arena_t* arena) {
  mi_assert_internal(mi_atomic_read_relaxed(&arena->field_count)!=0);
  return (mi_atomic_read_relaxed(&arena->field_count)==1 ? &arena->blocks_map.field : arena->blocks_map.bitmap);
}

static mi_bitmap_t mi_arena_bitmap_com(mi_arena_t* arena) {
  mi_assert_internal(mi_atomic_read_relaxed(&arena->field_count)!=0);
  return (mi_atomic_read_relaxed(&arena->field_count)==1 ? &arena->blocks_com.field : arena->blocks_com.bitmap);
}

static mi_arena_t* mi_arena_get(mi_arena_kind_t kind, bool is_static, size_t* arena_count) {
  if (arena_count!=NULL) {
    *arena_count = mi_atomic_read_relaxed(is_static ? &mi_arenas_static_count[kind] : &mi_arenas_count[kind]);
  }
  // ofs = kind*(MI_MAX_ARENAS+MI_MAX_STATIC_ARENAS) + (is_static ? 0 : MI_MAX_STATIC_ARENAS)
  size_t ofs = 0;
  if (kind==mi_arena_committed)     ofs = MI_MAX_ARENAS+MI_MAX_STATIC_ARENAS;
  else if (kind==mi_arena_reserved) ofs = 2*(MI_MAX_ARENAS+MI_MAX_STATIC_ARENAS);
  if (!is_static) ofs += MI_MAX_STATIC_ARENAS;
  return &mi_arenas[ofs];
}

/* -----------------------------------------------------------
  Thread safe allocation in an arena
----------------------------------------------------------- */

static bool mi_arena_try_claim(mi_arena_t* const arena, int numa_node, size_t blocks, mi_bitmap_index_t* bitmap_idx)
{
  const size_t fcount = mi_atomic_read_relaxed(&arena->field_count);
  if (fcount==0) { return false; } // not yet initialized
  if (numa_node>=0) {
    const int nnode = arena->numa_node-1;
    if (nnode>=0&&nnode!=numa_node) { return false; } // arena is not on the same NUMA node
  }
  if (mi_likely(fcount==1)) {
    // single field bitmap
    return mi_bitmap_try_claim_field(&arena->blocks_map.field, 0 /*idx*/, blocks, bitmap_idx);
  }
  else {
    // larger bitmap (like for reserved huge OS pages
    // TODO: handle large object allocations
    return mi_bitmap_try_claim(arena->blocks_map.bitmap, fcount, blocks, bitmap_idx);
  }
}

static bool mi_arenas_try_claim(mi_arena_t* arenas, size_t arena_count, size_t start_idx, int numa_node, size_t blocks, size_t* arena_idx, mi_bitmap_index_t* bitmap_idx) 
{
  size_t i = start_idx;
  for (size_t visited = 0; visited < arena_count; visited++, i++ ) {
    if (i>=arena_count) { i = 0; }  // wrap around
    if (mi_arena_try_claim(&arenas[i], numa_node, blocks, bitmap_idx)) {
      *arena_idx = i + (arenas - mi_arenas);
      return true;
    }
  }
  return false;
}

static inline bool mi_arena_kind_try_alloc(mi_arena_kind_t kind, bool is_static, int numa_node, size_t blocks, size_t* arena_idx, mi_bitmap_index_t* bitmap_idx, mi_os_tld_t* tld) 
{
  if (!is_static&&blocks>MI_ARENA_MAX_OBJ_SIZE) return false; // too large for non-static arena
  size_t arenas_count;
  mi_arena_t* arenas = mi_arena_get(kind, is_static, &arenas_count);
  if (arenas_count==0) return false;
  size_t start_idx = (is_static ? 0 : tld->arena_idx[kind]); // for non-static arena's remember the search index
  if (mi_arenas_try_claim(arenas, arenas_count, start_idx, numa_node, blocks, arena_idx, bitmap_idx)) {
    if (!is_static) tld->arena_idx[kind] = *arena_idx;
    return true;
  }
  else return false;
}


/* -----------------------------------------------------------
  Add fresh arena from the OS
----------------------------------------------------------- */

static bool mi_arena_add(mi_arena_kind_t kind, bool is_static, void* start, size_t blocks, int numa_node, 
                         size_t claim_blocks, mi_bitmap_t bitmap_map, mi_bitmap_t bitmap_com, size_t* arena_idx, mi_bitmap_index_t* bitmap_idx) 
{
  mi_assert_internal((is_static && claim_blocks==0 && bitmap_idx==NULL) ||
                     (!is_static && blocks==MI_ARENA_BLOCK_COUNT && bitmap_map==NULL && bitmap_com==NULL));
  mi_assert_internal(claim_blocks<=blocks);
  mi_assert_internal(claim_blocks==0 || bitmap_idx!=NULL);
  mi_assert_internal(blocks<=0xFFFF);
  size_t idx = mi_atomic_increment(is_static ? &mi_arenas_static_count[kind] : &mi_arenas_count[kind]);
  if (idx>=(is_static ? MI_MAX_STATIC_ARENAS : MI_MAX_ARENAS)) {
    mi_atomic_decrement(is_static ? &mi_arenas_static_count[kind] : &mi_arenas_count[kind]);
    return false;
  }
  mi_arena_t* arenas = mi_arena_get(kind, is_static, NULL);
  mi_arena_t* arena = &arenas[idx];
  mi_assert_internal(arena->start==NULL &&mi_atomic_read_relaxed(&arena->field_count)==0);
  if (bitmap_map!=NULL) {
    mi_assert_internal(bitmap_com!=NULL);
    mi_assert_internal(claim_blocks==0);
    arena->blocks_map.bitmap = bitmap_map;
    arena->blocks_com.bitmap = bitmap_com;
  }
  else if (claim_blocks>0) {
    mi_assert_internal(claim_blocks<=blocks&&bitmap_map==NULL&&bitmap_idx != NULL);
    *bitmap_idx = 0;
    mi_bitmap_claim(&arena->blocks_map.field, 1, claim_blocks, *bitmap_idx, NULL);
  }
  arena->is_committed = (kind<=mi_arena_committed);
  arena->is_fixed = (kind<=mi_arena_fixed);
  arena->is_zero_init = true; // TODO: allow non-zero initialization?
  arena->numa_node = (numa_node + 1);
  arena->block_count = (uint16_t)blocks;
  arena->start = start;
  // after this write other threads can claim 
  
  mi_atomic_write(&arena->field_count, _mi_divide_up(blocks, MI_BITMAP_FIELD_BITS));  

  if (arena_idx!=NULL) { *arena_idx = (arena-mi_arenas); }
  return true;
}


static bool mi_arena_try_alloc_from_os(size_t blocks, bool commit, bool allow_large, size_t* arena_idx, mi_bitmap_index_t* bitmap_idx, mi_os_tld_t* tld) {
  if (blocks*MI_ARENA_BLOCK_SIZE > MI_ARENA_MAX_OBJ_SIZE) return false; // not too large?
  mi_assert_internal(MI_ARENA_SIZE>=blocks*MI_ARENA_BLOCK_SIZE);
  bool do_commit = commit && mi_option_is_enabled(mi_option_eager_region_commit);
  bool is_large = allow_large && mi_option_is_enabled(mi_option_large_os_pages);
  void* start = _mi_os_alloc_aligned(MI_ARENA_SIZE, MI_SEGMENT_ALIGN, do_commit, &is_large, tld);
  if (start==NULL) return false;
  mi_arena_kind_t kind = (is_large ? mi_arena_fixed : (do_commit ? mi_arena_committed : mi_arena_reserved));
  if (!mi_arena_add(kind, false, start, MI_ARENA_BLOCK_COUNT, _mi_os_numa_node(tld), blocks, NULL, NULL, arena_idx, bitmap_idx)) {
    _mi_os_free(start, MI_ARENA_SIZE, tld->stats);
    return false;
  }
  return true;
}


/* -----------------------------------------------------------
  Allocation from an arena
----------------------------------------------------------- */

static bool mi_arena_try_allocx(size_t blocks, bool commit, bool allow_fixed, size_t* arena_idx, mi_bitmap_index_t* bitmap_idx, mi_os_tld_t* tld)
{
  int numa_node = (_mi_os_numa_node_count()<=1 ? -1 : _mi_os_numa_node(tld));

  // try to find a slot in an existiting arena
  if (commit) {
    if (allow_fixed) {
      if (mi_arena_kind_try_alloc(mi_arena_fixed, true, numa_node, blocks, arena_idx, bitmap_idx, tld)) return true;
      if (mi_arena_kind_try_alloc(mi_arena_fixed, false, numa_node, blocks, arena_idx, bitmap_idx, tld)) return true;
    }
    if (mi_arena_kind_try_alloc(mi_arena_committed, true, numa_node, blocks, arena_idx, bitmap_idx, tld)) return true;
    if (mi_arena_kind_try_alloc(mi_arena_committed, false, numa_node, blocks, arena_idx, bitmap_idx, tld)) return true;
  }
  if (mi_arena_kind_try_alloc(mi_arena_reserved, true, numa_node, blocks, arena_idx, bitmap_idx, tld)) return true;
  if (mi_arena_kind_try_alloc(mi_arena_reserved, false, numa_node, blocks, arena_idx, bitmap_idx, tld)) return true;

  // failed to allocate, try to allocate on a different numa node for static arenas
  if (numa_node>=0) {
    if (commit) {
      if (allow_fixed) {
        if (mi_arena_kind_try_alloc(mi_arena_fixed, true, -1, blocks, arena_idx, bitmap_idx, tld)) return true;
      }
      if (mi_arena_kind_try_alloc(mi_arena_committed, true, -1, blocks, arena_idx, bitmap_idx, tld)) return true;
    }
    if (mi_arena_kind_try_alloc(mi_arena_reserved, true, -1, blocks, arena_idx, bitmap_idx, tld)) return true;
  }

  // failed to allocate, try a fresh OS allocated arena 
  if (mi_arena_try_alloc_from_os(blocks,commit,allow_fixed,arena_idx,bitmap_idx,tld)) return true;

  // ah, completely failed
  return false;
}


/* -----------------------------------------------------------
  Arena Allocation
----------------------------------------------------------- */
static void* mi_arena_try_alloc(size_t blocks, bool* commit, bool* large, bool* is_zero, size_t* memid, mi_os_tld_t* tld) 
{
  if (blocks>MI_BITMAP_FIELD_BITS) return false; // TODO: for now, we cannot allocate blocks larger than this..
  if (blocks>0xFFFF) return false; // blocks must fit into a uint16_t

  // try to find a slot in an arena
  mi_bitmap_index_t bitmap_idx;
  size_t arena_idx;
  if (!mi_arena_try_allocx(blocks, *commit, *large, &arena_idx, &bitmap_idx, tld)) return NULL;

  // success!
  mi_arena_t* arena = &mi_arenas[arena_idx];
  mi_assert_internal(arena->start!=NULL&&mi_atomic_read_relaxed(&arena->field_count)>0&&arena->block_count>=blocks);
  mi_assert_internal(mi_bitmap_is_claimed(mi_arena_bitmap_map(arena), mi_atomic_read_relaxed(&arena->field_count), blocks, bitmap_idx));
  
  void* p = (uint8_t*)arena->start + (mi_bitmap_index_bit(bitmap_idx)*MI_ARENA_BLOCK_SIZE);
  *large = arena->is_fixed;
  *memid = mi_memid_create(arena_idx, bitmap_idx);

  bool any_zero = false;
  *is_zero = mi_bitmap_claim(mi_arena_bitmap_com(arena), mi_atomic_read_relaxed(&arena->field_count), blocks, bitmap_idx, &any_zero);
  if (!mi_option_is_enabled(mi_option_eager_commit)) { any_zero = true; } // if no eager commit, even dirty segments can be partially committed
  if (arena->is_committed) {
    *commit = true;
  }
  else if (*commit && any_zero) {
    bool commit_zero;
    _mi_os_commit(p, blocks*MI_ARENA_BLOCK_SIZE, &commit_zero, tld->stats);
  }

  return p;  
}

void* _mi_arena_alloc_aligned(size_t size, size_t alignment, 
                              bool* commit, bool* large, bool* is_zero, 
                              size_t* memid, mi_os_tld_t* tld) 
{
  mi_assert_internal(memid != NULL && tld != NULL);
  mi_assert_internal(size > 0);
  *memid   = MI_MEMID_OS;
  *is_zero = false;
  bool default_large = false;
  if (large==NULL) large = &default_large;  // ensure `large != NULL`

  // try to allocate in an arena if the alignment is small enough
  // and the object is not too large or too small.
  if (alignment<=MI_SEGMENT_ALIGN) {
    const size_t blocks = mi_block_count_of_size(size);
    void* p = mi_arena_try_alloc(blocks, commit, large, is_zero, memid, tld);
    if (p!=NULL) return p;
  }

  // otherwise, fall back to the OS
  *is_zero = true;
  *memid   = MI_MEMID_OS;
  if (*large) {
    *large = mi_option_is_enabled(mi_option_large_os_pages); // try large OS pages only if enabled and allowed
  }
  return _mi_os_alloc_aligned(size, alignment, *commit, large, tld);
}

void* _mi_arena_alloc(size_t size, bool* commit, bool* large, bool* is_zero, size_t* memid, mi_os_tld_t* tld)
{
  return _mi_arena_alloc_aligned(size, MI_SEGMENT_ALIGN, commit, large, is_zero, memid, tld);
}


/* -----------------------------------------------------------
  Arena free
----------------------------------------------------------- */

void _mi_arena_free(void* p, size_t size, size_t memid, mi_stats_t* stats) {
  mi_assert_internal(size > 0 && stats != NULL);
  if (p==NULL) return;
  if (size==0) return;
  if (memid == MI_MEMID_OS) {
    // was a direct OS allocation, pass through
    _mi_os_free(p, size, stats);
  }
  else {
    // allocated in an arena
    size_t arena_idx;
    size_t bitmap_idx;
    mi_memid_indices(memid, &arena_idx, &bitmap_idx);
    mi_assert_internal(arena_idx < MI_MAX_ARENAS_COUNT);
    mi_arena_t* arena = (mi_arena_t*)&mi_arenas[arena_idx];
    uintptr_t field_count = mi_atomic_read_relaxed(&arena->field_count);
    mi_assert_internal(arena != NULL);
    mi_assert_internal(field_count > mi_bitmap_index_field(bitmap_idx));
    if (field_count <= mi_bitmap_index_field(bitmap_idx)) {
      _mi_fatal_error("trying to free from non-existent arena block: %p, size %zu, memid: 0x%zx\n", p, size, memid);
      return;
    }
    const size_t blocks = mi_block_count_of_size(size);
    bool all_ones = mi_bitmap_unclaim(mi_arena_bitmap_map(arena), field_count, blocks, bitmap_idx);
    if (!all_ones) {
      _mi_fatal_error("trying to free an already freed block: %p, size %zu\n", p, size);
      return;
    };
  }
}


/* -----------------------------------------------------------
  Reserve a huge page arena.
----------------------------------------------------------- */
#include <errno.h> // ENOMEM

// reserve at a specific numa node
int mi_reserve_huge_os_pages_at(size_t pages, int numa_node, size_t timeout_msecs) mi_attr_noexcept {
  if (pages==0) return 0;
  if (numa_node < -1) numa_node = -1;
  if (numa_node >= 0) numa_node = numa_node % _mi_os_numa_node_count();
  size_t hsize = 0;
  size_t pages_reserved = 0;
  void* p = _mi_os_alloc_huge_os_pages(pages, numa_node, timeout_msecs, &pages_reserved, &hsize);
  if (p==NULL || pages_reserved==0) {
    _mi_warning_message("failed to reserve %zu gb huge pages\n", pages);
    return ENOMEM;
  }
  _mi_verbose_message("reserved %zu gb huge pages\n", pages_reserved);
  
  size_t bcount = mi_block_count_of_size(hsize);
  size_t fields = _mi_divide_up(bcount , MI_BITMAP_FIELD_BITS);
  size_t bmsize = fields*sizeof(mi_bitmap_field_t);  
  mi_bitmap_t bm = (mi_bitmap_t)_mi_os_alloc(2*bmsize, &_mi_stats_main); // TODO: can we avoid allocating from the OS?
  if (bm == NULL) {
    _mi_os_free_huge_pages(p, hsize, &_mi_stats_main);
    return ENOMEM;
  }
  mi_bitmap_t bm_map = bm;
  mi_bitmap_t bm_com = (mi_bitmap_t)((uint8_t*)bm + bmsize);
  // the bitmaps are already zero initialized due to os_alloc
  // just claim leftover blocks if needed
  size_t post = (fields*MI_BITMAP_FIELD_BITS)-bcount;
  if (post>0) {
    // don't use leftover bits at the end
    mi_bitmap_index_t postidx = mi_bitmap_index_create(fields-1, MI_BITMAP_FIELD_BITS-post);
    mi_bitmap_claim(bm_map, fields, post, postidx, NULL);
    mi_bitmap_claim(bm_com, fields, post, postidx, NULL);
  }
  if (!mi_arena_add(mi_arena_fixed, true, p, bcount, numa_node, 0, bm_map, bm_com, NULL, NULL)) {
    // out of arenas!
    _mi_os_free((void*)bm, 2*bmsize, &_mi_stats_main);
    _mi_os_free_huge_pages(p, hsize, &_mi_stats_main);
    return ENOMEM;
  }
  return 0;
}


// reserve huge pages evenly among all numa nodes. 
int mi_reserve_huge_os_pages_interleave(size_t pages, size_t timeout_msecs) mi_attr_noexcept {
  if (pages == 0) return 0;

  // pages per numa node
  int numa_count = _mi_os_numa_node_count();
  if (numa_count <= 0) numa_count = 1;
  const size_t pages_per = pages / numa_count;
  const size_t pages_mod = pages % numa_count;
  const size_t timeout_per = (timeout_msecs / numa_count) + 50;
  
  // reserve evenly among numa nodes
  for (int numa_node = 0; numa_node < numa_count && pages > 0; numa_node++) {
    size_t node_pages = pages_per;  // can be 0
    if ((size_t)numa_node < pages_mod) node_pages++;
    int err = mi_reserve_huge_os_pages_at(node_pages, numa_node, timeout_per);
    if (err) return err;
    if (pages < node_pages) {
      pages = 0;
    }
    else {
      pages -= node_pages;
    }
  }

  return 0;
}

int mi_reserve_huge_os_pages(size_t pages, double max_secs, size_t* pages_reserved) mi_attr_noexcept {
  UNUSED(max_secs);
  _mi_warning_message("mi_reserve_huge_os_pages is deprecated: use mi_reserve_huge_os_pages_interleave/at instead\n");
  if (pages_reserved != NULL) *pages_reserved = 0;
  int err = mi_reserve_huge_os_pages_interleave(pages, (size_t)(max_secs * 1000.0));  
  if (err==0 && pages_reserved!=NULL) *pages_reserved = pages;
  return err;
}


static bool mi_arena_contains(mi_arena_kind_t kind, bool is_static, const void* p) {
  size_t arena_count;
  mi_arena_t* arena = mi_arena_get(kind, is_static, &arena_count);
  for (; arena_count>0; arena++, arena_count--) {
    void* start = arena->start;
    void* end = (uint8_t*)start + (arena->block_count*MI_ARENA_BLOCK_SIZE);
    if (p >= start && p <= end) return true;
  }
  return false;
}

bool mi_is_in_heap_region(const void* p) mi_attr_noexcept {
  for (int kind = 0; kind < MI_ARENA_KINDS; kind++) {
    if (mi_arena_contains((mi_arena_kind_t)kind, true, p)) return true;
    if (mi_arena_contains((mi_arena_kind_t)kind, false, p)) return true;
  }
  return false;
}

void _mi_arena_collect(mi_stats_t* stats) {
  // TODO
  return;
}
