/* ----------------------------------------------------------------------------
Copyright (c) 2018,2019, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/

#include "mimalloc.h"
#include "mimalloc-internal.h"
#include "mimalloc-atomic.h"

#ifdef MI_TRACE

#if (MI_INTPTR_SIZE != 8)
#error "tracing can only be enabled for 64-bit compilation"
#endif

typedef enum mi_event_kind_e {
  mi_ev_none = 0,
  mi_ev_free,
  mi_ev_malloc_small,
  mi_ev_free_mt,
  mi_ev_malloc,
  mi_ev_realloc,
  mi_ev_malloc_aligned,
  mi_ev_realloc_aligned
} mi_event_kind_t;

typedef enum mi_barrier_e {
  mi_barrier_generic = 1,
  mi_barrier_thread_new,
  mi_barrier_thread_done
} mi_barrier_t;

typedef struct mi_event_s {
  uintptr_t kind:3;
  uintptr_t ptr:45;     
  uintptr_t size_lo:16; // size is the logical time for free_mt
  // only for realloc, malloc large sizes, or multi-threaded free
  uintptr_t padding:3;
  uintptr_t ptr_in:45;  // ptr_in is the thread id for free_mt 
  uintptr_t size_hi:16;
  // only for aligned malloc/realloc
  uintptr_t align:32;
  uintptr_t offset:32;
} mi_event_t;

typedef enum mi_trace_kind_e {
  mi_trace_normal
} mi_trace_kind_t;

#define MI_CHUNK_HEADER_SIZE    (32)
#define MI_CHUNK_FULL_SIZE      (8*1024)  // 8KiB per chunk
#define MI_CHUNK_MAX_DATASIZE   (MI_CHUNK_FULL_SIZE - MI_CHUNK_HEADER_SIZE)  

typedef struct mi_trace_chunk_s {
  uint64_t datasize;
  uint64_t timestamp;
  uint64_t thread_id;
  uint32_t kind;
  uint32_t adler;
  uint8_t  data[MI_CHUNK_MAX_DATASIZE];    
} mi_trace_chunk_t;


static uintptr_t mi_trace_timestamp() {
  static volatile uintptr_t timestamp = 0;
  return mi_atomic_increment(&timestamp);
}

static mi_trace_chunk_t* mi_trace_chunk(const mi_tld_t* tld) {
  return (tld==NULL ? NULL : (mi_trace_chunk_t*)tld->trace); // tld can be NULL during cleanup in heap_done.
}

static mi_trace_chunk_t* mi_heap_chunk(const mi_heap_t* heap) {
  return mi_trace_chunk(heap->tld);
}

// --------------------------------------------------------
// Tracing
// --------------------------------------------------------

static FILE* ftrace = NULL;


static void mi_trace_write(mi_trace_chunk_t* chunk) {
  if (chunk == NULL || ftrace == NULL) return;
  if (chunk->datasize < MI_CHUNK_MAX_DATASIZE && chunk->datasize + 32 > MI_CHUNK_MAX_DATASIZE) {
    // pad it out
    while (chunk->datasize < MI_CHUNK_MAX_DATASIZE) {
      chunk->data[chunk->datasize++] = 0;
    }
  }
  // todo: set adler
  size_t towrite = chunk->datasize + MI_CHUNK_HEADER_SIZE;
  size_t written = fwrite(chunk, 1, towrite, ftrace);
  if (written != towrite) _mi_warning_message("could not write to trace file\n");
  fflush(ftrace); 
  chunk->datasize = 0;
  chunk->timestamp = mi_trace_timestamp();
}

static void mi_event_emit(mi_trace_chunk_t* chunk, const mi_event_t* ev, size_t evsize) {
  if (chunk==NULL) return;
  if (chunk->datasize + evsize >= MI_CHUNK_MAX_DATASIZE) {
    mi_trace_write(chunk);
    mi_assert(chunk->datasize + evsize < MI_CHUNK_MAX_DATASIZE);
  }
  memcpy(&chunk->data[chunk->datasize], ev, evsize);
  chunk->datasize += evsize;
  mi_assert(chunk->datasize < MI_CHUNK_MAX_DATASIZE);
}

static void mi_trace(mi_trace_chunk_t* chunk, mi_event_kind_t ekind, const void* p, size_t size, const void* pin, size_t align, size_t offset) {
  mi_assert_internal( ((uintptr_t)p % MI_INTPTR_SIZE) == 0 ); // pointer must be aligned
  if (chunk==NULL) return; 
  if (size > UINT32_MAX) size = UINT32_MAX;  // trace at most 4Gb sizes for now.

  size_t size_lo = size & 0xFFFF;
  size_t size_hi = (size >> 16);
  if (ekind == mi_ev_malloc && size_hi == 0) ekind = mi_ev_malloc_small;

  // make compressed event record
  size_t evsize = sizeof(uintptr_t);
  mi_event_t ev;
  ev.kind = ekind;
  ev.ptr  = ((uintptr_t)p) >> MI_INTPTR_SHIFT;
  ev.size_lo = size_lo;
  if (ekind > mi_ev_malloc_small) {
    ev.padding = 0;
    ev.ptr_in  = ((uintptr_t)pin) >> MI_INTPTR_SHIFT;
    ev.size_hi = size_hi;
    if (ekind < mi_ev_malloc_aligned) {
      evsize = 2*evsize;
    }
    else {
      evsize = 3*evsize;
      ev.align =align;
      ev.offset = offset;
    }
  }

  // and "write" it out
  mi_event_emit(chunk,&ev,evsize);
}

// --------------------------------------------------------
// Trace API
// --------------------------------------------------------

void _mi_trace_free(mi_heap_t* heap, const void* p) {
  mi_trace(mi_heap_chunk(heap), mi_ev_free, p, 0, NULL, 0, 0);
}

void _mi_trace_malloc(mi_heap_t* heap, const void* p, size_t size) {
  mi_trace(mi_heap_chunk(heap), mi_ev_malloc, p, size, NULL, 0, 0);
}

void _mi_trace_realloc(mi_heap_t* heap, const void* p, const void* pin, size_t newsize) {
  mi_trace(mi_heap_chunk(heap), mi_ev_realloc, p, newsize, pin, 0, 0);
}

void _mi_trace_malloc_aligned(mi_heap_t* heap, const void* p, size_t size, size_t align, size_t offset) {
  mi_trace(mi_heap_chunk(heap), mi_ev_malloc_aligned, p, size, NULL, align, offset);
}

void _mi_trace_realloc_aligned(mi_heap_t* heap, const void* p, const void* pin, size_t size, size_t align, size_t offset) {
  mi_trace(mi_heap_chunk(heap), mi_ev_realloc_aligned, p, size, pin, align, offset);
}


// Multi-threaded free increments the logical clock; use special pointers for 
// thread new, done, and the slow path (to replay more in sync).
void _mi_trace_free_mt(mi_heap_t* heap, const void* p, uintptr_t thread_id) {
  static volatile uintptr_t lclock = 0;
  uintptr_t ltime = mi_atomic_increment(&lclock);
  mi_trace(mi_heap_chunk(heap), mi_ev_free_mt, p, ltime - 1, (const void*)thread_id, 0, 0);
}

static void mi_trace_barrierx(mi_heap_t* heap, mi_barrier_t barrier ) {
  _mi_trace_free_mt(heap, (void*)((uintptr_t)barrier<<MI_INTPTR_SHIFT), _mi_thread_id());
}

void _mi_trace_barrier() { 
  mi_trace_barrierx(mi_get_default_heap(),mi_barrier_generic);
}


// --------------------------------------------------------
// Initialization
// --------------------------------------------------------

void _mi_trace_process_init() {
  #pragma warning(suppress:4996)
  const char* fname = getenv("MIMALLOC_TRACE_FILE");
  if (fname==NULL || fname[0]=='0') return;
  #pragma warning(suppress:4996)
  ftrace = fopen(fname, "wb");
  if (ftrace==NULL) _mi_warning_message("unable to open trace file for writing: %s\n", fname);
  const char* header = "mimalloctrace64"; // 16 bytes with 0
  fwrite(header,1,strlen(header)+1,ftrace);
  _mi_trace_thread_init();
}

void _mi_trace_process_done() {
  _mi_trace_thread_done();
  if (ftrace==NULL) return;
  fflush(ftrace);
  fclose(ftrace);
}

void _mi_trace_thread_init() {
  mi_heap_t* heap = mi_get_default_heap();
  mi_tld_t* tld = heap->tld;
  tld->trace = NULL;
  if (ftrace==NULL) return;
  mi_trace_chunk_t* chunk = _mi_os_alloc(MI_CHUNK_FULL_SIZE, &tld->stats);
  chunk->datasize = 0;
  chunk->thread_id = _mi_thread_id();
  chunk->timestamp = mi_trace_timestamp();
  chunk->adler = 0;
  chunk->kind = 0;
  tld->trace = chunk;
  // encode thread new as a free_mt on a special pointer value
  mi_trace_barrierx(heap,mi_barrier_thread_new);
}

void _mi_trace_thread_done() {
  mi_heap_t* heap = mi_get_default_heap();
  mi_tld_t* tld = heap->tld;
  mi_trace_chunk_t* chunk = mi_trace_chunk(tld);
  if (chunk==NULL) return;
  
  // encode thread done as a free_mt on a special pointer value
  mi_trace_barrierx(heap,mi_barrier_thread_done);

  // and free the chunk
  mi_trace_write(chunk);
  _mi_os_free(chunk, MI_CHUNK_FULL_SIZE, &tld->stats);
  tld->trace = NULL;
}



#endif // MI_TRACE
