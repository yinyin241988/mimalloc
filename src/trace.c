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
  mi_ev_malloc,
  mi_ev_realloc,
  mi_ev_malloc_aligned,
  mi_ev_realloc_aligned
} mi_event_kind_t;

typedef struct mi_event_s {
  uintptr_t kind:3;
  uintptr_t ptr:45;
  uintptr_t size_lo:16;
  // only for realloc and large sizes
  uintptr_t padding:3;
  uintptr_t ptr_in:45;
  uintptr_t size_hi:16;
  // only for aligned malloc/realloc
  uintptr_t align:32;
  uintptr_t offset:32;
} mi_event_t;

typedef enum mi_trace_kind_e {
  mi_trace_normal
} mi_trace_kind_t;

#define MI_CHUNK_HEADER_SIZE    (32)
#define MI_CHUNK_FULL_SIZE      (64*1024)
#define MI_CHUNK_MAX_DATASIZE   (MI_CHUNK_FULL_SIZE - MI_CHUNK_HEADER_SIZE)  // 64KiB per chunk

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
  return (mi_trace_chunk_t*)tld->trace;
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
  size_t size_lo = (size >> MI_INTPTR_SHIFT) & 0xFFFF;
  size_t size_hi = (size >> (16 + MI_INTPTR_SHIFT));
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
}

void _mi_trace_process_done() {
  if (ftrace==NULL) return;
  fflush(ftrace);
  fclose(ftrace);
}

void _mi_trace_thread_init(mi_tld_t* tld) {
  tld->trace = NULL;
  if (ftrace==NULL) return;
  mi_trace_chunk_t* chunk = _mi_os_alloc(MI_CHUNK_FULL_SIZE, &tld->stats);
  chunk->datasize = 0;
  chunk->thread_id = _mi_thread_id();
  chunk->timestamp = mi_trace_timestamp();
  chunk->adler = 0;
  chunk->kind = 0;
  tld->trace = chunk;
  mi_trace(chunk, mi_ev_free, NULL, 1, NULL, 0, 0); // start-of-thread
}

void _mi_trace_thread_done(mi_tld_t* tld) {
  mi_trace_chunk_t* chunk = mi_trace_chunk(tld);
  if (chunk==NULL) return;
  mi_trace(chunk, mi_ev_free, NULL, 2, NULL, 0, 0); // end-of-thread
  mi_trace_write(chunk);
  _mi_os_free(chunk, MI_CHUNK_FULL_SIZE, &tld->stats);
  tld->trace = NULL;
}



#endif // MI_TRACE
