/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTIME_AMD_TOOL_EVENTS_H_
#define HSA_RUNTIME_AMD_TOOL_EVENTS_H_

// Insert license header

#include <stddef.h>
#include <stdint.h>
#include "hsa.h"


typedef enum {
  HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_NONE = 0,
  HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_USE_ONCE =
      (1 << 0),  // This scratch allocation is only valid for 1 dispatch.
  HSA_AMD_EVENT_SCRATCH_ALLOC_FLAG_ALT =
      (1 << 1),  // Used alternate scratch instead of main scratch
} hsa_amd_event_scratch_alloc_flag_t;

typedef enum {
  HSA_AMD_TOOL_EVENT_MIN = 0,

  // Scratch memory tracking
  HSA_AMD_TOOL_EVENT_SCRATCH_ALLOC_START,
  HSA_AMD_TOOL_EVENT_SCRATCH_ALLOC_END,
  HSA_AMD_TOOL_EVENT_SCRATCH_FREE_START,
  HSA_AMD_TOOL_EVENT_SCRATCH_FREE_END,
  HSA_AMD_TOOL_EVENT_SCRATCH_ASYNC_RECLAIM_START,
  HSA_AMD_TOOL_EVENT_SCRATCH_ASYNC_RECLAIM_END,

  // Queries.  Unlike the notifications above, the runtime reads an answer back out of
  // the event structure.  A tool that does not implement a query must leave the out
  // fields untouched; the runtime pre-fills them with the answer that preserves its
  // behaviour in the absence of any tool.
  HSA_AMD_TOOL_EVENT_QUERY_SIGNAL_HOST_RMW,

  // Add new events above ^
  HSA_AMD_TOOL_EVENT_MAX
} hsa_amd_tool_event_kind_t;

typedef struct {
  hsa_amd_tool_event_kind_t kind;
} hsa_amd_tool_event_none_t;

typedef struct {
  hsa_amd_tool_event_kind_t kind;
  const hsa_queue_t* queue;
  hsa_amd_event_scratch_alloc_flag_t flags;
  uint64_t dispatch_id;  // Dispatch ID of the AQL packet that needs more scratch memory
} hsa_amd_event_scratch_alloc_start_t;

typedef struct {
  hsa_amd_tool_event_kind_t kind;
  const hsa_queue_t* queue;
  hsa_amd_event_scratch_alloc_flag_t flags;
  uint64_t dispatch_id;  // Dispatch ID of the AQL packet that needs more scratch memory
  size_t size;           // Amount of scratch allocated - in bytes
  size_t num_slots;      // limit of number of waves
} hsa_amd_event_scratch_alloc_end_t;

typedef struct {
  hsa_amd_tool_event_kind_t kind;
  const hsa_queue_t* queue;
  hsa_amd_event_scratch_alloc_flag_t flags;
} hsa_amd_event_scratch_free_start_t;

typedef struct {
  hsa_amd_tool_event_kind_t kind;
  const hsa_queue_t* queue;
  hsa_amd_event_scratch_alloc_flag_t flags;
} hsa_amd_event_scratch_free_end_t;

typedef struct {
  hsa_amd_tool_event_kind_t kind;
  const hsa_queue_t* queue;
  hsa_amd_event_scratch_alloc_flag_t flags;
} hsa_amd_event_scratch_async_reclaim_start_t;

typedef struct {
  hsa_amd_tool_event_kind_t kind;
  const hsa_queue_t* queue;
  hsa_amd_event_scratch_alloc_flag_t flags;
} hsa_amd_event_scratch_async_reclaim_end_t;

/**
 * Asks the tool whether it may perform a host read-modify-write atomic on a completion
 * signal that the application placed on a packet it submitted.  Tools that interpose a
 * queue and reference-count the packets they observe do exactly that.
 *
 * This matters because a signal's value word can live in device memory
 * (::hsa_amd_signal_create_v2 with ::HSA_AMD_SIGNAL_CREATE_DEVICE_MEM_VALUE_WORD).  A
 * host lock-prefixed RMW against a device memory aperture is not promoted to a PCIe
 * atomic on x86: it degrades to a non-atomic read-then-write and can lose the update the
 * GPU made in between.  A producer that knows a tool will do this can keep its signals
 * out of the packets the tool rewrites, at some cost, and a producer that knows no tool
 * will do it can avoid paying that cost.
 *
 * The answer is a property of the tool, not of any one signal or queue, and the runtime
 * asks on demand rather than caching, so a tool that changes its mind is reflected the
 * next time the question is asked.  The answer must be CONSERVATIVE: a tool answers true
 * if it may do this at any point for as long as the answer stands, not only if it is
 * doing it at the instant it is asked.
 */
typedef struct {
  hsa_amd_tool_event_kind_t kind;
  /* OUT.  Non-zero iff the tool may host-RMW an application completion signal.  The
   * runtime pre-fills this with 0 and a tool that does not implement the query leaves
   * it that way, which is the answer that preserves pre-query behaviour. */
  uint8_t host_rmw_on_completion_signal;
} hsa_amd_tool_event_query_signal_host_rmw_t;

typedef union {
  const hsa_amd_tool_event_none_t* none;
  const hsa_amd_event_scratch_alloc_start_t* scratch_alloc_start;
  const hsa_amd_event_scratch_alloc_end_t* scratch_alloc_end;
  const hsa_amd_event_scratch_free_start_t* scratch_free_start;
  const hsa_amd_event_scratch_free_end_t* scratch_free_end;
  const hsa_amd_event_scratch_async_reclaim_start_t* scratch_async_reclaim_start;
  const hsa_amd_event_scratch_async_reclaim_end_t* scratch_async_reclaim_end;
  /* Not const: a query event carries the tool's answer back to the runtime. */
  hsa_amd_tool_event_query_signal_host_rmw_t* query_signal_host_rmw;
} hsa_amd_tool_event_t;

typedef hsa_status_t (*hsa_amd_tool_event)(hsa_amd_tool_event_t);


#endif