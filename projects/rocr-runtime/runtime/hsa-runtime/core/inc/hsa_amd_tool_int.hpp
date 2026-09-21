#ifndef HSA_RUNTIME_INC_HSA_TOOL_HOOK_IMPL_H
#define HSA_RUNTIME_INC_HSA_TOOL_HOOK_IMPL_H

#include "inc/hsa_amd_tool.h"
#include "runtime.h"

// namespace rocr::AMD::tool {  // C++17
namespace rocr { namespace AMD { namespace tool {

using scratch_alloc_flag = hsa_amd_event_scratch_alloc_flag_t;

__forceinline void notify_event_scratch_alloc_start(const hsa_queue_t* queue,
                                                    scratch_alloc_flag flag, uint64_t dispatch_id);

__forceinline void notify_event_scratch_alloc_end(const hsa_queue_t* queue, scratch_alloc_flag flag,
                                                  uint64_t dispatch_id, size_t size,
                                                  size_t num_slots);

__forceinline void notify_event_scratch_free_start(const hsa_queue_t* queue,
                                                   scratch_alloc_flag flag);

__forceinline void notify_event_scratch_free_end(const hsa_queue_t* queue, scratch_alloc_flag flag);

__forceinline void notify_event_scratch_async_reclaim_start(const hsa_queue_t* queue,
                                                            scratch_alloc_flag flag);

__forceinline void notify_event_scratch_async_reclaim_end(const hsa_queue_t* queue,
                                                          scratch_alloc_flag flag);


/// Ask the loaded tool, if any, whether it may perform a host read-modify-write atomic
/// on an application completion signal.  False when no tool is loaded, when the tool
/// predates the query, or when the tool answers no.  Asked on demand and never cached
/// here: the answer is the tool's to change.
__forceinline bool query_signal_host_rmw();


// Impl

__forceinline void notify_event_scratch_alloc_start(const hsa_queue_t* queue,
                                                    scratch_alloc_flag flags,
                                                    uint64_t dispatch_id) {
  const auto& tool_table = core::hsa_api_table().tools_api;
  if (!tool_table.hsa_amd_tool_scratch_event_alloc_start_fn) {
    return;
  }

  auto event = hsa_amd_event_scratch_alloc_start_t{.kind = HSA_AMD_TOOL_EVENT_SCRATCH_ALLOC_START,
                                                   .queue = queue,
                                                   .flags = flags,
                                                   .dispatch_id = dispatch_id};

  tool_table.hsa_amd_tool_scratch_event_alloc_start_fn(
      hsa_amd_tool_event_t{.scratch_alloc_start = &event});
}

__forceinline void notify_event_scratch_alloc_end(const hsa_queue_t* queue,
                                                  scratch_alloc_flag flags, uint64_t dispatch_id,
                                                  size_t size, size_t num_slots) {
  const auto& tool_table = core::hsa_api_table().tools_api;
  if (!tool_table.hsa_amd_tool_scratch_event_alloc_end_fn) {
    return;
  }

  auto event = hsa_amd_event_scratch_alloc_end_t{
      .kind = HSA_AMD_TOOL_EVENT_SCRATCH_ALLOC_END,
      .queue = queue,
      .flags = flags,
      .dispatch_id = dispatch_id,
      .size = size,
      .num_slots = num_slots,
  };

  tool_table.hsa_amd_tool_scratch_event_alloc_end_fn(
      hsa_amd_tool_event_t{.scratch_alloc_end = &event});
}

__forceinline void notify_event_scratch_free_start(const hsa_queue_t* queue,
                                                   scratch_alloc_flag flags) {
  const auto& tool_table = core::hsa_api_table().tools_api;
  if (!tool_table.hsa_amd_tool_scratch_event_free_start_fn) {
    return;
  }

  auto event = hsa_amd_event_scratch_free_start_t{
      .kind = HSA_AMD_TOOL_EVENT_SCRATCH_FREE_START,
      .queue = queue,
      .flags = flags,
  };

  tool_table.hsa_amd_tool_scratch_event_free_start_fn(
      hsa_amd_tool_event_t{.scratch_free_start = &event});
}

__forceinline void notify_event_scratch_free_end(const hsa_queue_t* queue,
                                                 scratch_alloc_flag flags) {
  const auto& tool_table = core::hsa_api_table().tools_api;
  if (!tool_table.hsa_amd_tool_scratch_event_free_end_fn) {
    return;
  }

  auto event = hsa_amd_event_scratch_free_end_t{
      .kind = HSA_AMD_TOOL_EVENT_SCRATCH_FREE_END,
      .queue = queue,
      .flags = flags,
  };

  tool_table.hsa_amd_tool_scratch_event_free_end_fn(
      hsa_amd_tool_event_t{.scratch_free_end = &event});
}

__forceinline void notify_event_scratch_async_reclaim_start(const hsa_queue_t* queue,
                                                            scratch_alloc_flag flags) {
  const auto& tool_table = core::hsa_api_table().tools_api;
  if (!tool_table.hsa_amd_tool_scratch_event_async_reclaim_start_fn) {
    return;
  }

  auto event = hsa_amd_event_scratch_async_reclaim_start_t{
      .kind = HSA_AMD_TOOL_EVENT_SCRATCH_ASYNC_RECLAIM_START,
      .queue = queue,
      .flags = flags,
  };

  tool_table.hsa_amd_tool_scratch_event_async_reclaim_start_fn(
      hsa_amd_tool_event_t{.scratch_async_reclaim_start = &event});
}

__forceinline void notify_event_scratch_async_reclaim_end(const hsa_queue_t* queue,
                                                          scratch_alloc_flag flags) {
  const auto& tool_table = core::hsa_api_table().tools_api;
  if (!tool_table.hsa_amd_tool_scratch_event_async_reclaim_end_fn) {
    return;
  }

  auto event = hsa_amd_event_scratch_async_reclaim_end_t{
      .kind = HSA_AMD_TOOL_EVENT_SCRATCH_ASYNC_RECLAIM_END,
      .queue = queue,
      .flags = flags,
  };

  tool_table.hsa_amd_tool_scratch_event_async_reclaim_end_fn(
      hsa_amd_tool_event_t{.scratch_async_reclaim_end = &event});
}

__forceinline bool query_signal_host_rmw() {
  const auto& tool_table = core::hsa_api_table().tools_api;

  // No size check here on purpose: this table is the runtime's own instance, so its
  // size is this build's sizeof(ToolsApiTable) by construction and any check against it
  // could not fail.  The version skew that is real runs the other way -- a tool built
  // against an older header -- and that tool simply never writes the slot, leaving it
  // null.  The tool side is where the size guard belongs, and is where it is.
  if (!tool_table.hsa_amd_tool_query_signal_host_rmw_fn) {
    return false;
  }

  // Pre-filled with the answer that preserves behaviour in the absence of a tool, so a
  // tool that returns an error, or one that ignores the out field, cannot turn the
  // feature on by accident.
  auto event = hsa_amd_tool_event_query_signal_host_rmw_t{
      .kind = HSA_AMD_TOOL_EVENT_QUERY_SIGNAL_HOST_RMW,
      .host_rmw_on_completion_signal = 0,
  };

  if (tool_table.hsa_amd_tool_query_signal_host_rmw_fn(
          hsa_amd_tool_event_t{.query_signal_host_rmw = &event}) != HSA_STATUS_SUCCESS) {
    return false;
  }
  return event.host_rmw_on_completion_signal != 0;
}

// }  // namespace rocr::AMD::tool
}  // namespace rocr
}  // namespace AMD
}  // namespace tool

#endif