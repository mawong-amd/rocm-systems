/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#ifndef HSA_RUNTIME_CORE_INC_INTERCEPT_QUEUE_LOGIC_H_
#define HSA_RUNTIME_CORE_INC_INTERCEPT_QUEUE_LOGIC_H_

#include <cstdint>

// Pure decision logic for InterceptQueue::Submit, factored out so it can be unit
// tested without the HSA runtime or a GPU (regression coverage for the retry/overflow fix).
namespace rocr {
namespace core {
namespace intercept_queue_logic {

struct SubmitPlan {
  // Number of (non-marker) packets to copy to the wrapped queue this iteration.
  uint64_t submitted_count;
  // Whether a fresh retry barrier must be inserted to drain the remaining packets.
  bool insert_retry_barrier;
};

// Decide how many packets fit now and whether a retry barrier is needed. write/read are read
// non-atomically, so a transient read > write is clamped to "queue full" (no underflow/OOB).
//
// Wake-up rule: once packets wait in overflow_, only the completion handler of the retry
// barrier resumes forwarding, and when it runs it must see a free slot (to submit into, or to
// arm the next barrier in); a full ring leaves it with no way to make progress and nothing
// rings the doorbell again. The handler runs after the barrier was consumed and with a read
// index that may lag the command processor, so every plan leaves one spare slot free as seen
// through the read index it used: data placed behind a freshly inserted barrier and top-ups
// made while a retry is pending never fill the ring. The one exception is a barrier that
// takes the only free slot (see insert_retry_barrier): the handler then relies on the read
// index having advanced by one slot, which holds for any ring where it trails the command
// processor by fewer than a ring's worth of packets (GPU rings have at least 64 slots).
inline SubmitPlan PlanSubmit(uint64_t write, uint64_t read, uint64_t qsize, uint64_t count,
                             uint64_t marker_count, bool pending_retry_point,
                             bool overflow_nonempty) {
  uint64_t inflight = (write >= read) ? (write - read) : qsize;
  if (inflight > qsize) inflight = qsize;
  uint64_t free_slots = qsize - inflight;

  const uint64_t non_marker = count - marker_count;
  uint64_t submitted_count = non_marker;

  // Slots this plan leaves unused: the spare above, plus the slot of the retry barrier when one
  // is inserted (none is while a retry is pending).
  const uint64_t reserve = pending_retry_point ? 1 : 2;

  if (submitted_count >= qsize) {
    // Submit what fits, keeping the reserve (saturating: free_slots may be 0).
    submitted_count = (free_slots > reserve) ? (free_slots - reserve) : 0;
  } else if (free_slots < submitted_count + 1) {
    // Out of space: prefer all-or-nothing, but when draining overflow submit what fits.
    // Require strictly more free slots than the reserve so at least one data packet is
    // actually drained.
    submitted_count = (overflow_nonempty && free_slots > reserve) ? (free_slots - reserve) : 0;
  }

  // packets[] has exactly non_marker non-marker entries; never copy more than that.
  if (submitted_count > non_marker) submitted_count = non_marker;

  // Need a replacement barrier only when packets remain, none is pending, and a slot is
  // free (a completed-but-unread barrier may still occupy its slot). The barrier may take the
  // last free slot: refusing would leave the remaining packets with no wake-up at all.
  const bool insert_retry_barrier =
      submitted_count < non_marker && !pending_retry_point && free_slots >= 1;

  return SubmitPlan{submitted_count, insert_retry_barrier};
}

}  // namespace intercept_queue_logic
}  // namespace core
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_INTERCEPT_QUEUE_LOGIC_H_
