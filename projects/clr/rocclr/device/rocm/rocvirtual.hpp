/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "platform/commandqueue.hpp"
#include "rocdefs.hpp"
#include "rocdevice.hpp"
#include "utils/flags.hpp"
#include "utils/nontemporal.hpp"
#include "utils/util.hpp"
#include "rocprintf.hpp"
#include "rocsched.hpp"
#include "device/device.hpp"
#include "os/os.hpp"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stack>
#include <string>
#include <thread>
#include <vector>

namespace amd::roc {
class Device;
class Memory;
struct ProfilingSignal;
class Timestamp;

constexpr static uint64_t kInvalidAqlSlot = std::numeric_limits<uint64_t>::max();

struct AqlSlotReservation {
  uint64_t start_slot;
  size_t packet_count;
  uint64_t barrier_bit_slot_before_reservation;
};

//! True while the calling thread is inside HsaAmdSignalHandler (async-events thread).
bool InAsyncSignalHandler();

// Initial HSA signal value
constexpr static hsa_signal_value_t kInitSignalValueOne = 1;

// Timeouts for HSA signal wait
constexpr static uint64_t kTimeout100us = 100 * K;
constexpr static uint64_t kUnlimitedWait = std::numeric_limits<uint64_t>::max();
constexpr static uint64_t kInvalidQueueIndex = std::numeric_limits<uint64_t>::max();

constexpr static uint64_t kTimeout4Secs = 4 * M;

inline bool WaitForSignal(hsa_signal_t signal, bool active_wait = false, bool yield = false) {
  hsa_wait_state_t wait_state = HSA_WAIT_STATE_BLOCKED;
  if (active_wait) {
    wait_state = HSA_WAIT_STATE_ACTIVE;
  }

  if (Hsa::signal_load_relaxed(signal) > 0) {
    // When it is blocked wait, we wait in active state for 100 us before proceeding to wait in
    // blocked state indefinitely.
    if (!active_wait) {
      ClPrint(amd::LOG_INFO, amd::LOG_SIG, "Host active wait for Signal = (0x%lx) for %d ns",
              signal.handle, kTimeout100us);
      if (Hsa::signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, kInitSignalValueOne,
                                    kTimeout100us, HSA_WAIT_STATE_ACTIVE) != 0) {
        if (HIP_SKIP_ABORT_ON_GPU_ERROR && amd::Device::IsGPUInError()) {
          ClPrint(amd::LOG_ERROR, amd::LOG_SIG,
                  "Device not Stable, while waiting for Signal ="
                  "(0x%lx) for %d ns",
                  signal.handle, kTimeout100us);
          return true;
        }
      }
    }

    // This is unlimited wait, but we wait for 4 secs and check if the device is
    // unstable, if so we return, otherwise we continue to wait in the while loop.
    while (Hsa::signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, kInitSignalValueOne,
                                     kTimeout4Secs, wait_state) != 0) {
      if (HIP_SKIP_ABORT_ON_GPU_ERROR && amd::Device::IsGPUInError()) {
        ClPrint(amd::LOG_ERROR, amd::LOG_SIG,
                "Device not Stable, while waiting for Signal ="
                "(0x%lx) for %d ns",
                signal.handle, kTimeout4Secs);
        return true;
      }
      if (yield && wait_state == HSA_WAIT_STATE_ACTIVE) {
        amd::Os::yield();
      }
    }
  }

  return true;
}

inline void fetchSignalTime(hsa_signal_t signal, hsa_agent_t gpu_device, uint64_t* start,
                            uint64_t* end) {
  if (start != nullptr && end != nullptr) {
    hsa_amd_profiling_dispatch_time_t time = {};
    Hsa::profiling_get_dispatch_time(gpu_device, signal, &time);
    *start = time.start;
    *end = time.end;
  }
}

// Timestamp for keeping track of some profiling information for various commands
// including EnqueueNDRangeKernel and clEnqueueCopyBuffer.
class Timestamp : public amd::ReferenceCountedObject {
 private:
  static double ticksToTime_;

  uint64_t start_;
  uint64_t end_;
  VirtualGPU* gpu_;                        //!< Virtual GPU, associated with this timestamp
  amd::Command& command_;                  //!< Command, associated with this timestamp
  amd::Command* parsedCommand_;            //!< Command down the list, considering command_ as head
  std::vector<ProfilingSignal*> signals_;  //!< The list of all signals, associated with the TS
  hsa_signal_t callback_signal_;  //!< Signal associated with a callback for possible later update
  std::recursive_mutex lock_;     //!< Serialize timestamp update
  bool accum_ena_ = false;        //!< If TRUE then the accumulation of execution times has started
  bool hasHwProfiling_ = false;   //!< If TRUE then HwProfiling is enabled for the command
  bool blocking_ = true;          //!< If TRUE callback is blocking

  //! Extract timing from a single signal and update accumulators
  void ExtractSignalTiming(ProfilingSignal* signal,
                           uint64_t& start, uint64_t& end,
                           uint64_t& sdmaStart, uint64_t& sdmaEnd);

  Timestamp(const Timestamp&) = delete;
  Timestamp& operator=(const Timestamp&) = delete;

 public:
  Timestamp(VirtualGPU* gpu, amd::Command& command)
      : start_(std::numeric_limits<uint64_t>::max()),
        end_(0),
        gpu_(gpu),
        command_(command),
        parsedCommand_(nullptr),
        callback_signal_(hsa_signal_t{}) {}

  ~Timestamp() {}

  void getTime(uint64_t* start, uint64_t* end) {
    checkGpuTime();
    *start = start_;
    *end = end_;
  }

  void AddProfilingSignal(ProfilingSignal* signal) {
    signals_.push_back(signal);
    hasHwProfiling_ = true;
  }

  const std::vector<ProfilingSignal*>& Signals() const { return signals_; }

  const bool HwProfiling() const { return hasHwProfiling_; }

  //! Finds execution ticks on GPU
  //! If single_signal is nullptr, processes all signals and clears the list
  //! If single_signal is provided, processes only that signal with merge enabled
  void checkGpuTime(ProfilingSignal* single_signal = nullptr);

  // Start a timestamp (get timestamp from OS)
  void start() { start_ = amd::Os::timeNanos(); }

  // End a timestamp (get timestamp from OS)
  void end() {
    // Timestamp value can be updated by HW profiling if current command had a stall.
    // Although CPU TS should be still valid in this situation, there are cases in VM mode
    // when CPU timeline is out of sync with GPU timeline and shifted time can be reported
    if (end_ == 0) {
      end_ = amd::Os::timeNanos();
    }
  }

  static void setGpuTicksToTime(double ticksToTime) { ticksToTime_ = ticksToTime; }
  static double getGpuTicksToTime() { return ticksToTime_; }

  //! Returns amd::command assigned to this timestamp
  amd::Command& command() const { return command_; }

  //! Sets the parsed command
  void setParsedCommand(amd::Command* command) { parsedCommand_ = command; }

  //! Gets the parsed command
  amd::Command* getParsedCommand() const { return parsedCommand_; }

  //! Returns virtual GPU device, used with this timestamp
  VirtualGPU* gpu() const { return gpu_; }

  //! Updates the callback signal
  void SetCallbackSignal(hsa_signal_t callback_signal, bool blocking = true) {
    callback_signal_ = callback_signal;
    blocking_ = blocking;
  }
  //! Returns the callback signal
  hsa_signal_t GetCallbackSignal() const { return callback_signal_; }

  //! Return if callback is blocking/non-blocking
  bool GetBlocking() { return blocking_; }
};

class VirtualGPU : public device::VirtualDevice {
 public:
  class ManagedBuffer : public amd::EmbeddedObject {
   public:
    //! The number of chunks the arg pool will be divided
    ManagedBuffer(VirtualGPU& gpu, uint32_t pool_size, uint32_t num_signals)
        : gpu_(gpu), pool_size_(pool_size), pool_signal_(num_signals),
          num_chunk_signals_(num_signals) {}
    ~ManagedBuffer();

    //! Allocates all necessary resources to manage memory
    bool Create(amd::Device::MemorySegment mem_segment);

    //! Acquires memory for use on the gpu
    address Acquire(uint32_t size);

    //! Acquires custom aligned memory for use on the gpu
    address Acquire(uint32_t size, uint32_t alignment);

    //! Reset mem pool
    void ResetPool();

   private:
    VirtualGPU& gpu_;                        //!< Queue object for ROCm device
    address pool_base_ = nullptr;            //!< Memory pool base address
    uint32_t pool_size_;                     //!< Memory pool base size
    uint32_t pool_chunk_end_ = 0;            //!< The end offset of the current chunk
    uint32_t active_chunk_ = 0;              //!< The index of the current active chunk
    uint32_t pool_cur_offset_ = 0;           //!< Current active offset for update
    std::vector<hsa_signal_t> pool_signal_;  //!< Pool of HSA signals to manage multiple chunks
    uint32_t num_chunk_signals_;                   //!< Number of signals used per chunk
  };
  class MemoryDependency : public amd::EmbeddedObject {
   public:
    //! Default constructor
    MemoryDependency()
        : memObjectsInQueue_(nullptr), numMemObjectsInQueue_(0), maxMemObjectsInQueue_(0) {}

    ~MemoryDependency() { delete[] memObjectsInQueue_; }

    //! Creates memory dependency structure
    bool create(size_t numMemObj);

    //! Notify the tracker about new kernel
    void newKernel() { endMemObjectsInQueue_ = numMemObjectsInQueue_; }

    //! Validates memory object on dependency
    void validate(VirtualGPU& gpu, const Memory* memory, bool readOnly);

    //! Clear memory dependency
    void clear(bool all = true);

    //! Max number of mem objects in the queue
    size_t maxMemObjectsInQueue() const { return maxMemObjectsInQueue_; }

   private:
    struct MemoryState {
      uint64_t start_;  //! Busy memory start address
      uint64_t end_;    //! Busy memory end address
      bool readOnly_;   //! Current GPU state in the queue
    };

    MemoryState* memObjectsInQueue_;  //!< Memory object state in the queue
    size_t endMemObjectsInQueue_;     //!< End of mem objects in the queue
    size_t numMemObjectsInQueue_;     //!< Number of mem objects in the queue
    size_t maxMemObjectsInQueue_;     //!< Maximum number of mem objects in the queue
  };

  class HwQueueTracker : public amd::EmbeddedObject {
   public:
    HwQueueTracker(const VirtualGPU& gpu) : gpu_(gpu) {}

    ~HwQueueTracker();

    //! Creates a pool of signals for tracking of HW operations on the queue
    bool Create();

    //! Finds a free signal for the upcoming operation
    //! `phi_only`   -- signal exists ONLY for the estimator; mirror stock's no-signal answer by
    //!                 CLEARING the command's HW event. Correct on the single-dispatch path, where
    //!                 stock would have left that command with no HW event at all.
    //! `phi_detached` -- signal exists only for the estimator and belongs to NO command: touch
    //!                 `gpu_.command()` neither way. ⛔ REQUIRED on the graph batch path, where the
    //!                 AccumulateCommand legitimately owns a HW event from the last-slot
    //!                 `ActiveSignal()` a few lines earlier; `phi_only` there would RELEASE it.
    hsa_signal_t ActiveSignal(hsa_signal_value_t init_val = kInitSignalValueOne,
                              Timestamp* ts = nullptr, bool attach_signal = true,
                             bool is_dispatch = false, bool phi_only = false,
                             bool phi_detached = false);

    //! Wait for the curent active signal. Can idle the queue
    bool WaitCurrent();

    //! ⭐ Harvest `d` from signals that completed while the pool was GROWING rather than recycling.
    //! ⛔ NEVER waits: it tests each signal with a non-blocking load and skips any still running.
    //! Call only where completion has already been established by a wait the runtime was doing
    //! anyway (i.e. after WaitCurrent() in releaseGpuMemoryFence), never on the dispatch path.
    void PhiSweepCompleted();

    //! Update current active engine
    void SetActiveEngine(HwQueueEngine engine = HwQueueEngine::Compute) { engine_ = engine; }
    HwQueueEngine GetActiveEngine() const { return engine_; }

    //! Returns the last submitted signals for a wait.  aql_barrier_dep says the caller will
    //! put the result straight into an AQL packet executed by this queue's own command
    //! processor; only such a caller is offered a device resident ordering edge.
    std::vector<hsa_signal_t>& WaitingSignal(HwQueueEngine engine = HwQueueEngine::Compute,
                                             bool aql_barrier_dep = false);

    //! Resets current signal back to the previous one. It's necessary in a case of ROCr failure.
    void ResetCurrentSignal();

    //! Adds an external signal(submission in another queue) for dependency tracking
    void AddExternalSignal(ProfilingSignal* signal) { external_signals_.push_back(signal); }

    //! Get the last active signal on the queue
    ProfilingSignal* GetLastSignal() const { return signal_list_[current_id_]; }

    //! Clear external signals
    void ClearExternalSignals() { external_signals_.clear(); }

    //! Empty check for external signals
    bool IsExternalSignalListEmpty() const { return external_signals_.empty(); }

    //! Adds a raw signal for dependency tracking
    void AddDynamicQueueWait(hsa_signal_t signal) { dynamic_queue_waits_.push_back(signal); }

   private:
    //! Creates HSA signal with the specified scope
    bool CreateSignal(ProfilingSignal* signal, bool interrupt = false) const;

    //! Wait for the next active signal
    void WaitNext();

    //! Wait for the provided signal
    bool CpuWaitForSignal(ProfilingSignal* signal);

    HwQueueEngine engine_ = HwQueueEngine::Unknown;  //!< Engine used in the current operations
    std::stack<ProfilingSignal*> signal_pool_irq_;   //!< The pool of free signals with interrupts
    std::stack<ProfilingSignal*> signal_pool_;       //!< The pool of free signals without interrupt
    std::vector<ProfilingSignal*> signal_list_;      //!< The pool of all signals for processing
    size_t current_id_ = 0;                          //!< Last submitted signal
    const VirtualGPU& gpu_;                          //!< VirtualGPU, associated with this tracker
    std::vector<ProfilingSignal*> external_signals_;  //!< External signals for a wait in this queue
    std::vector<hsa_signal_t> dynamic_queue_waits_;   //!< Extra raw signals for a wait in this queue
    std::vector<hsa_signal_t> waiting_signals_;       //!< Current waiting signals in this queue
  };

  class MetaDataPreloader : public amd::EmbeddedObject {
    public:
      //! Set the metadata ring buffer base for the current queue.
      void SetQueueBase(void* ring_buffer, uint32_t version_header = 0) {
        queue_base_ = ring_buffer;
        if (queue_base_ != nullptr) {
          metadata_version_header_ = version_header;
        }
        pending_descriptor_ = nullptr;
        pending_preload_length_ = 0;
        pending_preload_offset_ = 0;
      }

      //! Stage the kernel descriptor and preload info for the next dispatch.
      //! Call before dispatchAqlPacket.
      void PrepareDispatch(const hsa_amd_metadata_kernel_descriptor_t* descriptor,
                           uint16_t preload_length, uint16_t preload_offset) {
        pending_descriptor_ = descriptor;
        pending_preload_length_ = preload_length;
        pending_preload_offset_ = preload_offset;
      }

      //! Write the metadata prefetch packet for the AQL slot at |index|.
      //! |use_movdir64b|: use atomic 64B writes (no sfence between body/header).
      //! |device_mem_ring_buf|: ring buffer is WC over PCIe — when MOVDIR64B is
      //!   unavailable, stages locally then NT-copies to avoid scattered WC stores.
      template <class AqlPacket>
      inline void SetMetadata(AqlPacket* packet, uint16_t header,
                              uint64_t index, bool use_movdir64b,
                              bool device_mem_ring_buf) {
        if (!HasMetadataQueue()) {
          return;
        }
        auto* dst = static_cast<uint8_t*>(queue_base_) + index * kMetadataPacketSize;
        FillMetadata(packet, header, dst, use_movdir64b, device_mem_ring_buf);
      }

      //! Build the metadata-prefetch packet for a captured (graph) dispatch into a
      //! caller-provided, zero-initialized host buffer.  Mirrors SetMetadata but
      //! targets plain host memory — no NT/MOVDIR64B stores or fences needed.
      //! The buffer is flattened and later bulk-copied to the metadata ring at
      //! graph-launch time.
      template <class AqlPacket>
      inline void CaptureMetadata(AqlPacket* packet, uint16_t header, uint8_t* dst) {
        if (!HasMetadataQueue() || dst == nullptr) {
          return;
        }
        FillMetadata(packet, header, dst, false, false);
      }

      //! Set the launch descriptor version (called once from VirtualGPU::create)
      void SetLaunchDescriptorVersion(uint8_t version) {
        launch_descriptor_version_ = version;
      }

      //! Copy the dynamic data prefetch config into the preloader state
      void SetDynDataPrefetchRegions(const amd::DynDataPrefetchConfig& cfg) {
        dyn_data_prefetch_enabled_ = true;
        dyn_data_prefetch_num_regions_ = cfg.numRegions;
        dyn_data_prefetch_hints_ = cfg.hints;
        for (uint32_t i = 0; i < cfg.numRegions && i < amd::kDynDataPrefetchMaxRegions; ++i) {
          dyn_data_prefetch_regions_[i] = cfg.regions[i];
        }
      }

      //! Reset the dynamic data prefetch state after dispatch
      void ClearDynDataPrefetchConfig() {
        dyn_data_prefetch_enabled_ = false;
      }

      //! Whether the current queue has a valid metadata ring buffer. This is the
      //! single gate for all metadata work: the ring base is an optional resource
      //! provided by ROCr/firmware (HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_RING_BUFFER)
      //! and is null when no queue is assigned or the queue lacks prefetch support.
      bool HasMetadataQueue() const { return queue_base_ != nullptr; }

      //! Returns the metadata ring buffer base (nullptr if no metadata queue)
      void* GetQueueBase() const { return queue_base_; }

      //! Returns the metadata packet slot for the given (masked) queue index,
      //! or nullptr if no metadata queue is attached. For logging/diagnostics.
      const hsa_amd_metadata_kernel_dispatch_packet_t* GetMetadataPacket(uint64_t index) const {
        if (!IsAttached()) {
          return nullptr;
        }
        return reinterpret_cast<const hsa_amd_metadata_kernel_dispatch_packet_t*>(
            static_cast<uint8_t*>(queue_base_) + index * kMetadataPacketSize);
      }

    private:
      //! Dispatch and barrier metadata packets share the ring, so their slot size
      //! must be identical for uniform indexing.
      static_assert(sizeof(hsa_amd_metadata_kernel_dispatch_packet_t) ==
                        sizeof(hsa_amd_metadata_barrier_packet_t),
                    "metadata packet types must share a uniform ring slot size");
      static constexpr size_t kMetadataPacketSize =
          sizeof(hsa_amd_metadata_kernel_dispatch_packet_t);

      //! Return whether the loader is attached to a gpu queue
      bool IsAttached() const { return queue_base_ != nullptr; }

      //! Populate the metadata packet at |dst| from the given AQL packet.
      //! Shared by SetMetadata() (ring-buffer destination with NT/MOVDIR64B
      //! writes) and CaptureMetadata() (host-buffer destination with plain stores).
      template <class AqlPacket>
      void FillMetadata(AqlPacket* packet, uint16_t header, uint8_t* dst,
                        bool use_movdir64b, bool device_mem_ring_buf) {
        if constexpr (std::is_same_v<AqlPacket, hsa_kernel_dispatch_packet_t> ||
                     std::is_same_v<AqlPacket, hsa_amd_ext_kernel_dispatch_packet_t>) {
          if (pending_descriptor_ == nullptr) {
            return;
          }
          auto* metadata = reinterpret_cast<hsa_amd_metadata_kernel_dispatch_packet_t*>(dst);
          auto* dispatch_packet = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet);
          SetPacket(dispatch_packet, header, metadata, use_movdir64b, device_mem_ring_buf);
        } else if constexpr (std::is_same_v<AqlPacket, hsa_barrier_and_packet_t> ||
                             std::is_same_v<AqlPacket, hsa_barrier_or_packet_t> ||
                             std::is_same_v<AqlPacket, hsa_amd_barrier_value_packet_t>) {
          auto* metadata = reinterpret_cast<hsa_amd_metadata_barrier_packet_t*>(dst);
          SetPacket(packet, header, metadata, use_movdir64b, device_mem_ring_buf);
        }
      }

      //! Get type from aql packet header
      uint8_t GetType(uint16_t header) const {
        return (header >> HSA_PACKET_HEADER_TYPE) & ((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1);
      }

      //! Write the metadata prefetch packet for kernel dispatch.
      //! Unified function covering MOVDIR64B, legacy+device-mem (staged NT),
      //! and legacy+system-mem (direct write) paths.  The assembly logic is
      //! shared; only the final write to the ring buffer differs.
      void SetPacket(hsa_kernel_dispatch_packet_t* aql, uint16_t header,
                     hsa_amd_metadata_kernel_dispatch_packet_t* metadata,
                     bool use_movdir64b, bool device_mem_ring_buf);

      //! Fill the body (everything except the 4 header dwords) of a kernel-dispatch
      //! metadata packet from |aql| and the staged descriptor/preload state. Returns
      //! the header dword value the caller should write into header0..header3.
      //! |target_is_zeroed| signals that |metadata| is already zero-initialized so
      //! redundant memsets of required-zero fields can be skipped. Shared by the
      //! queue-write path (SetPacket) and the graph-capture path (CaptureMetadata).
      uint32_t FillKernelDispatchMetadata(hsa_kernel_dispatch_packet_t* aql, uint16_t header,
                                          hsa_amd_metadata_kernel_dispatch_packet_t* metadata,
                                          bool target_is_zeroed);

      //! Write the metadata prefetch packet for barrier.
      //! Unified function covering MOVDIR64B, legacy+device-mem (staged NT),
      //! and legacy+system-mem (direct write) paths — analogous to the
      //! kernel-dispatch SetPacket.
      template <class AqlBarrierPacket>
      void SetPacket(AqlBarrierPacket* aql, uint16_t header,
                     hsa_amd_metadata_barrier_packet_t* metadata,
                     bool use_movdir64b, bool device_mem_ring_buf) const {
        const uint32_t metadata_header = GetType(header) | metadata_version_header_;
        uint32_t event_id = 0;
        if (aql->completion_signal.handle) {
          auto* signal = reinterpret_cast<amd_signal_t*>(aql->completion_signal.handle);
          event_id = signal->event_id;
        }
        if (use_movdir64b) {
          alignas(64) uint8_t seg0[64] = {};
          *reinterpret_cast<uint32_t*>(seg0) = metadata_header;
          *reinterpret_cast<uint32_t*>(seg0 + 4) = event_id;
          amd::movdir64b_copy64(metadata, seg0);
        } else {
          metadata->event_id = event_id;
          if (device_mem_ring_buf) {
            amd::nontemporalStoreFence();
          }
          metadata->header0 = metadata_header;
        }
      }

      void* queue_base_ = nullptr;        //!< The buffer base of prefetching queue
      uint32_t metadata_version_header_ = 0; //!< Pre-shifted version bits for metadata headers
      const hsa_amd_metadata_kernel_descriptor_t* pending_descriptor_ = nullptr;
      uint16_t pending_preload_length_ = 0;
      uint16_t pending_preload_offset_ = 0;

      uint8_t launch_descriptor_version_ = AMD_LAUNCH_DESCRIPTOR_VERSION_NONE;
      bool dyn_data_prefetch_enabled_ = false;
      uint8_t dyn_data_prefetch_hints_ = 0;
      uint32_t dyn_data_prefetch_num_regions_ = 0;
      amd::DynDataPrefetchRegion dyn_data_prefetch_regions_[amd::kDynDataPrefetchMaxRegions] = {};
  };

  VirtualGPU(Device& device, bool profiling = false, bool cooperative = false,
             const std::vector<uint32_t>& cuMask = {},
             amd::CommandQueue::Priority priority = amd::CommandQueue::Priority::Normal,
             bool dedicated_queue = false);
  ~VirtualGPU();

  bool create();
  const Device& dev() const { return roc_device_; }

  void profilingBegin(amd::Command& command, bool sdmaProfiling = false);
  void profilingEnd(bool clearHwEvent = false, bool publishOrderingEdge = true);

  void updateCommandsState(amd::Command* list) const;

  void submitReadMemory(amd::ReadMemoryCommand& cmd);
  void submitWriteMemory(amd::WriteMemoryCommand& cmd);
  void submitCopyMemory(amd::CopyMemoryCommand& cmd);
  void submitCopyMemoryP2P(amd::CopyMemoryP2PCommand& cmd);
  void submitBatchCopyMemory(amd::BatchCopyMemoryCommand& cmd);
  void SubmitBatchWriteMemory(amd::BatchWriteMemoryCommand& cmd);
  void SubmitBatchReadMemory(amd::BatchReadMemoryCommand& cmd);
  void submitMapMemory(amd::MapMemoryCommand& cmd);
  void submitUnmapMemory(amd::UnmapMemoryCommand& cmd);
  void submitKernel(amd::NDRangeKernelCommand& cmd);
  bool submitKernelInternal(
      const amd::NDRangeContainer& sizes,                  //!< Workload sizes
      const amd::Kernel& kernel,                           //!< Kernel for execution
      const_address parameters,                            //!< Parameters for the kernel
      void* event_handle,                                  //!< Handle to OCL event for debugging
      uint32_t sharedMemBytes = 0,                         //!< Shared memory size
      amd::NDRangeKernelCommand* vcmd = nullptr,           //!< Original launch command
      hsa_kernel_dispatch_packet_t* aql_packet = nullptr,  //!< Scheduler launch
      bool attach_signal = false);
  void submitNativeFn(amd::NativeFnCommand& cmd);
  void submitMarker(amd::Marker& cmd);

  //! Appends a barrier packet whose completion signal is a device resident ordering edge,
  //! naming a twin of the current command's completion signal.  Eligibility is tested by
  //! the caller; see the call sites.
  void PublishOrderingEdge();
  void submitAccumulate(amd::AccumulateCommand& cmd);
  void submitAcquireExtObjects(amd::AcquireExtObjectsCommand& cmd);
  void submitReleaseExtObjects(amd::ReleaseExtObjectsCommand& cmd);
  void submitPerfCounter(amd::PerfCounterCommand& cmd);

  void flush(amd::Command* list = nullptr, bool wait = false);
  void submitFillMemory(amd::FillMemoryCommand& cmd);
  void submitStreamOperation(amd::StreamOperationCommand& cmd);
  void submitBatchMemoryOperation(amd::BatchMemoryOperationCommand& cmd);
  void submitVirtualMap(amd::VirtualMapCommand& cmd);
  void submitMigrateMemObjects(amd::MigrateMemObjectsCommand& cmd);

  void submitSvmFreeMemory(amd::SvmFreeMemoryCommand& cmd);
  void submitSvmCopyMemory(amd::SvmCopyMemoryCommand& cmd);
  void submitSvmFillMemory(amd::SvmFillMemoryCommand& cmd);
  void submitSvmMapMemory(amd::SvmMapMemoryCommand& cmd);
  void submitSvmUnmapMemory(amd::SvmUnmapMemoryCommand& cmd);
  void submitSvmPrefetchAsync(amd::SvmPrefetchAsyncCommand& cmd);
  void SubmitSvmPrefetchBatchAsync(amd::SvmPrefetchBatchAsyncCommand& cmd);
  void SubmitSvmDiscardBatchAsync(amd::SvmDiscardBatchAsyncCommand& cmd);
  virtual void submitSignal(amd::SignalCommand& cmd) {}
  virtual void submitMakeBuffersResident(amd::MakeBuffersResidentCommand& cmd) {}

  void submitThreadTraceMemObjects(amd::ThreadTraceMemObjectsCommand& cmd) {}
  void submitThreadTrace(amd::ThreadTraceCommand& vcmd) {}

  virtual void submitExternalSemaphoreCmd(amd::ExternalSemaphoreCmd& cmd) override;

  virtual address allocKernelArguments(size_t size, size_t alignment) final;
  virtual void ReleaseSdmaEngines() final;  //!< Release SDMA engine assignments
  virtual void ReleaseAllHwQueues() final;
  virtual void ReleaseHwQueue() final;

  /**
   * @brief Waits on an outstanding kernel without regard to how
   * it was dispatched - with or without a signal
   *
   * @return bool true if Wait returned successfully, false otherwise
   */
  bool releaseGpuMemoryFence(bool skip_copy_wait = false);

  hsa_agent_t gpu_device() const { return gpu_device_; }
  hsa_queue_t* gpu_queue() { return gpu_queue_; }
  hsa_queue_t* gpu_queue() const { return gpu_queue_; }

  //! Set the active HW queue and keep the metadata preloader in sync.
  void SetGpuQueue(hsa_queue_t* queue);

  //! Ensure a HW queue is held, acquiring one (with the last_hwq_ affinity hint) if a
  //! dynamic-queue reclaim released it. No-op for dedicated queues or when one is already held.
  //! Caller must hold the execution() lock.
  void AcquireHwQueueIfNeeded();

  //! Snapshot the current HW queue as preferred for future re-acquisition (used by graph launch).
  //! Only updates if the queue is still valid — avoids clobbering a hint saved by ReleaseHwQueue.
  void SetPreferredQueue() override {
    std::scoped_lock lock(execution());
    if (gpu_queue_ != nullptr) {
      last_hwq_ = gpu_queue_;
    }
  }
  //! Acquire a HW queue using the preferred hint, then clear the hint
  void AcquireQueueWithPreference() override;

  //! Pin the HW queue so ReleaseHwQueue() becomes a no-op (used by graph internal streams)
  void PinQueue() override { queue_pinned_ = true; }
  //! Unpin the HW queue, allowing ReleaseHwQueue() to release it again
  void UnpinQueue() override { queue_pinned_ = false; }
  //! Release current HW queue and acquire a new one, avoiding queues with IDs in the excluded set
  bool ReacquireQueueExcluding(const std::unordered_set<uint64_t>& excluded_ids) override;

  // Return pointer to PrintfDbg
  PrintfDbg* printfDbg() const { return printfdbg_; }

  //! Returns memory dependency class
  MemoryDependency& memoryDependency() { return memoryDependency_; }

  //! Detects memory dependency for HSA kernels and uses appropriate AQL header
  bool processMemObjects(const amd::Kernel& kernel,  //!< AMD kernel object for execution
                         const_address params,       //!< Pointer to the param's store
                         size_t& ldsAddress,         //!< LDS usage
                         bool cooperativeGroups,     //!< Dispatch with cooperative groups
                         bool& imageBufferWrtBack,   //!< Image buffer write back is required
                         std::vector<device::Memory*>& wrtBackImageBuffer  //!< Images for writeback
  );

  //! Returns a managed buffer for staging copies
  ManagedBuffer& Staging() { return managed_buffer_; }

  //! Adds a pinned memory object into a map
  void addPinnedMem(amd::Memory* mem);

  void enableSyncBlit() const;

  void hasPendingDispatch() { hasPendingDispatch_ = true; }
  //! Per-stream estimator readout. PUBLIC because ~Device prints it over the LIVE vgpu list;
  //! printing from ~VirtualGPU omits every stream the program never destroyed.
  void PhiReport() const;

  bool IsPendingDispatch() const { return (hasPendingDispatch_) ? true : false; }
  void addSystemScope() override {
    addSystemScope_ = true;
    fence_state_ = amd::Device::CacheState::kCacheStateInvalid;
  }
  void SetCopyCommandType(cl_command_type type) { copy_command_type_ = type; }

  HwQueueTracker& Barriers() { return barriers_; }

  Timestamp* timestamp() const { return timestamp_; }
  amd::Command* command() const { return command_; }

  void* allocKernArg(size_t size, size_t alignment);
  //! Returns the size of one managed kernarg pool chunk.
  size_t KernArgPoolChunkSize() const;
  bool isFenceDirty() const { return fence_dirty_.load(std::memory_order_acquire); }
  void setFenceDirty(bool state) { fence_dirty_.store(state, std::memory_order_release); }
  void WaitCompleteSignal(hsa_signal_t signal);

  void HiddenHeapInit();
  uint64_t getQueueID();

  //! Add completion signal to the scheduler queue thread's event list.
  //! Wakes the scheduler queue thread if it's sleeping.
  void addSchedulerEvent(hsa_signal_t signal) {
    {
      std::lock_guard<std::mutex> lock(scheduler_mutex_);
      pendingSchedulerEvents_.push_back(signal);
    }
    scheduler_cv_.notify_one();
  }

  //! Returns true if the scheduler queue thread is running
  bool isSchedulerQueueThreadRunning() const {
    return schedulerQueueThreadRunning_.load(std::memory_order_relaxed);
  }

  //! Start the scheduler queue thread on first use
  void startSchedulerQueueThread();

  //! Analyzes a crashed AQL queue to find a broken AQL packet.
  //! Returns the faulting kernel name ("<not identified>" if not found).
  std::string AnalyzeAqlQueue() const;

  //! Emits the hang report itself. Called by AnalyzeAqlQueue, which brackets it
  //! with the banner.
  std::string AnalyzeAqlQueueBody() const;
  bool ForceIrq() const { return force_irq_; }

  //! SDMA engine affinity management
  uint32_t AssignedSdmaEngine() const {
    return assigned_sdma_engine_;
  }
  void SetAssignedSdmaEngine(uint32_t engine_mask) {
    assigned_sdma_engine_ = engine_mask;
  }
  void ClearAssignedSdmaEngine() {
    assigned_sdma_engine_ = 0;
  }
  bool hasAssignedSdmaEngine() const {
    return assigned_sdma_engine_ != 0;
  }

  void* getOrCreateHostcallBuffer();

 private:
  //! Release pinned memory after previously submitted work on the queue has completed.
  void SchedulePinnedMemoryRelease(amd::HostQueue& queue, std::vector<amd::Memory*> pinned_memory);

  //! Apply fence-scope adjustments to the AQL header (system scope promotion,
  //! consecutive-system-scope optimization, fence_state_ tracking).
  void adjustHeader(uint16_t& header);

  //! Dispatches a barrier with blocking HSA signals
  void dispatchBlockingWait(hsa_kernel_dispatch_packet_t* packet);

  //! Dispatch (or capture, when graph-capturing) a kernel dispatch packet.
  //! Handles both hsa_kernel_dispatch_packet_t and hsa_amd_ext_kernel_dispatch_packet_t.
  template <typename AqlPacket> bool dispatchAqlPacket(AqlPacket* packet, uint16_t header,
                                                       uint16_t rest, bool blocking = true,
                                                       bool attach_signal = false);

  //! Fast-path dispatch: pre-built flat contiguous buffer
  bool dispatchAqlPacketBatchFlat(const amd::AlignedVector64<uint8_t>& flatPacketData,
                                  const std::vector<uint32_t>& validFullHeaders,
                                  amd::AccumulateCommand* vcmd = nullptr,
                                  bool attach_signal = false,
                                  bool pre_patched = false,
                                  bool blocking = false,
                                  const std::vector<uint8_t>* flatMetadataData = nullptr) override;

  //! ⛔⛔ `is_dispatch` DEFAULTS FALSE ON PURPOSE -- IT MUST FAIL CLOSED. This function serves BOTH
  //! the kernel-dispatch path and `dispatchCounterAqlPacket`, which pushes a PM4 perf-counter IB
  //! through it as a VENDOR_SPECIFIC packet. Tagging that as a dispatch folds a perf-counter
  //! packet's duration into the `d` estimate -- the barrier-mixture bug this flag exists to
  //! prevent. A wrong `false` loses a sample; a wrong `true` corrupts an EWMA for ~250 samples.
  //! ⛔ Do NOT re-derive this from `header`/`rest` here: the batch path packs setup in the HIGH
  //! half of a full_header dword while this path takes `setup` from `rest >> 8`. The two
  //! conventions differ, and guessing between them is how this bug got here.
  template <typename AqlPacket> bool dispatchGenericAqlPacket(AqlPacket* packet, uint16_t header,
                                                              uint16_t rest, bool blocking,
                                                              bool attach_signal = false,
                                                              bool is_dispatch = false);

  bool dispatchCounterAqlPacket(hsa_ext_amd_aql_pm4_packet_t* packet, const uint32_t gfxVersion,
                                bool blocking, const hsa_ven_amd_aqlprofile_1_00_pfn_t* extApi);
  void dispatchBarrierPacket(uint16_t packetHeader, bool skipSignal = false,
                             hsa_signal_t signal = hsa_signal_t{0});
  void dispatchBarrierValuePacket(uint16_t packetHeader, bool resolveDepSignal = false,
                                  hsa_signal_t signal = hsa_signal_t{0},
                                  hsa_signal_value_t value = 0, hsa_signal_value_t mask = 0,
                                  hsa_signal_condition32_t cond = HSA_SIGNAL_CONDITION_EQ,
                                  bool skipTs = false,
                                  hsa_signal_t completionSignal = hsa_signal_t{0});
  void initializeDispatchPacket(hsa_kernel_dispatch_packet_t* packet, amd::NDRangeContainer& sizes);

  //! Write an AQL packet to the ring buffer with metadata prefetch.
  //! Selects MOVDIR64B only for device-memory queues with CPU support;
  //! otherwise uses the legacy NT-store path.
  template <typename AqlPacket>
  void writePacketToRingBuffer(AqlPacket* aql_loc, AqlPacket* packet,
                               uint16_t header, uint16_t rest, uint64_t slot_index);

  //! Ring the queue doorbell via direct UC store or ROCr signal.
  void ringQueueDoorbell(uint64_t index);

  //! Snapshot shared barrier state before atomically reserving AQL queue slots.
  AqlSlotReservation ReserveAqlSlots(size_t packet_count);

  //! Clear a caller-requested barrier when prior queue state already preserves stream ordering.
  void OptimizeStreamOrderingBarrier(uint16_t& header,
                                     const AqlSlotReservation& reservation) const;

  //! Record a final packet header in the shared HW-queue barrier state.
  void RecordAqlPacketHeader(const AqlSlotReservation& reservation, size_t packet_offset,
                             uint16_t header);

  //! Record the final slot in a submission as this VirtualGPU's previous packet.
  void CompleteAqlSubmission(const AqlSlotReservation& reservation);

  void resetKernArgPool() { managed_kernarg_buffer_.ResetPool(); }

  uint64_t getVQVirtualAddress();

  bool createSchedulerParam();

  //! Returns TRUE if virtual queue was successfully allocated
  bool createVirtualQueue(uint deviceQueueSize);

  //! Common function for fill memory used by both svm Fill and non-svm fill
  bool fillMemory(cl_command_type type,         //!< the command type
                  amd::Memory* amdMemory,       //!< memory object to fill
                  const void* pattern,          //!< pattern to fill the memory
                  size_t patternSize,           //!< pattern size
                  const amd::Coord3D& surface,  //!< Whole Surface of mem object.
                  const amd::Coord3D& origin,   //!< memory origin
                  const amd::Coord3D& size,     //!< memory size for filling
                  bool forceBlit = false        //!< force shader blit path
  );

  //! Common function for memory copy used by both svm Copy and non-svm Copy
  bool copyMemory(cl_command_type type,            //!< the command type
                  amd::Memory& srcMem,             //!< source memory object
                  amd::Memory& dstMem,             //!< destination memory object
                  bool entire,                     //!< flag of entire memory copy
                  const amd::Coord3D& srcOrigin,   //!< source memory origin
                  const amd::Coord3D& dstOrigin,   //!< destination memory object
                  const amd::Coord3D& size,        //!< copy size
                  const amd::BufferRect& srcRect,  //!< region of source for copy
                  const amd::BufferRect& dstRect,  //!< region of destination for copy
                  amd::CopyMetadata copyMetadata = amd::CopyMetadata()  //!< Memory copy MetaData
  );

  //! Updates AQL header for the upcoming dispatch
  void setAqlHeader(uint16_t header) { aqlHeader_ = header; }

  //! Resets the current queue state. Note: should be called after AQL queue becomes idle
  void ResetQueueStates();

  //! Record the last write index and whether the last packet is idle-trackable. Only tracker-owned
  //! signals (from Barriers().ActiveSignal()) qualify; set skip_signal for externally-provided or
  //! absent completion signals. IsQueueIdle() reads the tracker-owned signal at check time.
  template <typename AqlPacket>
  inline void TrackQueueProgress(const AqlPacket& packet, uint64_t index,
                                 bool skip_signal = false) {
    last_write_index_ = index;
    if (!skip_signal && packet.completion_signal.handle != 0) {
      last_packet_with_signal_index_ = index;
    }
  }

  //! Returns true if the queue is considered as idle, i.e. all submitted packets are complete.
  //! Note: it doesn't track the state of caches.
  bool IsQueueIdle() const;

  //! True if this marker records the same event as the preceding barrier with no
  //! intervening dispatch or sync. Caller must hold the execution() lock.
  bool ShouldCoalesceMarker(const amd::Marker& vcmd) const {
    // A zero coalesceEvent() means the record didn't opt in or isn't a record,
    // so it can never be coalesced.
    uint64_t event_id = vcmd.coalesceEvent();
    if (event_id == 0 || event_id != last_barrier_coalesce_event_) {
      return false;
    }
    if (IsPendingDispatch() || vcmd.syncedSinceRecord()) {
      return false;
    }
    return true;
  }

  //! Set the coalescing window to the given event and its barrier's HwEvent,
  //! retaining the new signal and releasing the previously held one. Pass
  //! (0, nullptr) to end the window. Caller must hold the execution() lock.
  void SetCoalesceWindow(uint64_t event_id, void* hw_event);

  //! Attach a ProfilingSignal to a command as its HwEvent, releasing any prior
  //! one and retaining the new one. No-op if cmd or hw_event is null.
  static void AttachHwEvent(amd::Command* cmd, void* hw_event);

  //! Spin-wait until queue has space for a packet at \p write_index.
  //! Uses cached_read_dispatch_id_ to avoid DRAM traffic on the fast path;
  //! only re-reads the hardware read_dispatch_id when the cached value
  //! indicates the queue might be full.
  void WaitForQueueSlot(uint64_t write_index, uint32_t capacity) {
    if ((write_index - cached_read_dispatch_id_) < capacity) {
      return;
    }
    do {
      cached_read_dispatch_id_ = Hsa::queue_load_read_index_scacquire(gpu_queue_);
      if ((write_index - cached_read_dispatch_id_) < capacity) {
        return;
      }
      amd::Os::yield();
    } while (true);
  }

  //! Queue state flags
  union {
    struct {
      uint32_t hasPendingDispatch_ : 1;     //!< A kernel dispatch is outstanding
      uint32_t profiling_ : 1;              //!< Profiling is enabled
      uint32_t cooperative_ : 1;            //!< Cooperative launch is enabled
      uint32_t addSystemScope_ : 1;         //!< Insert a system scope to the next aql
      uint32_t tracking_created_ : 1;       //!< Enabled if tracking object was properly initialized
      uint32_t retainExternalSignals_ : 1;  //!< Indicate to retain external signal array
      uint32_t force_irq_ : 1;              //!< Forces interrupt on the signal completion
    };
    uint32_t state_;
  };

  //! ⭐ PLACEMENT-POLICY STATE. Phi = sum_rings T_q*H_q with T_q = sum rho*d and H_q = sum rho/d is
  //! DIMENSIONLESS -- T carries units of d, H carries 1/d -- so `d` is kept in RAW AGENT TICKS and
  //! never translated. Deliberate: every tick<->ns conversion is somewhere to put a units bug and
  //! the model does not need one.
  //! ⭐ rho/d is EXACTLY the dispatch rate (rho = c*d/W, so rho/d = c/W), so H_q needs no `d`.
  //! ⛔ BUT A RATE IS NOT A COUNT: it needs `phi_dispatches_` AND a time base. This comment used to
  //! claim H_q needed "only `phi_dispatches_` -- a counter, no timestamps"; that was WRONG and
  //! contradicted the note on the epochs below. See `phi_ep_*` and `PhiRate()`.
  //! const + mutable: the tracker holds `gpu_` by const reference, and this is
  //! accounting/estimator state rather than observable queue state.
  void PhiSampleDuration(ProfilingSignal* sig) const;  //!< fold one completed packet into `d`
  void PhiPublishSlot() const;  //!< mirror this stream's state where the shadow selector can read it
  //! Count a recycled signal the `is_dispatch` tag DECLINED -- the positive control for that tag.
  void PhiCountSkipped() const { phi_d_skipped_.fetch_add(1, std::memory_order_relaxed); }
  //! High-water mark of the drain-point sweep, so its budget is observed rather than assumed.
  void PhiNoteSweepDepth(uint64_t v) const {
    uint64_t m = phi_sweep_max_.load(std::memory_order_relaxed);
    while (v > m && !phi_sweep_max_.compare_exchange_weak(m, v, std::memory_order_relaxed)) {
    }
  }

  //! Widen the observation window to include `end`.
  //! ⛔⛔ MIN/MAX, NOT FIRST/LAST. Samples do NOT arrive in chronological order: the drain-point
  //! sweep walks the signal pool BACKWARD, so it delivers DECREASING `end` timestamps. A first/last
  //! window therefore inverts and the span reads 0 -- MEASURED `win_ticks=0` on every deep-backlog
  //! run, i.e. the rate was lost in exactly the regime the sweep exists to serve. The epoch start
  //! is likewise only ever advanced by a STRICTLY LATER `end`.
  void PhiNoteWindow(uint64_t end) const {
    const uint64_t last = phi_ep_last_seen_.load(std::memory_order_relaxed);
    if (end > last) {
      phi_ep_last_seen_.store(end, std::memory_order_relaxed);
    }
    const uint64_t cur = phi_ep_cur_start_.load(std::memory_order_relaxed);
    if (cur == 0) {
      const uint64_t d0 = phi_dispatches_.load(std::memory_order_relaxed);
      phi_ep_cur_start_.store(end, std::memory_order_relaxed);
      phi_ep_cur_disp_.store(d0, std::memory_order_relaxed);
      return;
    }
    if (end > cur && (end - cur) > kPhiEpochTicks) {
      // Roll. ⛔ Publish the base BEFORE the start, so a PhiRate() that catches the pair mid-update
      // sees a `base_disp` that is too OLD (harmless: a slightly low rate) rather than too new
      // (which underflows the subtraction).
      phi_ep_prev_disp_.store(phi_ep_cur_disp_.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
      phi_ep_prev_start_.store(cur, std::memory_order_relaxed);
      phi_ep_cur_disp_.store(phi_dispatches_.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
      phi_ep_cur_start_.store(end, std::memory_order_relaxed);
    }
  }

  //! ⛔ ATOMIC because slice C will read ANOTHER stream's counters from
  //! Device::getQueueFromPool, which runs under `active_queue_access_` -- a DIFFERENT monitor from
  //! this vgpu's execution(). Writers are single-threaded today, so this is not a live race; it
  //! becomes one the moment the selector lands, and it would present as a policy bug.
  mutable std::atomic<uint64_t> phi_dispatches_{0};  //!< dispatch count for this stream
  mutable std::atomic<uint64_t> phi_d_ticks_{0};     //!< EWMA of packet duration, agent ticks
  mutable std::atomic<uint64_t> phi_d_samples_{0};   //!< samples folded in
  mutable std::atomic<uint64_t> phi_d_rejected_{0};  //!< timings rejected as unusable
  mutable std::atomic<uint64_t> phi_d_skipped_{0};   //!< recycles declined by the is_dispatch tag
  mutable std::atomic<uint64_t> phi_sweep_max_{0};  //!< deepest drain-point sweep walk
  //! ⛔ STABLE slot for the shadow selector, claimed at construction and released at destruction.
  //! NOT `index()`: ~VirtualGPU decrements the index of every LATER vgpu, so an index-keyed slot
  //! silently re-points to a different stream the moment any stream is destroyed.
  uint32_t phi_slot_ = 0xFFFFFFFFu;
  //! ⭐ ROTATION CURSOR for graph-batch `d` sampling. One Phi-only signal per batch, on a DIFFERENT
  //! kernel each launch. ⛔ Rotation is not a nicety: a FIXED index sees one kernel of the graph
  //! forever, and `d` enters `T_q` squared, so a pinned unrepresentative kernel is the failure that
  //! killed the pre-patched-signal route (unbounded BOTH ways -- 24.4x over, 0.010x under).
  //! Rotating makes the EWMA converge to the mean over kernels, which is the quantity `T_q` wants.
  //! ⛔ Rotating ALONE is not sufficient, and the cursor must be PER SHAPE -- see below.
  //! ⛔⛔ THE BLOCK-MEAN, AND IT IS NOT AN OPTIMISATION -- IT CLOSES AN ALIASING DEFECT. The cursor
  //! walks the graph with period `eligible` while the EWMA remembers ~8 samples. When kernel
  //! duration is CORRELATED WITH INDEX AT LOW FREQUENCY (first half cheap, second half dear) the
  //! EWMA tracks whichever half the cursor is in: MEASURED 33,067 vs 7,518 ticks on IDENTICAL work,
  //! a 4.40x swing decided by nothing but how many launches had happened -- and `d` enters `T_q`
  //! SQUARED, so 19.4x in the Phi term. Averaging one FULL period first makes each fold a
  //! STRATIFIED sample of the whole graph regardless of phase.
  //! ⛔ Hashing the index instead only reaches 1.46x: randomising is not stratifying.
  //! ⛔⛔ The alternating (period-2) case the first version validated with is the ONE pattern this
  //! aliasing CANNOT bite. Never re-validate with an alternating graph, or a single shape.
  //!
  //! ⛔⛔⛔ AND THE STATE MUST BE PER SHAPE, NOT PER STREAM. A single `cycle_len` per vgpu was the
  //! first version's defect: a stream replaying a 64-kernel graph and a 32-kernel graph has no one
  //! rotation period, the blocks are malformed, and the 4.40x aliasing comes straight back -- in
  //! exactly production's geometry, where vLLM keeps one hipGraph per batch-size bucket and replays
  //! them on one stream. Slots are direct-mapped by `eligible` and reset on key mismatch; a
  //! collision between two shapes costs one discarded block, never a mixed one.
  //! ⛔⛔ SLOTS ARE LINEAR-PROBED, NOT DIRECT-MAPPED, AND THAT IS NOT A REFINEMENT. Direct mapping
  //! by `eligible % kPhiShapes` collides: a 64-kernel graph and a 32-kernel graph give
  //! `63 % 4 == 31 % 4 == 3`. Two colliding shapes ALTERNATING on one stream flip the key every
  //! launch, so the block resets every launch, NO block ever completes, and `d` freezes at an early
  //! partial -- MEASURED 6,200-6,440 ticks against ~20,000 for either shape alone, a 3.2x
  //! under-estimate, which is the MERGE direction and therefore the unsafe one. Probing gives every
  //! distinct shape its own slot while there is room.
  //! ⭐ Sized generously because it is cheap: one slot is 40 B, so 256 slots is ~10 KB per vgpu,
  //! and the COST OF A LOOKUP DOES NOT SCALE WITH THE TABLE -- the probe is bounded at
  //! `kPhiProbe`, so a hit is O(1) and a miss is O(16) whatever the capacity. Growing it only buys
  //! headroom; it cannot make the dispatch path slower.
  static constexpr size_t kPhiShapes = 256;
  static constexpr size_t kPhiProbe = 16;
  struct PhiShapeSlot {
    std::atomic<uint64_t> key{0};  //!< `eligible` this slot tracks; 0 = free
    std::atomic<uint64_t> rot{0};  //!< rotation cursor for THIS shape
    //! Block state below is touched ONLY by PhiSampleDuration, single-threaded per vgpu (it runs
    //! under execution() on every path), so it needs no atomics -- and the non-atomic
    //! read-modify-write is bounded anyway: `sum/n` always lies inside the block's own min/max.
    uint64_t blk_sum = 0;
    uint64_t blk_n = 0;
    uint64_t folds = 0;
  };
  mutable PhiShapeSlot phi_shapes_[kPhiShapes];
  mutable std::atomic<uint64_t> phi_shape_overflow_{0};  //!< shapes that found no slot
  mutable std::atomic<uint64_t> phi_batch_rot_fallback_{0};  //!< cursor for overflowed shapes

  //! Slot for `eligible`: the existing one, else a free one, else **nullptr**.
  //! ⛔⛔ TABLE-FULL RETURNS nullptr AND THE CALLER SKIPS THE BLOCK-MEAN. It used to EVICT the home
  //! slot, and that was the direct-mapped collision defect wearing a different hat: with more
  //! distinct shapes than slots, the evicted shape never completes a block, `folds` stays 0, and
  //! `d` freezes at a one-sample partial -- MEASURED 5,120-7,044 ticks against a truth of
  //! 19,600-20,280 as soon as the shape count exceeded the table, a 2.8-3.9x UNDER-estimate, i.e.
  //! the MERGE direction. Degrading to "no block-mean for this shape" instead is strictly the
  //! pre-block-mean behaviour: aliased, but bounded and never 3.7x low.
  //! ⚠️ vLLM keeps one hipGraph per batch-size bucket, so >8 shapes is the expected case, not a
  //! corner: sized for that, and the caller must handle nullptr.
  PhiShapeSlot* PhiShape(uint64_t eligible) const {
    // ⛔ NOT `eligible % kPhiShapes`. Keys are kernel counts -- 31, 63, 127, 255 -- which share
    // their low bits and so collide hard on a power-of-two modulus, exactly the clustering that
    // starved the table before. Fibonacci hashing spreads the high bits instead.
    const size_t home =
        static_cast<size_t>((eligible * 0x9E3779B97F4A7C15ull) >> 56) % kPhiShapes;
    for (size_t i = 0; i < kPhiProbe; ++i) {
      PhiShapeSlot& s = phi_shapes_[(home + i) % kPhiShapes];
      const uint64_t k = s.key.load(std::memory_order_relaxed);
      if (k == eligible) {
        return &s;
      }
      if (k == 0) {
        s.key.store(eligible, std::memory_order_relaxed);
        s.rot.store(0, std::memory_order_relaxed);
        s.blk_sum = 0;
        s.blk_n = 0;
        s.folds = 0;
        return &s;
      }
    }
    phi_shape_overflow_.fetch_add(1, std::memory_order_relaxed);
    return nullptr;  // caller: no block-mean for this shape, NOT an evicted (and so starved) slot
  }
  //! ⭐⭐ THE TIME BASE FOR `H_q`, AND IT IS FREE. `H_q = sum rho/d = c/W` is a RATE, and
  //! `phi_dispatches_` above is a MONOTONIC COUNT. Without a base a stream that issued a million
  //! dispatches an hour ago and is now idle is indistinguishable from a saturating one -- the same
  //! per-binding defect this campaign already retired once (idle bound streams counted at full
  //! weight => Phi spreads when it should merge).
  //! ⭐ Nearly free: `PhiSampleDuration` already has `end` from the completed signal.
  //! ⛔ WHY IT MATTERS EVEN THOUGH OUR PROBES CANNOT SEE IT: under a COMMON window the missing base
  //! cancels EXACTLY out of a ring-vs-ring comparison (3,000 randomised trials, every apparent flip
  //! at relative gap <= 6e-14, i.e. roundoff). Under PER-STREAM windows 25.8% of argmins flip.
  //! Every stream in our probes starts and stops together, so this is STRUCTURALLY INVISIBLE in all
  //! of our data and live in production, where streams do not.
  //! ⛔⛔ THREE THINGS A NAIVE first/last WINDOW GETS WRONG, all biting in the MERGE (unsafe)
  //! direction, which is why this is epochs and not two timestamps:
  //!  1. **No dispatch baseline.** `phi_dispatches_` counts from process start, so dividing it by a
  //!     partial window over-states the rate. Hence `*_disp`, the count AT the epoch start.
  //!  2. **A window that never forgets.** An idle-then-busy stream averages over all history and
  //!     under-reports its current rate. Hence epochs that roll.
  //!  3. **A per-stream right edge.** Using each stream's own last sample makes an IDLE stream's
  //!     window short and its rate look HIGH. The selector supplies one common `t_ref` for every
  //!     stream at decision time -- free, and it makes idle streams decay by construction.
  //! ⭐ TWO epochs, not one: the rate is measured from the PREVIOUS epoch's start, so there is
  //! always between one and two epochs of history. A single rolling epoch collapses to near-zero
  //! width right after each roll, the selector reads "unknown", and it flaps.
  static constexpr uint64_t kPhiEpochTicks = 50000000;  //!< ~50 ms at ~1 tick/ns
  mutable std::atomic<uint64_t> phi_ep_cur_start_{0};   //!< current epoch start, agent ticks
  mutable std::atomic<uint64_t> phi_ep_cur_disp_{0};    //!< phi_dispatches_ at that instant
  mutable std::atomic<uint64_t> phi_ep_prev_start_{0};  //!< previous epoch start = the rate's base
  mutable std::atomic<uint64_t> phi_ep_prev_disp_{0};
  mutable std::atomic<uint64_t> phi_ep_last_seen_{0};   //!< newest sample `end`; readout only

 public:
  //! Now, in the SAME agent-tick domain as `d` and the epochs -- no conversion, no units bug.
  //! The selector reads this ONCE per decision and shares it across streams.
  static uint64_t PhiNowTicks();

  //! Dispatch rate over [epoch base, t_ref] as an exact fraction, so no division happens here.
  //! Returns false for "rate UNKNOWN", which the caller must NOT treat as a rate of zero.
  bool PhiRate(uint64_t t_ref, uint64_t& disp_delta, uint64_t& tick_delta) const {
    // ⛔ READ ORDER: epoch base FIRST, `phi_dispatches_` SECOND. ⚠️ CORRECTION to what an earlier
    // comment of mine claimed: the wrong order does NOT produce a 1e19 rate, because the
    // `disp_now < base_disp` guard below catches it on the very next line. What the order actually
    // buys is AVAILABILITY -- wrong order means the guard fires, the rate reads UNKNOWN, and the
    // selector flaps, which is the thing the two-epoch design exists to prevent. Defence in depth,
    // not a safety property.
    uint64_t base_start = phi_ep_prev_start_.load(std::memory_order_relaxed);
    uint64_t base_disp = phi_ep_prev_disp_.load(std::memory_order_relaxed);
    if (base_start == 0) {  // fewer than two epochs yet; fall back to the current one
      base_start = phi_ep_cur_start_.load(std::memory_order_relaxed);
      base_disp = phi_ep_cur_disp_.load(std::memory_order_relaxed);
    }
    const uint64_t disp_now = phi_dispatches_.load(std::memory_order_relaxed);
    if (base_start == 0 || t_ref <= base_start || disp_now < base_disp) {
      return false;  // never sampled, no common edge, or a torn pair -- UNKNOWN, never 0
    }
    disp_delta = disp_now - base_disp;
    tick_delta = t_ref - base_start;
    return true;
  }

  //! ⚠️ On the rho clamp (in the shadow reader): `rho = rate * d` is clamped to 1, and when it
  //! fires -- rho > 1 <=> rate > 1/d -- the stream's contribution to `H_q` changes from `rate` to
  //! `1/d`, which is SMALLER. So a saturated or `d`-over-estimated stream is priced BELOW its own
  //! measured rate. Defensible (rho <= 1 is physics) but it silently changes WHICH quantity ranks
  //! the ring. MEASURED: 0 clamps in 1,092 opportunities, so this is a note, not a live effect.
  //! ⚠️ And a stream slower than one dispatch per epoch rolls on EVERY sample, so `disp_delta` ~ 1
  //! and its rate is very coarse. An honest limit of the epoch design, not a bug.
  //! The stream's contended service interval `d`, in raw agent ticks. 0 = UNKNOWN.
  //! ⛔ 0 is NOT "instant": a caller that treats it as a duration prices this stream's ring at
  //! T_q = 0, i.e. FREE, and Phi then piles every other stream onto it.
  uint64_t PhiDTicks() const { return phi_d_ticks_.load(std::memory_order_relaxed); }

  //! The rate BASE (epoch start, and the dispatch count at that instant) plus the live count, for
  //! a reader that will divide against its own common `t_ref`.
  void PhiRateBase(uint64_t& base_start, uint64_t& base_disp, uint64_t& disp_now) const {
    base_start = phi_ep_prev_start_.load(std::memory_order_relaxed);
    base_disp = phi_ep_prev_disp_.load(std::memory_order_relaxed);
    if (base_start == 0) {
      base_start = phi_ep_cur_start_.load(std::memory_order_relaxed);
      base_disp = phi_ep_cur_disp_.load(std::memory_order_relaxed);
    }
    disp_now = phi_dispatches_.load(std::memory_order_relaxed);
  }

  //! Span of the live observation window in ticks; 0 = unknown. Readout only.
  uint64_t PhiWindowTicks() const {
    uint64_t base = phi_ep_prev_start_.load(std::memory_order_relaxed);
    if (base == 0) {
      base = phi_ep_cur_start_.load(std::memory_order_relaxed);
    }
    const uint64_t last = phi_ep_last_seen_.load(std::memory_order_relaxed);
    return (last > base) ? (last - base) : 0;
  }

 private:

  Timestamp* timestamp_;
  bool sdma_profiling_for_cmd_ = false;  //!< SDMA profiling enabled for current command
  amd::Command* command_;   //!< Current command
  //! Monotonic client coalesce id of the last barrier from submitMarker, used to
  //! coalesce consecutive records. Execution() lock only. 0 = no window open.
  uint64_t last_barrier_coalesce_event_ = 0;
  //! Retained ProfilingSignal of that last barrier; a coalesced record reuses it
  //! as its HwEvent so query/sync observe correct readiness. Released on reset.
  void* last_barrier_hw_event_ = nullptr;
  hsa_agent_t gpu_device_;  //!< Physical device
  hsa_queue_t* gpu_queue_;                //!< Active queue associated with a vgpu
  bool device_mem_ring_buf_ = false;           //!< Queue ring buffer is in device memory
  bool use_movdir64b_ = false;                 //!< Use MOVDIR64B for AQL packet writes
  //! Cached hardware doorbell for the active queue (UC MMIO). Non-null only when
  //! DEBUG_CLR_DIRECT_DOORBELL is enabled and the doorbell id query succeeded.
  volatile uint64_t* doorbell_ptr_ = nullptr;
  //! Largest barrier-bit slot shared by every VirtualGPU using the physical HW queue.
  std::shared_ptr<std::atomic<uint64_t>> largest_aql_barrier_bit_slot_;
  //! Final AQL slot submitted by this stream, or kInvalidAqlSlot before its first packet.
  uint64_t last_aql_packet_slot_ = kInvalidAqlSlot;
  alignas(64) hsa_barrier_and_packet_t barrier_packet_ {};
  alignas(64) hsa_amd_barrier_value_packet_t barrier_value_packet_ {};

  uint64_t cached_read_dispatch_id_ = 0;  //!< Cached read_dispatch_id to avoid DRAM reads
                                          //!< when queue is not full. GPU updates to
                                          //!< amd_queue_t.read_dispatch_id probe-invalidate
                                          //!< CPU caches; this local copy stays in L1/L2.
  uint32_t skippedDispatches_;  //!< Count of consecutive dispatches that skipped the doorbell flush.
  uint32_t dispatch_id_;  //!< This variable must be updated atomically.
  Device& roc_device_;    //!< roc device object
  PrintfDbg* printfdbg_;
  MemoryDependency memoryDependency_;  //!< Memory dependency class
  uint16_t aqlHeader_;                 //!< AQL header for dispatch

  amd::Memory* virtualQueue_;  //!< Virtual device queue
  uint deviceQueueSize_;       //!< Device queue size
  uint maskGroups_;            //!< The number of mask groups processed in the scheduler by
                               //!< one thread
  uint schedulerThreads_;      //!< The number of scheduler threads

  hsa_queue_t* schedulerQueue_;
  //! Cached hardware doorbell for the scheduler queue (UC MMIO). Non-null only when
  //! DEBUG_CLR_DIRECT_DOORBELL is enabled and the doorbell id query succeeded.
  volatile uint64_t* schedulerDoorbell_ = nullptr;

  std::thread schedulerQueueThread_;                  //!< Host thread that monitors the scheduler queue
  std::atomic<bool> schedulerQueueThreadRunning_;     //!< Flag to indicate if the thread is running
  std::mutex scheduler_mutex_;                        //!< Lock to synchronize scheduler thread
  std::condition_variable scheduler_cv_;              //!< Condition to wake scheduler thread
  std::once_flag scheduler_thread_init_;              //!< Ensures thread is initialized exactly once
  std::vector<hsa_signal_t> pendingSchedulerEvents_;  //!< Pending scheduler completion signals

  HwQueueTracker barriers_;  //!< Tracks active barriers in ROCr

  ManagedBuffer managed_buffer_;          //!< Memory manager for staging copies
  ManagedBuffer managed_kernarg_buffer_;  //!< Managed memory for kernel args

  static constexpr uint32_t kStagingPoolNumSignals = 4; //!< Hsa Signal count for Staging Buffer
  static constexpr uint32_t kKernArgPoolNumSignals = 16; //!< Hsa Signal count for KernArg Buffer
  MetaDataPreloader metadata_preloader_; //!< Proloader of kernel meta data

  friend class Timestamp;

  //  PM4 packet for gfx8 performance counter
  enum {
    SLOT_PM4_SIZE_DW = HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE / sizeof(uint32_t),
    SLOT_PM4_SIZE_AQLP = HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE / 64
  };

  uint16_t dispatchPacketHeaderNoSync_;
  uint16_t dispatchPacketHeader_;

  //!< bit-vector representing the CU mask. Each active bit represents using one CU
  const std::vector<uint32_t> cuMask_;
  amd::CommandQueue::Priority priority_;  //!< The priority for the hsa queue
  bool dedicated_queue_;                  //!< TRUE if this VirtualGPU has a dedicated queue (e.g., null stream)
  bool queue_pinned_ = false;             //!< TRUE if queue is pinned by graph (blocks ReleaseHwQueue)
  hsa_queue_t* last_hwq_ = nullptr;       //!< Last HW queue used, for preferred re-acquisition hint

  cl_command_type copy_command_type_;  //!< Type of the copy command, used for ROC profiler
                                       //!< OCL doesn't distinguish different copy types,
                                       //!< but ROC profiler expects D2H or H2D detection
  int fence_state_;                    //!< Fence scope
                                       //!< kUnknown/kFlushedToDevice/kFlushedToSystem
  std::atomic<bool> fence_dirty_;      //!< Fence modified flag
  bool heap_init_fence_emitted_ = false;  //!< True once this queue has emitted system scope
                                          //!< fence after hidden heap init.

  uint64_t last_write_index_ = kInvalidQueueIndex; //!< The last HW queue write index for any packet
  uint64_t last_packet_with_signal_index_ = kInvalidQueueIndex; //!< The last HW queue write index for a packet
                                              //!< with a completion signal

  //! SDMA engine affinity tracking for this VirtualGPU/stream
  uint32_t assigned_sdma_engine_ = 0;           //!< Assigned SDMA engine mask for all operations

  void* hostcallBuffer_;        //!< Hostcall buffer
  size_t hostcallBufferSize_ = 0; //!< Byte size of hostcallBuffer_, for hostFree

  using KernelArgImpl = device::Settings::KernelArgImpl;
};
}  // namespace amd::roc
