/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

/*! \addtogroup HSA OCL Stub Implementation
 *  @{
 */

//! HSA OCL STUB Implementation
namespace amd::roc {

//! Device settings
class Settings : public device::Settings {
 public:
  enum Hmm : uint32_t {
    EnableSystemMemory = 0x01,    //!< Forces system memory preference by default
    EnableMallocPrefetch = 0x02,  //!< Skips default prefetch after allocation
    EnableSvmTracking = 0x04,     //!< Enables SW SVM tracking
    EnableDebugSvm = 0x08         //!< Extra debug flag (reserved for runtime developers)
  };

  union {
    struct {
      uint doublePrecision_ : 1;       //!< Enables double precision support
      uint enableLocalMemory_ : 1;     //!< Enable GPUVM memory
      uint enableNCMode_ : 1;          //!< Enable Non Coherent mode for system memory
      uint imageDMA_ : 1;              //!< Enable direct image DMA transfers
      uint imageBufferWar_ : 1;        //!< Image buffer workaround for Gfx10
      uint cpu_wait_for_signal_ : 1;   //!< Wait for HSA signal on CPU
      uint system_scope_signal_ : 1;   //!< HSA signal is visibile to the entire system
      uint fgs_kernel_arg_ : 1;        //!< Use fine grain kernel arg segment
      uint barrier_value_packet_ : 1;  //!< Barrier value packet functionality
      uint dynamic_queues_ : 2;        //!< Dynamic queues: 0=off, 1=Depth, 2=1+dedicated null
      uint blocking_blit_ : 1;         //!< Blit ops can be blocking on CPU
      uint queue_pipe_dist_ : 1;       //!< gfx94x queue pipe distribution
      uint ext_dispatch_packet_ : 1;   //!< Uses new ext dispatch packet for all launches
      uint aql_barrier_opt_ : 1;       //!< Per-stream barrier-bit optimization
      //! Total-slowdown placement policy: 0 off, 1 observe only, 2 select.
      //! ⛔ 2 BITS, and DEBUG_CLR_QUEUE_PHI is a uint -- it MUST be clamped where it is
      //! assigned. dynamic_queues_ has the same width and is NOT clamped, so
      //! DEBUG_HIP_DYNAMIC_QUEUES=4 silently reads as 0 (off) with no diagnostic. Do not
      //! reproduce that here: a mode that silently degrades to a stock baseline is
      //! indistinguishable from a working control.
      //! ⛔ WIDENED 2 -> 3 BITS for the live mode. At 2 bits `DEBUG_CLR_QUEUE_PHI=4` would have
      //! stored as 0, i.e. silently STOCK -- exactly the failure the note above describes for
      //! `dynamic_queues_`. The clamp in rocsettings.cpp must stay <= (1 << width) - 1.
      uint queue_phi_ : 3;
      uint reserved_ : 14;
    };
    uint value_;
  };

  //! Default max workgroup size for 1D
  int maxWorkGroupSize_;

  //! Preferred workgroup size
  uint preferredWorkGroupSize_;

  uint kernargPoolSize_;
  uint numDeviceEvents_;  //!< The number of device events
  uint numWaitEvents_;    //!< The number of wait events for device enqueue

  size_t xferBufSize_;        //!< Transfer buffer size for image copy optimization
  size_t pinnedXferSize_;     //!< Pinned buffer size for transfer
  size_t pinnedMinXferSize_;  //!< Minimal buffer size for pinned transfer

  size_t sdmaCopyThreshold_;   //!< Use SDMA to copy above this size
  size_t sdma_p2p_threshold_;  //!< Use SDMA in P2P above this size

  uint32_t hmmFlags_;       //!< HMM functionality control flags
  uint32_t limit_blit_wg_;  //!< The number of workgroups for blit execution
  uint32_t max_hw_queues_;  //!< Effective maximum HW queues (accounts for null stream reservation)

  //! Default constructor
  Settings();

  //! Creates settings
  bool create(bool fullProfile, const amd::Isa& isa, bool enableXNACK, bool coop_groups = false,
              bool isXgmi = false);

 private:
  //! Disable copy constructor
  Settings(const Settings&);

  //! Disable assignment
  Settings& operator=(const Settings&);

  //! Overrides current settings based on registry/environment
  void override();

  //! Determine how kernel arguments should be implemented given ASIC (host
  //! memory, device memory, device memory with memory ordering workaround)
  void setKernelArgImpl(const amd::Isa& isa, bool isXgmi);
};

/*@}*/  // namespace amd::roc
}  // namespace amd::roc

