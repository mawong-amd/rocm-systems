/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "top.hpp"
#include "CL/cl.h"
#include "device/device.hpp"
#include "platform/command.hpp"
#include "platform/program.hpp"
#include "platform/perfctr.hpp"
#include "platform/memory.hpp"
#include "utils/concurrent.hpp"
#include "thread/thread.hpp"
#include "thread/monitor.hpp"
#include "utils/versions.hpp"

#include "device/rocm/rocrctx.hpp"
#include "device/rocm/rocsettings.hpp"
#include "device/rocm/rocvirtual.hpp"
#include "device/rocm/rocdefs.hpp"
#include "device/rocm/rocprintf.hpp"
#include "device/rocm/rocglinterop.hpp"


#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

/*! \addtogroup HSA
 *  @{
 */

//! HSA Device Implementation
namespace amd::roc {

/**
 * @brief List of environment variables that could be used to
 * configure the behavior of Hsa Runtime
 */
#define ENVVAR_HSA_POLL_KERNEL_COMPLETION "HSA_POLL_COMPLETION"

//! Forward declarations
class Command;
class Device;
class GpuCommand;
class Heap;
class HeapBlock;
class Program;
class Kernel;
class Memory;
class Resource;
class VirtualDevice;
class PrintfDbg;

class ProfilingSignal : public amd::ReferenceCountedObject {
 public:
  //! Sentinel for dispatch_slot_ when this signal doesn't time a reported graph dispatch.
  static constexpr uint32_t kNoDispatchSlot = std::numeric_limits<uint32_t>::max();

  hsa_signal_t signal_;   //!< HSA signal to track profiling information
  Timestamp* ts_;         //!< Timestamp object associated with the signal
  HwQueueEngine engine_;  //!< Engine used with this signal
  //! AccumulateCommand dispatch record this signal supplies timing for.
  uint32_t dispatch_slot_ = kNoDispatchSlot;
  //! ⭐ True when this signal timed a KERNEL DISPATCH rather than a barrier/marker. The `d`
  //! estimator folds in dispatches only: a barrier's start->end is not a service interval, and
  //! mixing the two was MEASURED to inflate `d` by a fixed ~8.6 us while leaving the slope right.
  bool phi_is_dispatch_ = false;
  //! ⭐ True once the `d` estimator has taken this signal's timing, so the recycle-point harvest and
  //! the drain-point sweep cannot double-count the same packet. Defaults TRUE: a signal nobody armed
  //! for Phi must never look like an unharvested sample.
  std::atomic<bool> phi_harvested_{true};
  //! ⭐ The rotation period (targetable kernels) of the BATCH this signal was armed for. The sample
  //! is harvested long after the batch, so the block-mean cannot know the shape unless the signal
  //! carries it. 0 = not a rotating graph sample.
  uint32_t phi_cycle_len_ = 0;
  std::recursive_mutex lock_;  //!< Signal lock for update

  typedef union {
    struct {
      uint32_t done_ : 1;       //!< True if signal is done
      uint32_t interrupt_ : 1;  //!< True if the signal will trigger an interrupt
      uint32_t reserved_ : 30;
    };
    uint32_t data_;
  } Flags;

  Flags flags_;

  //! Handle of a device resident twin of signal_, published by the producing command so
  //! another queue can name it in barrier_packet_.dep_signal[] instead of signal_; zero when
  //! none was published.  The consumer reads it without this object's lock: the release store
  //! in VirtualGPU::PublishOrderingEdge() publishes edge_owner_ and edge_slot_ with it, and
  //! orders it after the arming store on the slot.
  std::atomic<uint64_t> edge_handle_{0};
  const Device* edge_owner_ = nullptr;  //!< Device whose pool owns edge_slot_
  uint32_t edge_slot_ = 0;              //!< Index of that slot inside the pool

  //! Returns a published edge slot to its owner's free list.  Called from the destructor and
  //! from the point in ActiveSignal() where this object is about to be re-armed - the two
  //! places clr already knows no command holds it.  That is the property slot reuse needs: a
  //! consumer's barrier packet can only name a slot while the command that waits still holds
  //! the producing command, and that command holds this object.
  void ReleaseOrderingEdge();

  //! Cached timing data - populated when signal completes, avoids repeated HSA calls
  struct CachedTiming {
    uint64_t start_ = 0;   //!< Cached start timestamp from HSA
    uint64_t end_ = 0;     //!< Cached end timestamp from HSA
    bool valid_ = false;   //!< True if timing data has been cached
  };
  CachedTiming cached_timing_;

  ProfilingSignal() : ts_(nullptr), engine_(HwQueueEngine::Compute) {
    signal_.handle = 0;
    flags_.data_ = 0;
    flags_.done_ = true;
    dispatch_slot_ = kNoDispatchSlot;
  }

  virtual ~ProfilingSignal();
  std::recursive_mutex& LockSignalOps() { return lock_; }

  //! Cache timing data from HSA for this signal (called once when signal completes)
  void CacheTimingData(hsa_agent_t gpu_device);

  //! Reset cached timing for signal reuse
  void ResetCachedTiming() {
    std::scoped_lock lock(lock_);
    cached_timing_.start_ = 0;
    cached_timing_.end_ = 0;
    cached_timing_.valid_ = false;
    dispatch_slot_ = kNoDispatchSlot;
  }

  //! Check if timing is already cached
  bool IsTimingCached() const { return cached_timing_.valid_; }

  //! Get cached timing values
  void GetCachedTiming(uint64_t& start, uint64_t& end) {
    std::scoped_lock lock(lock_);
    start = cached_timing_.start_;
    end = cached_timing_.end_;
  }
};

class Sampler : public device::Sampler {
 public:
  //! Constructor
  Sampler(const Device& dev) : dev_(dev) {}

  //! Default destructor for the device memory object
  virtual ~Sampler();

  //! Creates a device sampler from the OCL sampler state
  bool create(const amd::Sampler& owner  //!< AMD sampler object
  );

 private:
  void fillSampleDescriptor(hsa_ext_sampler_descriptor_v2_t& samplerDescriptor,
                            const amd::Sampler& sampler) const;
  Sampler& operator=(const Sampler&);

  //! Disable operator=
  Sampler(const Sampler&);

  const Device& dev_;  //!< Device object associated with the sampler

  hsa_ext_sampler_t hsa_sampler;
};

// A NULL Device type used only for offline compilation
// Only functions that are used for compilation will be in this device
class NullDevice : public amd::Device {
 public:
  //! constructor
  NullDevice(){};

  //! create the device
  bool create(const amd::Isa& isa);

  //! Initialise all the offline devices that can be used for compilation
  static bool init();
  //! Teardown for offline devices
  static void tearDown();

  //! Destructor for the Null device
  virtual ~NullDevice();

  const Settings& settings() const { return static_cast<Settings&>(*settings_); }

  //! Construct an device program object from the ELF assuming it is valid
  device::Program* createProgram(amd::Program& owner,
                                 amd::option::Options* options = nullptr) override;

  // List of dummy functions which are disabled for NullDevice

  //! Create a new virtual device environment.
  device::VirtualDevice* createVirtualDevice(amd::CommandQueue* queue = nullptr) override {
    ShouldNotReachHere();
    return nullptr;
  }

  virtual bool registerSvmMemory(void* ptr, size_t size) const {
    ShouldNotReachHere();
    return false;
  }

  virtual void deregisterSvmMemory(void* ptr) const { ShouldNotReachHere(); }

  //! Just returns nullptr for the dummy device
  device::Memory* createMemory(amd::Memory& owner) const override {
    ShouldNotReachHere();
    return nullptr;
  }
  device::Memory* createMemory(size_t size, size_t alignment = 0) const override {
    ShouldNotReachHere();
    return nullptr;
  }
  //! Sampler object allocation
  bool createSampler(const amd::Sampler& owner,  //!< abstraction layer sampler object
                     device::Sampler** sampler   //!< device sampler object
  ) const override {
    ShouldNotReachHere();
    return true;
  }

  //! Just returns nullptr for the dummy device
  device::Memory* createView(
      amd::Memory& owner,           //!< Owner memory object
      const device::Memory& parent  //!< Parent device memory object for the view
  ) const override {
    ShouldNotReachHere();
    return nullptr;
  }

  device::Signal* createSignal() const override {
    ShouldNotReachHere();
    return nullptr;
  }

  //! Just returns nullptr for the dummy device
  void* svmAlloc(amd::Context& context,   //!< The context used to create a buffer
                 size_t size,             //!< size of svm spaces
                 size_t alignment,        //!< alignment requirement of svm spaces
                 cl_svm_mem_flags flags,  //!< flags of creation svm spaces
                 void* svmPtr             //!< existing svm pointer for mGPU case
  ) const override {
    ShouldNotReachHere();
    return nullptr;
  }

  //! Just returns nullptr for the dummy device
  void svmFree(void* ptr  //!< svm pointer needed to be freed
  ) const override {
    ShouldNotReachHere();
    return;
  }

  void* virtualAlloc(void* req_addr, size_t size, size_t alignment) override {
    ShouldNotReachHere();
    return nullptr;
  }

  bool virtualFree(void* addr) override {
    ShouldNotReachHere();
    return true;
  }

  cl_int virtualMap(void* va, size_t size, amd::Memory* phys) override {
    ShouldNotReachHere();
    return CL_INVALID_OPERATION;
  }

  cl_int virtualUnmap(void* va, size_t size) override {
    ShouldNotReachHere();
    return CL_INVALID_OPERATION;
  }

  virtual bool SetMemAccess(void* va_addr, size_t va_size, VmmAccess access_flags,
                            VmmLocationType = VmmLocationType::kDevice,
                            int numaNode = -1) override {
    ShouldNotReachHere();
    return false;
  }

  virtual bool GetMemAccess(void* va_addr, VmmAccess* access_flags_ptr) const override {
    ShouldNotReachHere();
    return false;
  }

  virtual bool ValidateMemAccess(amd::Memory& mem, bool read_write) const override {
    ShouldNotReachHere();
    return true;
  }

  //! Determine if we can use device memory for SVM
  const bool forceFineGrain(amd::Memory* memory) const {
    return (memory->getContext().devices().size() > 1);
  }

  virtual bool importExtSemaphore(void** extSemahore, const amd::Os::FileDesc& handle,
                                  amd::ExternalSemaphoreHandleType sem_handle_type) override {
    ShouldNotReachHere();
    return false;
  }

  void DestroyExtSemaphore(void* extSemaphore) override { ShouldNotReachHere(); }

  //! Acquire external graphics API object in the host thread
  //! Needed for OpenGL objects on CPU device

  bool bindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                          bool validateOnly) override {
    ShouldNotReachHere();
    return false;
  }

  bool unbindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                            bool validateOnly) override {
    ShouldNotReachHere();
    return false;
  }

  //! Releases non-blocking map target memory
  virtual void freeMapTarget(amd::Memory& mem, void* target) { ShouldNotReachHere(); }

  //! Empty implementation on Null device
  bool globalFreeMemory(size_t* freeMemory) const override {
    ShouldNotReachHere();
    return false;
  }

  //! Empty implementation on Null device
  bool amdFileRead(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                uint64_t* size_copied, int32_t* status) override {
    ShouldNotReachHere();
    return false;
  }

  //! Empty implementation on Null device
  bool amdFileWrite(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                 uint64_t* size_copied, int32_t* status) override {
    ShouldNotReachHere();
    return false;
  }

  bool SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput,
                    cl_set_device_clock_mode_output_amd* pSetClockModeOutput) override {
    return true;
  }

  bool IsHwEventReady(const amd::Event& event, bool wait = false,
                      amd::SyncPolicy policy = amd::SyncPolicy::Auto) const override {
    return false;
  }

  void getHwEventTime(const amd::Event& event, uint64_t* start, uint64_t* end) const override {};
  void ReleaseGlobalSignal(void* signal) const override {}

#if defined(__clang__)
#if __has_feature(address_sanitizer)
  virtual device::UriLocator* createUriLocator() const {
    ShouldNotReachHere();
    return nullptr;
  }
#endif
#endif

 private:
  static constexpr bool offlineDevice_ = true;
};

struct AgentInfo {
  hsa_agent_t agent;
  hsa_amd_memory_pool_t fine_grain_pool;
  hsa_amd_memory_pool_t coarse_grain_pool;
  hsa_amd_memory_pool_t kern_arg_pool;
  hsa_amd_memory_pool_t ext_fine_grain_pool;
};

//! A HSA device ordinal (physical HSA device)
class Device : public NullDevice {
 public:
  //! Initialise the whole HSA device subsystem (init, device enumeration, etc).
  static bool init();
  static void tearDown();

  //! Lookup all AMD HSA devices and memory regions.
  static hsa_status_t iterateAgentCallback(hsa_agent_t agent, void* data);
  static hsa_status_t iterateGpuMemoryPoolCallback(hsa_amd_memory_pool_t region, void* data);
  static hsa_status_t iterateCpuMemoryPoolCallback(hsa_amd_memory_pool_t region, void* data);
  static hsa_status_t loaderQueryHostAddress(const void* device, const void** host);

  //! Returns the AMD HSA loader extension function table.
  static const hsa_ven_amd_loader_1_03_pfn_t& loaderExtensionTable() {
    return amd_loader_ext_table;
  }

  static bool loadHsaModules();

  hsa_agent_t getBackendDevice() const { return bkendDevice_; }
  //! Get the CPU agent with the least NUMA distance to this GPU
  const hsa_agent_t& getCpuAgent() const { return cpu_agent_info_->agent; }

  //! Maps an HSA agent to a stable global index shared across all devices. GPU agents
  //! occupy [0, numGpuAgents); CPU agents occupy [numGpuAgents, numGpuAgents + numCpuAgents).
  //! Returns -1 if the agent is not one of the enumerated agents.
  static int agentGlobalIndex(hsa_agent_t agent);

  //! Get the CPU agent that is in a 'index' NUMA node
  const hsa_agent_t getCpuAgent(int index) const {
    if ((index < 0) || (index >= cpu_agents_.size())) {
      // Return default CPU agent
      return cpu_agent_info_->agent;
    }
    return cpu_agents_[index].agent;
  }

  void setupCpuAgent();  // Setup the CPU agent which has the least NUMA distance to this GPU

  void checkAtomicSupport();  //!< Check the support for pcie atomics

  //! Destructor for the physical HSA device
  virtual ~Device();

  // Temporary, delete it later when HSA Runtime and KFD is fully fucntional.
  void fake_device();

  ///////////////////////////////////////////////////////////////////////////////
  // TODO: Below are all mocked up virtual functions from amd::Device, they may
  // need real implementation.
  ///////////////////////////////////////////////////////////////////////////////

  //! Instantiate a new virtual device
  virtual device::VirtualDevice* createVirtualDevice(amd::CommandQueue* queue = nullptr) override;

  //! Construct an device program object from the ELF assuming it is valid
  virtual device::Program* createProgram(amd::Program& owner,
                                         amd::option::Options* options = nullptr) override;

  virtual device::Memory* createMemory(amd::Memory& owner) const override;
  virtual device::Memory* createMemory(size_t size, size_t alignment = 0) const override;
  //! Sampler object allocation
  virtual bool createSampler(const amd::Sampler& owner,  //!< abstraction layer sampler object
                             device::Sampler** sampler   //!< device sampler object
  ) const override;

  //! Just returns nullptr for the dummy device
  virtual device::Memory* createView(
      amd::Memory& owner,           //!< Owner memory object
      const device::Memory& parent  //!< Parent device memory object for the view
  ) const override {
    return nullptr;
  }

  virtual device::Signal* createSignal() const override;
  virtual device::Signal* createIpcSignal() const override;

  //! Acquire external graphics API object in the host thread
  //! Needed for OpenGL objects on CPU device
  virtual bool bindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                  bool validateOnly) override;

  /**
   * @brief Removes the external device as an available device.
   *
   * @note: The current implementation is to avoid build break
   * and does not represent actual / correct implementation. This
   * needs to be done.
   */
  bool unbindExternalDevice(
      uint flags,               //!< Enum val. for ext.API type: GL, D3D10, etc.
      void* const gfxDevice[],  //!< D3D device do D3D, HDC/Display handle of X Window for GL
      void* gfxContext,         //!< HGLRC/GLXContext handle
      bool validateOnly         //!< Only validate if the device can inter-operate with
                                //!< pDevice/pContext, do not bind.
  ) override;

  //! Gets free memory on a GPU device
  virtual bool globalFreeMemory(size_t* freeMemory) const override;
  virtual void* hostAlloc(size_t size, size_t alignment,
                          MemorySegment mem_seg = MemorySegment::kNoAtomics,
                          const void* agentInfo = nullptr, bool allowAllAgentsAccess = true) const override;  // nullptr uses default CPU agent
  virtual void hostFree(void* ptr, size_t size = 0) const override;

  virtual bool amdFileRead(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                        uint64_t* size_copied, int32_t* status) override;
  virtual bool amdFileWrite(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                         uint64_t* size_copied, int32_t* status) override;

  bool deviceAllowAccess(void* dst) const override;

  bool allowPeerAccess(device::Memory* memory) const override;
  void deviceVmemRelease(uint64_t mem_handle) const;
  uint64_t deviceVmemAlloc(size_t size, uint64_t flags) const;

  //! Allocate a host-resident VMM handle on a CPU NUMA pool. numaNode < 0 resolves
  //! to the calling thread's current node (HostNumaCurrent). Returns 0 on failure.
  uint64_t hostVmemAlloc(size_t size, uint64_t flags, int numaNode) const;

  void* deviceLocalAlloc(size_t size,
                        const AllocationFlags& flags = AllocationFlags{}, bool allowAllAgentsAccess = true) const override;
  void* reserveMemory(size_t size, size_t alignment) const;
  void releaseMemory(void* ptr, size_t size) const;
  void memFree(void* ptr, size_t size) const;

  virtual void* svmAlloc(amd::Context& context, size_t size, size_t alignment,
                         cl_svm_mem_flags flags = CL_MEM_READ_WRITE, void* svmPtr = nullptr) const override;

  virtual void svmFree(void* ptr) const override;

  virtual bool SetSvmAttributes(const void* dev_ptr, size_t count, amd::MemoryAdvice advice,
                                bool use_cpu = false, int numa_id = kDefaultNumaNode) const override;
  virtual bool GetSvmAttributes(void** data, size_t* data_sizes, int* attributes,
                                size_t num_attributes, const void* dev_ptr, size_t count) const override;
  virtual size_t ScratchLimitCurrent() const final;
  virtual bool UpdateScratchLimitCurrent(size_t limit) const final;
  virtual void* virtualAlloc(void* req_addr, size_t size, size_t alignment) override;
  virtual bool virtualFree(void* addr) override;

  virtual cl_int virtualMap(void* va, size_t size, amd::Memory* phys) override;
  virtual cl_int virtualUnmap(void* va, size_t size) override;

  virtual bool SetMemAccess(void* va_addr, size_t va_size, VmmAccess access_flags,
                            VmmLocationType = VmmLocationType::kDevice,
                            int numaNode = -1) override;
  virtual bool GetMemAccess(void* va_addr, VmmAccess* access_flags_ptr) const override;
  virtual bool ValidateMemAccess(amd::Memory& mem, bool read_write) const override { return true; }

  virtual VmmExportStatus ExportShareableVMMHandle(amd::Memory& amd_mem_obj, int flags,
                                                 void* shareableHandle,
                                                 amd::Memory::HandleType handle_type) override;

  bool ImportShareableHSAHandle(void* osHandle, uint64_t* hsa_handle_ptr,
                                amd::Memory::HandleType handle_type) const;

  virtual amd::Memory* ImportShareableVMMHandle(void* osHandle,
                                                amd::Memory::HandleType handle_type) override;

  virtual bool SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput,
                            cl_set_device_clock_mode_output_amd* pSetClockModeOutput) override;

  virtual bool IsHwEventReady(const amd::Event& event, bool wait = false,
                              amd::SyncPolicy policy = amd::SyncPolicy::Auto) const override;
  virtual void getHwEventTime(const amd::Event& event, uint64_t* start, uint64_t* end) const override;
  virtual void ReleaseGlobalSignal(void* signal) const override;
  virtual void RetainGlobalSignal(void* signal) const override;
  virtual bool CreateHwEvents(int count, std::vector<void*>& hw_events) const override;
  virtual void DestroyHwEvent(void* hw_event) const override;
  virtual void ResetHwEvents(const std::vector<void*>& hw_events) const override;
  virtual void QuiesceHwEvents(const std::vector<void*>& hw_events) const override;
  virtual uint8_t* CreateBarrierPacket() const override;
  virtual void ApplyHwEventPatches(const std::vector<HwEventPatch>& patches,
                                   const std::vector<void*>& hw_events) const override;
  virtual bool CreateUserEvent(amd::UserEvent* event) const override;
  virtual void SetUserEvent(amd::UserEvent* event) const override;

  virtual bool importExtSemaphore(void** extSemaphore, const amd::Os::FileDesc& handle,
                                  amd::ExternalSemaphoreHandleType sem_handle_type) override;
  virtual void DestroyExtSemaphore(void* extSemaphore) override;

  //! Allocate host memory in terms of numa policy set by user
  void* hostNumaAlloc(size_t size, size_t alignment, MemorySegment mem_seg) const;

  //! Pin a host pointer allocated by C/C++ or OS allocator (i.e. ordinary system DRAM) and
  //! return a new device pointer accessible by the GPU agent.
  void* hostLock(void* hostMem, size_t size, MemorySegment memSegment) const;

  //! Returns transfer engine object
  const device::BlitManager& xferMgr() const { return xferQueue()->blitMgr(); }

  const size_t alloc_granularity() const { return alloc_granularity_; }

  const hsa_profile_t agent_profile() const { return agent_profile_; }

  //! Finds an appropriate map target
  amd::Memory* findMapTarget(size_t size) const;

  //! Adds a map target to the cache
  bool addMapTarget(amd::Memory* memory) const;

  //! Returns a ROC memory object from AMD memory object
  roc::Memory* getRocMemory(amd::Memory* mem  //!< Pointer to AMD memory object
  ) const;

  //! Create internal blit program
  bool createBlitProgram();

  // P2P agents avaialble for this device
  const std::vector<hsa_agent_t>& p2pAgents() const { return p2p_agents_; }

  //! Returns the list of HSA agents used for IPC memory attach
  const hsa_agent_t* IpcAgents() const { return p2p_agents_list_; }

  // User enabled peer devices
  const bool isP2pEnabled() const { return (enabled_p2p_devices_.size() > 0) ? true : false; }

  // Update the global free memory size
  void updateFreeMemory(size_t size, bool free);

  //! Returns the lock object for the virtual gpus list
  std::recursive_mutex& vgpusAccess() const { return vgpusAccess_; }

#ifdef _WIN32
  //! D3D interop accessors - return adapter LUID for device matching
  const LUID& getDeviceLUID() const { return deviceLuid_; }
  bool hasValidLUID() const { return luidValid_; }
#endif

  typedef std::vector<VirtualGPU*> VirtualGPUs;
  //! Returns the list of all virtual GPUs running on this device
  const VirtualGPUs& vgpus() const { return vgpus_; }
  VirtualGPUs vgpus_;  //!< The list of all running virtual gpus (lock protected)

  VirtualGPU* xferQueue() const;

  struct QueueExtras {
    void* metadataRingBuffer = nullptr;
    //! Cached hardware doorbell (UC MMIO), only set when DEBUG_CLR_DIRECT_DOORBELL is enabled.
    volatile uint64_t* doorbellPtr = nullptr;
    bool deviceMemRingBuf = false;
    //! Largest barrier-bit slot shared by every VirtualGPU using this physical queue.
    std::shared_ptr<std::atomic<uint64_t>> largestAqlBarrierBitSlot;
  };

  //! Acquire HSA queue. This method can create a new HSA queue or
  hsa_queue_t* acquireQueue(
      uint32_t queue_size_hint, bool coop_queue = false, const std::vector<uint32_t>& cuMask = {},
      amd::CommandQueue::Priority priority = amd::CommandQueue::Priority::Normal,
      bool managed = false, bool dedicated_queue = false,
      hsa_queue_t* preferred = nullptr,
      const std::unordered_set<uint64_t>* excluded_ids = nullptr,
      void** metadata_ring_buffer = nullptr);

  //! Release HSA queue
  void releaseQueue(hsa_queue_t*, const std::vector<uint32_t>& cuMask = {}, bool coop_queue = false,
                    bool managed = false);

  hsa_queue_t* AcquireActiveQueue(amd::CommandQueue::Priority priority,
                                   hsa_queue_t* preferred = nullptr,
                                   const std::unordered_set<uint64_t>* excluded_ids = nullptr);
  bool ReleaseActiveQueue(hsa_queue_t* queue, amd::CommandQueue::Priority priority);

  //! Look up per-queue extras (metadata ring buffer and placement).
  QueueExtras GetQueueExtras(hsa_queue_t* queue);

  //! Return the pre-computed metadata packet version header bits
  uint32_t MetadataVersionHeader() const { return metadata_version_header_; }

  //! Return multi GPU grid launch sync buffer
  address MGSync() const { return mg_sync_; }

  //! Returns value for corresponding Link Attributes in a vector, given other device
  virtual bool findLinkInfo(const amd::Device& other_device, std::vector<LinkAttrType>* link_attr) override;

  //! Returns a GPU memory object from AMD memory object
  roc::Memory* getGpuMemory(amd::Memory* mem  //!< Pointer to AMD memory object
  ) const;

  //! Initialize memory in AMD HMM on the current device or keeps it in the host memory
  bool SvmAllocInit(void* memory, size_t size) const;

  void getGlobalCUMask(std::string_view cuMaskStr);

  static hsa_status_t BackendErrorCallBackHandler(const hsa_amd_event_t* event, void* data);

  static void RegisterBackendErrorCb();

  virtual amd::Memory* GetArenaMemObj(const void* ptr, size_t& offset, size_t size = 0) override;

  virtual uint32_t getPreferredNumaNode() const final { return preferred_numa_node_; }

  virtual uint32_t numHostNumaNodes() const final {
    return static_cast<uint32_t>(cpu_agents_.size());
  }

  const bool isFineGrainSupported() const override;

  //! Returns True if memory pointer is known to ROCr (excludes HMM allocations)
  bool IsValidAllocation(const void* dev_ptr, size_t size, hsa_amd_pointer_info_t* ptr_info);

  //! Allocates hidden heap for device memory allocations
  void HiddenHeapAlloc(const VirtualGPU& gpu);
  //! Init hidden heap for device memory allocations
  void HiddenHeapInit(const VirtualGPU& gpu);

  //! True if this agent can host the value word of an ordering edge signal.  Answered once,
  //! at create() time, by HSA_AMD_AGENT_INFO_ORDERING_EDGE_SIGNAL_SUPPORTED.
  bool orderingEdgeSignals() const { return ordering_edge_signals_; }

  //! Takes a free ordering edge slot, arms it and returns its handle, or {0} if none is
  //! available.  Never blocks, never allocates and never grows the pool: a caller that
  //! cannot get a slot keeps today's host resident dependency for that one event.
  hsa_signal_t AcquireOrderingEdge(uint32_t* slot) const;

  //! Returns a slot taken by AcquireOrderingEdge() to the free list.
  void ReleaseOrderingEdge(uint32_t slot) const;
  bool isXgmi() const override { return isXgmi_; }

  //! SDMA engine allocation for per-stream affinity
  uint32_t AllocateSdmaEngine(VirtualGPU* vgpu, HwQueueEngine engine_type,
                              hsa_agent_t peerAgent, hsa_agent_t copyAgent) const {
    return sdma_engine_allocator_.AllocateEngine(vgpu, engine_type, peerAgent, copyAgent);
  }
  void ReleaseSdmaEngine(VirtualGPU* vgpu) const {
    sdma_engine_allocator_.ReleaseEngine(vgpu);
  }
  //! Returns the map of code objects to kernels
  const auto& KernelMap() const { return kernel_map_; }
  //! Adds a kernel to the kernel map
  void AddKernel(Kernel& gpuKernel) const;
  //! Removes a kernel from the kernel map
  void RemoveKernel(Kernel& gpuKernel) const;

  // Returns the number of allocated queues for a given priority on this device
  uint32_t NumQueues(uint qIndex) const { return num_queues_[qIndex].load(); }

  //! enum for keeping the total and available queue priorities
  enum QueuePriority : uint { Low = 0, Normal = 1, High = 2, Total = 3 };

  //! Returns the number of hardware pipes
  uint32_t NumHwPipes() const { return numHwPipes_; }

  //! ⭐ THE ONE PLACEMENT-POLICY GATE. Every Phi site must ask this and nothing else.
  //! ⛔ It exists because the gate was previously spelled out at each site and the spellings
  //! DIVERGED: signal forcing tested the regime, the `d` sampler tested only `queue_phi_ != 0`.
  //! That asymmetry was inert only because no ordinary dispatch carries a caller-requested signal --
  //! which is the very assumption forcing exists to work around, so the two would have disagreed
  //! exactly where it mattered. One predicate, one place.
  //! Regime: cap <= numHwPipes_ means at most one queue per pipe, so W(n) == 0 and only ring
  //! sharing is priced. Outside it the policy declines (counted as `declined_regime`).
  bool PhiActive() const {
    return settings().queue_phi_ != 0 && settings().max_hw_queues_ <= numHwPipes_;
  }

  //! ⭐ SHADOW: evaluate the policy and LOG what it would have chosen, changing nothing. This is
  //! how we find out whether the policy has any reach at all before a line of decision code
  //! exists -- and it is also where the cross-thread read of another stream's counters gets shaken
  //! out, because here a torn read costs a wrong LOG LINE rather than a wrong placement.
  bool PhiShadow() const { return PhiActive() && settings().queue_phi_ >= 2; }
  //! ⭐ PHI=3: additionally TIME the shadow body. ⛔ The cost of this instrument is a first-class
  //! quantity -- it runs inside `active_queue_access_`, which serialises every queue acquire and
  //! release device-wide, and in the target workload decisions are ~15x more frequent than in any
  //! toy we have. Today there is no way to answer "what did the shadow cost" from a production log
  //! at all. It is its OWN mode because the timestamp read is ~58-110 ns against a body of ~200-300
  //! ns, i.e. it would be a ~30% observer effect if it were always on.
  bool PhiTimed() const { return PhiActive() && settings().queue_phi_ >= 3; }

  //! Returns true if PM4 emulation is enabled
  bool IsPm4Emulation() const { return pm4_emulation_; }

  //! Waits until all VirtualGPU QueuedAsyncHandlers are zero (30s timeout).
  void WaitForHsaAsyncHandlersIdle() override;

  //! Destroy all queues whose destroy was deferred from the async-events thread.
  //! Must only be called on an app thread (e.g. acquireQueue, ~Device).
  void DrainDeferredQueueDestroys();

  //! Current number of queues pending deferred destroy.
  size_t DeferredQueueCount();

 private:
  bool create();

  //! Construct a new physical HSA device
  Device(hsa_agent_t bkendDevice);

  static constexpr int kDefaultNumaNode = -1;

  //! Queues with destroy deferred from an async-handler-driven ~VirtualGPU, drained on app threads.
  static constexpr size_t kDeferredQueueDrainThreshold = 8;
  std::vector<hsa_queue_t*> deferredQueueDestroy_;
  std::mutex deferredQueueDestroyLock_;

  bool SetSvmAttributesInt(const void* dev_ptr, size_t count, amd::MemoryAdvice advice,
                           bool first_alloc = false, bool use_cpu = false,
                           int numa_id = kDefaultNumaNode) const;
  static constexpr hsa_signal_value_t InitSignalValue = 1;

  static hsa_ven_amd_loader_1_03_pfn_t amd_loader_ext_table;

  std::recursive_mutex* mapCacheOps_;    //!< Lock to serialise cache for the map resources
  std::vector<amd::Memory*>* mapCache_;  //!< Map cache info structure

  bool populateOCLDeviceConstants();
  static bool isHsaInitialized_;
  static bool hostVmemSupported_;
  static std::vector<hsa_agent_t> gpu_agents_;
  static std::vector<AgentInfo> cpu_agents_;
  uint32_t preferred_numa_node_;
  std::vector<hsa_agent_t> p2p_agents_;   //!< List of P2P agents available for this device
  mutable std::mutex lock_allow_access_;  //!< To serialize allow_access calls
  hsa_agent_t bkendDevice_;
  uint32_t pciDeviceId_;
  hsa_agent_t* p2p_agents_list_ = nullptr;
  hsa_profile_t agent_profile_;
  hsa_amd_memory_pool_t group_segment_;

  AgentInfo* cpu_agent_info_;

  hsa_amd_memory_pool_t gpuvm_segment_;
  hsa_amd_memory_pool_t gpu_fine_grained_segment_;
  hsa_amd_memory_pool_t gpu_ext_fine_grained_segment_;
  hsa_signal_t prefetch_signal_;  //!< Prefetch signal, used to explicitly prefetch SVM on device
  std::atomic<int> cache_state_;  //!< State of cache, kUnknown/kFlushedToDevice/kFlushedToSystem

  size_t gpuvm_segment_max_alloc_;
  size_t alloc_granularity_;
  static constexpr bool offlineDevice_ = false;
  VirtualGPU* xferQueue_;  //!< Transfer queue, created on demand
  mutable std::once_flag xferQueueOnce_;  //!< Serialises lazy creation of xferQueue_

  std::atomic<size_t> freeMem_;       //!< Total of free memory available
  mutable std::recursive_mutex vgpusAccess_;  //!< Lock to serialise virtual gpu list access
  bool hsa_exclusive_gpu_access_;  //!< TRUE if current device was moved into exclusive GPU access
                                   //!< mode
  static address mg_sync_;         //!< MGPU grid launch sync memory (SVM location)

  //! Pre-computed metadata packet version header bits
  uint32_t metadata_version_header_ = 0;
  bool metadata_version_queried_ = false;

#ifdef _WIN32
  // D3D interop device properties
  LUID deviceLuid_;     //!< Adapter LUID for D3D interop validation
  bool luidValid_;      //!< True if LUID was successfully extracted from HSA
#endif

  struct QueueInfo {
    int refCount;             //! Reference counter. Shows how many time the queue was shared
    bool hasDedicatedQueue_;  //! True if this queue is a dedicated queue (e.g., null stream)

    // Constructor
    QueueInfo() : refCount(0), hasDedicatedQueue_(false) {}

    //! Get the current hardware queue depth (wptr - rptr)
    static uint64_t GetHwQueueDepth(hsa_queue_t* queue) {
      uint64_t wptr = Hsa::queue_load_write_index_relaxed(queue);
      uint64_t rptr = Hsa::queue_load_read_index_relaxed(queue);
      return wptr - rptr;
    }

    //! Get a combined metric for queue selection (lower is better)
    uint64_t GetLoadMetric(hsa_queue_t* queue, uint32_t mode = 1) const {
      auto depth = GetHwQueueDepth(queue);

      // Dedicated queue penalty: prefer regular queues, but use dedicated if regular queues
      // have depth > ~128 packets. Penalty = 128 << 4 = 2048.
      uint64_t dedicated_queue_penalty = hasDedicatedQueue_ ? 2048 : 0;

      // Advanced weighted metric: Give queue depth significantly more weight than refCount
      uint64_t metric = dedicated_queue_penalty + (depth << 4) + static_cast<uint64_t>(refCount);
      return metric;
    }
  };

  struct QueueCompare {
    // Customized queue compare operator to make sure the queues are sorted in the creation order
    bool operator()(hsa_queue_t* lhs, hsa_queue_t* rhs) const { return lhs->id < rhs->id; }
  };
  //! a vector for keeping Pool of HSA queues with low, normal and high priorities for recycling
  std::vector<std::map<hsa_queue_t*, QueueInfo, QueueCompare>> queuePool_;

  //! ⭐ Placement-policy accounting. EVERY path out of the gate is counted separately, including
  //! the declines, because a policy that silently does not run is indistinguishable from one that
  //! runs and does nothing -- and that ambiguity has cost this work more than any wrong constant.
  //! Printed at device teardown whenever queue_phi_ != 0.
  struct PhiStats {
    std::atomic<uint64_t> reached{0};        //!< getQueueFromPool consulted the selector at all
    std::atomic<uint64_t> eligible{0};       //!< ... and the policy was live and in-regime
    std::atomic<uint64_t> declined_regime{0};//!< ... declined: max_hw_queues_ > numHwPipes_
    std::atomic<uint64_t> bypass_preferred{0};//!< returned via the `preferred` hint, selector unused
  };
  mutable PhiStats phi_stats_;

  //! ⛔⛔ WHY THIS EXISTS INSTEAD OF ITERATING `vgpus()`. The shadow report runs inside
  //! `getQueueFromPool`, which holds ONLY `active_queue_access_`. But `vgpus_` is a
  //! `std::vector<VirtualGPU*>` that VirtualGPU's constructor `resize()`s and ~VirtualGPU
  //! `erase()`s -- both under `vgpusAccess_`, a DIFFERENT monitor. Iterating it from here is
  //! reallocation under a live iterator, i.e. a use-after-free, not the benign counter-tearing an
  //! earlier comment of mine claimed. ⛔ And it cannot be fixed by taking `vgpusAccess_` here:
  //! `createVirtualDevice` establishes the order `vgpusAccess_ -> active_queue_access_`, so that
  //! would deadlock.
  //! ⭐ A FIXED-CAPACITY array owned by the device sidesteps both: it is never resized, so there is
  //! nothing to invalidate, and each slot is written only by its owning vgpu and read without any
  //! lock. A dead vgpu clears its own slot. Worst case a reader sees a stale sample, which costs a
  //! wrong LOG LINE -- which is what the original comment claimed and is now actually true.
  //! ⚠️ Our probes could never have caught this: they create every stream up front and destroy them
  //! all at the end. vLLM creates and destroys streams throughout a run.
  static constexpr size_t kPhiMaxStreams = 256;
  static constexpr uint32_t kPhiNoSlot = 0xFFFFFFFFu;
  struct PhiStreamSlot {
    std::atomic<uint64_t> queue_id{0};    //!< 0 = not bound / not live
    std::atomic<uint64_t> d_ticks{0};
    //! ⭐ The RATE BASE, not a computed rate. The reader divides against ITS OWN `t_ref`, so all
    //! streams keep one common right edge and an idle stream DECAYS instead of going stale --
    //! which is the whole point of the epoch machinery and would have been silently undone by
    //! publishing a finished rate here.
    //! ⛔ SEQLOCKED as a PAIR. Read separately, a tear gives an OLD start with a NEW disp -- window
    //! too wide, count too small, so the rate reads too LOW. A low rate lowers rho, lowers `H_q`,
    //! and makes the ring look MORE attractive: the MERGE direction, which is the unsafe one. An
    //! earlier comment of mine called that "harmless: a slightly low rate"; the magnitude is
    //! bounded to one epoch but the DIRECTION is the wrong one, so it is closed rather than noted.
    std::atomic<uint64_t> seq{0};  //!< odd = write in progress
    std::atomic<uint64_t> base_start{0};
    std::atomic<uint64_t> base_disp{0};
    std::atomic<uint64_t> disp_now{0};
    std::atomic<bool> claimed{false};
  };
  mutable PhiStreamSlot phi_slots_[kPhiMaxStreams];
  mutable std::atomic<uint64_t> phi_slot_overflow_{0};
  //! ⛔ Highest claimed slot + 1. Without it the reader scans all `kPhiMaxStreams` slots per
  //! candidate -- 1024 atomic loads per decision at 4 candidates, REGARDLESS of how many streams
  //! are live, which is worse than the vgpu scan it replaced at small stream counts. Monotone: it
  //! never shrinks when a slot is released, so a long-lived process pays for its peak, not its
  //! current, stream count. Acceptable and bounded; noted rather than optimised.
  mutable std::atomic<uint32_t> phi_slot_hi_{0};

 public:
  //! ⛔ A STABLE slot, claimed for the vgpu's lifetime. `VirtualGPU::index()` CANNOT be used as the
  //! key: ~VirtualGPU decrements the index of every later vgpu, so slots would silently re-point to
  //! a different stream on any destruction.
  uint32_t PhiClaimSlot() const {
    for (size_t i = 0; i < kPhiMaxStreams; ++i) {
      bool expected = false;
      if (phi_slots_[i].claimed.compare_exchange_strong(expected, true,
                                                        std::memory_order_acq_rel)) {
        phi_slots_[i].queue_id.store(0, std::memory_order_relaxed);
        uint32_t hi = phi_slot_hi_.load(std::memory_order_relaxed);
        while (hi < i + 1 && !phi_slot_hi_.compare_exchange_weak(hi, static_cast<uint32_t>(i + 1),
                                                                std::memory_order_relaxed)) {
        }
        return static_cast<uint32_t>(i);
      }
    }
    phi_slot_overflow_.fetch_add(1, std::memory_order_relaxed);
    return kPhiNoSlot;  // more live streams than slots: they are simply invisible to the shadow
  }
  void PhiReleaseSlot(uint32_t idx) const {
    if (idx >= kPhiMaxStreams) return;
    // ⛔ Clear the key BEFORE releasing the claim, so a reader can never attribute this stream's
    // payload to whoever claims the slot next.
    phi_slots_[idx].queue_id.store(0, std::memory_order_release);
    phi_slots_[idx].claimed.store(false, std::memory_order_release);
  }
  void PhiPublishStream(uint32_t idx, uint64_t queue_id, uint64_t d_ticks, uint64_t base_start,
                        uint64_t base_disp, uint64_t disp_now) const {
    if (idx >= kPhiMaxStreams) return;
    PhiStreamSlot& s = phi_slots_[idx];
    const uint64_t g = s.seq.load(std::memory_order_relaxed);
    s.seq.store(g + 1, std::memory_order_release);  // odd: write in progress
    s.d_ticks.store(d_ticks, std::memory_order_relaxed);
    s.base_disp.store(base_disp, std::memory_order_relaxed);
    s.base_start.store(base_start, std::memory_order_relaxed);
    s.disp_now.store(disp_now, std::memory_order_relaxed);
    s.seq.store(g + 2, std::memory_order_release);  // even: consistent
    // Payload before key: a reader seeing a live `queue_id` sees payload at least as new.
    s.queue_id.store(queue_id, std::memory_order_release);
  }
  //! ⛔⛔ `disp_now` MUST MOVE WITH `phi_dispatches_`, NOT WITH THE SAMPLER. Published only from
  //! PhiSampleDuration, the numerator of the reader's rate is as-of the last COMPLETED sample while
  //! the denominator `t_ref - base_start` is as-of NOW, so a busy stream whose samples are rare
  //! reads a rate too LOW by exactly the sampling lag. A low rate lowers rho, lowers `H_q`, and
  //! makes the ring look MORE attractive: the MERGE direction -- the same direction the seqlock was
  //! added to close, at a far larger magnitude (one sample period, not two instructions).
  //! ⛔ It bites hardest in the workload this is aimed at: a graph batch tags ONE Phi-only signal,
  //! and the block-mean holds even that back until a whole rotation of `L` completes.
  //! ⭐ Deliberately OUTSIDE the seqlock. `disp_now` is monotone and independent of the base pair;
  //! a reader pairing a NEW `disp_now` with an OLD base over-counts the numerator, i.e. reads the
  //! rate too HIGH, which over-prices and therefore SPREADS -- the safe direction. The pair that
  //! must not tear is (base_start, base_disp), and that is exactly what `seq` covers.
  void PhiPublishDisp(uint32_t idx, uint64_t disp_now) const {
    if (idx >= kPhiMaxStreams) return;
    phi_slots_[idx].disp_now.store(disp_now, std::memory_order_relaxed);
  }
  //! Streams that found no slot, i.e. are invisible to the shadow census. ⛔ Must be REPORTED: an
  //! invisible stream lowers a ring's `n` and can make an occupied ring read as free, which
  //! corrupts the all-occupied fraction -- the single number this design hinges on -- silently.
  uint64_t PhiSlotOverflow() const {
    return phi_slot_overflow_.load(std::memory_order_relaxed);
  }

 private:
  mutable std::atomic<uint64_t> phi_sel_seq_{0};  //!< decision key for T313SEL / T313SELQ

  //! ⛔⛔ WHY A RING AND NOT ClPrint. Emitting the trace inline costs a MEASURED 7,887 ns per
  //! decision -- a synchronous fprintf+fflush per line, 35x the cost of the computation being
  //! traced -- plus 824 B/decision. At vLLM dispatch rates that both drowns the log and perturbs
  //! the very timing the trace is meant to characterise, which would make the measurement a
  //! measurement of the instrument. Records go to a preallocated ring with a relaxed atomic bump
  //! and NO I/O on the decision path; the whole thing is formatted once at device teardown.
  //! ⚠️ A ring KEEPS THE LAST N and counts what it dropped. `phi_trace_drop_` must be reported, not
  //! silently absorbed -- a truncated trace that looks complete is worse than no trace.
  //! ⚠️ An unclean kill (SIGKILL of a server) loses the trace entirely. Arrange a clean shutdown.
  struct PhiSelRec {
    //! ⭐ `t_ref` is FREE: PhiShadowReport already reads it once per decision for the rate's common
    //! right edge, so recording it is one store of a value we have. It is also the single
    //! highest-value field: without it the trace CANNOT BE SEGMENTED IN TIME, and the in-workload
    //! window's boundary moves the unweighted headline from 21.4% to 48.8% or 2.0%.
    uint64_t t_ref;
    uint64_t seq, stock, phi_w, phi_u;
    uint32_t cands, n_free, n_meas, mult_w, mult_u;
    int8_t eval_w, eval_u, agree_w, agree_u, wu_same, via_bypass;
  };
  //! ⭐ `rc`/`ded` are stock's OWN view of the same candidate at the same instant (read before the
  //! refCount increment). They are here so the shadow census can be validated IN BAND: `n` is what
  //! the slot array can see, `rc` is ground truth for "is this ring occupied", and `n != rc` is a
  //! census miss that would otherwise be invisible. `ded` is stock's 2048-penalty term, which Phi
  //! has no analogue for -- without it a disagreement caused by that term is indistinguishable from
  //! one caused by the ranking.
  struct PhiSelQRec {
    uint64_t seq, q;
    uint32_t n, n_unk, rc;
    uint8_t ded;
    //! ⭐ WHERE `d` REACHES THE BIND-TIME DECISION. Everywhere else it cancels identically:
    //! `rho/d = (rate*d)/d = rate`, so `H_w = sum rate` regardless of `d`. It survives only when
    //! (a) the rho clamp fires (`rate > 1/d` => the weighted term becomes `1/d`, i.e. the general
    //! form is `min(rate, 1/d)` and the clamp is the `1/d` half of it), or (c) `d` is known but the
    //! RATE is not, so rho defaults to 1 and the contribution is `1/d`. (c) is common at STARTUP,
    //! before an epoch base exists — a candidate explanation for lifetime-vs-window divergence.
    //! ⛔⛔ `n_nod` IS NOT A DIAGNOSTIC COUNTER, IT IS PART OF THE VALUE. Streams with a rate and no
    //! `d` contribute `rate` to `H_w` and NOTHING to `T_w`/`T_u`/`H_u`, so those three are
    //! aggregates over `n - (n_unk + n_nod)` streams, not over `n`. Any scorer that compares them
    //! across rings without normalising by that count systematically prefers the ring with more
    //! unmeasured streams — the merge-ward trap `n_unk` already exists to guard. The count travels
    //! with the value; do not drop it from a reader.
    //! ⚠️ WIRE LAYOUT UNCHANGED. This byte was `n_imput` in the trace-v2 layout (18fa2f2611), which
    //! no collected trace carries — the DSV4 production traces are v1, where all three bytes are
    //! reader padding. Renamed rather than added, so `PhiSelQRec` stays 48 B and every existing
    //! reader is unaffected.
    uint8_t n_clamp, n_nod, n_norate;
    float Tw, Hw, Tu, Hu;
  };
  //! ⭐ PER-STREAM SNAPSHOT. The ring aggregates above cannot answer "what is the duty
  //! distribution" or "which ring is this stream on over time" — both of which we have already had
  //! to reconstruct by one-off analysis. Written periodically, amortised to ~10 B/decision.
  //! ⭐ FUTURE-PROOFING FOR REBIND: there `d` stops cancelling (`T_q = sum rate*d^2`, and dPhi needs
  //! the joiner's own `d`), the `dPhi(leave)` term needs each stream's current ring, and
  //! oscillation detection needs the per-stream sequence of rings. All three come from this record.
  //! The published rate BASE is stored rather than a computed rate, so a reader can evaluate it
  //! against any `t_ref` it likes.
  struct PhiSlotRec {
    uint64_t t_ref, queue_id, d_ticks, base_start, base_disp, disp_now;
    uint32_t slot;
    uint32_t pad_;
  };
  static constexpr size_t kPhiTraceSel = 1u << 16;   //!< 64 Ki decisions   (~2.6 MB)
  static constexpr size_t kPhiTraceSelQ = 1u << 18;  //!< 256 Ki candidates (~13 MB)
  static constexpr size_t kPhiTraceSlot = 1u << 16;  //!< 64 Ki slot samples (~3.7 MB)
  //! Snapshot cadence in DECISIONS, overridable with `DEBUG_CLR_PHI_SNAP`. ⚠️ The default of 1000
  //! yields only ONE snapshot on a 331-decision probe and four on a 3,660-decision server run --
  //! too thin to characterise a duty distribution. Bounded by decisions rather than time so the
  //! cost stays proportional to activity.
  static constexpr uint64_t kPhiSnapEveryDefault = 100;
  mutable uint64_t phi_snap_every_ = kPhiSnapEveryDefault;
  mutable PhiSlotRec* phi_slot_buf_ = nullptr;
  mutable std::atomic<uint64_t> phi_trace_slot_n_{0};
  //! ⛔⛔ THE TRACE MUST SURVIVE A SIGNAL. MEASURED: with the ring in heap memory and dumped from
  //! ~Device, a mid-run **SIGTERM** loses the ENTIRE trace (SELSUM=0, SEL=0), as does SIGKILL,
  //! while a clean exit yields SELSUM=1. SIGTERM matters because that is how a server is normally
  //! stopped -- so even a GRACEFUL shutdown lost everything, and under vLLM's multiprocess executor
  //! ~Device may not run at all.
  //! ⭐ When `DEBUG_CLR_PHI_TRACE` names a directory, the rings are backed by an mmap'd file
  //! instead. Records land in the page cache as they are written, so nothing is lost to a signal,
  //! and the decision path pays exactly what it paid before -- a store to mapped memory.
  //! ⭐ One file per PROCESS and device, named with the pid: eight TP ranks are eight processes with
  //! eight independent `seq` counters, and a scorer that pools them across ranks is wrong. Separate
  //! files make that mistake impossible rather than merely documented.
  mutable std::vector<PhiSelRec> phi_trace_sel_;
  mutable std::vector<PhiSelQRec> phi_trace_selq_;
  //! Non-null when the mmap sink is active; these alias the mapping, not the vectors.
  mutable PhiSelRec* phi_sel_buf_ = nullptr;
  mutable PhiSelQRec* phi_selq_buf_ = nullptr;
  mutable void* phi_map_ = nullptr;
  mutable size_t phi_map_len_ = 0;
  //! File header, mapped at offset 0, so a reader knows what it has without our process.
  struct PhiTraceHdr {
    uint64_t magic, version, sel_cap, selq_cap, sel_n, selq_n, pid, dev;
    uint64_t slot_cap, slot_n, snap_every, reserved_;  //!< v2
  };
  mutable PhiTraceHdr* phi_hdr_ = nullptr;
  //! Open the mmap sink if DEBUG_CLR_PHI_TRACE is set. Returns false to fall back to the heap ring.
  bool PhiTraceMapOpen() const;
  mutable std::atomic<uint64_t> phi_trace_sel_n_{0};
  mutable std::atomic<uint64_t> phi_trace_selq_n_{0};
  mutable std::atomic<uint64_t> phi_body_ticks_{0};  //!< PHI=3 only: summed shadow-body duration
  mutable std::atomic<uint64_t> phi_body_max_{0};    //!< PHI=3 only: worst single body

 public:
  //! Format and emit the whole trace. Called once, at device teardown.
  void PhiTraceDump() const;
  //! Write one PhiSlotRec per live slot. Called every kPhiSnapEvery decisions.
  void PhiTraceSnapshot(uint64_t t_ref) const;

 private:

  //! ⭐ Per-stream estimator snapshots, deposited by ~VirtualGPU. ⛔ Printing only from
  //! ~VirtualGPU misses every stream the program never destroyed; printing only from ~Device
  //! misses ALL of them, because `vgpus_` is already empty by then (verified: the device line
  //! prints, the per-stream lines do not). The union of the two covers both.
  struct PhiStreamSnapshot { uint64_t dispatches, d_ticks, samples, rejected, skipped, win,
                             rate_disp, rate_ticks, sweep_max, shape_ovf, shape_live,
                             shape_open; };
  mutable std::vector<PhiStreamSnapshot> phi_streams_;
  mutable amd::Monitor phi_streams_lock_;

 public:
  void PhiRecordStream(uint64_t dispatches, uint64_t d_ticks, uint64_t samples, uint64_t rejected,
                       uint64_t skipped, uint64_t win, uint64_t rate_disp, uint64_t rate_ticks,
                       uint64_t sweep_max, uint64_t shape_ovf, uint64_t shape_live,
                       uint64_t shape_open) const {
    amd::ScopedLock l(phi_streams_lock_);
    phi_streams_.push_back({dispatches, d_ticks, samples, rejected, skipped, win, rate_disp,
                            rate_ticks, sweep_max, shape_ovf, shape_live, shape_open});
  }

 private:
  amd::Monitor active_queue_access_;            //!< Lock to serialise virtual gpu list access
  std::atomic<uint32_t> num_queues_[QueuePriority::Total] = {};  //!< Per-priority queue counters

  //! Use dynamic queues mode to get a queue from pool
  //! Emit T313SEL: what stock chose vs what Phi would choose, duty-weighted and unweighted.
  //! ⛔ Must be side-effect free. Called with `active_queue_access_` held.
  //! `via_bypass` distinguishes the `preferred`-hint path (which returns WITHOUT evaluating any
  //! metric) from the comparator path. Without it the two are indistinguishable in the trace, and
  //! a policy that only replaces the comparator would look effective while never running on the
  //! path that matters.
  void PhiShadowReport(const uint qIndex, const hsa_queue_t* stock_choice,
                       const std::unordered_set<uint64_t>* excluded_ids,
                       bool via_bypass) const;

  hsa_queue_t* getQueueFromPool(const uint qIndex, bool force_reuse = false,
                                hsa_queue_t* preferred = nullptr,
                                const std::unordered_set<uint64_t>* excluded_ids = nullptr);

  //! Per-queue extras (metadata ring buffer and placement), keyed by queue pointer.
  //! Populated at queue creation, erased when non-pooled queues are destroyed.
  std::unordered_map<hsa_queue_t*, QueueExtras> queue_extras_;

  //! returns value for corresponding LinkAttrbutes in a vector given Memory pool.
  virtual bool findLinkInfo(const hsa_amd_memory_pool_t& pool,
                            std::vector<LinkAttrType>* link_attr);

  hsa_amd_memory_pool_t getHostMemoryPool(MemorySegment mem_seg,
                                          const AgentInfo* agentInfo = nullptr) const;
  //! Read and Write mask for device<->host
  uint32_t maxSdmaReadMask_;
  uint32_t maxSdmaWriteMask_;
  bool isXgmi_;  //!< Flag to indicate if there is XGMI between CPU<->GPU
  bool ordering_edge_signals_ = false;  //!< Agent can host an ordering edge signal value word
  //! Device resident ordering edge signals.  Owned by the device, not by a queue, so that
  //! they outlive every command processor that can name one and so that their number does
  //! not scale with the number of streams an application creates.
  std::vector<hsa_signal_t> edge_signals_;
  mutable std::vector<uint32_t> edge_free_;  //!< Indices of the slots nobody holds
  mutable amd::Monitor edge_pool_lock_;      //!< Serialises the two lines above
  bool pm4_emulation_ = false;  //!< Flag to indicate if PM4 emulation is enabled
  uint32_t numHwPipes_;  //!< Number of hardware pipes

  //! SDMA engine allocator for per-stream affinity
  struct SdmaEngineAllocator {
    amd::Monitor lock_;  //!< Protects the allocation state
    std::unordered_map<VirtualGPU*, uint32_t> vgpu_to_engine_;  //!< VirtualGPU -> engine mask
    std::atomic<uint32_t> next_rr_engine_{0};  //!< RR counter for sdma engine selection
    const Device& device_;  //!< Reference to parent device for accessing masks

    SdmaEngineAllocator(const Device& device) : device_(device) {}

    //! Allocate an SDMA engine for a VirtualGPU
    //! Queries HSA for engine status and preferred engines, then allocates
    //! For inter-GPU copies, strongly prefers recommended engines even if already allocated
    uint32_t AllocateEngine(VirtualGPU* vgpu, HwQueueEngine engine_type,
                           hsa_agent_t peerAgent, hsa_agent_t copyAgent);

    //! Release engine allocation for a VirtualGPU
    void ReleaseEngine(VirtualGPU* vgpu);
  };
  mutable SdmaEngineAllocator sdma_engine_allocator_;

  //! Code object to kernel info map (used in the crash dump analysis)
  mutable std::map<uint64_t, Kernel&> kernel_map_;

  //! Friend function callbackQueue can access and set device class variables.
  friend void callbackQueue(hsa_status_t status, hsa_queue_t* queue, void* data);

 public:
  std::atomic<uint> numOfVgpus_;  //!< Virtual gpu unique index

  //! Returns the valid SDMA engine bitmask for the given operation type.
  uint32_t GetSdmaValidMask(HwQueueEngine engine_type) const {
    return (engine_type == HwQueueEngine::SdmaD2H) ? maxSdmaReadMask_ : maxSdmaWriteMask_;
  }

#if defined(__clang__)
#if __has_feature(address_sanitizer)
  virtual device::UriLocator* createUriLocator() const;
#endif
#endif
};  // class roc::Device

void callbackQueue(hsa_status_t status, hsa_queue_t* queue, void* data);

}  // namespace amd::roc

/**
 * @}
 */
