#pragma once
#include <assert.h>
#include <vector>
#include <map>
#include <stdint.h>
#include <memory>
#include <iostream>

// Synapse :: Habana Synapse training API
#include <synapse_api.h>
// HCCL :: Habana Collective Communications Library
#include "hccl_common.hpp"
#include "hccl_env.hpp"


#ifdef NDEBUG
#define HASSERT(x) x
#else
#define HASSERT(x) assert(x)
#endif


typedef std::shared_ptr<InternalStreamHandle> hpu_shared_stream_t;

struct HpuDeleter {
  const uint32_t device_id_;
  HpuDeleter(uint32_t _id) : device_id_(_id) {}

  void operator()(void* obj) const {
    if (obj) {
      auto status = synDeviceFree(device_id_, (uint64_t)obj, 0);
      HASSERT(status == synSuccess && "Failed to synDeviceFree");
      // printf("Free HPU memory\n");
    }
  }
};

struct HostDeleter {
  const uint32_t device_id_;
  HostDeleter(uint32_t _id) : device_id_(_id) {}

  void operator()(void* obj) const {
    if (obj) {
      auto status = synHostFree(device_id_, obj, 0);
      HASSERT(status == synSuccess && "Failed to synDeviceFree");
      // printf("Free Host memory\n");
    }
  }
};

template <typename _T>
struct hpu_vector {
  hpu_vector() : size_(0) {}

  void resize(uint64_t _size, uint32_t _id) {
    size_ = _size;
    _T* tmp = nullptr;
    auto status = synDeviceMalloc(_id, _size * sizeof(_T), 0, 0, (uint64_t*)&tmp);
    HASSERT(status == synSuccess && "Failed to synDeviceMalloc");
    ptr_ = std::shared_ptr<_T>(tmp, HpuDeleter(_id));
  }

  inline uint64_t size() { return size_; }

  inline _T* data() { return ptr_.get(); }

  std::shared_ptr<_T> ptr_;
  uint64_t size_;
};

template <typename _T>
struct host_vector {
  host_vector() : size_(0) {}

  void resize(uint64_t _size, uint32_t _id) {
    size_ = _size;
    _T* tmp = nullptr;
    auto status = synHostMalloc(_id, _size * sizeof(_T), 0, (void**)&tmp);
    HASSERT(status == synSuccess && "Failed to synHostMalloc");
    ptr_ = std::shared_ptr<_T>(tmp, HostDeleter(_id));
  }

  inline uint64_t size() { return size_; }

  inline _T* data() { return ptr_.get(); }

  std::shared_ptr<_T> ptr_;
  uint64_t size_;
};

struct HpuStreamDeleter {
  void operator()(synStreamHandle obj) const {
    if (obj) {
      synStreamDestroy(obj);
      printf("%s, %d, synStreamDestroy\n",__func__, __LINE__);
    }
  }
};

static inline hpu_shared_stream_t createSmartStream(uint32_t _dev_id) {
    synStreamHandle tmp = NULL;
    HASSERT(synSuccess == synStreamCreateGeneric(&tmp, _dev_id, 0) && "synStreamCreateGeneric");
    return hpu_shared_stream_t(tmp, HpuStreamDeleter());
};

static constexpr double bytes2G() { return (double)1024 * 1024 * 1024; }

class hpu_device {
 public:

  synDeviceType device_type;
  uint32_t deviceId = -1;
  int initSynapse = 0;
  uint32_t sramSize;
  uint64_t dramSize;
  uint32_t deviceCount = 0;
  hpu_shared_stream_t default_stream_;


  hpu_device() {
    _initSynapse();
    _acquireDevice();
    if (deviceId == -1) {
      printf("No device");
      exit(-1);
    }
    printf("Using DeviceID: %d, DeviceName: %s\n", deviceId, getDeviceName().c_str());
    default_stream_ = createSmartStream(deviceId);

  };

    ~hpu_device() {
        _releaseDeviceId();
        _destroySynapse();
    }

  std::string getDeviceName() {
    std::string deviceNames[5] = {"GOYA", "GRECO", "GAUDI", "GAUDI HL2000M", "GAUDI2"};
    return deviceNames[device_type];
  }

  void memcpyH2D_async(void* src, void* dst, size_t _size) {
    synMemCopyAsync(default_stream_.get(), (uint64_t)src, _size, (uint64_t)dst, HOST_TO_DRAM);
  }

  void memcpyD2H_async(void* src, void* dst, size_t _size) {
    synMemCopyAsync(default_stream_.get(), (uint64_t)src, _size, (uint64_t)dst, DRAM_TO_HOST);
  }

  void memcpyD2D_async(void* src, void* dst, size_t _size) {
    synMemCopyAsync(default_stream_.get(), (uint64_t)src, _size, (uint64_t)dst, DRAM_TO_DRAM);
  }

  void zero(void* dst, size_t _size) { synMemsetD8Async((uint64_t)dst, 0, _size, default_stream_.get()); }

  void sync() { synStreamSynchronize(default_stream_.get()); }

  template <typename T>
  hpu_vector<T> create_hpumem(size_t _size) {
    hpu_vector<T> tmp;
    tmp.resize(_size, deviceId);
    return tmp;
  }

  template <typename T>
  host_vector<T> to_host(hpu_vector<T>& dev) {
    return hpu2host<T>(dev.data(), dev.size(), default_stream_.get(), deviceId);
  }

  template <typename T>
  host_vector<T> to_host(const T* devptr, size_t _size) {
    return hpu2host<T>(devptr, _size, default_stream_.get(), deviceId);
  }

  template <typename T>
  host_vector<T> create_hostmem(size_t _size) {
    host_vector<T> tmp;
    tmp.resize(_size, deviceId);
    return tmp;
  }


  void setDeviceType(const uint32_t deviceId) {
    synDeviceInfo deviceInfo;
    // requests the device info and extracts the type
    HASSERT(synSuccess == synDeviceGetInfo(deviceId, &deviceInfo));
    device_type = deviceInfo.deviceType;
    sramSize = deviceInfo.sramSize;
    dramSize = deviceInfo.dramSize;
    printf("sramSize = %d, dramSize = %ld\n", sramSize / bytes2G(), dramSize / bytes2G());
  }  // setDeviceType

  void _acquireDevice() {
    HASSERT(synSuccess == synDeviceAcquire(&deviceId, nullptr));
    synStatus status = synDeviceGetCount(&deviceCount);
    printf("Gaudi device count: %u\n", deviceCount);

    setDeviceType(deviceId);
  }  // _acquireDevice

  uint32_t _acquireDeviceId() {
    if (deviceId == -1) {
      _acquireDevice();
    }

    return deviceId;
  }  // _acquireDeviceId

  void _acquireDeviceModuleId() {
    const char* var = std::getenv("ID");
    if (var == NULL || strlen(var) == 0) {
      std::cout << "There is no env var with the NAME ID ## Test didnt run ##" << std::endl;
      return;
    }
    synStatus status = synDeviceAcquireByModuleId(&deviceId, std::stoi(var));
    delete[] var;
    HASSERT(synSuccess == status);
  }

  void _releaseDeviceId() {
    if (deviceId != -1) {
      HASSERT(synSuccess == synDeviceRelease(deviceId));
      deviceId = -1;
    }
  }

  void _initSynapse() {
    auto status = synInitialize();
    if (status != synSuccess) {
        printf("Failed to open Gaudi device: %d\n", status);
    } else {
        initSynapse = 1;
        printf("Gaudi device initial success: %d\n", status);
    }
  } 

  void _destroySynapse() {
    if (initSynapse != 0) {
      HASSERT(synSuccess == synDestroy());
      initSynapse = 0;
    }
  }

  };
