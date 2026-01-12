// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// How to run:
// etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls
// http://10.0.0.1:2379
// ./rdma_transport_test --mode=target  --metadata_server=127.0.0.1:2379
//   --local_server_name=127.0.0.2:12345 --device_name=erdma_0
// ./rdma_transport_test --metadata_server=127.0.0.1:2379
//   --segment_id=127.0.0.2:12345 --local_server_name=127.0.0.3:12346
//   --device_name=erdma_1

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sys/time.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>


#include "transfer_engine.h"
#include "transport/transport.h"
#include "common.h"

#ifdef USE_CUDA
#include <bits/stdint-uintn.h>
#include <cuda_runtime.h>

#ifdef USE_NVMEOF
#include <cufile.h>
#endif

#include <cassert>

static void checkCudaError(cudaError_t result, const char *message) {
    if (result != cudaSuccess) {
        LOG(ERROR) << message << " (Error code: " << result << " - "
                   << cudaGetErrorString(result) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

#ifdef USE_HPU
#include <synapse_api.h>
#include "hpu_device.hpp"
#endif

#define NR_SOCKETS (1)

DEFINE_string(local_server_name, mooncake::getHostname(),
              "Local server name for segment discovery");
DEFINE_string(metadata_server, "192.168.3.77:2379", "etcd server host address");
DEFINE_string(mode, "initiator",
              "Running mode: initiator or target. Initiator node read/write "
              "data blocks from target node");
DEFINE_string(operation, "read", "Operation type: read or write");

DEFINE_string(protocol, "rdma", "Transfer protocol: rdma|tcp");

DEFINE_string(device_name, "mlx5_2",
              "Device name to use, valid if protocol=rdma");
DEFINE_string(nic_priority_matrix, "",
              "Path to RDMA NIC priority matrix file (Advanced)");

DEFINE_string(segment_id, "192.168.3.76", "Segment ID to access data");

#ifdef USE_CUDA
DEFINE_bool(use_vram, true, "Allocate memory from GPU VRAM");
DEFINE_int32(gpu_id, 0, "GPU ID to use");
#endif

#ifdef USE_HPU
DEFINE_bool(use_vram, true, "Allocate memory from GPU VRAM");
DEFINE_int32(hpu_id, 0, "GPU ID to use, -1 for all GPUs");
#endif

using namespace mooncake;


static void *allocateMemoryPool(size_t size, int socket_id,
                                bool from_vram = false) {
#ifdef USE_CUDA
    if (from_vram) {
        int gpu_id = FLAGS_gpu_id;
        void *d_buf;
        checkCudaError(cudaSetDevice(gpu_id), "Failed to set device");
        checkCudaError(cudaMalloc(&d_buf, size),
                       "Failed to allocate device memory");
        return d_buf;
    }
#endif

#ifdef USE_HPU
    if (from_vram) {
        synStatus status;
        synDeviceId deviceId = 0;
        uint64_t* hpu_buf = nullptr;
        uint32_t flags = 0;
        uint64_t reqAddr = 0;
        status = synDeviceMalloc(deviceId, size, flags, reqAddr,  (uint64_t*)&hpu_buf);
        if (status != synSuccess) {
            std::cerr << "Failed to synDeviceMalloc" << std::endl;
            return nullptr;
        }
        std::cout << hpu_buf << std::endl;
        printf("HPU memory allocated at address: %p (size=%zu bytes)\n", hpu_buf, size);
        // return (void*)hpu_buf;
    }

    Buffers buffers;



#endif

    return numa_alloc_onnode(size, socket_id);
}

static void freeMemoryPool(void *addr, size_t size) {
#ifdef USE_CUDA
    // check pointer on GPU
    cudaPointerAttributes attributes;
    checkCudaError(cudaPointerGetAttributes(&attributes, addr),
                   "Failed to get pointer attributes");

    if (attributes.type == cudaMemoryTypeDevice) {
        cudaFree(addr);
    } else if (attributes.type == cudaMemoryTypeHost) {
        numa_free(addr, size);
    } else {
        LOG(ERROR) << "Unknown memory type";
    }
#else
    numa_free(addr, size);
#endif
}


static HCL_Rank handleCustomComm(EnvData& envData, DeviceResources& resources)
{
    // Custom comm is not supported for send_recv and scale_validation
    if (envData.testType == "send_recv" || envData.testType == "scale_validation")
    {
        throw std::runtime_error {"Custom comm is not supported for this test type"};
    }

    std::vector<HCL_Rank> peers = envData.customComm;

    // Choosing new root rank if it is not part of the custom comm.
    std::vector<HCL_Rank>::iterator rootIt = find(peers.begin(), peers.end(), envData.root);
    if (rootIt == peers.end())
    {
        rootIt       = peers.begin();
        envData.root = *peers.begin();
        if (isRoot(envData))
        {
            log() << "While building a new custom communicator, the root rank is automatically set to "
                  << *peers.begin() << "." << std::endl;
        }
    }

    // Check if the current rank is part of the custom comm
    std::vector<HCL_Rank>::iterator rankIt = find(peers.begin(), peers.end(), envData.rank);
    if (rankIt == peers.end())
    {
        log() << "HCCL demo process id (" << envData.rank << ") will not participate in the custom communicator"
              << std::endl;
#if MPI_ENABLED
        hcclUniqueId uniqueID {};
        CHECK_MPI_STATUS(MPI_Bcast(&uniqueID, sizeof(uniqueID), MPI_BYTE, envData.root, MPI_COMM_WORLD));
        CHECK_MPI_STATUS(MPI_Finalize());
#endif
        exit(0);
    }

    // In the custom comm - override params to match new custom comm
    envData.nranks     = peers.size();
    resources.commRoot = distance(peers.begin(), rootIt);

    return distance(peers.begin(), rankIt);
}

static void initDevice(EnvData& envData, DeviceResources& resources)
{
    HCL_Rank commRank = envData.rank;
    if (envData.customComm.size() == 0)
    {
        // Generate HCCL comm world
        resources.commRoot = envData.root;
        for (HCL_Rank i = 0; i < envData.nranks; i++)
        {
            envData.customComm.push_back(i);
        }
    }
    else
    {
        commRank = handleCustomComm(envData, resources);
    }

    // Initialize Synapse API context
    CHECK_SYNAPSE_STATUS(synInitialize());

    // Acquire device
    synModuleId deviceModuleID = envData.rank % envData.ranksPerNode;
    synStatus   rc             = synDeviceAcquireByModuleId(&resources.deviceHandle, deviceModuleID);
    if (rc != synSuccess)
    {
        deviceModuleID = INVALID_MODULE_ID;
        CHECK_SYNAPSE_STATUS(synDeviceAcquire(&resources.deviceHandle, nullptr));
    }

#if AFFINITY_ENABLED
    if (setupAffinity(deviceModuleID) != 0)
    {
        throw std::runtime_error {"Affinity setting for HCCL demo failed."};
    }
#endif

    // Generate unique id
    hcclUniqueId uniqueID {};
    if (isRoot(envData))
    {
        CHECK_HCCL_STATUS(hcclGetUniqueId(&uniqueID));
    }

#if MPI_ENABLED
    CHECK_MPI_STATUS(MPI_Bcast(&uniqueID, sizeof(uniqueID), MPI_BYTE, envData.root, MPI_COMM_WORLD));
#endif  // MPI_ENABLED

    // Create new HCCL communicator
    std::cout << "envData.nranks: " << envData.nranks << " envData.rank : "<< envData.rank << " commRank: " << commRank <<std::endl;
    CHECK_HCCL_STATUS(hcclCommInitRank(&resources.comm, envData.nranks, uniqueID, commRank));

    // Create Streams
    CHECK_SYNAPSE_STATUS(synStreamCreateGeneric(&resources.collectiveStream, resources.deviceHandle, 0));
    CHECK_SYNAPSE_STATUS(synStreamCreateGeneric(&resources.deviceToHostStream, resources.deviceHandle, 0));
    CHECK_SYNAPSE_STATUS(synStreamCreateGeneric(&resources.hostToDeviceStream, resources.deviceHandle, 0));
}







int initiatorWorker(TransferEngine *engine, SegmentID segment_id, int thread_id,
                    void *addr) {
    bindToSocket(0);
    auto segment_desc = engine->getMetadata()->getSegmentDescByID(segment_id);
    uint64_t remote_base = (uint64_t)segment_desc->buffers[0].addr;
    const size_t kDataLength = 4096000;
    {
        LOG(INFO) << "Stage 1: Write Data";
        for (size_t offset = 0; offset < kDataLength; ++offset)
            *((char *)(addr) + offset) = 'Z' + lrand48() % 26;

        LOG(INFO) << "Write Data: " << std::string((char *)(addr), 16) << "...";

        auto batch_id = engine->allocateBatchID(1);
        Status s;

        TransferRequest entry;
        entry.opcode = TransferRequest::WRITE;
        entry.length = kDataLength;
        entry.source = (uint8_t *)(addr);
        entry.target_id = segment_id;
        entry.target_offset = remote_base;
        s = engine->submitTransfer(batch_id, {entry});
        LOG_ASSERT(s.ok());
        bool completed = false;
        TransferStatus status;
        while (!completed) {
            Status s = engine->getTransferStatus(batch_id, 0, status);
            LOG_ASSERT(s.ok());
            if (status.s == TransferStatusEnum::COMPLETED)
                completed = true;
            else if (status.s == TransferStatusEnum::FAILED) {
                LOG(INFO) << "FAILED";
                completed = true;
            }
        }
        s = engine->freeBatchID(batch_id);
        LOG_ASSERT(s.ok());
    }

    {
        LOG(INFO) << "Stage 2: Read Data";
        auto batch_id = engine->allocateBatchID(1);
        Status s;

        TransferRequest entry;
        entry.opcode = TransferRequest::READ;
        entry.length = kDataLength;
        entry.source = (uint8_t *)(addr) + kDataLength;
        entry.target_id = segment_id;
        entry.target_offset = remote_base;
        s = engine->submitTransfer(batch_id, {entry});
        LOG_ASSERT(s.ok());
        bool completed = false;
        TransferStatus status;
        while (!completed) {
            Status s = engine->getTransferStatus(batch_id, 0, status);
            LOG_ASSERT(s.ok());
            if (status.s == TransferStatusEnum::COMPLETED)
                completed = true;
            else if (status.s == TransferStatusEnum::FAILED) {
                LOG(INFO) << "FAILED";
                completed = true;
            }
        }
        s = engine->freeBatchID(batch_id);
        LOG_ASSERT(s.ok());
    }

    int ret =
        memcmp((uint8_t *)(addr), (uint8_t *)(addr) + kDataLength, kDataLength);
    LOG(INFO) << "Read Data: " << std::string((char *)(addr) + kDataLength, 16)
              << "...";
    LOG(INFO) << "Compare: " << (ret == 0 ? "OK" : "FAILED");

    return 0;
}

std::string formatDeviceNames(const std::string &device_names) {
    std::stringstream ss(device_names);
    std::string item;
    std::vector<std::string> tokens;
    while (getline(ss, item, ',')) {
        tokens.push_back(item);
    }

    std::string formatted;
    for (size_t i = 0; i < tokens.size(); ++i) {
        formatted += "\"" + tokens[i] + "\"";
        if (i < tokens.size() - 1) {
            formatted += ",";
        }
    }
    return formatted;
}

std::string loadNicPriorityMatrix() {
    if (!FLAGS_nic_priority_matrix.empty()) {
        std::ifstream file(FLAGS_nic_priority_matrix);
        if (file.is_open()) {
            std::string content((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
            file.close();
            return content;
        }
    }
    // Build JSON Data
    auto device_names = formatDeviceNames(FLAGS_device_name);
    return "{\"cpu:0\": [[" + device_names +
           "], []], "
           " \"cpu:1\": [[" +
           device_names +
           "], []], "
           " \"cuda:0\": [[" +
           device_names + "], []]}";
}

static void
prepareBuffers(const EnvData& envData, const DeviceResources& resources, const uint64_t size, Buffers& buffers)
{
    // Calculate buffers sizes
    buffers.inputSize = size;
    if (envData.testType == "all_gather")
    {
        buffers.outputSize = size * envData.nranks;
    }
    else if (envData.testType == "reduce_scatter")
    {
        buffers.outputSize = size / envData.nranks;
    }
    else
    {
        buffers.outputSize = size;
    }

    // Validate calculated buffer size.
    if (buffers.inputSize < getDataTypeSize(envData) || buffers.outputSize < getDataTypeSize(envData))
    {
        throw std::runtime_error {"Invalid buffer size"};
    }

    // Calculate number of buffers
    const uint64_t maxBufferSize   = std::max(buffers.inputSize, buffers.outputSize);
    uint64_t       numberOfBuffers = 1;
    if (!envData.useSameBuffers)
    {
        if (maxBufferSize <= ALLOCATED_HBM_SIZE)
        {
            numberOfBuffers = (ALLOCATED_HBM_SIZE / maxBufferSize) <= AMOUNT_JUMBO_BUFFERS
                                  ? AMOUNT_JUMBO_BUFFERS
                                  : ALLOCATED_HBM_SIZE / maxBufferSize;
        }
        else  // use at least 2 buffers for performance
        {
            numberOfBuffers = 2;
        }
    }
    numberOfBuffers = std::min(numberOfBuffers, MAX_BUFFER_COUNT);

    // Allocate buffers on the device
    uint64_t inputDevPtr      = 0;
    uint64_t outputDevPtr     = 0;
    buffers.correctnessDevPtr = 0;
    CHECK_SYNAPSE_STATUS(
        synDeviceMalloc(resources.deviceHandle, buffers.inputSize * numberOfBuffers, 0, 0, &inputDevPtr));
    CHECK_SYNAPSE_STATUS(
        synDeviceMalloc(resources.deviceHandle, buffers.outputSize * numberOfBuffers, 0, 0, &outputDevPtr));

    for (uint64_t index = 0; index < numberOfBuffers; index++)
    {
        buffers.inputDevPtrs.push_back(inputDevPtr + (index * buffers.inputSize));
        buffers.outputDevPtrs.push_back(outputDevPtr + (index * buffers.outputSize));
    }

    // Set default correctness buffer on the device
    buffers.correctnessDevPtr = buffers.outputDevPtrs[0];
}

static hcclResult_t sendRecvTest(const EnvData&         envData,
                                 const DeviceResources& resources,
                                 const HCL_Rank         recvFromRank,
                                 const HCL_Rank         sendToRank,
                                 const size_t           count,
                                 const void*            sendbuff,
                                 void*                  recvbuff)
{
    hcclGroupStart();

    CHECK_HCCL_STATUS(
        hcclSend(sendbuff, count, getDataType(envData), sendToRank, resources.comm, resources.collectiveStream));
    CHECK_HCCL_STATUS(
        hcclRecv(recvbuff, count, getDataType(envData), recvFromRank, resources.comm, resources.collectiveStream));

    hcclGroupEnd();

    return hcclSuccess;
}

void sendRecvTestDefaultDriver(const EnvData&         envData,
                               const DeviceResources& resources,
                               Buffers&               buffers,
                               const uint64_t         size,
                               Stats&                 stats)
{
    // The flow of the test is as follows:
    // For single box, exchange buffer with adjacent rank. If odd number of ranks then last rank does self send/recv.
    // For scale-out test, exchange buffer with next peer rank in ring manner.
    //
    // Example:
    // 4 boxes: R0 -> R8 & R0 <- R24, R8 <- R0 & R8 -> R16, R16 <- R8 & R16 -> R24, R24 <- R16 & R24 ->R0 etc.
    // 2 boxes: R0 <> R8, R1 <> R9, etc.
    //
    // In both cases, each rank does 1 send and 1 recv from another (same) rank.
    const size_t scaleupGroupSize = envData.scaleupGroupSize;
    const size_t numOfRanks       = envData.nranks;
    size_t       numOfBoxes       = envData.nranks / envData.scaleupGroupSize;
    if (numOfRanks % scaleupGroupSize > 0)
    {
        numOfBoxes++;
    }
    const size_t ranksPerBox = numOfRanks / numOfBoxes;

    const HCL_Rank myRank   = envData.rank;
    const size_t   myBoxNum = myRank / scaleupGroupSize;

    HCL_Rank sendToRank;
    HCL_Rank recvFromRank;
    if (numOfBoxes > 1)
    // scaleout
    {
        // Do ring with adjacent boxes
        const size_t targetSendBox = myBoxNum == numOfBoxes - 1 ? 0 : myBoxNum + 1;
        sendToRank                 = targetSendBox * ranksPerBox + (myRank % ranksPerBox);
        const size_t targetRecvBox = myBoxNum == 0 ? numOfBoxes - 1 : myBoxNum - 1;
        recvFromRank               = targetRecvBox * ranksPerBox + (myRank % ranksPerBox);
    }
    else
    // single box
    {
        // send / recv from adjacent even/odd pairs ranks, i.e. R0 <>R1, R2<>R3.
        // in case of odd number of ranks - last rank will do send/recv with self.
        sendToRank   = (myRank % 2) != 0                                       ? myRank - 1
                       : ((numOfRanks % 2) && (myRank == numOfRanks - 1)) != 0 ? myRank
                                                                               : myRank + 1;
        recvFromRank = sendToRank;
    }

    int iter = 1;
    uint64_t index = iter % buffers.inputDevPtrs.size();
    CHECK_HCCL_STATUS(sendRecvTest(envData,
                                           resources,
                                           recvFromRank,
                                           sendToRank,
                                           buffers.inputSize / getDataTypeSize(envData),
                                           (const void*)buffers.inputDevPtrs[index],
                                           (void*)buffers.outputDevPtrs[index]))

    // stats.rankDurationInSec = benchmark(
    //     envData,
    //     resources,
    //     [&](uint64_t iter) {
    //         uint64_t index = iter % buffers.inputDevPtrs.size();
    //         CHECK_HCCL_STATUS(sendRecvTest(envData,
    //                                        resources,
    //                                        recvFromRank,
    //                                        sendToRank,
    //                                        buffers.inputSize / getDataTypeSize(envData),
    //                                        (const void*)buffers.inputDevPtrs[index],
    //                                        (void*)buffers.outputDevPtrs[index]));
    //     },
    //     [&]() {
    //         CHECK_HCCL_STATUS(sendRecvTest(envData,
    //                                        resources,
    //                                        recvFromRank,
    //                                        sendToRank,
    //                                        buffers.inputSize / getDataTypeSize(envData),
    //                                        (const void*)buffers.inputDevPtrs[0],
    //                                        (void*)buffers.correctnessDevPtr));
    //     });

    // // Calculate expected results for correctness check
    // if (envData.shouldCheckCorrectness)
    // {
    //     for (size_t i = 0; i < buffers.outputSize / getDataTypeSize(envData); i++)
    //     {
    //         stats.expectedOutputs.push_back(getInput(recvFromRank, envData.nranks, i));
    //     }
    // }

    stats.isDescribing = true;
}

int initiator() {
    const size_t ram_buffer_size = 1ull << 30;
    // disable topology auto discovery for testing.
    auto engine = std::make_unique<TransferEngine>(false);

    auto hostname_port = parseHostNameWithPort(FLAGS_local_server_name);
    engine->init(FLAGS_metadata_server, FLAGS_local_server_name.c_str(),
                 hostname_port.first.c_str(), hostname_port.second);

    Transport *xport = nullptr;
    if (FLAGS_protocol == "rdma") {
        auto nic_priority_matrix = loadNicPriorityMatrix();
        void **args = (void **)malloc(2 * sizeof(void *));
        args[0] = (void *)nic_priority_matrix.c_str();
        args[1] = nullptr;
        xport = engine->installTransport("rdma", args);
    } else if (FLAGS_protocol == "tcp") {
        xport = engine->installTransport("tcp", nullptr);
    } else if (FLAGS_protocol == "nvmeof") {
        xport = engine->installTransport("nvmeof", nullptr);
    } else {
        LOG(ERROR) << "Unsupported protocol";
    }

    LOG_ASSERT(xport);

    void *addr = nullptr;
#ifdef USE_CUDA
    addr = allocateMemoryPool(ram_buffer_size, 0, FLAGS_use_vram);
    std::string name_prefix = FLAGS_use_vram ? "cuda:" : "cpu:";
    int name_suffix = FLAGS_use_vram ? FLAGS_gpu_id : 0;
    int rc = engine->registerLocalMemory(
        addr, ram_buffer_size, name_prefix + std::to_string(name_suffix));
    LOG_ASSERT(!rc);
#elif defined(USE_HPU)

    hpu_device ctx;
    // auto envData = getenvData();
    DeviceResources resources;
    // initDevice(envData, resources);

    // double size = 1024;
    Buffers buffers;
    // prepareBuffers(envData, resources, size, buffers);

    printf("%s, %d, USE_HPU\n", __func__, __LINE__);
    hcclUniqueId uniqueID {};
    CHECK_HCCL_STATUS(hcclGetUniqueId(&uniqueID));
    HCL_Rank commRank = 1;
    hcclComm_t      comm;
  // Create new HCCL communicator
    CHECK_HCCL_STATUS(hcclCommInitRank(&comm, 1, uniqueID, 0));

    printf("%s, %d, USE_HPU\n", __func__, __LINE__);
    // Create Streams
    synStreamHandle stream;
    CHECK_SYNAPSE_STATUS(synStreamCreateGeneric(&stream, 0, 0));

    uint64_t inputDevPtr      = 0;
    uint64_t outputDevPtr     = 0;
    buffers.inputSize = 16;
    buffers.outputSize = 16;
    buffers.correctnessDevPtr = 0;
    uint64_t       numberOfBuffers = 1;


    CHECK_SYNAPSE_STATUS(
        synDeviceMalloc(0, buffers.inputSize * numberOfBuffers, 0, 0, &inputDevPtr));
    CHECK_SYNAPSE_STATUS(
        synDeviceMalloc(0, buffers.outputSize * numberOfBuffers, 0, 0, &outputDevPtr));
    CHECK_SYNAPSE_STATUS(synDeviceMalloc(0, buffers.outputSize, 0, 0, &buffers.correctnessDevPtr));

    printf("%s, %d, USE_HPU\n", __func__, __LINE__);
    for (uint64_t index = 0; index < numberOfBuffers; index++)
    {
        buffers.inputDevPtrs.push_back(inputDevPtr + (index * buffers.inputSize));
        buffers.outputDevPtrs.push_back(outputDevPtr + (index * buffers.outputSize));
    }

    std::vector<float> inputHostData(buffers.inputSize / sizeof(float));
    void*          inputHostDataPtr = reinterpret_cast<void*>(inputHostData.data());

    // Create inputs
    for (size_t i = 0; i < inputHostData.size(); i++)
    {
        inputHostData[i] = i + 100;
    }

    CHECK_SYNAPSE_STATUS(synHostMap(0, buffers.inputSize, inputHostDataPtr));
    CHECK_SYNAPSE_STATUS(synMemCopyAsync(stream,
                                         (uint64_t)inputHostDataPtr,
                                         buffers.inputSize,
                                         buffers.inputDevPtrs[0],
                                         HOST_TO_DRAM));
    CHECK_SYNAPSE_STATUS(synStreamSynchronize(stream));

    for (size_t i = 0; i < inputHostData.size(); i++) {
        std::cout << "=======inputHostData======== " << inputHostData[i] << std::endl;
    }


    printf("%s, %d, USE_HPU\n", __func__, __LINE__);


    auto sendbuff = (const void*)buffers.inputDevPtrs[0];
    auto recvbuff = (void*)buffers.outputDevPtrs[0];
    //auto count = buffers.inputSize / getDataTypeSize(envData);
    auto count = buffers.inputSize / sizeof(float);

    std::cout << "sendbuff:  " << sendbuff << std::endl;
    std::cout << "recvbuff: " << recvbuff << std::endl;

    printf("%s, %d, USE_HPU\n", __func__, __LINE__);

    hcclGroupStart();

    CHECK_HCCL_STATUS(
        hcclSend(sendbuff, count,  hcclFloat32, 0, comm, stream));
    printf("%s, %d, USE_HPU\n", __func__, __LINE__);
    CHECK_HCCL_STATUS(
        hcclRecv(recvbuff, count,  hcclFloat32, 0, comm, stream));

    hcclGroupEnd();

    printf("%s, %d, USE_HPU\n", __func__, __LINE__);

    CHECK_SYNAPSE_STATUS(synStreamSynchronize(stream));
    auto        outputHostData    = std::vector<float>(buffers.outputSize / sizeof(float));
    const void* outputHostDataPtr = reinterpret_cast<void*>(outputHostData.data());

    printf("%s, %d, USE_HPU\n", __func__, __LINE__);
    CHECK_SYNAPSE_STATUS(synHostMap(0, buffers.outputSize, outputHostDataPtr));
    printf("%s, %d, USE_HPU\n", __func__, __LINE__);
    CHECK_SYNAPSE_STATUS(synMemCopyAsync(stream, (uint64_t)recvbuff, buffers.outputSize, (uint64_t)outputHostDataPtr, DRAM_TO_HOST));
    printf("%s, %d, USE_HPU\n", __func__, __LINE__);
    CHECK_SYNAPSE_STATUS(synStreamSynchronize(stream));


    // CHECK_SYNAPSE_STATUS(synMemCopyAsync(stream, (uint64_t)dev_src.data(), ele_size * sizeof(float), (uint64_t)host_dst.data(), DRAM_TO_HOST));
    // CHECK_SYNAPSE_STATUS(synStreamSynchronize(stream));
    for (size_t i = 0; i < outputHostData.size(); i++) {
        std::cout << "=======outputHostData======== " << outputHostData[i] << std::endl;
    }



    // auto ele_size = 8;
    // hpu_vector<float> dev_src = ctx.create_hpumem<float>(ele_size);
    // host_vector<float> host_src = ctx.create_hostmem<float>(ele_size);
    // for (size_t i = 0; i < dev_src.size(); i++) {
    //     host_src.data()[i] = i + 10;
    // }

    // for (size_t i = 0; i < host_src.size(); i++) {
    //     std::cout << "=============== " << host_src.data()[i] << std::endl;
    // }

    // std::cout << " dev_src.data() :" << dev_src.data() << std::endl;
    // std::cout << " host_src.data() : " << host_src.data() << std::endl;
    // CHECK_SYNAPSE_STATUS(synMemCopyAsync(stream, (uint64_t)host_src.data(), ele_size * sizeof(float),
    //                 (uint64_t)dev_src.data(), HOST_TO_DRAM));

    // CHECK_HCCL_STATUS(
    //     hcclSend(dev_src.data(), ele_size,  hcclFloat32, 0, comm, stream));
    
    // // 测试一张卡上内存搬运。
    // host_vector<float> host_dst = ctx.create_hostmem<float>(ele_size);
    // CHECK_SYNAPSE_STATUS(synMemCopyAsync(stream, (uint64_t)dev_src.data(), ele_size * sizeof(float), (uint64_t)host_dst.data(), DRAM_TO_HOST));
    // CHECK_SYNAPSE_STATUS(synStreamSynchronize(stream));
    // for (size_t i = 0; i < host_dst.size(); i++) {
    //     std::cout << "=======host_dst======== " << host_dst.data()[i] << std::endl;
    // }




    // printf("%s, %d, USE_HPU\n", __func__, __LINE__);
    // hpu_vector<float> dev_dst = ctx.create_hpumem<float>(ele_size);
    // host_vector<float> host_dst = ctx.create_hostmem<float>(ele_size);

    // CHECK_HCCL_STATUS(
    //     hcclRecv(dev_dst.data(), ele_size,  hcclFloat32, 0, comm, stream));


    // CHECK_SYNAPSE_STATUS(synMemCopyAsync(stream, (uint64_t)dev_dst.data(), ele_size * sizeof(float), (uint64_t)host_dst.data(), DRAM_TO_HOST));

    // CHECK_SYNAPSE_STATUS(synStreamSynchronize(stream));
    // for (size_t i = 0; i < host_dst.size(); i++) {
    //     std::cout << "=============== " << host_dst.data()[i] << std::endl;
    // }















    printf("%s, %d, USE_HPU\n", __func__, __LINE__);
    addr = allocateMemoryPool(ram_buffer_size, 0, FLAGS_use_vram);
    printf("HPU allocated address: %p\n", addr);
    std::string name_prefix = FLAGS_use_vram ? "hpu:" : "cpu:";
    int name_suffix = FLAGS_use_vram ? FLAGS_hpu_id : 0;
    int rc = engine->registerLocalMemory(
        addr, ram_buffer_size, name_prefix + std::to_string(name_suffix));
    LOG_ASSERT(!rc);
#else
    addr = allocateMemoryPool(ram_buffer_size, 0, false);
    printf("CPU allocated address: %p\n", addr);
    int rc = engine->registerLocalMemory(addr, ram_buffer_size, kWildcardLocation);
    LOG_ASSERT(!rc);
#endif

    auto segment_id = engine->openSegment(FLAGS_segment_id.c_str());
    std::thread workers(initiatorWorker, engine.get(), segment_id, 0, addr);
    workers.join();
    engine->unregisterLocalMemory(addr);
    freeMemoryPool(addr, ram_buffer_size);
    return 0;
}

int target() {
    const size_t ram_buffer_size = 1ull << 30;
    // disable topology auto discovery for testing.
    auto engine = std::make_unique<TransferEngine>(false);

    auto hostname_port = parseHostNameWithPort(FLAGS_local_server_name);
    engine->init(FLAGS_metadata_server, FLAGS_local_server_name.c_str(),
                 hostname_port.first.c_str(), hostname_port.second);

    if (FLAGS_protocol == "rdma") {
        auto nic_priority_matrix = loadNicPriorityMatrix();
        void **args = (void **)malloc(2 * sizeof(void *));
        args[0] = (void *)nic_priority_matrix.c_str();
        args[1] = nullptr;
        engine->installTransport("rdma", args);
    } else if (FLAGS_protocol == "tcp") {
        engine->installTransport("tcp", nullptr);
    } else if (FLAGS_protocol == "nvmeof") {
        engine->installTransport("nvmeof", nullptr);
    } else {
        LOG(ERROR) << "Unsupported protocol";
    }

    void *addr = nullptr;
    addr = allocateMemoryPool(ram_buffer_size, 0);
    int rc = engine->registerLocalMemory(addr, ram_buffer_size, "cpu:0");
    LOG_ASSERT(!rc);

    while (true) sleep(1);

    engine->unregisterLocalMemory(addr);
    freeMemoryPool(addr, ram_buffer_size);
    return 0;
}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);

    
    if (FLAGS_mode == "initiator")
        return initiator();
    else if (FLAGS_mode == "target")
        return target();

    LOG(ERROR) << "Unsupported mode: must be 'initiator' or 'target'";
    exit(EXIT_FAILURE);
}
