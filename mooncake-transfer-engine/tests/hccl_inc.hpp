#pragma once
#include <hccl.h>
#include <cstdint>


#define CHECK_SYNAPSE_STATUS(x)                                                                                        \
    {                                                                                                                  \
        const auto _res = (x);                                                                                         \
        if (_res != synSuccess)                                                                                        \
            throw std::runtime_error {"In function " + std::string {__FUNCTION__} +                                    \
                                      "(): " #x " failed with synapse error: " + std::to_string((_res))};              \
    }


using HCL_Rank = uint32_t;

static constexpr HCL_Rank HCL_INVALID_RANK = static_cast<HCL_Rank>(-1);
// Constants
static constexpr int      DATA_ELEMENTS_MAX    = 13;
static constexpr uint64_t ALLOCATED_HBM_SIZE   = (2UL * 1024 * 1024 * 1024);  // 2GB
static constexpr uint64_t AMOUNT_JUMBO_BUFFERS = (2);
static constexpr uint64_t MAX_BUFFER_COUNT     = (33UL);


struct Buffers
{
    uint64_t              inputSize;
    uint64_t              outputSize;
    std::vector<uint64_t> inputDevPtrs;
    std::vector<uint64_t> outputDevPtrs;
    uint64_t              correctnessDevPtr;
};

struct DeviceResources
{
    synDeviceId     deviceHandle;
    hcclComm_t      comm;
    HCL_Rank        commRoot;
    synStreamHandle collectiveStream;
    synStreamHandle deviceToHostStream;
    synStreamHandle hostToDeviceStream;
};

struct EnvData
{
    HCL_Rank              root;
    std::string           testType;
    std::string           dataType;
    uint64_t              sizeMin;
    uint64_t              sizeMax;
    uint64_t              sizeInc;
    std::string           redop;
    size_t                numIters;
    bool                  useSameBuffers;
    bool                  shouldCheckCorrectness;
    std::string           dataCSVPath;
    std::string           resultsCSVPath;
    std::string           ranksList;
    uint64_t              expectedScaleoutBW;
    HCL_Rank              rank;
    size_t                nranks;
    size_t                ranksPerNode;
    size_t                scaleupGroupSize;
    std::vector<HCL_Rank> customComm;
};

inline uint64_t getDataTypeSize(const EnvData& envData)
{
    static const std::unordered_map<std::string, uint64_t> dataTypeMap = {
        {"float", sizeof(float)},
        {"bfloat16", sizeof(uint16_t)},
    };

    auto it = dataTypeMap.find(envData.dataType);
    if (it != dataTypeMap.end())
    {
        return it->second;
    }
    else
    {
        throw std::runtime_error("Unknown data type.");
    }
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
