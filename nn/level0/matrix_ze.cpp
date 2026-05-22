//go:build gpu
#include <level_zero/ze_api.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include "matrix_ze.h"

#define CHECK_ZE(a) do { \
    ze_result_t res = (a); \
    if (res != ZE_RESULT_SUCCESS) { \
        std::cerr << "Level Zero Error: " << std::hex << res << " at " << #a << std::endl; \
        exit(1); \
    } \
} while (0)

static ze_context_handle_t context = nullptr;
static ze_device_handle_t device = nullptr;
static ze_command_queue_handle_t queue = nullptr;
static ze_module_handle_t module = nullptr;
static ze_kernel_handle_t kernel = nullptr;
static ze_driver_handle_t driver = nullptr;

static std::vector<uint8_t> read_file(const std::string& filename) {
    std::ifstream is(filename, std::ios::binary | std::ios::ate);
    if (!is.good()) return {};
    std::streamsize size = is.tellg();
    is.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if (!is.read((char*)buffer.data(), size)) return {};
    return buffer;
}

extern "C" void ze_init() {
    if (context) return; // Already initialized

    CHECK_ZE(zeInit(0));

    uint32_t driverCount = 0;
    CHECK_ZE(zeDriverGet(&driverCount, nullptr));
    std::vector<ze_driver_handle_t> allDrivers(driverCount);
    CHECK_ZE(zeDriverGet(&driverCount, allDrivers.data()));
    driver = allDrivers[0];

    uint32_t deviceCount = 0;
    CHECK_ZE(zeDeviceGet(driver, &deviceCount, nullptr));
    std::vector<ze_device_handle_t> allDevices(deviceCount);
    CHECK_ZE(zeDeviceGet(driver, &deviceCount, allDevices.data()));
    
    for(auto d : allDevices) {
        ze_device_properties_t props = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
        CHECK_ZE(zeDeviceGetProperties(d, &props));
        if (props.type == ZE_DEVICE_TYPE_GPU) {
            device = d;
            std::cout << "Level Zero Initialized on: " << props.name << std::endl;
            break;
        }
    }

    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    CHECK_ZE(zeContextCreate(driver, &contextDesc, &context));

    uint32_t queueGroupCount = 0;
    CHECK_ZE(zeDeviceGetCommandQueueGroupProperties(device, &queueGroupCount, nullptr));
    std::vector<ze_command_queue_group_properties_t> queueGroupProps(queueGroupCount);
    for(auto& p : queueGroupProps) p.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    CHECK_ZE(zeDeviceGetCommandQueueGroupProperties(device, &queueGroupCount, queueGroupProps.data()));

    uint32_t computeQueueGroupIndex = -1;
    for (uint32_t i = 0; i < queueGroupCount; i++) {
        if (queueGroupProps[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            computeQueueGroupIndex = i;
            break;
        }
    }

    ze_command_queue_desc_t queueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
    queueDesc.ordinal = computeQueueGroupIndex;
    queueDesc.index = 0;
    queueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    CHECK_ZE(zeCommandQueueCreate(context, device, &queueDesc, &queue));

    auto spirv = read_file("nn/level0/kernel.spv_bmg.spv");
    if (spirv.empty()) {
        std::cerr << "Error: Could not read kernel.spv_bmg.spv" << std::endl;
        exit(1);
    }
    ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC};
    moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    moduleDesc.inputSize = spirv.size();
    moduleDesc.pInputModule = spirv.data();
    CHECK_ZE(zeModuleCreate(context, device, &moduleDesc, &module, nullptr));

    ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC};
    kernelDesc.pKernelName = "matrix_multiply";
    CHECK_ZE(zeKernelCreate(module, &kernelDesc, &kernel));
}

extern "C" void ze_cleanup() {
    if (kernel) zeKernelDestroy(kernel);
    if (module) zeModuleDestroy(module);
    if (queue) zeCommandQueueDestroy(queue);
    if (context) zeContextDestroy(context);
    kernel = nullptr; module = nullptr; queue = nullptr; context = nullptr;
}

extern "C" void ze_matrix_multiply(const float* A, const float* B, float* C, int N) {
    if (!context) ze_init();

    ze_device_mem_alloc_desc_t deviceDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    ze_host_mem_alloc_desc_t hostDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    float *zeA, *zeB, *zeC;
    size_t matrixSize = N * N * sizeof(float);
    CHECK_ZE(zeMemAllocShared(context, &deviceDesc, &hostDesc, matrixSize, 1, device, (void**)&zeA));
    CHECK_ZE(zeMemAllocShared(context, &deviceDesc, &hostDesc, matrixSize, 1, device, (void**)&zeB));
    CHECK_ZE(zeMemAllocShared(context, &deviceDesc, &hostDesc, matrixSize, 1, device, (void**)&zeC));

    memcpy(zeA, A, matrixSize);
    memcpy(zeB, B, matrixSize);

    CHECK_ZE(zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &zeA));
    CHECK_ZE(zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &zeB));
    CHECK_ZE(zeKernelSetArgumentValue(kernel, 2, sizeof(void*), &zeC));
    CHECK_ZE(zeKernelSetArgumentValue(kernel, 3, sizeof(int), &N));

    ze_command_list_desc_t listDesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    ze_command_list_handle_t cmdList;
    CHECK_ZE(zeCommandListCreate(context, device, &listDesc, &cmdList));

    uint32_t groupSizeX = 16, groupSizeY = 16;
    CHECK_ZE(zeKernelSetGroupSize(kernel, groupSizeX, groupSizeY, 1));
    ze_group_count_t dispatch = { (uint32_t)N / groupSizeX, (uint32_t)N / groupSizeY, 1 };

    CHECK_ZE(zeCommandListAppendLaunchKernel(cmdList, kernel, &dispatch, nullptr, 0, nullptr));
    CHECK_ZE(zeCommandListClose(cmdList));
    CHECK_ZE(zeCommandQueueExecuteCommandLists(queue, 1, &cmdList, nullptr));
    CHECK_ZE(zeCommandQueueSynchronize(queue, UINT64_MAX));

    memcpy(C, zeC, matrixSize);

    zeMemFree(context, zeA);
    zeMemFree(context, zeB);
    zeMemFree(context, zeC);
    zeCommandListDestroy(cmdList);
}
