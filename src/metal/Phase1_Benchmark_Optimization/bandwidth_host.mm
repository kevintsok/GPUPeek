// Metal Bandwidth Test Host Code
// 用于在Apple M系列GPU上运行带宽基准测试

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/QuartzCore.h>

// MARK: - 辅助函数

static inline uint64_t getTimeNanos() {
    return mach_absolute_time();
}

static inline double getTimeInterval(uint64_t start, uint64_t end) {
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    uint64_t elapsed = end - start;
    return (double)elapsed * (double)timebase.numer / (double)timebase.denom / 1e9;
}

static inline void checkError(NSError *error, const char *operation) {
    if (error) {
        NSLog(@"Error %s: %@", operation, error);
        exit(1);
    }
}

// MARK: - 设备信息

void printDeviceInfo(id<MTLDevice> device) {
    NSLog(@"\n=== Apple Metal GPU Info ===");
    NSLog(@"Device Name: %s", [[device name] UTF8String]);

    // 获取GPU支持的功能
    MTLFeatureSet featureSet = MTLFeatureSet_iOS_GPUFamily1_v1;
    if (@available(iOS 13, *)) {
        featureSet = MTLFeatureSet_iOS_GPUFamily3_v1;
    }

    // 统一的内存信息
    NSLog(@"Unified Memory: Yes (Shared with CPU)");

    // 查询设备属性
    NSLog(@"Supports Family: Apple GPU");

    // 缓冲区对齐要求
    NSLog(@"Buffer Alignment: %lu bytes", (unsigned long)[device minBufferAlignment]);

    // SIMD组大小
    NSLog(@"Max Threadgroup Memory: %lu KB", (unsigned long)[device maxThreadgroupMemoryLength] / 1024);

    // 并行计算支持
    NSLog(@"Supports Compute: %s", device.supportsFamily(MTLFamilyApple4) ? "Yes" : "No");

    NSLog(@"\n");
}

// MARK: - 带宽测试

void testBandwidthCopy(id<MTLDevice> device, id<MTLLibrary> library) {
    NSLog(@"=== Memory Copy Bandwidth Test ===");

    // 创建命令队列
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        NSLog(@"Failed to create command queue");
        return;
    }

    // 创建计算管道
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bandwidth_copy"] error:&error];
    checkError(error, "create pipeline");

    // 测试配置
    const size_t bufferSize = 256 * 1024 * 1024; // 256MB
    const uint32_t iterations = 100;

    // 分配缓冲区
    id<MTLBuffer> srcBuffer = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> dstBuffer = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];

    // 初始化数据
    float *srcData = (float *)srcBuffer.contents;
    for (size_t i = 0; i < bufferSize / sizeof(float); i++) {
        srcData[i] = (float)i;
    }

    // 预热
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:srcBuffer offset:0 atIndex:0];
        [encoder setBuffer:dstBuffer offset:0 atIndex:1];
        [encoder setBytes:&bufferSize length:sizeof(bufferSize) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(bufferSize / sizeof(float), 1, 1)
                          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    // 实际测试
    uint64_t start = getTimeNanos();

    for (int i = 0; i < iterations; i++) {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:srcBuffer offset:0 atIndex:0];
        [encoder setBuffer:dstBuffer offset:0 atIndex:1];
        [encoder setBytes:&bufferSize length:sizeof(bufferSize) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(bufferSize / sizeof(float), 1, 1)
                          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    uint64_t end = getTimeNanos();
    double elapsed = getTimeInterval(start, end);

    double totalBytes = (double)bufferSize * iterations;
    double bandwidthGBs = (totalBytes / elapsed) / 1e9;

    NSLog(@"Buffer Size: %.2f MB", bufferSize / (1024.0 * 1024.0));
    NSLog(@"Iterations: %d", iterations);
    NSLog(@"Total Time: %.3f ms", elapsed * 1000);
    NSLog(@"Bandwidth: %.2f GB/s", bandwidthGBs);
    NSLog(@"\n");
}

void testBandwidthSet(id<MTLDevice> device, id<MTLLibrary> library) {
    NSLog(@"=== Memory Set Bandwidth Test ===");

    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bandwidth_set"] error:&error];
    checkError(error, "create pipeline");

    const size_t bufferSize = 256 * 1024 * 1024;
    const uint32_t iterations = 100;
    const float value = 3.14159f;

    id<MTLBuffer> dstBuffer = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];

    uint64_t start = getTimeNanos();

    for (int i = 0; i < iterations; i++) {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:dstBuffer offset:0 atIndex:0];
        [encoder setBytes:&value length:sizeof(value) atIndex:1];
        [encoder setBytes:&bufferSize length:sizeof(bufferSize) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(bufferSize / sizeof(float), 1, 1)
                          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    uint64_t end = getTimeNanos();
    double elapsed = getTimeInterval(start, end);

    double totalBytes = (double)bufferSize * iterations;
    double bandwidthGBs = (totalBytes / elapsed) / 1e9;

    NSLog(@"Bandwidth: %.2f GB/s", bandwidthGBs);
    NSLog(@"\n");
}

void testVectorAdd(id<MTLDevice> device, id<MTLLibrary> library) {
    NSLog(@"=== Vector Add Bandwidth Test ===");

    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"vector_add"] error:&error];
    checkError(error, "create pipeline");

    const size_t bufferSize = 64 * 1024 * 1024; // 64M elements
    const uint32_t iterations = 50;

    id<MTLBuffer> aBuffer = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> bBuffer = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> resultBuffer = [device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];

    float *a = (float *)aBuffer.contents;
    float *b = (float *)bBuffer.contents;
    for (size_t i = 0; i < bufferSize / sizeof(float); i++) {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    uint64_t start = getTimeNanos();

    for (int i = 0; i < iterations; i++) {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:aBuffer offset:0 atIndex:0];
        [encoder setBuffer:bBuffer offset:0 atIndex:1];
        [encoder setBuffer:resultBuffer offset:0 atIndex:2];
        [encoder setBytes:&bufferSize length:sizeof(bufferSize) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(bufferSize / sizeof(float), 1, 1)
                          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    uint64_t end = getTimeNanos();
    double elapsed = getTimeInterval(start, end);

    // 每次操作读取2个float，写入1个float = 12 bytes per element
    double totalBytes = (double)bufferSize * iterations * 3;
    double bandwidthGBs = (totalBytes / elapsed) / 1e9;

    NSLog(@"Elements: %zu M", (bufferSize / sizeof(float)) / 1024 / 1024);
    NSLog(@"Bandwidth: %.2f GB/s", bandwidthGBs);
    NSLog(@"\n");
}

// MARK: - 主函数

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSLog(@"Apple Metal GPU Bandwidth Benchmark\n");

        // 获取默认设备
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Metal is not supported on this device");
            return 1;
        }

        printDeviceInfo(device);

        // 创建库
        NSString *shaderSource = @""
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "\n"
            "kernel void bandwidth_copy(device const float* src [[buffer(0)]],\n"
            "                          device float* dst [[buffer(1)]],\n"
            "                          constant uint& size [[buffer(2)]],\n"
            "                          uint id [[thread_position_in_grid]]) {\n"
            "    dst[id] = src[id];\n"
            "}\n"
            "\n"
            "kernel void bandwidth_set(device float* dst [[buffer(0)]],\n"
            "                          constant float& value [[buffer(1)]],\n"
            "                          constant uint& size [[buffer(2)]],\n"
            "                          uint id [[thread_position_in_grid]]) {\n"
            "    dst[id] = value;\n"
            "}\n"
            "\n"
            "kernel void vector_add(device const float* a [[buffer(0)]],\n"
            "                       device const float* b [[buffer(1)]],\n"
            "                       device float* result [[buffer(2)]],\n"
            "                       constant uint& size [[buffer(3)]],\n"
            "                       uint id [[thread_position_in_grid]]) {\n"
            "    result[id] = a[id] + b[id];\n"
            "}\n";

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource
                                                      options:nil
                                                        error:&error];
        checkError(error, "create library");

        // 运行测试
        testBandwidthCopy(device, library);
        testBandwidthSet(device, library);
        testVectorAdd(device, library);

        NSLog(@"Benchmark completed.\n");
    }
    return 0;
}
