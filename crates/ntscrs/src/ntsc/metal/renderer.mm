#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

struct Params { uint32_t width,height,a,b,c,d,args[8]; float values[16]; };
struct Command { uint32_t kernel,metadata; Params p; };
static_assert(sizeof(Command)==128);
static NSString* names[] = {@"load_rgb",@"store_rgb",@"filter_scan",@"modulate",@"demod_luma",@"demod_chroma",
    @"noise",@"rotate_chroma",@"shift_plane",@"add_sparse",@"vertical_blend",@"drop_chroma",@"box_luma",@"filter_scan_1",@"filter_scan_2",@"filter_serial"};
struct DeviceState {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    std::vector<id<MTLComputePipelineState>> pipelines;
    std::mutex poolMutex;
    std::vector<id<MTLBuffer>> pool;
    id<MTLBuffer> acquire(size_t bytes) {
        std::lock_guard<std::mutex> lock(poolMutex);
        for(size_t i=0;i<pool.size();i++) if(pool[i].length>=bytes) {
            auto result=pool[i];pool.erase(pool.begin()+i);return result;
        }
        return [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    }
    void release(id<MTLBuffer> buffer) {
        std::lock_guard<std::mutex> lock(poolMutex);
        // Bound cached scratch memory, including after rendering larger resolutions.
        if(pool.size()<2 && buffer.length<=3840ull*2160*6*4) pool.push_back(buffer);
    }
};
static std::shared_ptr<DeviceState> stateFor(id<MTLDevice> device,const char* source,size_t length) {
    static std::mutex mutex;
    static std::unordered_map<uint64_t,std::shared_ptr<DeviceState>> states;
    std::lock_guard<std::mutex> lock(mutex);
    auto existing=states.find(device.registryID);
    if(existing!=states.end()) return existing->second;
    auto state=std::make_shared<DeviceState>();state->device=device;state->queue=[device newCommandQueue];
    NSError* error=nil;
    auto options=[MTLCompileOptions new]; options.fastMathEnabled=NO;
    NSString* text=[[NSString alloc] initWithBytes:source length:length encoding:NSUTF8StringEncoding];
    id<MTLLibrary> library=[device newLibraryWithSource:text options:options error:&error];
    if(!library) {fprintf(stderr,"NTSC-rs Metal: %s\n",error.description.UTF8String);return nullptr;}
    for(NSString* name:names) {
        id<MTLFunction> function=[library newFunctionWithName:name];
        id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
        if(!pipeline || pipeline.threadExecutionWidth!=32) {
            fprintf(stderr,"NTSC-rs Metal: unsupported pipeline %s (%s)\n",name.UTF8String,error.description.UTF8String);return nullptr;
        }
        state->pipelines.push_back(pipeline);
    }
    states.emplace(device.registryID,state);return state;
}
extern "C" int ntscrs_metal_render(void* hostQueue,void* source,void* destination,
    size_t sourceBytes,size_t destinationBytes,size_t arenaBytes,
    const Command* commands,size_t count,const float* metadata,size_t metadataCount,
    const char* shader,size_t shaderLength,double* gpuMs) {
    @autoreleasepool {
        @try {
            id<MTLCommandQueue> queue=(__bridge id<MTLCommandQueue>)hostQueue;
            id<MTLDevice> device=queue?queue.device:MTLCreateSystemDefaultDevice();
            if(!device) return 1;
            auto state=stateFor(device,shader,shaderLength);if(!state) return 2;
            if(!queue) queue=state->queue;
            id<MTLBuffer> input=hostQueue?(__bridge id<MTLBuffer>)source:
                [device newBufferWithBytes:source length:sourceBytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> output=hostQueue?(__bridge id<MTLBuffer>)destination:
                [device newBufferWithLength:destinationBytes options:MTLResourceStorageModeShared];
            if(!input || !output || input.device!=device || output.device!=device || input.length<sourceBytes || output.length<destinationBytes) return 3;
            id<MTLBuffer> arena=state->acquire(arenaBytes);
            id<MTLBuffer> meta=[device newBufferWithLength:std::max(size_t(256),metadataCount*4+256) options:MTLResourceStorageModeShared];
            if(!arena || !meta) return 4;
            if(metadataCount) memcpy(meta.contents,metadata,metadataCount*4);
            id<MTLCommandBuffer> buffer=[queue commandBuffer];buffer.label=@"NTSC-rs Metal frame";
            id<MTLComputeCommandEncoder> encoder=[buffer computeCommandEncoder];
            if(!buffer || !encoder) return 5;
            [encoder setBuffer:arena offset:0 atIndex:0];
            for(size_t i=0;i<count;i++) {
                auto& op=commands[i];if(op.kernel>=state->pipelines.size()) return 6;
                auto pipeline=state->pipelines[op.kernel];
                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&op.p length:sizeof(Params) atIndex:1];
                [encoder setBuffer:meta offset:op.metadata atIndex:2];
                [encoder setBuffer:op.kernel==1?output:input offset:0 atIndex:3];
                if(op.kernel==2 || op.kernel==13 || op.kernel==14) {
                    [encoder dispatchThreadgroups:MTLSizeMake((op.p.height+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                } else {
                    size_t threads=size_t(op.p.width)*op.p.height;
                    if(op.kernel==1 && op.p.args[6]) threads=size_t(op.p.width)*op.p.args[0];
                    if(op.kernel==9) threads=op.p.args[0];
                    if(op.kernel==12 || op.kernel==15) threads=op.p.height;
                    size_t group=std::min(NSUInteger(256),pipeline.maxTotalThreadsPerThreadgroup);
                    [encoder dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(group,1,1)];
                }
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [encoder endEncoding];
            [buffer addCompletedHandler:^(id<MTLCommandBuffer> completed){
                if(completed.status==MTLCommandBufferStatusError) fprintf(stderr,"NTSC-rs Metal command failed: %s\n",completed.error.description.UTF8String);
                static std::atomic_bool reported{false};
                if(hostQueue && completed.status==MTLCommandBufferStatusCompleted && !reported.exchange(true))
                    fprintf(stderr,"NTSC-rs Metal: GPU render completed on %s\n",state->device.name.UTF8String);
                state->release(arena);
            }];
            [buffer commit];
            if(!hostQueue) {
                [buffer waitUntilCompleted];
                if(buffer.status==MTLCommandBufferStatusError) return 7;
                memcpy(destination,output.contents,destinationBytes);
                *gpuMs=(buffer.GPUEndTime-buffer.GPUStartTime)*1000.0;
            }
            return 0;
        } @catch(NSException* error) {fprintf(stderr,"NTSC-rs Metal exception: %s\n",error.description.UTF8String);return 8;}
    }
}

// Reusable Metal images for the standalone host-path verification harness.
struct TestImages { id<MTLCommandQueue> queue; id<MTLBuffer> source,destination; };
extern "C" void* ntscrs_metal_test_images(const void* source,size_t bytes,void** queue,void** src,void** dst) {
    @autoreleasepool {
        auto device=MTLCreateSystemDefaultDevice();if(!device)return nullptr;
        auto images=new TestImages{[device newCommandQueue],
            [device newBufferWithBytes:source length:bytes options:MTLResourceStorageModeShared],
            [device newBufferWithLength:bytes options:MTLResourceStorageModeShared]};
        *queue=(__bridge void*)images->queue;*src=(__bridge void*)images->source;*dst=(__bridge void*)images->destination;
        return images;
    }
}
extern "C" int ntscrs_metal_test_wait(void* handle,void* destination,size_t bytes) {
    @autoreleasepool {
        auto images=static_cast<TestImages*>(handle);auto buffer=[images->queue commandBuffer];
        [buffer commit];[buffer waitUntilCompleted];
        if(buffer.status==MTLCommandBufferStatusError)return 1;
        if(destination && bytes<=images->destination.length)memcpy(destination,images->destination.contents,bytes);
        return 0;
    }
}
extern "C" void ntscrs_metal_test_free(void* handle){delete static_cast<TestImages*>(handle);}
