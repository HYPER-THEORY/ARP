/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

/** Hierarchical Z (HZB) pyramid build — port of UnrealEngine's
    Engine/Shaders/Private/HZB.usf (`HZBBuildCS`), pure-depth path only.

    One 8x8 thread group consumes a 16x16 texel footprint of the parent mip and
    produces a 8x8 texel footprint of the destination mip, then reduces that in
    LDS down to a single texel for up to three further mip levels. The batch
    therefore writes DIM_MIP_LEVEL_COUNT mip levels in a single dispatch, which
    is UE's MAX_MIP_BATCH_SIZE == 4 scheme.

    Depth convention
    ----------------
    UE uses a reverse-Z device depth (1 = near, 0 = far) and so stores
    min(deviceZ) in the "furthest" HZB and max(deviceZ) in the "closest" HZB.
    Falcor's projection (math::perspective) is forward-Z (0 = near, 1 = far), so
    the mapping is mirrored:

        furthest = max(deviceZ over the footprint)
        closest  = min(deviceZ over the footprint)

    Both names keep UE's meaning ("the furthest/closest surface in this tile"),
    only the comparison direction flips.

    Parent source
    -------------
    UE reads the parent mip through an SRV and a point sampler (Gather4). The
    first batch reads the scene depth, later batches read the previous batch's
    last HZB mip. Falcor inserts whole-resource barriers when binding resources,
    so binding one HZB texture as an SRV (parent mip) and a UAV (destination
    mips) in the same dispatch would leave the texture in a single state and
    make the SRV read invalid. HZB_PARENT_IS_DEPTH selects the parent
    declaration: the scene depth stays an SRV (a depth format cannot be a UAV),
    while HZB parents are read through a UAV. UAV reads and writes of different
    mip levels of one resource are legal and need no barrier between them.

    The Gather4 footprint is reproduced with four explicit Loads. min/max are
    commutative and associative, so the order of the four taps does not affect
    the result and the reduction stays bit-identical to the sampled version.
*/

#include "ReductionCommon.slang"

#define GROUP_TILE_SIZE 8
#define MAX_MIP_BATCH_SIZE 4

cbuffer PerFrameCB
{
    float4 gDispatchThreadIdToBufferUV; ///< xy = 2 / parent size, zw = ViewRect.Min / parent size
    float2 gInvSize;                    ///< 1 / parent size
    float2 gInputViewportMaxBound;      ///< UV of the last valid texel centre of the parent
    uint2  gParentSize;                 ///< Parent texture dimensions
};

#if HZB_PARENT_IS_DEPTH
    // Batch 0: the parent is the scene depth, which cannot be bound as a UAV.
    Texture2D<float> gParentTexture;
#else
    // Later batches: the parent is the previous batch's last HZB mip.
    RWTexture2D<float> gParentTexture;
#endif
#if HZB_WRITE_FURTHEST
RWTexture2D<float> gFurthestHZBOutput[DIM_MIP_LEVEL_COUNT];
#endif
#if HZB_WRITE_CLOSEST
RWTexture2D<float> gClosestHZBOutput[DIM_MIP_LEVEL_COUNT];
#endif

groupshared float gSharedFurthestDeviceZ[GROUP_TILE_SIZE * GROUP_TILE_SIZE];
groupshared float gSharedClosestDeviceZ[GROUP_TILE_SIZE * GROUP_TILE_SIZE];

/** Writes one output texel of the given destination mip level.

    The mip index is dispatched with an explicit branch chain (as UE's
    OutputMipLevel does) instead of indexing the array with the loop variable,
    so every descriptor index is a compile-time constant.

    Note there is no rounding of the closest value here, unlike UE. UE stores
    the pyramid in PF_R16F and so biases its closest bound with RoundUpF16
    (HZB.usf) — an fp16 store rounds to nearest, which can land on the far side
    of the true value and make an occlusion test cull something visible, so UE
    pre-rounds onto the fp16 grid one ulp toward the conservative side. The
    pyramids here are R32Float: the store is exact, there is no quantization to
    counteract, and applying the bias anyway would just snap values back onto a
    coarser grid. Storing the exact per-tile min/max is the tightest correct
    bound, and any tolerance a consumer needs (surface thickness, ray-march
    epsilon) belongs to that consumer's own hit test.
*/
void outputMipLevel(uint mipLevel, uint2 outputPixelPos, float furthestDeviceZ, float closestDeviceZ)
{
#if DIM_MIP_LEVEL_COUNT >= 1
    if (mipLevel == 0)
    {
#if HZB_WRITE_FURTHEST
        gFurthestHZBOutput[0][outputPixelPos] = furthestDeviceZ;
#endif
#if HZB_WRITE_CLOSEST
        gClosestHZBOutput[0][outputPixelPos] = closestDeviceZ;
#endif
    }
#endif
#if DIM_MIP_LEVEL_COUNT >= 2
    else if (mipLevel == 1)
    {
#if HZB_WRITE_FURTHEST
        gFurthestHZBOutput[1][outputPixelPos] = furthestDeviceZ;
#endif
#if HZB_WRITE_CLOSEST
        gClosestHZBOutput[1][outputPixelPos] = closestDeviceZ;
#endif
    }
#endif
#if DIM_MIP_LEVEL_COUNT >= 3
    else if (mipLevel == 2)
    {
#if HZB_WRITE_FURTHEST
        gFurthestHZBOutput[2][outputPixelPos] = furthestDeviceZ;
#endif
#if HZB_WRITE_CLOSEST
        gClosestHZBOutput[2][outputPixelPos] = closestDeviceZ;
#endif
    }
#endif
#if DIM_MIP_LEVEL_COUNT >= 4
    else if (mipLevel == 3)
    {
#if HZB_WRITE_FURTHEST
        gFurthestHZBOutput[3][outputPixelPos] = furthestDeviceZ;
#endif
#if HZB_WRITE_CLOSEST
        gClosestHZBOutput[3][outputPixelPos] = closestDeviceZ;
#endif
    }
#endif
}

/** Reads one texel of the parent mip. `p` must be in bounds.

    The coordinate type differs between the two parents: a 2D Texture2D's Load
    takes an int3 (xy plus mip index, which is 0 because the SRV is created for a
    single mip), while an RWTexture2D's Load takes an int2.
*/
float sampleParent(int2 p)
{
#if HZB_PARENT_IS_DEPTH
    return gParentTexture.Load(int3(p, 0));
#else
    return gParentTexture.Load(p);
#endif
}

/** Reads the 2x2 texel block of the parent mip that this thread's output texel
    covers, as UE's Gather4 does.

    UE clamps the sample UV to InputViewportMaxBound - InvSize rather than
    clamping the texel coordinate, because the hardware sampler is in clamp
    addressing mode. The two are equivalent here: the offset by -0.25 texels
    biases the resolved texel coordinate to 2 * dispatchThreadId + 0.25, whose
    floor is exactly 2 * dispatchThreadId, so each gather covers
    [2 * dtid, 2 * dtid + 1] and only the clamped edge of the viewport collapses
    onto the last texel.
*/
float4 gatherParentDeviceZ(uint2 dispatchThreadId)
{
    float2 bufferUV = (float2(dispatchThreadId) + 0.5f) * gDispatchThreadIdToBufferUV.xy + gDispatchThreadIdToBufferUV.zw;

    // min(..., InputViewportMaxBound) because we don't want to sample outside of
    // the viewport when the view size has odd dimensions on X/Y axis. (UE)
    float2 uv = min(bufferUV + float2(-0.25f, -0.25f) * gInvSize, gInputViewportMaxBound - gInvSize);

    int2 maxTexel = int2(gParentSize) - int2(1, 1);
    int2 p0 = clamp(int2(floor(uv * float2(gParentSize) - 0.5f)), int2(0, 0), maxTexel);
    int2 p1 = min(p0 + int2(1, 1), maxTexel);

    return float4(
        sampleParent(int2(p0.x, p0.y)),
        sampleParent(int2(p1.x, p0.y)),
        sampleParent(int2(p0.x, p1.y)),
        sampleParent(int2(p1.x, p1.y)));
}

[numthreads(GROUP_TILE_SIZE, GROUP_TILE_SIZE, 1)]
void csMain(uint2 groupId : SV_GroupID, uint groupThreadIndex : SV_GroupIndex)
{
    // With a single destination mip no LDS reduction happens, and UE uses a
    // plain row-major mapping instead of the swizzle (HZB.usf). The two only
    // differ in which thread handles which texel, not in the set of texels
    // covered, but the row-major form avoids reading uninitialised LDS.
#if DIM_MIP_LEVEL_COUNT == 1
    uint2 groupThreadId = uint2(groupThreadIndex % GROUP_TILE_SIZE, groupThreadIndex / GROUP_TILE_SIZE);
#else
    uint2 groupThreadId = initialTilePixelPositionForReduction2x2(MAX_MIP_BATCH_SIZE - 1, groupThreadIndex);
#endif

    const uint2 groupOffset = GROUP_TILE_SIZE * groupId;
    const uint2 dispatchThreadId = groupOffset + groupThreadId;

    const float4 deviceZ = gatherParentDeviceZ(dispatchThreadId);

    float furthestDeviceZ = max(max(deviceZ.x, deviceZ.y), max(deviceZ.z, deviceZ.w));
    float closestDeviceZ = min(min(deviceZ.x, deviceZ.y), min(deviceZ.z, deviceZ.w));

    uint2 outputPixelPos = dispatchThreadId;

    outputMipLevel(0, outputPixelPos, furthestDeviceZ, closestDeviceZ);

#if DIM_MIP_LEVEL_COUNT == 1
    // Nothing more to do: a lone destination mip has no in-group reduction.
#else
    gSharedFurthestDeviceZ[groupThreadIndex] = furthestDeviceZ;
    gSharedClosestDeviceZ[groupThreadIndex] = closestDeviceZ;

    [unroll]
    for (uint mipLevel = 1; mipLevel < DIM_MIP_LEVEL_COUNT; ++mipLevel)
    {
        const uint tileSize = uint(GROUP_TILE_SIZE) >> mipLevel;
        const uint reduceBankSize = tileSize * tileSize;

        // UE only syncs when (ReduceBankSize << 2) > LaneCount, i.e. when the
        // values a thread is about to read may have been written by another
        // wave. The condition depends on the hardware wave size, so it is a
        // uniform branch; syncing unconditionally is always correct and only
        // costs a barrier, which is why it is not replicated here.
        GroupMemoryBarrierWithGroupSync();

        if (groupThreadIndex < reduceBankSize)
        {
            // The swizzled thread order puts the four LDS entries at stride
            // reduceBankSize on a real 2x2 pixel block of this mip.
            float4 parentFurthestDeviceZ;
            float4 parentClosestDeviceZ;
            parentFurthestDeviceZ[0] = furthestDeviceZ;
            parentClosestDeviceZ[0] = closestDeviceZ;

            [unroll]
            for (uint i = 1; i < 4; i++)
            {
                const uint ldsIndex = groupThreadIndex + i * reduceBankSize;
                parentFurthestDeviceZ[i] = gSharedFurthestDeviceZ[ldsIndex];
                parentClosestDeviceZ[i] = gSharedClosestDeviceZ[ldsIndex];
            }

            furthestDeviceZ = max(
                max(parentFurthestDeviceZ.x, parentFurthestDeviceZ.y),
                max(parentFurthestDeviceZ.z, parentFurthestDeviceZ.w));
            closestDeviceZ = min(
                min(parentClosestDeviceZ.x, parentClosestDeviceZ.y),
                min(parentClosestDeviceZ.z, parentClosestDeviceZ.w));

            outputPixelPos = outputPixelPos >> 1;

            outputMipLevel(mipLevel, outputPixelPos, furthestDeviceZ, closestDeviceZ);

            gSharedFurthestDeviceZ[groupThreadIndex] = furthestDeviceZ;
            gSharedClosestDeviceZ[groupThreadIndex] = closestDeviceZ;
        }
    }
#endif
}
