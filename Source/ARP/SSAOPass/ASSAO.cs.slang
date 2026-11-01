/// Copyright (C) 2026-2027 ARP (Auto Render Pipeline). All Rights Reserved.

import ACommon;
import Scene.Scene;
import Utils.Math.MathHelpers;
import Arctic.GBufferPass.GBufferHelpers;

/** Differentiable SSAO — the reverse-mode companion of SSAOPass.

    The AO computation and early-outs mirror SSAO.cs.slang's `__A` path. Only
    the four learnable parameters are carried in a differentiable struct; all
    per-pixel inputs remain constant through the reverse pass.

    Entry points
    ------------
      - `main`     : the primal forward pass. Writes the AO image to the graph
                     texture output.
      - `backward` : seeds each pixel with its dL/dAO from `gdLdI`, propagates it
                     through the same computeAO with `bwd_diff`, and reduces one
                     sum per thread group into the partial buffer.
      - `reduce`   : sums the per-group partials into one gradient.

    The pass knows nothing about the loss. It computes the vector-Jacobian
    product dL/dparam = sum_pixels dLdI * dAO/dparam, where dLdI is dL/dAO as
    whoever consumes the AO image computed it, normalization included. A mean
    over pixels therefore puts its 1/N into dLdI, not here.
*/

cbuffer CB
{
    float  intensity;
    float  power;
    float  radius;
    float  bias;
    float  fadeMul;
    float  fadeAdd;
    float  mipLevelFactor;
    float  radiusInWorldSpace;

    uint2  frameDim;
    float2 temporalOffset;
    float2 hzbUVFactor;

    /// Thread groups per axis in the backward dispatch, i.e. ceil(frameDim /
    /// kGroupDim). `reduce` uses it to bound its stride over the partials.
    uint2  groupCount;
};

Texture2D<float4>   gGBufferA;
Texture2D<float>    gSceneDepth;
Texture2D<float>    gZFurthest;
Texture2D<float2>   gRandomNormal;
SamplerState        gRandomNormalSampler;
Texture2D<float>    gdLdI;              // dL/dAO per pixel, read by `backward`.

RWTexture2D<float>        gAOMap;           // Primal AO, graph output.
RWStructuredBuffer<float> gParamGrads;      // dL/dparam, kParamCount floats.

/// Per-group partials, written by `backward` and summed by `reduce`. Reducing in
/// shared memory first keeps the global traffic to one gradient per group.
RWStructuredBuffer<float> gGradPartials;

/// Number of differentiable parameters, in the order ASSAOPass reads them back.
static const uint kParamCount = 4;

/// Threads per reversing group, i.e. the shared-memory reduction width. Declared
/// as separate scalars because `[numthreads]` rejects a component of a vector
/// constant. Keep in step with ASSAOPass::kGroupDimX / kGroupDimY, which size the
/// per-group partial buffer.
static const uint kGroupDimX = 8;
static const uint kGroupDimY = 8;
static const uint kGroupSize = kGroupDimX * kGroupDimY;

#define SAMPLE_STEPS 3
#define SAMPLESET_ARRAY_SIZE 6

static const float2 OcclusionSamplesOffsets[SAMPLESET_ARRAY_SIZE] = {
    float2( 0.000,  0.200),
    float2( 0.325,  0.101),
    float2( 0.272, -0.396),
    float2(-0.385, -0.488),
    float2(-0.711,  0.274),
    float2( 0.060,  0.900),
};

/// The differentiable unknowns, in the same order as the gradient buffer.
/// `mipLevelFactor` and the fade pair are read from the constant buffer
/// directly in computeAO, which is what makes them constants to the reverse
/// pass.
struct SSAOParams : IDifferentiable
{
    float intensity;
    float power;
    float radius;
    float bias;
};

float getDeviceZ(uint2 pixelPos)
{
    return gSceneDepth.Load(int3(pixelPos, 0)).r;
}

float3 getWorldNormal(uint2 pixelPos)
{
    return decodeNormal(gGBufferA.Load(int3(pixelPos, 0)).rgb);
}

float2 getRandomNormal(float2 randomUV)
{
    return gRandomNormal.SampleLevel(gRandomNormalSampler, randomUV, 0).rg * 2.0 - 1.0;
}

[Differentiable]
float convertFromDeviceZ(float deviceZ)
{
    float nearZ = gScene.camera.data.nearZ;
    float farZ = gScene.camera.data.farZ;
    return (nearZ * farZ) / (farZ - deviceZ * (farZ - nearZ));
}

float2 uvToScreenPos(float2 uv)
{
    return uv * float2(2.0, -2.0) + float2(-1.0, 1.0);
}

[Differentiable]
float2 screenPosToUV(float2 screenPos)
{
    return screenPos * float2(0.5, -0.5) + float2(0.5, 0.5);
}

[Differentiable]
float3 screenToViewPos(float2 screenPos, float linearZ)
{
    float P00 = gScene.camera.data.projMat[0][0];
    float P11 = gScene.camera.data.projMat[1][1];
    float P02 = gScene.camera.data.projMat[0][2];
    float P12 = gScene.camera.data.projMat[1][2];
    return float3((screenPos.x + P02) / P00 * linearZ, (screenPos.y + P12) / P11 * linearZ, -linearZ);
}

float computeMipLevel(int sampleid, int step)
{
    float samplePos = (float(sampleid) + 0.5) / SAMPLESET_ARRAY_SIZE;
    float scale = float(step + 1) / SAMPLE_STEPS;
    return log2(mipLevelFactor * scale * samplePos);
}

[Differentiable]
float getHZBDepth(float2 screenPos, float mipLevel)
{
    float2 hzbUV = screenPosToUV(screenPos) * hzbUVFactor;
    float deviceZ = aTrilinearSample(gZFurthest, hzbUV, mipLevel).r;
    return convertFromDeviceZ(deviceZ);
}

[Differentiable]
float3 wedgeWithNormal(float2 screenPos, float2 localRandom, float3 viewPos, float3 scaledViewNormal, float mipLevel)
{
    float2 screenPosL = screenPos + localRandom;
    float2 screenPosR = screenPos - localRandom;

    float linearZL = getHZBDepth(screenPosL, mipLevel);
    float linearZR = getHZBDepth(screenPosR, mipLevel);

    float3 samplePosL = screenToViewPos(screenPosL, linearZL);
    float3 samplePosR = screenToViewPos(screenPosR, linearZR);

    float3 deltaL = samplePosL - viewPos;
    float3 deltaR = samplePosR - viewPos;

    float invNormAngleL = saturate(dot(deltaL, scaledViewNormal) / dot(deltaL, deltaL));
    float invNormAngleR = saturate(dot(deltaR, scaledViewNormal) / dot(deltaR, deltaR));

    return float3(invNormAngleL, invNormAngleR, 1.0);
}

/// The full AO integral. Differentiable in `p`; everything else is constant.
[Differentiable]
float computeAO(no_diff float2 uv, no_diff float2 screenPos, no_diff uint2 pixelPos, no_diff float linearZ, SSAOParams p)
{
    float3 fovFix = float3(gScene.camera.data.projMat[0][0], gScene.camera.data.projMat[1][1], 1.0);

    float3 worldNormal = getWorldNormal(pixelPos);
    float3 viewNormal = normalize(mul(float3x3(gScene.camera.data.viewMat), worldNormal));
    float3 viewPos = screenToViewPos(screenPos, linearZ);

    float actualAORadius = p.radius * lerp(linearZ, 1.0, radiusInWorldSpace);

    viewPos += p.bias * linearZ * viewNormal;

    float2 randomUV = temporalOffset + uv * float2(frameDim) / 64.0;
    float2 randomVec = no_diff(getRandomNormal(randomUV)) * actualAORadius;
    float2 fovFixXY = fovFix.xy * (1.0 / linearZ);
    float4 randomBase = float4(randomVec, -randomVec.y, randomVec.x) * fovFixXY.xyxy;

    float3 scaledViewNormal = viewNormal * 0.08 * lerp(linearZ, 1000.0, radiusInWorldSpace);

    float2 weightAccumulator = float2(0.0001, 0.0001);

    [unroll]
    for (int i = 0; i < SAMPLESET_ARRAY_SIZE; ++i)
    {
        float2 unrotatedRandom = OcclusionSamplesOffsets[i].xy;
        float2 localRandom = unrotatedRandom.x * randomBase.xy + unrotatedRandom.y * randomBase.zw;

        float3 localAccumulator = float3(0.0, 0.0, 0.0);

        [unroll]
        for (int step = 0; step < SAMPLE_STEPS; ++step)
        {
            float scale = float(step + 1) / SAMPLE_STEPS;
            float mipLevel = computeMipLevel(i, step);

            float3 stepSample = wedgeWithNormal(screenPos, scale * localRandom, viewPos, scaledViewNormal, mipLevel);

            localAccumulator = lerp(localAccumulator, float3(max(localAccumulator.xy, stepSample.xy), 1.0), stepSample.z);
        }

        weightAccumulator += float2(sqr(1 - localAccumulator.x) * localAccumulator.z, localAccumulator.z);
        weightAccumulator += float2(sqr(1 - localAccumulator.y) * localAccumulator.z, localAccumulator.z);
    }

    float result = weightAccumulator.x / weightAccumulator.y;

    return 1.0 - (1.0 - pow(lerp(result, 1.0, saturate(linearZ * fadeMul + fadeAdd)), p.power)) * p.intensity;
}

SSAOParams readParams()
{
    SSAOParams p;
    p.intensity = intensity;
    p.power = power;
    p.radius = radius;
    p.bias = bias;
    return p;
}

/// Forward: evaluate AO at the current parameters.
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixelPos = dispatchThreadID.xy;

    if (any(pixelPos >= frameDim))
    {
        return;
    }

    float2 uv = (float2(pixelPos) + 0.5) / float2(frameDim);
    float2 screenPos = uvToScreenPos(uv);

    float deviceZ = getDeviceZ(pixelPos);
    float linearZ = convertFromDeviceZ(deviceZ);

    if (deviceZ >= 1.0 || linearZ * fadeMul + fadeAdd >= 1.0)
    {
        gAOMap[pixelPos] = 1.0;
    }
    else
    {
        SSAOParams p = readParams();
        gAOMap[pixelPos] = computeAO(uv, screenPos, pixelPos, linearZ, p);
    }
}

/// Shared staging for the two-level reduction below. One slot per wave, sized
/// for the smallest wave any D3D12 device reports.
#define kMaxWaves (kGroupSize / 4)

/// `groupshared`, not `shared`: Slang's `shared` is a memory-space qualifier
/// that yields a non-assignable view.
groupshared float sWaveGrad[kMaxWaves][kParamCount];

/// Reduces this thread's contribution across its wave and stores the wave total
/// in the shared staging. Every thread of the group must reach this call, and
/// the group barrier after it, so dead pixels call it with a zero contribution
/// rather than returning early.
///
/// The wave width is a hardware property (32, 64, or the SIMD width on Intel),
/// so the slot index and the "one lane stores" test come from the Wave
/// intrinsics instead of being hardcoded to 32.
void reduceWave(uint gid, SSAOParams grad)
{
    const float waveIntensity = WaveActiveSum(grad.intensity);
    const float wavePower     = WaveActiveSum(grad.power);
    const float waveRadius    = WaveActiveSum(grad.radius);
    const float waveBias      = WaveActiveSum(grad.bias);

    // WaveActiveSum gives every lane the total, so exactly one of them stores it.
    if (WaveIsFirstLane())
    {
        const uint wave = gid / WaveGetLaneCount();
        sWaveGrad[wave][0] = waveIntensity;
        sWaveGrad[wave][1] = wavePower;
        sWaveGrad[wave][2] = waveRadius;
        sWaveGrad[wave][3] = waveBias;
    }
}

/// Sums the staged wave totals. Only valid after the group barrier, and only for
/// thread 0 of the group.
void combineWaves(uint waveCount, out SSAOParams outGrad)
{
    outGrad = (SSAOParams)0;
    for (uint w = 0; w < waveCount; ++w)
    {
        outGrad.intensity += sWaveGrad[w][0];
        outGrad.power     += sWaveGrad[w][1];
        outGrad.radius    += sWaveGrad[w][2];
        outGrad.bias      += sWaveGrad[w][3];
    }
}

/// Backward: propagate each pixel's dL/dAO back to the four parameters through
/// bwd_diff, and reduce the per-pixel results into one gradient per group.
/// `reduce` then sums the groups.
///
/// bwd_diff re-evaluates the integral at the current parameters rather than
/// reading the forward's output: the reverse sweep needs the primal
/// intermediates, not just the final AO. So the parameters must be the ones the
/// forward that produced this dL/dAO ran at.
[numthreads(kGroupDimX, kGroupDimY, 1)]
void backward(uint3 dispatchThreadID : SV_DispatchThreadID,
              uint3 groupID : SV_GroupID,
              uint groupIndex : SV_GroupIndex)
{
    uint2 pixelPos = dispatchThreadID.xy;
    SSAOParams grad = (SSAOParams)0;

    if (all(pixelPos < frameDim))
    {
        float2 uv = (float2(pixelPos) + 0.5) / float2(frameDim);
        float2 screenPos = uvToScreenPos(uv);

        float deviceZ = getDeviceZ(pixelPos);
        float linearZ = convertFromDeviceZ(deviceZ);

        // These pixels have AO fixed to 1 in main, hence zero parameter gradient.
        bool fixedAtOne = deviceZ >= 1.0 || linearZ * fadeMul + fadeAdd >= 1.0;
        if (!fixedAtOne)
        {
            SSAOParams p = readParams();

            // The seed is this pixel's dL/dAO, so dp.d comes back as
            // dL/dAO * dAO/dparam, the pixel's term of the vector-Jacobian product.
            DifferentialPair<SSAOParams> dp = diffPair(p, (SSAOParams)0);
            bwd_diff(computeAO)(uv, screenPos, pixelPos, linearZ, dp, gdLdI[pixelPos]);
            grad = dp.d;
        }
    }

    reduceWave(groupIndex, grad);
    GroupMemoryBarrierWithGroupSync();

    if (groupIndex != 0)
        return;

    SSAOParams groupGrad;
    combineWaves((kGroupSize + WaveGetLaneCount() - 1) / WaveGetLaneCount(), groupGrad);

    // SV_GroupID is the group's index in the dispatch grid, which is what the
    // partial buffer is indexed by. The stride is groupCount.x groups per row.
    const uint partialIndex = groupID.y * groupCount.x + groupID.x;
    gGradPartials[partialIndex * kParamCount + 0] = groupGrad.intensity;
    gGradPartials[partialIndex * kParamCount + 1] = groupGrad.power;
    gGradPartials[partialIndex * kParamCount + 2] = groupGrad.radius;
    gGradPartials[partialIndex * kParamCount + 3] = groupGrad.bias;
}

/// Collapse the per-group partials into a single gradient vector. Dispatched as
/// one group of kGroupSize threads, so it reduces through the same warp
/// primitive as the reverse pass, and the stride is bounded by the group count
/// so it is correct at every resolution.
[numthreads(kGroupSize, 1, 1)]
void reduce(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint partialCount = groupCount.x * groupCount.y;
    const uint tid = dispatchThreadID.x;

    SSAOParams grad = (SSAOParams)0;

    for (uint g = tid; g < partialCount; g += kGroupSize)
    {
        grad.intensity += gGradPartials[g * kParamCount + 0];
        grad.power     += gGradPartials[g * kParamCount + 1];
        grad.radius    += gGradPartials[g * kParamCount + 2];
        grad.bias      += gGradPartials[g * kParamCount + 3];
    }

    reduceWave(tid, grad);
    GroupMemoryBarrierWithGroupSync();

    if (tid != 0)
        return;

    // This pass is a single group, so its wave totals are the answer.
    SSAOParams totalGrad;
    combineWaves((kGroupSize + WaveGetLaneCount() - 1) / WaveGetLaneCount(), totalGrad);

    // The output is overwritten, never accumulated, so a caller never has to
    // clear it between iterations.
    gParamGrads[0] = totalGrad.intensity;
    gParamGrads[1] = totalGrad.power;
    gParamGrads[2] = totalGrad.radius;
    gParamGrads[3] = totalGrad.bias;
}
