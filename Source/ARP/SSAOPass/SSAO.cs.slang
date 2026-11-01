/// Copyright (C) 2026-2027 ARP (Auto Render Pipeline). All Rights Reserved.

import ACommon;
import Scene.Scene;
import Utils.Math.MathHelpers;
import Arctic.GBufferPass.GBufferHelpers;

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
};

Texture2D<float4>  gGBufferA;
Texture2D<float>   gSceneDepth;
Texture2D<float>   gZFurthest;
SamplerState       gZFurthestSampler;
Texture2D<float2>  gRandomNormal;
SamplerState       gRandomNormalSampler;
RWTexture2D<float> gAOMap;

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

float getDeviceZ(uint2 pixelPos)
{
    return gSceneDepth.Load(int3(pixelPos, 0)).r;
}

float3 getWorldNormal(uint2 pixelPos)
{
    return decodeNormal(gGBufferA.Load(int3(pixelPos, 0)).rgb);
}

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

float2 screenPosToUV(float2 screenPos)
{
    return screenPos * float2(0.5, -0.5) + float2(0.5, 0.5);
}

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

float getHZBDepth(float2 screenPos, float mipLevel)
{
    float2 hzbUV = screenPosToUV(screenPos) * hzbUVFactor;
#ifdef __A
    float deviceZ = aTrilinearSample(gZFurthest, hzbUV, mipLevel).r;
#else
    float deviceZ = gZFurthest.SampleLevel(gZFurthestSampler, hzbUV, mipLevel).r;
#endif
    return convertFromDeviceZ(deviceZ);
}

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

float computeAO(float2 uv, float2 screenPos, uint2 pixelPos, float linearZ)
{
    float3 fovFix = float3(gScene.camera.data.projMat[0][0], gScene.camera.data.projMat[1][1], 1.0);

    float3 worldNormal = getWorldNormal(pixelPos);
    float3 viewNormal = normalize(mul(float3x3(gScene.camera.data.viewMat), worldNormal));
    float3 viewPos = screenToViewPos(screenPos, linearZ);

    float actualAORadius = radius * lerp(linearZ, 1.0, radiusInWorldSpace);

    viewPos += bias * linearZ * viewNormal;

    float2 randomUV = temporalOffset + uv * float2(frameDim) / 64.0;
    float2 randomVec = (gRandomNormal.SampleLevel(gRandomNormalSampler, randomUV, 0).rg * 2.0 - 1.0) * actualAORadius;
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

    return 1.0 - (1.0 - pow(lerp(result, 1.0, saturate(linearZ * fadeMul + fadeAdd)), power)) * intensity;
}

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
        gAOMap[pixelPos] = computeAO(uv, screenPos, pixelPos, linearZ);
    }
}
