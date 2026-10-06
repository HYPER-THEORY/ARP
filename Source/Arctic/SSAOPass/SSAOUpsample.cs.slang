/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific written permission.
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

/** Bilateral upsample: half-res AO -> full-res.

    Port of UE's ComputeUpsampleContribution (PostProcessAmbientOcclusionCommon.ush)
    4-tap path (AO_UPSAMPLE_QUALITY == 0). Each tap is weighted by its bilinear
    footprint, by depth similarity between the target full-res pixel and the
    full-res pixel the tap's AO was computed from, and by normal similarity to
    fight bleeding across silhouettes. No temporal accumulation (SSAO uses the
    smooth pass instead).

    This is also where UE applies the distance fade and the user
    AmbientOcclusionPower / Intensity remap (PostProcessAmbientOcclusion.usf):
    full-res, after recombination — never at the resolution AO was computed at.

    Inputs (half-res): gAOMapHalf (RGBA32Float)
    Inputs (full-res): gGBufferA (world normal), gSceneDepth
    Outputs (full-res): gAOMap (RGBA32Float, r=AO)
*/

import Scene.Scene;
import Scene.Camera.Camera;
import Arctic.GBufferPass.GBufferHelpers;

cbuffer PerFrameCB
{
    uint2 gFullFrameDim;
    uint2 gHalfFrameDim;
    float gPower;                  // AmbientOcclusionPower — contrast power
    float gIntensity;              // AmbientOcclusionIntensity
    float gFadeMul;                // UE ScreenSpaceAOParams[4].x — 1/fadeRadius
    float gFadeAdd;                // UE ScreenSpaceAOParams[4].y — -fadeDistance/fadeRadius
    float gDepthSimilarityScale;   // UE's 0.003 tweak constant, in 1/meters
};

Texture2D<float4> gAOMapHalf;
Texture2D<float4> gGBufferA;
Texture2D<float>  gSceneDepth;

RWTexture2D<float4> gAOMap;

float ComputeDepthSimilarity(float DepthA, float DepthB, float TweakScale)
{
    return saturate(1.0f - abs(DepthA - DepthB) * TweakScale);
}

// Convert Falcor device Z (NDC, forward-Z: 0=near, 1=far) to linear view depth.
float ConvertFromDeviceZ(float deviceZ)
{
    float nearZ = gScene.camera.data.nearZ;
    float farZ = gScene.camera.data.farZ;
    if (isinf(farZ) || farZ > 1e8f)
    {
        return nearZ / max(deviceZ, 1e-6f);
    }
    return (nearZ * farZ) / max(farZ - deviceZ * (farZ - nearZ), 1e-6f);
}

[numthreads(8, 8, 1)]
void csMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFullFrameDim)) return;

    float deviceZ = gSceneDepth.Load(int3(pixel, 0)).r;

    // Background pixel: pass through (AO = 1).
    if (deviceZ >= 1.0f)
    {
        gAOMap[pixel] = float4(1.0f, 1.0f, 1.0f, 1.0f);
        return;
    }

    float centerLinearDepth = ConvertFromDeviceZ(deviceZ);
    float3 centerWorldNormal = decodeNormal(gGBufferA.Load(int3(pixel, 0)).rgb);

    // Locate this full-res pixel in half-res texel space. SSAOPass.cs.slang
    // evaluates half-res pixel `hp` at full-res pixel `2*hp + 1` (a consequence
    // of its QuarterOffset centre UV), so that tap's centre sits at full-res
    // coordinate 2*hp + 1.5. Solving 2*h + 1.5 == pixel + 0.5 for h gives the
    // exact fractional half-res coordinate — no +/-0.5 texel guesswork.
    float2 h = (float2(pixel) - 1.0f) * 0.5f;
    int2 base = int2(floor(h));
    float2 f = h - float2(base);

    // Bilinear tap footprints. UE reads the downsampled AO with a hardware
    // bilinear Texture2DSample, which it can afford because its bilateral weights
    // come from a matched half-res normal+depth texture (the setup pass we don't
    // have). We keep four discrete taps so each one can still be rejected on its
    // own depth/normal, and fold the bilinear weight in by hand: with equal tap
    // weights the result only shifted with pixel parity, which read as 2x2 blocks.
    const int2 kTapOffset[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };
    float4 tapBilinear = float4(
        (1.0f - f.x) * (1.0f - f.y),
        f.x * (1.0f - f.y),
        (1.0f - f.x) * f.y,
        f.x * f.y);

    float SmallValue = 0.0001f;
    float WeightSum = SmallValue;
    float AOSum = SmallValue;

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        int2 sampleHalfPx = clamp(base + kTapOffset[i], int2(0, 0), int2(gHalfFrameDim) - 1);

        float sampleAO = gAOMapHalf.Load(int3(sampleHalfPx, 0)).r;

        // The full-res pixel this half-res AO value was actually computed from.
        int2 sampleFullPx = clamp(sampleHalfPx * 2 + 1, int2(0, 0), int2(gFullFrameDim) - 1);
        float sampleDeviceZ = gSceneDepth.Load(int3(sampleFullPx, 0)).r;
        float sampleLinearDepth = (sampleDeviceZ >= 1.0f) ? centerLinearDepth : ConvertFromDeviceZ(sampleDeviceZ);

        float weight = tapBilinear[i];
        weight *= ComputeDepthSimilarity(sampleLinearDepth, centerLinearDepth, gDepthSimilarityScale);

        // Normal similarity: penalize samples whose AO came from a differently-oriented surface.
        float3 sampleWorldNormal = decodeNormal(gGBufferA.Load(int3(sampleFullPx, 0)).rgb);
        weight *= saturate(dot(sampleWorldNormal, centerWorldNormal));

        AOSum += sampleAO * weight;
        WeightSum += weight;
    }

    float outAO = AOSum / WeightSum;

    // Soft distance fade (PostProcessAmbientOcclusion.usf). A hard
    // depth cutoff leaves a visible seam at the cutoff distance; this ramps AO
    // to 1 over [fadeDistance, fadeDistance + fadeRadius]. The half-res pass
    // early-outs on the same expression, so the two agree.
    outAO = lerp(outAO, 1.0f, saturate(centerLinearDepth * gFadeMul + gFadeAdd));

    // User-adjustable contrast + intensity (PostProcessAmbientOcclusion.usf).
    // abs() to prevent a shader warning, as in UE.
    outAO = 1.0f - (1.0f - pow(abs(outAO), gPower)) * gIntensity;
    outAO = saturate(outAO);

    gAOMap[pixel] = float4(outAO, outAO, outAO, 1.0f);
}
