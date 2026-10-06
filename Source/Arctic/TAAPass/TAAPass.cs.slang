/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived from
 #    this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 # LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

// Temporal anti-aliasing resolve, ported from UnrealEngine's TemporalAA.usf
// (TemporalAASample). Mirrors UE's *main* TAA permutation, i.e.
// TAA_PASS_CONFIG == 0 / TAA_QUALITY_HIGH, which resolves to:
//
//   AA_SAMPLES 5   (plus-shaped, from the #ifndef default -- NOT 9)
//   AA_HISTORY_CLAMPING_BOX  HISTORY_CLAMPING_BOX_MIN_MAX  (NOT variance)
//   AA_YCOCG 1  AA_TONE 1  AA_BICUBIC 1  AA_CROSS 2
//   AA_CLIP 0  AA_ROUND 0  AA_UPSAMPLE 0  AA_LERP 0
//
// Implemented here:
//   - 5-tap (plus) current-frame reconstruction filter: jitter-aware Gaussian
//     spatial weight (UE sigma = 0.47) x HDR weight
//   - plus-shaped hard min/max history clamp box, in YCgCo
//   - nearest-depth velocity dilation over a +-2 px "X" (UE AA_CROSS = 2)
//   - Catmull-Rom bicubic history resampling
//   - adaptive blend: velocity ramp + UE's minimum-contribution floor
//   - HDR-weighted final blend (UE WeightedLerpFactors) -- the firefly killer
//   - first-frame / camera-cut / off-screen history rejection
//
// Deliberately NOT implemented (disabled or inapplicable in UE's main config):
//   - the "add back aliasing to sharpen" block: UE gates it on
//     `AA_FILTERED && !AA_BICUBIC`, and the main config has AA_BICUBIC 1.
//   - responsive AA (stencil), dynamic anti-ghosting (history alpha), and
//     stochastic output quantization: moot for a static scene with an RGBA32Float history.
//
// Input motion vectors come from MotionVectorPass and are current->previous
// UV offsets, so history is sampled at UV = pixelUV + motion
// (motion already encodes the +jitter term).

import Utils.Color.ColorHelpers;

cbuffer PerFrameCB
{
    uint2  gFrameDim;
    float2 gJitterPixels;      // this frame's sub-pixel sample offset, in pixels (see below)
    float  gCurrentFrameWeight; // baseline blend factor (UE CurrentFrameWeight)
    float  gExposureScale;      // UE FrameExposureScale * OneOverPreExposure
    uint   gHasHistory;         // 0 on first frame / after realloc
    uint   gCameraCut;          // 1 => force full reset this frame
};

Texture2D<float4>  gColor;        // current-frame HDR color (CompositePass.color)
Texture2D<float2>  gMotionVecs;   // MotionVectorPass output
Texture2D<float>   gSceneDepth;   // GBufferPass.depth (device Z: 0=near, 1=far)
Texture2D<float4>  gPrevColor;    // history (ping-pong)
SamplerState gLinearSampler;

RWTexture2D<float4> gColorOut;

// UE's plus pattern: kOffsets3x3 indices { 1, 3, 4, 5, 7 }. Index 2 below is the centre.
static const int2 kPlusOffsets[5] = { int2(0, -1), int2(-1, 0), int2(0, 0), int2(1, 0), int2(0, 1) };

// Gaussian reconstruction filter exponent. UE's TemporalAA.cpp builds
// SampleWeights/PlusWeights on the CPU as exp(-2.29 * d^2) -- a normal
// distribution with sigma = 0.47 -- at r.TemporalAAFilterSize = 1.
static const float kFilterGaussianExp = 2.29;

// UE AA_CROSS for the main TAA config: radius of the depth search "X", in pixels.
static const int kVelocityCrossRadius = 2;

// UE's HDR weight, GetSceneColorHdrWeight() with AA_YCOCG:
//     rcp(Luma4 * Exposure + 4),   Luma4 = R + 2G + B
// Falcor's RGBToYCgCo is the *normalized* basis (Y = 0.25R + 0.5G + 0.25B =
// Luma4 / 4), so the identical function is 0.25 * rcp(Y * Exposure + 1). The
// constant 0.25 cancels everywhere it is used (the filter and the final blend
// both renormalize by the weight sum), so drop it and keep the knee at Y = 1.
float hdrWeightY(float y)
{
    return rcp(max(y, 0.0) * gExposureScale + 1.0);
}

// UE WeightedLerpFactors(): lerp between two colors weighted by their HDR
// weights, i.e. a lerp in tonemapped space. Returns normalized (history, current).
float2 weightedLerpFactors(float weightA, float weightB, float blend)
{
    float blendA = (1.0 - blend) * weightA;
    float blendB = blend * weightB;
    return float2(blendA, blendB) * rcp(blendA + blendB);
}

// Catmull-Rom bicubic, 9-tap optimization (ported from Falcor's TAA.ps.slang,
// originally vec3.ca bicubic-in-fewer-taps).
float3 bicubicSampleCatmullRom(Texture2D tex, SamplerState samp, float2 samplePos, float2 texDim)
{
    float2 invTextureSize = 1.0 / texDim;
    float2 tc = floor(samplePos - 0.5) + 0.5;
    float2 f = samplePos - tc;
    float2 f2 = f * f;
    float2 f3 = f2 * f;

    float2 w0 = f2 - 0.5 * (f3 + f);
    float2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    float2 w3 = 0.5 * (f3 - f2);
    float2 w2 = 1.0 - w0 - w1 - w3;

    float2 w12 = w1 + w2;

    float2 tc0  = (tc - 1.0) * invTextureSize;
    float2 tc12 = (tc + w2 / w12) * invTextureSize;
    float2 tc3  = (tc + 2.0) * invTextureSize;

    // clang-format off
    float3 result =
        tex.SampleLevel(samp, float2(tc0.x,  tc0.y),  0.0).rgb * (w0.x  * w0.y) +
        tex.SampleLevel(samp, float2(tc0.x,  tc12.y), 0.0).rgb * (w0.x  * w12.y) +
        tex.SampleLevel(samp, float2(tc0.x,  tc3.y),  0.0).rgb * (w0.x  * w3.y) +
        tex.SampleLevel(samp, float2(tc12.x, tc0.y),  0.0).rgb * (w12.x * w0.y) +
        tex.SampleLevel(samp, float2(tc12.x, tc12.y), 0.0).rgb * (w12.x * w12.y) +
        tex.SampleLevel(samp, float2(tc12.x, tc3.y),  0.0).rgb * (w12.x * w3.y) +
        tex.SampleLevel(samp, float2(tc3.x,  tc0.y),  0.0).rgb * (w3.x  * w0.y) +
        tex.SampleLevel(samp, float2(tc3.x,  tc12.y), 0.0).rgb * (w3.x  * w12.y) +
        tex.SampleLevel(samp, float2(tc3.x,  tc3.y),  0.0).rgb * (w3.x  * w3.y);
    // clang-format on
    return result;
}

[numthreads(8, 8, 1)]
void csMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFrameDim)) return;

    float2 texDim = float2(gFrameDim);
    float2 pixelUV = (float2(pixel) + 0.5) / texDim;
    int2 maxCoord = int2(gFrameDim) - 1;

    // --- 1. Current-frame plus-tap reconstruction filter (UE AA_SAMPLES = 5) ---
    // The spatial weight is what turns TAA from "blur + temporal average" into
    // temporal supersampling: each tap's sample does not sit at its own pixel
    // centre but at (pixel centre + gJitterPixels) in the un-jittered pixel
    // grid, so its weight is the reconstruction kernel evaluated at the distance
    // from *this* output pixel's centre. UE does the same thing by folding the
    // jitter into the CPU-side PlusWeights array.
    float3 taps[5];
    float3 colorSum = float3(0.0, 0.0, 0.0);
    float weightSum = 0.0;

    [unroll]
    for (int i = 0; i < 5; ++i)
    {
        int2 sp = clamp(int2(pixel) + kPlusOffsets[i], int2(0, 0), maxCoord);
        // Filter in YCgCo (as UE does via TransformCurrentFrameSceneColor) so the
        // HDR weight can read the luma straight out of .x. YCgCo is linear, so
        // filtering here is identical to filtering in RGB and converting after.
        float3 yc = RGBToYCgCo(gColor.Load(int3(sp, 0)).rgb);
        taps[i] = yc;

        float2 d = float2(kPlusOffsets[i]) + gJitterPixels;
        float w = exp(-kFilterGaussianExp * dot(d, d)) * hdrWeightY(yc.x);
        colorSum += yc * w;
        weightSum += w;
    }
    // UE reweights by the accumulated weight because the HDR term makes the sum != 1.
    float3 filteredYCgCo = colorSum * rcp(max(weightSum, 1e-8));

    // --- 2. Neighborhood bounding box: hard plus-shaped min/max, in YCgCo ---
    // UE's HISTORY_CLAMPING_BOX_MIN_MAX over kOffsets3x3 {1,3,4,5,7}. No variance
    // and no min/max against the filtered colour: with a plus-only filter the
    // filtered colour is a convex combination of these same taps, so it already
    // lies inside the box.
    float3 colorMin = taps[0];
    float3 colorMax = taps[0];
    [unroll]
    for (int k = 1; k < 5; ++k)
    {
        colorMin = min(colorMin, taps[k]);
        colorMax = max(colorMax, taps[k]);
    }

    // --- 3. Velocity dilation: nearest depth in a +-2 px "X" (UE AA_CROSS = 2) ---
    // Picking the *nearest* neighbour (not the fastest-moving one) is what keeps
    // foreground silhouettes intact against a differently-moving background;
    // UE's comment: "Larger 2 pixel distance x works best (because AA dilates
    // surface)". Falcor's depth buffer is NOT reversed (0 = near, 1 = far), so
    // every comparison below is flipped w.r.t. UE's HAS_INVERTED_Z_BUFFER path.
    const int C = kVelocityCrossRadius;
    float centerDepth = gSceneDepth.Load(int3(pixel, 0)).r;
    float4 depths;
    depths.x = gSceneDepth.Load(int3(clamp(int2(pixel) + int2(-C, -C), int2(0, 0), maxCoord), 0)).r;
    depths.y = gSceneDepth.Load(int3(clamp(int2(pixel) + int2( C, -C), int2(0, 0), maxCoord), 0)).r;
    depths.z = gSceneDepth.Load(int3(clamp(int2(pixel) + int2(-C,  C), int2(0, 0), maxCoord), 0)).r;
    depths.w = gSceneDepth.Load(int3(clamp(int2(pixel) + int2( C,  C), int2(0, 0), maxCoord), 0)).r;

    int2 depthOffset = int2(C, C);
    int depthOffsetXx = C;
    if (depths.x < depths.y) depthOffsetXx = -C;
    if (depths.z < depths.w) depthOffset.x = -C;
    float depthsXY = min(depths.x, depths.y);
    float depthsZW = min(depths.z, depths.w);
    if (depthsXY < depthsZW)
    {
        depthOffset.y = -C;
        depthOffset.x = depthOffsetXx;
    }
    float depthsXYZW = min(depthsXY, depthsZW);

    int2 velocityOffset = int2(0, 0);
    if (depthsXYZW < centerDepth) velocityOffset = depthOffset;

    // UE re-derives camera motion at the centre pixel using the dilated *depth*;
    // with a precomputed velocity texture the equivalent is to fetch the dilated
    // pixel's motion vector (this is UE's own AA_DYNAMIC branch). The two differ
    // only by how much the motion field varies over 2 px on one surface, which is
    // negligible for camera-only motion.
    float2 motion = gMotionVecs.Load(int3(clamp(int2(pixel) + velocityOffset, int2(0, 0), maxCoord), 0)).xy;

    // --- 4. Reproject + sample history (Catmull-Rom bicubic) ---
    float2 historyUV = pixelUV + motion;
    bool offScreen = (any(historyUV < 0.0) || any(historyUV > 1.0));
    float3 history = bicubicSampleCatmullRom(gPrevColor, gLinearSampler, historyUV * texDim, texDim);
    history = RGBToYCgCo(history);

    // --- 5. Clamp history to the neighborhood box (per-channel clamp in YCgCo) ---
    // UE stashes the *pre-clamp* history luma for the blend heuristics below.
    float lumaHistory = history.x;
    float lumaFiltered = filteredYCgCo.x;
    float3 historyClamped = clamp(history, colorMin, colorMax);

    // --- 6. Blend factor ---
    float blend = gCurrentFrameWeight;

    // UE's Velocity is BackN (a ScreenPos delta, i.e. 2x the UV delta) times the
    // viewport size in pixels, so it is TWICE the motion in pixels -- see the
    // comment above UE's OffScreen test: "BackN is in units of
    // 2pixels/viewportWidthInPixels". Reproduce that scale so the /40 threshold
    // means what it means in UE (full ramp at 20 px/frame, not 40).
    float velocity = 2.0 * length(motion * texDim);
    blend = lerp(blend, 0.2, saturate(velocity / 40.0));

    // UE's "make sure to have at least some small contribution": a FLOOR that
    // raises the blend when history and current frame have nearly the same luma,
    // so a converged pixel can never get stuck. (UE's DistToClamp variant that
    // *reduces* the blend instead lives inside a `#if 0` and is not used.)
    // The +epsilon is a NaN guard for the 0/0 case; UE relies on saturate(NaN)==0.
    blend = max(blend, saturate(0.01 * lumaHistory / (abs(lumaFiltered - lumaHistory) + 1e-8)));

    // Seed / reset: first frame, camera cut, or off-screen history -> full current.
    // UE instead assigns History = Filtered on IgnoreHistory; with the weighted
    // lerp below, blend = 1 produces exactly the same result.
    if (gCameraCut != 0 || gHasHistory == 0 || offScreen)
        blend = 1.0;

    // --- 7. HDR-weighted temporal blend (UE WeightedLerpFactors) ---
    // Blending in tonemapped space is what suppresses fireflies and specular
    // flicker: a bright current-frame sample gets a small weight and cannot
    // dominate the history.
    float2 lerpWeights = weightedLerpFactors(hdrWeightY(historyClamped.x), hdrWeightY(filteredYCgCo.x), blend);
    float3 resultYCgCo = historyClamped * lerpWeights.x + filteredYCgCo * lerpWeights.y;

    float3 result = YCgCoToRGB(resultYCgCo);
    // UE AA_NAN: maps NaN *and* negatives to 0. max(x, 0) would let NaN through.
    result = -min(-result, 0.0);
    gColorOut[pixel] = float4(result, 1.0);
}
