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

/** Classic (unintegrated) SSAO — half-res compute pass.

    Direct port of UnrealEngine's MainPSandCS + WedgeWithNormal
    (Private/PostProcessAmbientOcclusion.usf and
     Private/PostProcessAmbientOcclusionCommon.ush) to Slang.
    Runs at half resolution reading full-res GBuffer world normal + scene
    depth, using UE's OPTIMIZATION_O1 wedge formula and the 4x4 random
    rotation texture. The half-res dither is cleaned up by SSAOSmooth.cs.slang
    and bilateral-upsampled to full-res by SSAOUpsample.cs.slang.

    Half-res output:
      - gAOMapHalf (RGBA32Float): AO in r (replicated to rgb, a=1)

    This follows UE's !OPTIMIZATION_O1 branch: an explicit distance falloff per
    wedge instead of the Alchemy-style dot(D,N)/dot(D,D) form. UE ships with
    OPTIMIZATION_O1 1, but its 0.08 * lerp(SceneDepth, 1000, ScaleRadiusInWorldSpace)
    normal scale collapses to a constant 80 UE-units (0.8 m) in world-space mode
    and therefore stops tracking the user radius entirely, so the falloff branch
    is the one worth porting.

    Notable deviations from UE:
      - Falcor has no HZB available, so we sample full-res depth with Load
        (point) instead of GetHZBDepth; ComputeMipLevel is unused.
      - Falcor has no RandomNormalTexture asset, so a 4x4 RGBA8 random tile is
        generated C++-side and bound as gRandomTexture.
      - Falcor device Z is forward-Z (0=near, 1=far); UE assumes the same
        (CalcSceneDepth returns 0..1, 1=far).
      - No multi-resolution AO pyramid (UE's USE_AO_SETUP_AS_INPUT chain), so
        ComputeUpsampleContribution/ComputeLerpFactor have no counterpart here;
        SSAOUpsample.cs.slang only lifts half-res AO to full-res.
      - AmbientOcclusionPower / Intensity and the distance fade are applied
        full-res in SSAOUpsample.cs.slang, matching where UE applies them
        (PostProcessAmbientOcclusion.usf) rather than at AO resolution.
*/

import Scene.Scene;
import Scene.Camera.Camera;
import Utils.Math.MathHelpers;
import Arctic.GBufferPass.GBufferHelpers;

cbuffer PerFrameCB
{
    uint2  gFullFrameDim;     // full-res dims
    uint2  gHalfFrameDim;     // half-res dims
    float  gRadius;           // base AO radius (view-space units / world units)
    float  gBias;             // view-space position bias along normal (reserved hook)
    float  gScaleRadiusInWorldSpace; // 0=view-space radius scaling, 1=world-space constant
    float  gFadeMul;          // UE ScreenSpaceAOParams[4].x — 1/fadeRadius
    float  gFadeAdd;          // UE ScreenSpaceAOParams[4].y — -fadeDistance/fadeRadius
    uint   gFrameIndex;       // for per-frame random rotation jitter
};

Texture2D<float4> gGBufferA;     // full-res: world normal (encoded) + per-object
Texture2D<float>  gSceneDepth;   // full-res: device Z (D32Float, forward-Z: 0=near 1=far)
Texture2D<float4> gRandomTexture; // 4x4 RGBA8: random 2D vectors in .rg (encoded v*0.5+0.5)

RWTexture2D<float4> gAOMapHalf;

static const float PI = 3.14159265358979f;

// ---- UE quality tiers (PostProcessAmbientOcclusion.usf) ----
// SHADER_QUALITY is a compile-time define injected from C++.

#if SHADER_QUALITY == 0
    // very low
    #define USE_SAMPLESET 1
    #define SAMPLE_STEPS 1
#elif SHADER_QUALITY == 1
    // low
    #define USE_SAMPLESET 1
    #define SAMPLE_STEPS 1
#elif SHADER_QUALITY == 2
    // medium
    #define USE_SAMPLESET 1
    #define SAMPLE_STEPS 2
#elif SHADER_QUALITY == 3
    // high
    #define USE_SAMPLESET 1
    #define SAMPLE_STEPS 3
#else // SHADER_QUALITY == 4
    // very high
    #define USE_SAMPLESET 3
    #define SAMPLE_STEPS 3
#endif

// Sample offset tables on the unit disc (PostProcessAmbientOcclusion.usf).
#if USE_SAMPLESET == 0
    #define SAMPLESET_ARRAY_SIZE 1
    static const float2 OcclusionSamplesOffsets[SAMPLESET_ARRAY_SIZE] =
    {
        float2(0.500, 0.500),
    };
#elif USE_SAMPLESET == 1
    #define SAMPLESET_ARRAY_SIZE 3
    static const float2 OcclusionSamplesOffsets[SAMPLESET_ARRAY_SIZE] =
    {
        // 3 points on the unit disc, spiral order and distance
        float2(0, -1.0f) * 0.43f,
        float2(0.58f, 0.814f) * 0.7f,
        float2(-0.58f, 0.814f),
    };
#elif USE_SAMPLESET == 2
    #define SAMPLESET_ARRAY_SIZE 5
    static const float2 OcclusionSamplesOffsets[SAMPLESET_ARRAY_SIZE] =
    {
        // 5 points on a ring
        float2(0.156434, 0.987688),
        float2(0.987688, 0.156434) * 0.9,
        float2(0.453990, -0.891007) * 0.8,
        float2(-0.707107, -0.707107) * 0.7,
        float2(-0.891006, 0.453991) * 0.65,
    };
#else // USE_SAMPLESET == 3
    #define SAMPLESET_ARRAY_SIZE 6
    static const float2 OcclusionSamplesOffsets[SAMPLESET_ARRAY_SIZE] =
    {
        // 6 points on the unit disc, spiral order and distance
        float2(0.000, 0.200),
        float2(0.325, 0.101),
        float2(0.272, -0.396),
        float2(-0.385, -0.488),
        float2(-0.711, 0.274),
        float2(0.060, 0.900),
    };
#endif

// ---- Depth / position reconstruction (mirrors GTAOPass.cs.slang) ----

// Point-sample scene depth at integer full-res pixel coords.
float LookupDeviceZ(int2 fullPx)
{
    fullPx = clamp(fullPx, int2(0, 0), int2(gFullFrameDim) - 1);
    return gSceneDepth.Load(int3(fullPx, 0)).r;
}

float LookupDeviceZUV(float2 uv)
{
    int2 px = int2(floor(uv * float2(gFullFrameDim)));
    return LookupDeviceZ(px);
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

// Inverse of ConvertFromDeviceZ: linear view depth -> device Z (forward-Z NDC).
float ConvertToDeviceZ(float linearDepth)
{
    float nearZ = gScene.camera.data.nearZ;
    float farZ = gScene.camera.data.farZ;
    if (isinf(farZ) || farZ > 1e8f)
    {
        return nearZ / max(linearDepth, 1e-6f);
    }
    return farZ * (linearDepth - nearZ) / max(linearDepth * (farZ - nearZ), 1e-6f);
}

// Reconstruct view-space position from a full-res UV and linear scene depth.
// Falcor textures are Y-down; NDC is Y-up, so flip Y.
float3 ScreenToViewPos(float2 uv, float linearDepth)
{
    float2 ndc;
    ndc.x = uv.x * 2.0f - 1.0f;
    ndc.y = 1.0f - uv.y * 2.0f;
    float deviceZ = ConvertToDeviceZ(linearDepth);
    float4 clip = float4(ndc, deviceZ, 1.0f);
    float4 worldH = mul(gScene.camera.data.invViewProj, clip);
    float3 worldPos = worldH.xyz / worldH.w;
    float4 viewH = mul(gScene.camera.data.viewMat, float4(worldPos, 1.0f));
    return viewH.xyz;
}

// Get view-space normal at given UV from GBufferA.
float3 GetViewSpaceNormal(float2 uv)
{
    int2 px = int2(floor(uv * float2(gFullFrameDim)));
    px = clamp(px, int2(0, 0), int2(gFullFrameDim) - 1);
    float4 a = gGBufferA.Load(int3(px, 0));
    float3 worldNormal = decodeNormal(a.rgb);
    float3 viewNormal = normalize(mul((float3x3)gScene.camera.data.viewMat, worldNormal));
    return viewNormal;
}

// Wedge sample: returns float3(elevL, elevR, weight) where `elev` is the
// elevation of the sample above the surface tangent plane (dot(delta,N)/|delta|,
// clamped to [0,1]; 0 = at/below the horizon, 1 = directly along the normal).
// `weight` is the *product* of the two sides' distance falloffs, matching UE
// (PostProcessAmbientOcclusionCommon.ush): a wedge with either side
// outside the halo contributes nothing at all. Averaging the two sides instead
// would let one out-of-range side keep half its influence, fabricating occluders
// at the radius boundary and against the sky.
float3 WedgeWithNormal(float2 centerNDC, float2 ndcOffset,
                       float3 centerViewPos,
                       float3 viewNormal, float radiusWorld)
{
    float2 sampleNDC_L = centerNDC + ndcOffset;
    float2 sampleNDC_R = centerNDC - ndcOffset;

    // NDC (-1..1, Y-up) -> full-res UV (Y-down).
    float2 uvL = float2(sampleNDC_L.x * 0.5f + 0.5f, (1.0f - sampleNDC_L.y) * 0.5f);
    float2 uvR = float2(sampleNDC_R.x * 0.5f + 0.5f, (1.0f - sampleNDC_R.y) * 0.5f);

    float deviceZ_L = LookupDeviceZUV(uvL);
    float deviceZ_R = LookupDeviceZUV(uvR);

    // A sample that lands on the sky (deviceZ >= 1, far plane) sees open
    // background — it is unoccluded and must NOT be treated as geometry.
    // The product weight below already discards the whole wedge once one side's
    // weight is zero, but we still test explicitly: ConvertFromDeviceZ's
    // infinite-far-plane branch does not map deviceZ=1 to a far distance, so the
    // falloff alone cannot be relied on to reject sky samples.
    bool skyL = (deviceZ_L >= 1.0f);
    bool skyR = (deviceZ_R >= 1.0f);

    float elevL = 0.0f, wL = 0.0f;
    float elevR = 0.0f, wR = 0.0f;

    [flatten]
    if (!skyL)
    {
        float3 samplePosL = ScreenToViewPos(uvL, ConvertFromDeviceZ(deviceZ_L));
        float3 deltaL = samplePosL - centerViewPos;
        float lenL = max(length(deltaL), 1e-6f);
        elevL = saturate(dot(deltaL, viewNormal) / lenL);
        wL = saturate(1.0f - lenL / max(radiusWorld, 1e-6f));
    }
    [flatten]
    if (!skyR)
    {
        float3 samplePosR = ScreenToViewPos(uvR, ConvertFromDeviceZ(deviceZ_R));
        float3 deltaR = samplePosR - centerViewPos;
        float lenR = max(length(deltaR), 1e-6f);
        elevR = saturate(dot(deltaR, viewNormal) / lenR);
        wR = saturate(1.0f - lenR / max(radiusWorld, 1e-6f));
    }

    return float3(elevL, elevR, wL * wR);
}

[numthreads(8, 8, 1)]
void csMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 halfPixel = int2(dispatchThreadID.xy);
    if (any(halfPixel >= int2(gHalfFrameDim))) return;

    // Half-res pixel -> full-res UV with UE's QuarterOffset trick (so the
    // half-res center maps to a consistent full-res sample location).
    float2 quarterOffset = 0.25f / float2(gFullFrameDim);
    float2 centerUV = (float2(halfPixel) + 0.5f) * 2.0f / float2(gFullFrameDim) + quarterOffset;
    centerUV = clamp(centerUV, float2(0.0f, 0.0f), float2(1.0f, 1.0f) - 1.0f / float2(gFullFrameDim));

    float deviceZ = LookupDeviceZUV(centerUV);
    float centerLinearZ = ConvertFromDeviceZ(deviceZ);

    // Sky / fully-faded early-out: AO = 1 (unoccluded). UE's MainCS gates on the
    // same expression it later uses for the soft fade, so the branch
    // and the fade agree on where AO stops mattering.
    if (deviceZ >= 1.0f || (centerLinearZ * gFadeMul + gFadeAdd) >= 1.0f)
    {
        gAOMapHalf[halfPixel] = float4(1.0f, 1.0f, 1.0f, 1.0f);
        return;
    }

    float3 centerViewPos = ScreenToViewPos(centerUV, centerLinearZ);
    float3 viewNormal = GetViewSpaceNormal(centerUV);

    // FOV / aspect correction. UE's FovFix = (InvTanHalfFov, Ratio*InvTanHalfFov, 1)
    // with a *horizontal* InvTanHalfFov and Ratio = width/height, which is exactly
    // the projection matrix diagonal: ReconstructCSPos yields
    // (ScreenPos*Z, Z) == ViewPos * (P00, P11, 1), so FovFix == (P00, P11, 1).
    // Deriving it from the vertical FOV instead would scale both components by
    // the aspect ratio, inflating the sampling disc past radiusWorld (1.78x at
    // 16:9) so that outer samples fall outside the falloff and are wasted.
    float3 fovFix = float3(gScene.camera.data.projMat[0][0], gScene.camera.data.projMat[1][1], 1.0f);

    // UE: ActualAORadius = AORadiusInShader * lerp(SceneDepth, 1, ScaleRadiusInWorldSpace).
    float actualAORadius = gRadius * lerp(centerLinearZ, 1.0f, gScaleRadiusInWorldSpace);

    // Per-pixel random rotation from the 4x4 tile, shifted per frame for jitter.
    // (UE adds TemporalOffset to the random UV; here we shift the tile index.)
    int2 rotOffset = int2(int(gFrameIndex) & 3, (int(gFrameIndex) / 4) & 3);
    int2 rp = (halfPixel + rotOffset) & 3;
    float2 randomVec = (gRandomTexture.Load(int3(rp, 0)).rg * 2.0f - 1.0f) * actualAORadius;

    // RandomBase converts the view-space random vector to an NDC offset
    // (FovFix.xy / viewZ), so localRandom is added directly to centerNDC.
    // Use the positive linear depth (Falcor view-space z is negative for
    // objects in front, which would flip the offset sign).
    float2 fovFixXY = fovFix.xy * (1.0f / max(centerLinearZ, 1e-6f));
    float4 randomBase = float4(randomVec, -randomVec.y, randomVec.x) * fovFixXY.xyxy;

    // centerNDC is the clip-space xy (Y up, -1..1), derived directly from the UV.
    float2 centerNDC = float2(centerUV.x * 2.0f - 1.0f, 1.0f - centerUV.y * 2.0f);

    // OPTIMIZATION_O1: pre-scale the normal by 0.08 * depth-scale to embed falloff.
    // (Replaced with an explicit distance falloff in WedgeWithNormal — the 0.08
    // heuristic is tuned for UE's view-space AORadiusInShader units and breaks
    // under world-space radius scaling.)

    // Optional view-space position bias along the normal (UE AmbientOcclusionBias).
    // UE multiplies the normal by FovFix because its ViewSpacePosition lives in
    // the FovFix-scaled "CSPos" space; centerViewPos here is a true view-space
    // point, so the normal is used unscaled. kScaleFactor is UE's ScaleFactor,
    // the full-res / AO-res ratio per dimension.
    const float kScaleFactor = 2.0f;
    centerViewPos += gBias * centerLinearZ * kScaleFactor * viewNormal;

    // World-space radius (meters) for the distance falloff. In view-space
    // scaling mode (gScaleRadiusInWorldSpace=0) actualAORadius already grows
    // with depth, so use it directly; in world-space mode it's constant.
    float radiusWorld = actualAORadius;

    // Accumulate visibility per direction (UE Square() contrast).
    float2 weightAccumulator = float2(0.0001f, 0.0001f);

    [unroll]
    for (int i = 0; i < SAMPLESET_ARRAY_SIZE; ++i)
    {
        float2 unrotatedRandom = OcclusionSamplesOffsets[i].xy;
        float2 localRandom = unrotatedRandom.x * randomBase.xy + unrotatedRandom.y * randomBase.zw;

        // .xy = horizon (max elevation) per direction, .z = coverage.
        float3 localAcc = float3(0.0f, 0.0f, 0.0f);

        [unroll]
        for (int step = 0; step < SAMPLE_STEPS; ++step)
        {
            float scale = (float(step) + 1.0f) / (float)SAMPLE_STEPS;
            float3 stepSample = WedgeWithNormal(centerNDC, scale * localRandom,
                                                centerViewPos,
                                                viewNormal, radiusWorld);

            // UE's lerp-accumulate (PostProcessAmbientOcclusion.usf): the
            // step's falloff weight attenuates how much of its horizon is taken,
            // and coverage accumulates as 1 - prod(1 - w_i). A plain max() would
            // let a sample sitting right at the radius boundary contribute its
            // full horizon, a strong noise source on dense thin geometry.
            localAcc = lerp(localAcc, float3(max(localAcc.xy, stepSample.xy), 1.0f), stepSample.z);
        }

        // Accumulate *visibility* Square(1 - horizon), not occlusion
        // (PostProcessAmbientOcclusion.usf) — Square() because the
        // unoccluded area scales quadratically with the horizon angle. Squaring
        // the occlusion instead would differ by 2*mean(horizon) (a 0.3 horizon
        // gives 0.49 here vs 0.91) and would make the degenerate "no valid
        // sample" case 0.0001/0.0001 resolve to 0 = black rather than
        // 1 = unoccluded, which shows up as black rims on geometry silhouetted
        // against the sky.
        float2 vis = 1.0f - localAcc.xy;
        vis *= vis;
        weightAccumulator += float2((vis.x + vis.y) * localAcc.z, localAcc.z * 2.0f);
    }

    // UE's OutColor.r: already AO visibility (1 = unoccluded). Power, intensity
    // and the distance fade are applied full-res in SSAOUpsample.cs.slang so the
    // smooth and upsample filters operate on linear visibility, as in UE.
    float ao = saturate(weightAccumulator.x / weightAccumulator.y);

    gAOMapHalf[halfPixel] = float4(ao, ao, ao, 1.0f);
}
