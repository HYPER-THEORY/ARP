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
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 # LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

import Scene.Scene;
import Scene.Camera.Camera;
import Utils.Math.MathHelpers;
import Utils.Math.HashUtils;
import Utils.Math.BitTricks;
import Arctic.GBufferPass.GBufferHelpers;

// ---------------------------------------------------------------------------
// Compile-time configuration (added by C++ via addDefine in execute()).
// Defaults must be provided so the shader still compiles at setScene() time,
// before any define has been added. They match the initial member values in
// ShadowProjectionPass.h.
// ---------------------------------------------------------------------------

// 0 = axis-aligned Gather4 PCF (UE ManualPCF), 1 = Sobol-rotated Vogel disk.
#ifndef SHADOW_FILTER_MODE
#define SHADOW_FILTER_MODE 1
#endif

// Gather4 PCF kernel width in texels: 1, 3, 5 or 7.
#ifndef PCF_KERNEL_SIZE
#define PCF_KERNEL_SIZE 5
#endif

// Number of disk taps for SHADOW_FILTER_MODE 1.
#ifndef DISK_TAP_COUNT
#define DISK_TAP_COUNT 12
#endif

cbuffer PerFrameCB
{
    float4x4 gLightViewProj;
    float3   gLightPos;
    float3   gLightDir;
    float    gMaxSubjectDepth;
    float    gConstantDepthBias;
    float    gReceiverBias;
    float    gSoftTransitionScale;
    // Scales density (from subsurfaceDensityFromOpacity) into DensityMulConstant,
    // mirroring UE's ProjectionDepthBiasParameters.w (ShadowProjectionPixelShader.usf).
    float    gDensityDepthScale;
    // UE ShadowSharpen: contrast expansion about 0.5 applied after filtering
    // (ShadowProjectionPixelShader.usf). 1 = no change.
    float    gShadowSharpen;
    // Radius of the disk filter in WORLD METERS (SHADOW_FILTER_MODE 1). Kept in
    // world units rather than texels so the look does not change when the
    // shadow resolution or the fitted frustum size changes.
    float    gFilterRadiusWorld;
    // World meters -> shadowmap UV along the light's right/up axes.
    // = 0.5 / orthoHalfExtent.
    float2   gShadowUVPerMeter;
    uint2    gFrameDim;
    uint2    gShadowDim;
    float2   gInvShadowDim;
    // Frame counter, drives the temporal component of the Sobol rotation.
    uint     gFrameIndex;
};

Texture2D<float>  gSceneDepth;
Texture2D<float>  gShadowDepth;
// GBuffer A: world normal encoded as N*0.5+0.5 in RGB10A2Unorm (xyz=normal, w=per-object data).
Texture2D<float4> gGBufferA;
// GBuffer B.a: ShadingModelID (to detect TWOSIDED_FOLIAGE for SSS transmission).
Texture2D<float4> gGBufferB;
// GBuffer D.a: foliage density/opacity (subsurface transmission strength).
Texture2D<float4> gGBufferD;
// .r = surface shadow, .g = SSS transmission shadow (UE-style two-channel pack).
RWTexture2D<float2> gShadowMask;

SamplerState gShadowSampler;

// ---------------------------------------------------------------------------
// World position reconstruction
// ---------------------------------------------------------------------------

// Convert Falcor device depth (NDC z in [0,1], 0=near) to a linear world-space
// position along the camera ray. We reconstruct NDC xy from the pixel coord,
// then use the camera's invViewProj to get the world position.
//
// invViewProj includes the TAA sub-pixel jitter, which is
// what we want: the depth buffer was rasterized with the jittered projection,
// so only the jittered inverse recovers the world point that was actually
// rasterized here.
float3 reconstructWorldPos(uint2 pixel)
{
    float deviceZ = gSceneDepth.Load(int3(pixel, 0)).r;
    // NDC in [-1,1].
    float2 ndc;
    ndc.x = (float(pixel.x) + 0.5f) / float(gFrameDim.x) * 2.f - 1.f;
    ndc.y = 1.f - (float(pixel.y) + 0.5f) / float(gFrameDim.y) * 2.f; // flip Y
    float4 clip = float4(ndc, deviceZ, 1.f);
    // Falcor matrices are column-vector: v' = M * v. Use mul(M, v), not mul(v, M).
    float4 worldH = mul(gScene.camera.data.invViewProj, clip);
    return worldH.xyz / worldH.w;
}

// ---------------------------------------------------------------------------
// Occlusion evaluation (port of UE FPCFSamplerSettings + CalculateOcclusion,
// ShadowFilteringCommon.ush)
// ---------------------------------------------------------------------------

// Mirrors UE's FPCFSamplerSettings so the surface and subsurface paths share
// one filter implementation instead of duplicating every kernel. Both call
// sites pass a compile-time-constant bSubsurface, so the branch in
// calcOcclusion4 folds away per specialization.
struct PCFSettings
{
    // Receiver depth in light space (meters).
    float sceneDepth;
    float transitionScale;
    // Set by the caller and constant for the whole filter.
    bool  bSubsurface;
    // Only used when bSubsurface is true.
    float densityMulConstant;
    // Screen pixel coordinate, seeds the per-pixel Sobol scramble in
    // SHADOW_FILTER_MODE 1. Mirrors FPCSSSamplerSettings::SvPosition
    // (ShadowPercentageCloserFiltering.ush).
    uint2 pixel;
};

// Vectorized UE CalculateOcclusion. Evaluated per TEXEL (before filtering) —
// comparing first and filtering after is what makes this a percentage-closer
// filter rather than a blur of depths.
float4 calcOcclusion4(float4 shadowmapDepth, PCFSettings s)
{
    // Cleared texels (nothing was rasterized) must never occlude. The shadowmap
    // is cleared far beyond gMaxSubjectDepth, which itself already includes the
    // maximum depth bias ShadowPass can add. UE's equivalent test is
    // `ShadowmapDepth > .99f` against its normalized depth.
    float4 unwritten = select(shadowmapDepth > gMaxSubjectDepth, float4(1.f), float4(0.f));

    if (s.bSubsurface)
    {
        // UE bSubsurface path (ShadowFilteringCommon.ush +
        // TransmissionCommon.ush). Light that penetrates the occluder
        // attenuates exponentially with the distance it travelled through it.
        // This assumes everything between this pixel and the light was also
        // subsurface material, so foliage leaks light by how thin it is.
        float4 occluderDistance = max(s.sceneDepth - shadowmapDepth, 0.f);
        float4 occlusion = saturate(exp(-occluderDistance * s.densityMulConstant));
        return saturate(occlusion + unwritten);
    }
    else
    {
        // The standard comparison is sceneDepth < shadowmapDepth. Using a soft
        // transition based on the depth difference offsets shadows slightly but
        // reduces self-shadowing artifacts considerably.
        //
        // Unoptimized: saturate((shadowmapDepth - sceneDepth) * scale + 1)
        // Rearranged so the per-pixel constant hoists out of the per-tap loop.
        float constantFactor = s.sceneDepth * s.transitionScale - 1.f;
        float4 shadowFactor = saturate(shadowmapDepth * s.transitionScale - constantFactor);
        return saturate(shadowFactor + unwritten);
    }
}

// ---------------------------------------------------------------------------
// Gather4 helpers
// ---------------------------------------------------------------------------

// Fetch the 2x2 texel block whose TOP-LEFT texel is `topLeft`, as occlusion
// values.
//
// Gather4 samples "at the following locations: (-,+),(+,+),(+,-),(-,-), where
// the magnitude of the deltas are always half a texel" (DX11 Functional Spec).
// A gather at UV = (topLeft + 1) / dim therefore lands exactly on the corner
// between texels `topLeft` and `topLeft + 1` in both axes, returning that 2x2
// block. This is the same addressing UE uses
// (ShadowFilteringCommon.ush `SamplePos = (TexelCorner + 1.0f) * ...`).
//
// Component mapping, with v increasing downward:
//   .x = (topLeft.x    , topLeft.y + 1)   bottom-left
//   .y = (topLeft.x + 1, topLeft.y + 1)   bottom-right
//   .z = (topLeft.x + 1, topLeft.y    )   top-right
//   .w = (topLeft.x    , topLeft.y    )   top-left
//
// The sampler uses BORDER addressing with a border depth of 1e9, so blocks that
// straddle the shadowmap edge read as "never written" and contribute no
// occlusion — which is the correct answer, since the frustum encloses
// the whole scene.
float4 gatherOcclusionBlock(int2 topLeft, PCFSettings s)
{
    float2 uv = (float2(topLeft) + 1.f) * gInvShadowDim;
    float4 depths = gShadowDepth.GatherRed(gShadowSampler, uv);
    return calcOcclusion4(depths, s);
}

// Bilinear 2x2 PCF at an arbitrary UV: compare all four texels, then bilerp the
// occlusion values. Port of UE Manual1x1PCF + PCF1x1
// (ShadowFilteringCommon.ush). One Gather4, four comparisons.
float bilinearPCF2x2(float2 shadowUV, PCFSettings s)
{
    // Bias by -0.5 so integer values land on texel centers, consistent with the
    // texture filtering hardware.
    float2 texelPos = shadowUV * float2(gShadowDim) - 0.5f;
    float2 fraction = frac(texelPos);
    int2 topLeft = int2(floor(texelPos));

    float4 o = gatherOcclusionBlock(topLeft, s);
    // o.w = top-left, o.z = top-right, o.x = bottom-left, o.y = bottom-right.
    float top    = lerp(o.w, o.z, fraction.x);
    float bottom = lerp(o.x, o.y, fraction.x);
    return lerp(top, bottom, fraction.y);
}

// ---------------------------------------------------------------------------
// Mode 0: axis-aligned Gather4 PCF (UE ManualPCF)
// ---------------------------------------------------------------------------

// For KERNEL_SIZE = N we weight an (N+1)x(N+1) texel grid as an NxN bilinear
// PCF kernel: the two edge rows/columns get (1 - fraction) and fraction, the
// interior gets 1. The per-axis weights sum to exactly N, so the total divides
// by N*N.
//
// KERNEL CENTERING (this was previously wrong). The grid must start at
//   floor(texelPos) - (N-1)/2
// and NOT at floor(texelPos). Starting at floor() puts the kernel's weighted
// centroid at texelPos + (N-1)/2 instead of texelPos, shifting the whole shadow
// diagonally by (N-1)/2 shadowmap texels, plus
// one-sided light leak that had to be paid for with extra constant bias.
//
// Verified against UE: Manual3x3PCF (ShadowFilteringCommon.ush) gathers
// at offsets (-1,-1)..(1,1) around (floor+1), covering texels
// floor-1 .. floor+2; Manual5x5PCF starts at offset (-2,-2),
// covering floor-2 .. floor+3. Both have centroid exactly floor + fraction.
//
// Because kGrid = N+1 is always even for odd N, the grid tiles exactly into
// (kGrid/2)^2 aligned 2x2 blocks, so every four taps cost a single Gather4.
float axisAlignedPCF(float2 shadowUV, PCFSettings s)
{
    float2 texelPos = shadowUV * float2(gShadowDim) - 0.5f;
    float2 fraction = frac(texelPos);

    const int kGrid = PCF_KERNEL_SIZE + 1;      // 2, 4, 6, 8
    const int kBlocks = kGrid / 2;              // 1, 2, 3, 4
    int2 base = int2(floor(texelPos)) - (PCF_KERNEL_SIZE - 1) / 2;

    float occlusion = 0.f;
    [unroll]
    for (int by = 0; by < kBlocks; ++by)
    {
        [unroll]
        for (int bx = 0; bx < kBlocks; ++bx)
        {
            float4 o = gatherOcclusionBlock(base + int2(bx * 2, by * 2), s);

            // Grid indices of this block's four texels. These are compile-time
            // constants, so the edge-weight selects below fold away entirely.
            const int gx0 = bx * 2, gx1 = gx0 + 1;
            const int gy0 = by * 2, gy1 = gy0 + 1;

            float wx0 = (gx0 == 0) ? (1.f - fraction.x) : ((gx0 == kGrid - 1) ? fraction.x : 1.f);
            float wx1 = (gx1 == 0) ? (1.f - fraction.x) : ((gx1 == kGrid - 1) ? fraction.x : 1.f);
            float wy0 = (gy0 == 0) ? (1.f - fraction.y) : ((gy0 == kGrid - 1) ? fraction.y : 1.f);
            float wy1 = (gy1 == 0) ? (1.f - fraction.y) : ((gy1 == kGrid - 1) ? fraction.y : 1.f);

            occlusion += o.w * wx0 * wy0;   // top-left
            occlusion += o.z * wx1 * wy0;   // top-right
            occlusion += o.x * wx0 * wy1;   // bottom-left
            occlusion += o.y * wx1 * wy1;   // bottom-right
        }
    }
    return occlusion / float(PCF_KERNEL_SIZE * PCF_KERNEL_SIZE);
}

// ---------------------------------------------------------------------------
// Mode 1: Sobol-rotated Vogel disk
// ---------------------------------------------------------------------------

// UE's Sobol direction numbers, ported verbatim from SobolRandom.ush.
static const uint2 kSobolNumbers[10] = {
    uint2(0x8680u, 0x4c80u), uint2(0xf240u, 0x9240u), uint2(0x8220u, 0x0e20u),
    uint2(0x4110u, 0x1610u), uint2(0xa608u, 0x7608u), uint2(0x8a02u, 0x280au),
    uint2(0xe204u, 0x9e04u), uint2(0xa400u, 0x4682u), uint2(0xe300u, 0xa74du),
    uint2(0xb700u, 0x9817u),
};

// Port of UE SobolIndex (SobolRandom.ush). Advances the per-pixel base by
// `index` in the Sobol sequence; `bits` bounds how many index bits matter.
// Returns a scrambled 2D position in [0, 0xffff].
uint2 sobolIndex(uint2 base, uint index, uint bits)
{
    uint2 result = base;
    [unroll]
    for (uint b = 0; b < 10u; ++b)
    {
        bool active = (b < bits) && ((index & (1u << b)) != 0u);
        result ^= active ? kSobolNumbers[b] : uint2(0u, 0u);
    }
    return result;
}

// Per-pixel Sobol scramble base. UE's SobolPixel (SobolRandom.ush) reads
// a 256x256 tile out of View.SobolSamplingTexture, which we do not have here;
// hashing the pixel coordinate with Falcor's TEA block cipher serves the same
// purpose (and decorrelates over the full screen rather than a 256px tile).
// The `& 0xff00` mask matches UE: the pixel picks the coarse position, the
// Sobol index supplies the fine bits.
uint2 sobolPixelScramble(uint2 pixel)
{
    return blockCipherTEA(interleave_32bit(pixel), 0x9E3779B9u) & 0xff00u;
}

// Filter over a disk of taps whose angular phase is rotated per-pixel and
// per-frame, and whose radii are stratified with a per-pixel jitter.
//
// Structure follows UE's DirectionalPCSS sampling stage
// (ShadowPercentageCloserFiltering.ush): a Sobol sequence seeded
// from the pixel, advanced by the frame index in its low bits so successive
// frames continue one progressive sequence instead of drawing independent sets
// (which is what lets TAA integrate it cleanly).
//
// Where UE draws every tap position from Sobol via
// UniformSampleDiskConcentricApprox, we instead use Sobol for the rotation and
// radial jitter only, and place the taps on a golden-angle (Vogel) spiral. For
// the small tap counts that make this mode worthwhile, a rotated Vogel
// disk is measurably more even than the same number of random points, so the
// residual noise TAA has to remove is lower.
//
// Each tap is a bilinear 2x2 PCF rather than a point sample, so a tap costs one
// Gather4 and the per-texel quantization inside the penumbra disappears.
// DISK_TAP_COUNT taps = DISK_TAP_COUNT texture ops, against ((N+1)/2)^2 for
// mode 0 (5x5 -> 9, 7x7 -> 16). The reason to pay for this mode is that the
// radius is decoupled from the tap count: widening the penumbra costs nothing,
// whereas mode 0 has to add taps quadratically.
//
// NOTE: this mode is designed to be resolved by TAAPass. With TAA disabled the
// residual per-pixel noise is visible.
float sobolDiskPCF(float2 shadowUV, PCFSettings s)
{
    // Two Sobol dimensions: .x -> rotation phase, .y -> radial jitter.
    uint2 sobolBase = sobolPixelScramble(s.pixel);
    uint2 sobolFrame = sobolIndex(sobolBase, gFrameIndex & 7u, 3u);
    float2 e = float2(sobolIndex(sobolFrame, 0u, 3u)) * (1.f / 65536.f) + (1.f / 131072.f);

    const float kGoldenAngle = 2.39996323f; // pi * (3 - sqrt(5))
    float phase = e.x * 6.28318531f;
    float radialJitter = e.y;

    // World meters -> UV. gShadowUVPerMeter differs per axis whenever the fitted
    // frustum is non-square, which keeps the disk circular in WORLD space.
    float2 radiusUV = gFilterRadiusWorld * gShadowUVPerMeter;

    float occlusion = 0.f;
    [unroll]
    for (int i = 0; i < DISK_TAP_COUNT; ++i)
    {
        // sqrt() for uniform area density; +radialJitter destroys the fixed
        // ring structure that would otherwise alias.
        float r = sqrt((float(i) + radialJitter) / float(DISK_TAP_COUNT));
        float theta = phase + float(i) * kGoldenAngle;
        float2 offset = r * float2(cos(theta), sin(theta)) * radiusUV;
        occlusion += bilinearPCF2x2(shadowUV + offset, s);
    }
    return occlusion / float(DISK_TAP_COUNT);
}

// ---------------------------------------------------------------------------
// Filter dispatch + post-filter shaping
// ---------------------------------------------------------------------------

float filterShadow(float2 shadowUV, PCFSettings s)
{
#if SHADOW_FILTER_MODE == 0
    return axisAlignedPCF(shadowUV, s);
#else
    return sobolDiskPCF(shadowUV, s);
#endif
}

// UE ApplyPCFOverBlurCorrection (ShadowProjectionPixelShader.usf).
// "PCF falloff is overly blurry, apply to get a more reasonable look
// artistically." UE applies this to BOTH the surface shadow and the
// subsurface transmission.
//
// Note this is NOT part of UE's light-attenuation encoding: EncodeLightAttenuation
// is sqrt and DecodeLightAttenuation is Square (Common.ush), which
// cancel. The squaring here is a real, separate darkening of the penumbra, and
// omitting it was why these shadows read as lighter and blurrier than UE's.
float applyPCFOverBlurCorrection(float shadow)
{
    return shadow * shadow;
}

// UE ShadowSharpen (ShadowProjectionPixelShader.usf): expand contrast about
// the 0.5 midpoint. 1 = identity.
float applyShadowSharpen(float shadow)
{
    return saturate((shadow - 0.5f) * gShadowSharpen + 0.5f);
}

// UE SubsurfaceDensityFromOpacity (DeferredShadingCommon.ush). Derives a
// density (extinction rate) from a 0..1 opacity via a log heuristic, tweaked
// for a useful falloff range. opacity 0 -> 0 (no extinction), opacity -> 1
// (clamped 0.99) -> ~0.23.
float subsurfaceDensityFromOpacity(float opacity)
{
    return -0.05f * log(1.f - min(opacity, 0.99f));
}

// ---------------------------------------------------------------------------

[numthreads(8, 8, 1)]
void csMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFrameDim)) return;

    float deviceZ = gSceneDepth.Load(int3(pixel, 0)).r;
    // Skip pixels with no geometry (far plane); they will be filled by SkyboxPass.
    if (deviceZ >= 1.f)
    {
        gShadowMask[pixel] = float2(1.f, 1.f);
        return;
    }

    float3 worldPos = reconstructWorldPos(pixel);
    // Project world position into light clip space to get the shadowmap UV.
    // Falcor matrices are column-vector: v' = M * v. Use mul(M, v), not mul(v, M).
    float4 shadowPosH = mul(gLightViewProj, float4(worldPos, 1.f));
    float3 shadowPos = shadowPosH.xyz / shadowPosH.w;

    // NDC xy is in [-1,1] with +1 at top. Texture UV is [0,1] with 0 at top,
    // so flip Y: uv = (ndc.x*0.5+0.5, 0.5 - ndc.y*0.5).
    float2 shadowUV = float2(shadowPos.x * 0.5f + 0.5f, 0.5f - shadowPos.y * 0.5f);

    // Receiver depth = light-space linear depth in METERS, matching the metric
    // ShadowPass writes (dot(gLightDir, posW - gLightPos)). Using the SAME
    // metric on both sides is what makes the bias and soft-transition values
    // meaningful in world units.
    float sceneDepth = dot(gLightDir, worldPos - gLightPos);

    // UE receiver bias: instead of subtracting a constant from sceneDepth, UE
    // modulates the soft transition scale by `lerp(ReceiverBias, 1.0, NoL)`
    // (ShadowProjectionPixelShader.usf). At grazing angles (NoL -> 0) the
    // transition narrows, suppressing acne without light-leaking at contact.
    // At perpendicular incidence (NoL -> 1) the full SoftTransitionScale is
    // used. This requires the receiver's world normal, which we decode from
    // GBufferA (stored as N*0.5+0.5).
    float3 worldNormal = normalize(gGBufferA.Load(int3(pixel, 0)).xyz * 2.f - 1.f);
    float3 toLight = normalize(-gLightDir);
    float NoL = saturate(dot(worldNormal, toLight));

    // Surface shadow (.r) — standard UE comparison PCF.
    PCFSettings surf;
    surf.sceneDepth = sceneDepth;
    surf.transitionScale = gSoftTransitionScale * lerp(gReceiverBias, 1.f, NoL);
    surf.bSubsurface = false;
    surf.densityMulConstant = 0.f;
    surf.pixel = pixel;

    float surfaceShadow = filterShadow(shadowUV, surf);
    surfaceShadow = applyShadowSharpen(surfaceShadow);
    surfaceShadow = applyPCFOverBlurCorrection(surfaceShadow);

    // SSS transmission shadow (.g) — only foliage participates. GBufferD.a holds
    // UE's material Opacity input, which subsurfaceDensityFromOpacity turns into
    // an extinction rate; DensityMulConstant then attenuates transmitted light by
    // how far the nearest occluder is in front of this pixel.
    // Defaults to 1 (unshadowed) for non-foliage, whose transmission is unused.
    float sssTransmission = 1.f;
    uint shadingModelID = decodeShadingModelId(gGBufferB.Load(int3(pixel, 0)).a);
    if (shadingModelID == SHADINGMODELID_TWOSIDED_FOLIAGE)
    {
        float opacity = gGBufferD.Load(int3(pixel, 0)).a;

        PCFSettings sss;
        // +gConstantDepthBias pushes the receiver slightly toward the light so
        // its own front face doesn't self-occlude (UE LightSpacePixelDepthForSSS
        // + ProjectionDepthBiasParameters.x, :325).
        sss.sceneDepth = sceneDepth + gConstantDepthBias;
        // UE uses the unmodulated SoftTransitionScale for the subsurface pass
        // (:324); it is unused by the subsurface branch of calcOcclusion4 anyway.
        sss.transitionScale = gSoftTransitionScale;
        sss.bSubsurface = true;
        sss.densityMulConstant = subsurfaceDensityFromOpacity(opacity) * gDensityDepthScale;
        sss.pixel = pixel;

        sssTransmission = filterShadow(shadowUV, sss);
        // UE applies the overblur correction to transmission too (:367), but
        // not ShadowSharpen (that is surface-only, :375).
        sssTransmission = applyPCFOverBlurCorrection(sssTransmission);
    }

    gShadowMask[pixel] = float2(surfaceShadow, sssTransmission);
}
