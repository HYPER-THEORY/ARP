/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms with or without
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

/** Chaos-style GBuffer common facilities.

    This module mirrors the structure of ref/chaos/shaders/core/gbuffer/gbuffer_mr.hlsli:
    a `MaterialParameters` aggregate that the material's `assembleMaterialParams`
    callback fills in, and an `encodeMaterialToGbuffer` encoder that writes the
    UE-layout MRTs consumed by DIPass.

    Unlike Falcor's IMaterial/IMaterialInstance/IBSDF three-layer interface, this
    is a flat "fill struct -> encode to MRT" pipeline, intentionally bypassing
    Falcor's material dispatch so the foliage shader can stay close to the
    original engine style.

    The *structure* is chaos', but the *encoding* is UE's, byte for byte. In
    particular `shadingModelID` holds a `SHADINGMODELID_*` value directly rather
    than a chaos `SHADE_MODE_*`: the two enums disagree (chaos numbers
    TwoSidedFoliage 10, UE numbers it 6) and the old hybrid `SHADE_MODE_*` here
    matched neither, with DEFAULT_LIT = 0 while UE's is 1. Using UE's IDs removes
    the mapping table and the class of bug that came with it.
*/

// Re-exported: MaterialParameters::shadingModelID is typed in terms of the
// SHADINGMODELID_* constants, so every material that fills the struct needs them
// in scope. Without `__exported`, Slang imports are not transitive and each
// assembler would have to import GBufferHelpers separately. MaterialHeader is
// re-exported for the same reason: it appears in every assembler's signature.
__exported import Arctic.GBufferPass.GBufferHelpers;
__exported import Scene.Material.MaterialData;

/** Per-pixel geometric and per-frame inputs to material assembly.

    Semantics follow chaos' `PixelParameters`
    (core/pixel_shader_factory/pixel_parameters.hlsli), trimmed to the fields
    Arctic can source. Together with `MaterialParameters` these are the two
    material-agnostic halves of the assembly signature:

        assembleMaterialParams_<Type>(materialHeader, <Type>MaterialData, pixelParams, materialParams)

    `customClip_<Type>` takes the same four arguments, so the pass dispatches
    both through one uniform shape and adding a material type does not change
    the call site's structure.

    The frame in `worldNormal` / `worldTangent` / `tbn` is the interpolated
    *geometric* vertex frame, before normal mapping. The normal-mapped result is
    an output of assembly and lands in `MaterialParameters::normal`.
*/
struct PixelParameters
{
    float2 uv;
    float4 worldNormal;                ///< Normalized. Interpolated vertex normal.
    float4 worldTangent;               ///< xyz normalized tangent, w handedness sign.
    float3x3 tbn;                      ///< Rows (T, B, N): tangent to world, so use mul(localVector, tbn).
    float4 worldPosition;
    /// SV_Position as the rasterizer delivers it: xy pixel centres, z device
    /// depth, and w the *reciprocal* of clip-space w.
    float4 svPosition;
    float linearDepth;                 ///< View-space depth, i.e. clip-space w.
    float3 viewDirection;              ///< Normalized, surface to camera.
    bool isTemporalEnable;
    int jitterFrameIndex;
    float ditherClipOffset;
    float transparencyDistanceThreshold;
    float4 jitterProjectionOffset;     ///< NDC projection offset. xy this frame, zw previous.
    float isFrontFaceFloat;            ///< +1 front-facing, -1 back-facing.
    float3 screenUVZ;                  ///< xy screen UV, z device depth.
    SamplerState sampler;              ///< The material's default texture sampler.
};

/** Aggregate of material parameters filled in by the material's
    `assembleMaterialParams` callback. This is the chaos-style equivalent of
    `MaterialParameters` in core/material/material_parameters.hlsli, trimmed to
    the fields used by the shading models Arctic produces.
*/
struct MaterialParameters
{
    float3 baseColor;
    float3 normal;                     ///< Normalized world-space shading normal.
    float  metallic;
    float  roughness;
    float  ao;
    float  dielectricSpecularMultiplier;
    float3 subsurfaceColor;            ///< Subsurface / foliage shading models. Linear, pre-encode.
    float  opacity;                    ///< UE material Opacity input. Range [0,1]. Drives SSS thickness.
    float3 emissiveColor;              ///< Emissive radiance. Written to its own HDR MRT, not the GBuffer.
    float  opacityMask;                ///< Alpha mask value before clip. Unrelated to `opacity`.
    uint   shadingModelID;             ///< SHADINGMODELID_* (see GBufferHelpers.slang).
};

/** The fallback path has no material data of its own. It carries the type
    anyway so its assembler keeps the common four-argument shape.
*/
struct UnsupportedMaterialData
{
};

/** Neutral DefaultLit parameters for a material this pass has no assembly for.

    Two call sites need the exact same values — the unsupported-MaterialType
    branch and StandardMaterial's SpecGloss shading model — so the defaults live
    here rather than being spelled out twice in GBufferPass.3d.slang.

    The pixel still gets valid, plausibly-shaded GBuffer data instead of being
    discarded, so an unsupported material reads as neutral grey rather than as a
    hole in the frame.
*/
void assembleMaterialParams_Unsupported(
    const MaterialHeader materialHeader,
    const UnsupportedMaterialData materialData,
    const PixelParameters pixelParams,
    inout MaterialParameters materialParams)
{
    materialParams.baseColor = float3(0.5f);
    materialParams.normal = pixelParams.tbn[2];
    materialParams.metallic = 0.f;
    materialParams.roughness = 0.5f;
    materialParams.ao = 1.f;
    materialParams.dielectricSpecularMultiplier = 0.5f;     // UE's default Specular.
    materialParams.subsurfaceColor = float3(0.f);
    materialParams.opacity = 0.f;
    materialParams.emissiveColor = float3(0.f);
    materialParams.opacityMask = 1.f;
    materialParams.shadingModelID = SHADINGMODELID_DEFAULT_LIT;
}

/** Encode MaterialParameters into the UE GBuffer MRT layout.

    Port of UE's EncodeGBuffer (DeferredShadingCommon.ush) plus the
    per-shading-model CustomData fill from SetGBufferForShadingModel
    (ShadingModelsMaterial.ush).

    Emissive is NOT part of UE's GBuffer: UE's base pass writes it straight into
    SceneColor. Arctic has no base-pass color target, so it goes to a separate
    HDR MRT that CompositePass adds back. It must not be folded into GBufferE,
    which is PrecomputedShadowFactors in UE and is an 8-bit UNORM target here
    (i.e. it would clamp any emissive above 1).
*/
void encodeMaterialToGbuffer(
    MaterialParameters mp,
    out float4 outGBufferA,
    out float4 outGBufferB,
    out float4 outGBufferC,
    out float4 outGBufferD,
    out float4 outGBufferE,
    out float4 outEmissive)
{
    // Emissive is orthogonal to the shading model: UE writes it for Unlit and
    // lit models alike, so it is filled before the Unlit early-out.
    outEmissive = float4(mp.emissiveColor, 0.f);

    if (mp.shadingModelID == SHADINGMODELID_UNLIT)
    {
        // UE's Unlit branch (DeferredShadingCommon.ush + SetGBufferForUnlit
        // :482-486): everything zero except the ID. Zeroing the normal matters —
        // it is what lets a downstream pass tell "unlit" from "unwritten".
        outGBufferA = float4(0.f);
        outGBufferB = float4(0.f, 0.f, 0.f,
                             encodeShadingModelIdAndSelectiveOutputMask(SHADINGMODELID_UNLIT, 0));
        outGBufferC = float4(0.f);
        outGBufferD = float4(0.f);
        outGBufferE = float4(0.f);
        return;
    }

    // GBufferA.a is UE's PerObjectGBufferData (decal receiver mask + selective
    // output). Arctic has no decals and no static lighting, so 0 is correct
    // rather than merely unimplemented. Same for SelectiveOutputMask in B.a:
    // no anisotropy, no precomputed shadows, no skipped velocity.
    outGBufferA = float4(encodeNormal(mp.normal), 0.f);

    outGBufferB = float4(mp.metallic, mp.dielectricSpecularMultiplier, mp.roughness,
                         encodeShadingModelIdAndSelectiveOutputMask(mp.shadingModelID, 0));

    outGBufferC = float4(encodeBaseColor(mp.baseColor), mp.ao);

    // CustomData. Only the (SubsurfaceColor, Opacity) family is produced here;
    // every other lit model writes 0 exactly as UE's DefaultLit does.
    if (hasSubsurfaceColorCustomData(mp.shadingModelID))
    {
        outGBufferD = float4(encodeSubsurfaceColor(mp.subsurfaceColor), saturate(mp.opacity));
    }
    else
    {
        outGBufferD = float4(0.f);
    }

    // PrecomputedShadowFactors. Arctic has no baked lighting, and UE writes 0
    // when ALLOW_STATIC_LIGHTING is off.
    outGBufferE = float4(0.f);
}
