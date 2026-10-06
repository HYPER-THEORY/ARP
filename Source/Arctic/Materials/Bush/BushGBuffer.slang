/** Chaos-style bush material assembly for the GBuffer (rasterization) pass.

    This is the *baseline* definition of what a bush surface is. It mirrors the
    original engine shader `foliage_bush_vertex_color.hlsl::assembleMaterialParams`
    and `customClip`. It deliberately bypasses
    Falcor's IMaterial/IMaterialInstance/IBSDF three-layer interface: instead of
    returning a BSDF, it fills a `MaterialParameters` struct that
    GBufferPass.3d.slang encodes into the UE-layout MRTs via
    `encodeMaterialToGbuffer`.

    The ray-tracing path (`BushMaterial.slang`) is written separately against the
    same three textures and the same `BushMaterialData`; it is a simplification of
    this file (no screen-space derivatives, no color variability). Only the pure
    math in `BushCommon.slang` is shared.

    This module deliberately lives OUTSIDE `BushMaterial.slang`. That file is
    registered as a material shader module (an anonymous Slang translation unit)
    by `BushMaterial::getShaderModules()`, so if GBufferPass also `import`ed it by
    name, Slang would load the file a second time and `struct BushMaterial` would
    exist twice in the linked global scope. The `{BushMaterial, IMaterial}` type
    conformance lookup then fails with "Type 'BushMaterial' in type conformance
    was not found."

    What is migrated from the original HLSL:
      - 3-texture sampling (baseColor/opacity, normal/specular/roughness,
        subsurfaceColor/opacity).
      - ColorCorrect on base color and subsurface color, independently
        (foliage_bush_vertex_color.hlsl).
      - ColorVariability (world-space mask + tint).
      - opacity_mask + density (the latter is UE's Opacity input; see below).
      - Normal decode + normal_strength.
      - Roughness with contrast/scale.
      - Specular via the packed specular channel multiplied by specular_value.
      - Temporal dither (`temporalDitherLerp`, temporal_dither.hlsli:5), used here
        to soften the leaf silhouette and to dissolve cards seen edge-on.

    What is intentionally NOT migrated (per project scope):
      - Wind (Wind-Branch / Wind-Frond).
      - Interactive-RT (character trail).
      - Rippling (wind waves).
      - near-clip hexahedron dither (no engine primitive available).
      - AO from vertex color: Falcor's vertex layout has no color channel, and
        the packed map has no free channel, so `ao` is left at 1 and the screen
        -space AO passes (SSAOPass/GTAOPass) are the only occlusion source.
      - blending the reconstructed ddx/ddy face normal into the *shading* normal
        by distance: the reference blend weight was 0, so the reconstructed
        frame never contributed there. The clip's grazing term does use that
        reconstruction, see `customClip_Bush`.
*/

import Scene.Scene;
import Scene.Material.ShadingUtils;
import Scene.Material.TextureSampler;
import Scene.ShadingData;
import Utils.Math.MathHelpers;
import Arctic.Materials.Bush.BushMaterialData;
import Arctic.Materials.Bush.BushCommon;
import Arctic.GBufferPass.GBufferCommon;

// World-space color variability mask texture. This is a *fourth* texture and is
// intentionally not a Material::TextureSlot: it is a global, scene-wide detail
// map rather than a per-material asset, so it would be bound by the pass. It is
// currently never bound by GBufferPass, which is why
// BushMaterialData::isColorVariabilityEnabled() defaults to false — with the
// feature off, the sample below is never reached and the unbound descriptor is
// never read.
Texture2D<float4> gColorVariabilityMaskMap;
SamplerState      gColorVariabilitySampler;

void assembleMaterialParams_Bush(
    const MaterialHeader materialHeader,
    const BushMaterialData materialData,
    const PixelParameters pixelParams,
    inout MaterialParameters materialParams)
{
    const SamplerState s = pixelParams.sampler;

    // No uniform fallbacks: an unbound texture reads 0 so the failure is visible.
    const float4 baseColor_Opacity       = gScene.materials.sampleTexture(materialData.texBaseColor, s, pixelParams.uv, float4(0), ImplicitLodTextureSampler());
    const float4 normal_Spec_Rough       = gScene.materials.sampleTexture(materialData.texNormalSpecRough, s, pixelParams.uv, float4(0), ImplicitLodTextureSampler());
    const float4 subsurfaceColor_Opacity = gScene.materials.sampleTexture(materialData.texSubsurfaceColor, s, pixelParams.uv, float4(0), ImplicitLodTextureSampler());

    float3 baseColor = bushColorCorrection(
        baseColor_Opacity.rgb,
        (float)materialData.baseColorSaturation,
        (float3)materialData.baseColorMultiplier.rgb,
        (float)materialData.baseColorBrightness,
        (float)materialData.baseColorContrast);

    float3 subsurfaceColor = bushColorCorrection(
        subsurfaceColor_Opacity.rgb,
        (float)materialData.subsurfaceColorSaturation,
        (float3)materialData.subsurfaceColorMultiplier.rgb,
        (float)materialData.subsurfaceColorBrightness,
        (float)materialData.subsurfaceColorContrast);

    if (materialData.isColorVariabilityEnabled())
    {
        float2 maskUV = pixelParams.worldPosition.xy / materialData.colorVariabilityTilling.xy + materialData.colorVariabilityTilling.zw;
        float mask = gColorVariabilityMaskMap.SampleLevel(gColorVariabilitySampler, maskUV, 0).r;
        mask = pow(mask, (float)materialData.colorVariabilityContrast);
        float3 tint = (float3)materialData.colorVariabilityTint.rgb;
        baseColor = lerp(baseColor * tint, baseColor, mask);
        subsurfaceColor = lerp(subsurfaceColor * tint, subsurfaceColor, mask);
    }

    const float opacityMask = baseColor_Opacity.a;
    const float opacity = subsurfaceColor_Opacity.a * (float)materialData.opacityValue;

    const float3 localNormal = bushDecodeNormal(normal_Spec_Rough.xy, (float)materialData.normalStrength);
    const float3 worldNormal = normalize(mul(localNormal, pixelParams.tbn));

    const float roughness = pow(normal_Spec_Rough.a * (float)materialData.roughnessScale, (float)materialData.roughnessContrast);
    const float dielectricSpecular = normal_Spec_Rough.b * (float)materialData.specularValue;

    materialParams.baseColor = baseColor;
    materialParams.normal = worldNormal;
    materialParams.metallic = 0.f;                                  // Foliage is always dielectric.
    materialParams.roughness = roughness;
    materialParams.ao = 1.f;                                        // See the note on vertex-color AO above.
    materialParams.dielectricSpecularMultiplier = dielectricSpecular;
    materialParams.subsurfaceColor = subsurfaceColor;
    materialParams.opacity = opacity;
    materialParams.emissiveColor = float3(0.f);
    materialParams.opacityMask = opacityMask;
    materialParams.shadingModelID = SHADINGMODELID_TWOSIDED_FOLIAGE;
}

// ---------------------------------------------------------------------------
// Temporal dither, ported from the engine's `temporalDitherLerp`
// (ref/chaos/shaders/core/common/temporal_dither.hlsli:5), which is the engine's
// port of UE's DitherTemporalAA material node.
//
// A deferred GBuffer cannot express partial coverage — there is no blending to do
// it with — so a partially covered surface has to be stippled. The only question
// is whether the stipple is *resolvable*. A pattern that is fixed (in screen
// space or on the surface) is not: TAA converges to the pattern itself and it
// reads as holes in the foliage. Animating the pattern with the frame index makes
// each pixel sample a different threshold every frame, so TAAPass' history
// integrates the surviving fraction and the stipple resolves into apparent
// semi-transparency.
//
// An earlier revision used the world-space hashed alpha test (Wyman & McGuire
// 2017, engine `hashedAlpha`, common_endecode.hlsli:210). Its selling point is
// scale stability — the noise is locked to the surface and does not swim — but
// that is exactly what makes it unresolvable: the pattern is static, TAA's
// sub-pixel jitter only crosses a hash cell boundary part of the time, and up
// close (where the alpha fade band and the grazing band cover much of the screen)
// the residual salt-and-pepper reads as many small holes. Screen-space ordered
// dither also has a strictly better per-frame spatial distribution than a hash,
// so the unconverged frame is less noisy.
// ---------------------------------------------------------------------------

/** Per-pixel stochastic alpha threshold in [0,1), animated by the frame index.

    A surface with coverage `c` survives where `c > threshold`, so the surviving
    fraction over a neighbourhood — and, because the pattern animates, over a
    pixel's TAA history — is `c`.

    \param[in] svPos SV_Position.xy, i.e. pixel centres at (x + 0.5, y + 0.5).
               The `- 1.5` below cancels those two half-pixel offsets so that
               `svPos.x + 2 * svPos.y - 1.5` is an integer and the ordered term
               lands exactly on {0, 1/8, ... 7/8}. Passing rounded-down integer
               coordinates instead would shift the whole pattern by -1.5/8.
    \param[in] frameIndex Frame counter; advances the ordered pattern by one step
               per frame. Only meaningful while there is a temporal history to
               integrate the pattern, which is why the caller gates on
               `isTemporalEnable` rather than dithering unconditionally.
*/
float bushTemporalDitherThreshold(float2 svPos, uint frameIndex)
{
    // Ordered 8-step pattern along the (1,2) diagonal, stepped by the frame index.
    const float ordered = frac((svPos.x + svPos.y * 2.0 - 1.5 + (float)frameIndex) / 8.0);
    // Per-pixel offset that breaks up the diagonal banding within one step.
    const float noise = frac(dot(float2(171.0, 231.0) / 71.0, svPos));

    // Deliberate deviation from the engine, which divides by 9, not 8: that caps
    // the threshold at 8/9 and inflates every coverage by 12.5% (a nominal 0.4
    // renders as 0.45). Dividing by 8 makes `ordered * 8` (an integer in [0,8))
    // plus `noise` cover [0,8) uniformly, so the surviving fraction equals the
    // requested coverage exactly and the 0.5*T / 15 deg / 7.5 deg fade
    // coordinates below mean what they say. Equals `ordered + noise / 8`.
    return (ordered * 8.0 + noise) * (1.0 / 8.0);
}

/** Alpha clip, softened into apparent semi-transparency by a temporal dither.

    Two coverage terms feed one dithered test:

    1. Alpha edge. Instead of a binary cut at the MaterialHeader threshold `T`,
       coverage ramps from 0 at `T * kBushAlphaFadeEndScale` up to 1 at `T`, so
       the leaf silhouette dissolves over that band rather than terminating on a
       hard, aliased edge.

    2. Grazing angle. A foliage card seen nearly edge-on degenerates into a
       one-pixel sliver of stretched texels that shimmers under any camera
       motion. Coverage ramps to 0 as the angle between the view ray and the
       *face plane* closes: 1 at 15 deg, 0 at 7.5 deg. Since
       |dot(V, faceN)| = sin(angle to the plane), the thresholds below are the
       sines of those angles, and the absolute value makes the test
       orientation-agnostic (two-sided foliage is drawn from both sides).

    The grazing term deliberately reaches 0: a card past 7.5 deg disappears
    entirely rather than settling at some minimum coverage. Close up, with nothing
    behind it, that still reads as a gap — that is the specified behaviour, not a
    dither artifact.

    The two are combined with min() and compared against a single dithered
    threshold, so the effects do not compound into extra transparency where they
    overlap.

    Both coverage terms only apply while a temporal history exists to integrate
    the stipple. With no history the pattern cannot resolve into
    semi-transparency, so the clip falls back to the plain hard alpha test and
    the silhouette is aliased but solid rather than riddled with a frozen
    stipple. This is the reference shader's own two-branch structure.

    The threshold `T` itself comes from the MaterialHeader, the same value
    `MaterialSystem::alphaTest` uses in ShadowPass. ShadowPass deliberately keeps
    the plain hard test: the grazing term is defined against the camera ray and
    is meaningless from the light's point of view, so the shadow silhouette
    tracks the un-softened alpha cut and is very slightly fuller than the
    GBuffer one.
*/
void customClip_Bush(
    const MaterialHeader materialHeader,
    const BushMaterialData materialData,
    const PixelParameters pixelParams,
    inout MaterialParameters materialParams)
{
    static const float kBushAlphaFadeEndScale = 0.75;      ///< Fully faded at T * this.
    static const float kBushGrazingFadeStartSin = 0.25881905; ///< sin(15 deg): fade begins.
    static const float kBushGrazingFadeEndSin = 0.13052619;   ///< sin(7.5 deg): fully faded.

    const float alphaThreshold = (float)materialHeader.getAlphaThreshold();
    const float alphaFadeStart = alphaThreshold * kBushAlphaFadeEndScale;
    const float alphaCoverage =
        saturate((materialParams.opacityMask - alphaFadeStart) / max(alphaThreshold - alphaFadeStart, 1e-6));

    // Face normal of this quad's plane, from the screen-space gradients of world
    // position. This is the geometric normal the grazing test must use, not the
    // authored vertex normal, which foliage cards deliberately bend away from the
    // plane for softer lighting. Computed outside the branch below so the
    // gradients are taken before any lane discards.
    const float3 facePlaneNormal =
        normalize(cross(ddx(pixelParams.worldPosition.xyz), ddy(pixelParams.worldPosition.xyz)));

    // |dot(V, N)| = cos(angle to the normal) = sin(angle to the face plane). The
    // absolute value makes the test orientation-agnostic, so the plane normal
    // needs no winding fix-up.
    const float sinToPlane = abs(dot(pixelParams.viewDirection, facePlaneNormal));
    const float grazingCoverage = saturate(
        (sinToPlane - kBushGrazingFadeEndSin) / (kBushGrazingFadeStartSin - kBushGrazingFadeEndSin));

    if (pixelParams.isTemporalEnable)
    {
        // `<=`, not `<`: the threshold's range is [0,1), so a fully faded surface
        // (coverage 0) would otherwise survive on the pixels where the dither is
        // exactly 0. Coverage 1 still always survives, since the threshold never
        // reaches 1.
        const float coverage = min(alphaCoverage, grazingCoverage);
        if (coverage <= bushTemporalDitherThreshold(pixelParams.svPosition.xy, (uint)pixelParams.jitterFrameIndex))
            discard;
    }
    else if (materialParams.opacityMask < alphaThreshold)
    {
        discard;
    }
}
