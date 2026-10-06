/** Chaos-style StandardMaterial assembly for the GBuffer (rasterization) pass.

    Falcor's own `StandardMaterial` is the DefaultLit surface for Arctic. This
    module is the bridge: it fills a `MaterialParameters` struct (no BSDF, no
    IMaterialInstance) that GBufferPass.3d.slang encodes into the UE-layout MRTs
    via `encodeMaterialToGbuffer`.

    It deliberately lives OUTSIDE `Rendering/Materials/StandardMaterial.slang`.
    That file is registered as a material shader module (an anonymous Slang
    translation unit) by `StandardMaterial::getShaderModules()`, so if GBufferPass
    also `import`ed it by name, Slang would load the file a second time and
    `struct StandardMaterial` would exist twice in the linked global scope. The
    `{StandardMaterial, IMaterial}` type conformance lookup then fails with
    "Type 'StandardMaterial' in type conformance was not found." This mirrors how
    `Arctic/Materials/Bush/BushGBuffer.slang` relates to
    `Arctic/Materials/Bush/BushMaterial.slang`.

    Channel mapping, MetalRough only (BasicMaterialData + MaterialHeader):
      - baseColor.rgb            -> GBufferC.rgb (sRGB-encoded by the format)
      - baseColor.a              -> alpha mask (see customClip_Standard)
      - specular.g               -> roughness      (GBufferB.b)
      - specular.b               -> metallic       (GBufferB.r)
      - IoR                      -> UE dielectric Specular (GBufferB.g), see below
      - emissive * emissiveFactor-> the dedicated HDR emissive MRT
      - texNormalMap             -> GBufferA.rgb

    Two UE GBuffer channels have no StandardMaterial field:

      - GBufferB.g is UE's dielectric Specular, a [0,1] knob that scales F0 by
        0.08 (F0 = 0.08 * Specular). Falcor stores the same quantity as an index
        of refraction instead, and derives F0 = ((n-1)/(n+1))^2
        (StandardMaterial.slang). Inverting that gives Specular = F0 / 0.08, and
        Falcor's default IoR lands exactly on UE's default Specular. Going
        through IoR keeps the raster GBuffer and Falcor's own ray-traced
        StandardMaterial BSDF agreeing on the same F0.

      - GBufferC.a is UE's GBufferAO. Falcor has no occlusion texture slot, and
        the Specular slot only enables the G|B channels in MetalRough
        (StandardMaterial.cpp), so glTF's packed ORM occlusion is dropped on
        import and there is nothing to read. It is left at 1 and the screen-space
        AO passes (SSAOPass/GTAOPass) are the only occlusion source, exactly as
        for BushMaterial.

    SpecGloss is not handled here: it encodes `specular` as (F0.rgb, glossiness),
    which the metal-rough GBuffer cannot carry without a lossy metallic recovery.
    GBufferPass routes SpecGloss materials to the unsupported-material fallback.
*/

import Scene.Scene;
import Scene.Material.BasicMaterialData;
import Scene.Material.ShadingUtils;
import Scene.Material.TextureSampler;
import Arctic.GBufferPass.GBufferCommon;

/** Assemble callback for StandardMaterial (MetalRough).

    GBufferPass.3d.slang calls this with the header and the BasicMaterialData
    already fetched from the scene, avoiding a __dynamic_cast on the IMaterial
    dispatch.
*/
void assembleMaterialParams_Standard(
    const MaterialHeader materialHeader,
    const BasicMaterialData materialData,
    const PixelParameters pixelParams,
    inout MaterialParameters materialParams)
{
    let lod = ImplicitLodTextureSampler();
    const SamplerState s = pixelParams.sampler;

    const float4 baseColorOpacity =
        gScene.materials.sampleTexture(materialData.texBaseColor, s, pixelParams.uv, materialData.baseColor, lod);

    // G - Roughness; B - Metallic. Same unpacking as StandardMaterial.slang.
    const float4 spec =
        gScene.materials.sampleTexture(materialData.texSpecular, s, pixelParams.uv, materialData.specular, lod);

    // UE dielectric Specular recovered from Falcor's IoR (see the header note).
    const float ior = (float)materialHeader.getIoR();
    const float f = (ior - 1.f) / (ior + 1.f);
    const float F0 = f * f;

    // Emissive is orthogonal to the shading model in UE: a DefaultLit material
    // can emit. It goes to GBufferPass' dedicated HDR emissive MRT, so it is not
    // clamped to [0,1] and emissiveFactor can push it well above 1. Gated on the
    // facing sign to match StandardMaterial.slang, which only emits from the
    // front side.
    float3 emissive = float3(0.f);
    if (pixelParams.isFrontFaceFloat > 0.f)
    {
        emissive = gScene.materials
                       .sampleTexture(
                           materialData.texEmissive, s, pixelParams.uv, float4(materialData.emissive, 1.f), lod)
                       .rgb *
                   materialData.emissiveFactor;
    }

    // Normal mapping, tangent space to world. This is the shading-normal half of
    // Falcor's `computeShadingFrameFromNormalMap`: only the perturbed normal is
    // needed here, not a full re-orthonormalized frame, and `mul(mapN, tbn)` is
    // that function's `frame.fromLocal(mapN)` written out. Leaving mapN at +z for
    // NormalMapType::None yields the geometric normal, so the unbound normal map
    // needs no separate guard.
    float3 localNormal = float3(0.f, 0.f, 1.f);
    const float3 encodedNormal =
        gScene.materials.sampleTexture(materialData.texNormalMap, s, pixelParams.uv, float4(0.f), lod).rgb;
    switch (materialData.getNormalMapType())
    {
    case NormalMapType::RGB:
        localNormal = rgbToNormal(encodedNormal);
        break;
    case NormalMapType::RG:
        localNormal = rgToNormal(encodedNormal.rg);
        break;
    }
    float3 worldNormal = normalize(mul(localNormal, pixelParams.tbn));

    // Falcor's `flipShadingNormal`: a double-sided surface seen from behind
    // shades with the reversed normal.
    if (pixelParams.isFrontFaceFloat < 0.f && materialHeader.isDoubleSided())
    {
        worldNormal = -worldNormal;
    }

    materialParams.baseColor = max(baseColorOpacity.rgb, float3(0.f));
    materialParams.normal = worldNormal;
    materialParams.metallic = saturate(spec.b);
    materialParams.roughness = saturate(spec.g);
    materialParams.ao = 1.f;                                // See the GBufferAO note above.
    materialParams.dielectricSpecularMultiplier = saturate(F0 / 0.08f);
    // DefaultLit has no subsurface inputs, so subsurfaceColor/opacity stay 0 and
    // the encoder writes GBufferD as 0, matching UE's DefaultLit CustomData.
    materialParams.subsurfaceColor = float3(0.f);
    materialParams.opacity = 0.f;
    materialParams.emissiveColor = emissive;
    materialParams.opacityMask = baseColorOpacity.a;
    materialParams.shadingModelID = SHADINGMODELID_DEFAULT_LIT;
}

/** Alpha clip.

    Both the mode and the threshold come from the MaterialHeader, the same two
    values `MaterialSystem::alphaTest` reads in ShadowPass. Keeping one source is
    what stops the shadow silhouette from disagreeing with the GBuffer
    silhouette. Unlike `customClip_Bush`, the mode check is not optional here:
    BushMaterial is always alpha-masked, while a StandardMaterial with
    AlphaMode::Opaque and baseColor.a below the threshold must not be clipped.
*/
void customClip_Standard(
    const MaterialHeader materialHeader,
    const BasicMaterialData materialData,
    const PixelParameters pixelParams,
    inout MaterialParameters materialParams)
{
    if (materialHeader.getAlphaMode() != AlphaMode::Mask) return;
    if (materialParams.opacityMask < (float)materialHeader.getAlphaThreshold()) discard;
}
