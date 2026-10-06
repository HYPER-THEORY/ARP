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
__exported import Rendering.Materials.IMaterial;
__exported import Arctic.Materials.Bush.BushMaterialInstance;
import Arctic.Materials.Bush.BushMaterialData;
import Arctic.Materials.Bush.BushCommon;
import Scene.Material.ShadingUtils;
import Scene.Scene;

/** Ray-tracing (IMaterial) path for the bush material.

    This reads the same three textures and the same `BushMaterialData` as the
    rasterization path in `BushGBuffer.slang`, applies the same color correction,
    normal decode, and roughness/specular remap, and feeds the result to
    `BushBRDF` — which is a term-for-term transcription of the deferred path's
    `DIPass.3d.slang::twoSidedBxDF`. So RT and raster agree on both the surface
    parameters and the shading model.

    Two deliberate simplifications relative to the raster path:

      1. No ColorVariability. The variability mask is a scene-wide detail texture
         bound by the pass, not a material texture slot; there is no pass to bind
         it on the RT side. The feature is off by default anyway.
      2. No ddx/ddy face-normal reconstruction. Ray tracing has no screen-space
         derivatives. The raster path's reconstructed frame was blended in with
         weight 0, so this costs nothing in practice.

    Notably absent: the shading normal is NOT flipped for backfacing hits, and
    there is no "subsurface weight" scalar. Both match the deferred path — see
    the header comment in BushBRDF.slang.
*/
struct BushMaterial : MaterialBase, IMaterial
{
    BushMaterialData data;

    typedef BushMaterialInstance MaterialInstance;

    BushMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        SamplerState s = ms.getTextureSampler(header.getDefaultTextureSamplerID());

        // No uniform fallbacks: an unbound texture reads 0 so the failure is visible.
        const float4 baseColor_Opacity       = ms.sampleTexture(data.texBaseColor, s, sd.uv, float4(0.f), lod);
        const float4 normal_Spec_Rough       = ms.sampleTexture(data.texNormalSpecRough, s, sd.uv, float4(0.f), lod);
        const float4 subsurfaceColor_Opacity = ms.sampleTexture(data.texSubsurfaceColor, s, sd.uv, float4(0.f), lod);

        BushBRDF brdf = {};

        brdf.diffuseColor = bushColorCorrection(
            baseColor_Opacity.rgb,
            (float)data.baseColorSaturation,
            (float3)data.baseColorMultiplier.rgb,
            (float)data.baseColorBrightness,
            (float)data.baseColorContrast);

        brdf.subsurfaceColor = bushColorCorrection(
            subsurfaceColor_Opacity.rgb,
            (float)data.subsurfaceColorSaturation,
            (float3)data.subsurfaceColorMultiplier.rgb,
            (float)data.subsurfaceColorBrightness,
            (float)data.subsurfaceColorContrast);

        brdf.roughness = pow(normal_Spec_Rough.a * (float)data.roughnessScale, (float)data.roughnessContrast);

        // UE convention: dielectric F0 = 0.08 * specular. Metallic is always 0
        // for foliage, so the metal branch of UE's lerp is unreachable.
        brdf.specularColor = float3(0.08f * normal_Spec_Rough.b * (float)data.specularValue);

        // Normal mapping. `computeShadingFrameFromNormalMap` is not used because
        // it cannot apply normalStrength, and because the packed map's B channel
        // is dielectric specular rather than normal Z — neither NormalMapType::RGB
        // nor ::RG would decode it with the strength applied.
        ShadingFrame sf = sd.frame;
        if (isNormalMappingEnabled(hints))
        {
            const float3 localNormal = bushDecodeNormal(normal_Spec_Rough.xy, (float)data.normalStrength);
            sf = ShadingFrame(normalize(sd.frame.fromLocal(localNormal)), sd.tangentW);
        }
        if (isAdjustShadingNormalEnabled(hints))
        {
            adjustShadingNormal(sd, sf);
        }

        return BushMaterialInstance(sf, brdf);
    }

    [Differentiable]
    BushMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        diffData = DiffMaterialData();
        return no_diff setupMaterialInstance(ms, sd, lod, hints);
    }
};
