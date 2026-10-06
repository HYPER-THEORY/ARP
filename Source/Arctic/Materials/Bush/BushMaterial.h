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
#pragma once
#include "Falcor.h"
#include "Scene/Material/Material.h"
#include "BushMaterialData.slang"

using namespace Falcor;

/** Bush (foliage) material.

    Derives directly from `Material`, not `BasicMaterial`. The earlier version
    derived from `BasicMaterial` and reused `BasicMaterialData`, which meant every
    bush parameter had to be smuggled through an unrelated field: the subsurface
    color map lived in `texTransmission`, the subsurface weight in
    `diffuseTransmission`, occlusion and roughness in `specular.rg`. Two shading
    paths then disagreed about what those fields meant. `BushMaterialData` gives
    each parameter one name and one meaning, and the rasterization and
    ray-tracing paths both read it directly.

    Three textures (see `BushMaterialData` for the channel layout):
        BaseColor    -> RGB albedo + A opacity mask       (sRGB)
        Normal       -> RG normal XY, B specular, A roughness (linear)
        Transmission -> RGB subsurface color + A opacity  (sRGB)

    The `Specular` and `Emissive` slots are intentionally left disabled: the
    packed Normal map already carries specular and roughness, and foliage does
    not emit.

    Alpha masking uses a single threshold, `MaterialHeader::getAlphaThreshold()`.
    Both the GBuffer clip (`BushGBuffer.slang::customClip_Bush`) and
    `MaterialSystem::alphaTest` (used by ShadowPass and VBufferRT) read it, so the
    shadow silhouette cannot drift far from the geometry silhouette. The GBuffer
    clip is the softer of the two: it dissolves the band below the threshold (and
    cards seen nearly edge-on) with a frame-animated dither that TAAPass resolves
    into apparent semi-transparency, so its silhouette is slightly thinner than
    the shadow's.

    Parameter grouping in the UI mirrors the `//#group` annotations in the
    original engine shader `foliage_bush_vertex_color.hlsl`.
*/
class BushMaterial : public Material
{
    FALCOR_OBJECT(BushMaterial)
public:
    static ref<BushMaterial> create(ref<Device> pDevice, const std::string& name) { return make_ref<BushMaterial>(pDevice, name); }

    BushMaterial(ref<Device> pDevice, const std::string& name);

    // ---- Material interface ----

    bool renderUI(Gui::Widgets& widget) override;
    Material::UpdateFlags update(MaterialSystem* pOwner) override;
    bool isEqual(const ref<Material>& pOther) const override;
    MaterialDataBlob getDataBlob() const override { return prepareDataBlob(mData); }
    ProgramDesc::ShaderModuleList getShaderModules() const override;
    TypeConformanceList getTypeConformances() const override;

    void setDefaultTextureSampler(const ref<Sampler>& pSampler) override;
    ref<Sampler> getDefaultTextureSampler() const override { return mpDefaultSampler; }

    // Alpha masking. These must be overridden: the `Material` base
    // implementations only log "does not support alpha" and do nothing (only
    // `BasicMaterial` overrode them), which would silently leave the threshold at
    // its default and break the leaf cutouts.
    void setAlphaMode(AlphaMode alphaMode) override;
    void setAlphaThreshold(float alphaThreshold) override;

    // ---- Albedo (chaos group "Albedo") ----

    void setBaseColorMultiplier(const float4& v);
    float4 getBaseColorMultiplier() const { return (float4)mData.baseColorMultiplier; }

    void setBaseColorBrightness(float v);
    float getBaseColorBrightness() const { return (float)mData.baseColorBrightness; }

    void setBaseColorContrast(float v);
    float getBaseColorContrast() const { return (float)mData.baseColorContrast; }

    void setBaseColorSaturation(float v);
    float getBaseColorSaturation() const { return (float)mData.baseColorSaturation; }

    // ---- Normal (chaos group "Normal") ----

    void setNormalStrength(float v);
    float getNormalStrength() const { return (float)mData.normalStrength; }

    // ---- OrsMask (chaos group "OrsMask") ----
    // Occlusion is not among these: the packed map has no free channel for it
    // and Falcor's vertex layout has no color channel, so occlusion comes from
    // the screen-space AO passes instead.

    void setRoughnessScale(float v);
    float getRoughnessScale() const { return (float)mData.roughnessScale; }

    void setRoughnessContrast(float v);
    float getRoughnessContrast() const { return (float)mData.roughnessContrast; }

    void setSpecularValue(float v);
    float getSpecularValue() const { return (float)mData.specularValue; }

    // ---- SubSurface (chaos group "SubSurface") ----

    void setSubsurfaceColorMultiplier(const float4& v);
    float4 getSubsurfaceColorMultiplier() const { return (float4)mData.subsurfaceColorMultiplier; }

    void setSubsurfaceColorBrightness(float v);
    float getSubsurfaceColorBrightness() const { return (float)mData.subsurfaceColorBrightness; }

    void setSubsurfaceColorContrast(float v);
    float getSubsurfaceColorContrast() const { return (float)mData.subsurfaceColorContrast; }

    void setSubsurfaceColorSaturation(float v);
    float getSubsurfaceColorSaturation() const { return (float)mData.subsurfaceColorSaturation; }

    void setOpacityValue(float v);
    float getOpacityValue() const { return (float)mData.opacityValue; }

    // ---- ColorVariability (chaos group "ColorVariability") ----
    // The mask texture is a scene-wide detail map bound by the render pass, not
    // a material texture slot. No pass binds it today, which is why this is
    // disabled by default; with it disabled the shader never samples the map.

    void setColorVariabilityEnabled(bool enabled);
    bool isColorVariabilityEnabled() const { return mData.isColorVariabilityEnabled(); }

    void setColorVariabilityTint(const float4& v);
    float4 getColorVariabilityTint() const { return (float4)mData.colorVariabilityTint; }

    void setColorVariabilityTilling(const float4& v);
    float4 getColorVariabilityTilling() const { return mData.colorVariabilityTilling; }

    void setColorVariabilityContrast(float v);
    float getColorVariabilityContrast() const { return (float)mData.colorVariabilityContrast; }

    const BushMaterialData& getData() const { return mData; }

private:
    /** Assign a float16 scalar, marking the material dirty only on change.
     */
    void setScalar(float16_t& dst, float value, float lo, float hi);

    /** Assign a float16 vector, marking the material dirty only on change.
     */
    void setColor(float16_t4& dst, const float4& value, float lo, float hi);

    BushMaterialData mData;         ///< Material parameters, uploaded as the material data blob.
    ref<Sampler> mpDefaultSampler;  ///< Sampler used for all three texture slots.
};
