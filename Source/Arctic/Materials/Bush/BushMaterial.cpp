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
#include "BushMaterial.h"
#include "Scene/Material/MaterialSystem.h"

namespace
{
    const char kShaderFile[] = "Arctic/Materials/Bush/BushMaterial.slang";
}

// The shader reads this struct via `reinterpret<BushMaterialData, MaterialPayload>`,
// which is a raw bit cast — so the C++ and Slang layouts must agree byte for byte.
// Slang gives float4 16B alignment and float16_t4 8B alignment, and
// BushMaterialData is ordered so that every vector sits on a matching boundary
// with no padding anywhere. Pinning the size here means a new field that breaks
// that ordering (and therefore introduces padding) fails to compile instead of
// silently shifting every subsequent parameter. If this fires, re-check the
// offset comments in BushMaterialData.slang rather than just updating the number.
static_assert(sizeof(BushMaterialData) == 80, "BushMaterialData layout changed; verify field alignment against the Slang side.");
static_assert(sizeof(MaterialHeader) + sizeof(BushMaterialData) <= sizeof(MaterialDataBlob), "BushMaterialData exceeds the material payload budget.");

BushMaterial::BushMaterial(ref<Device> pDevice, const std::string& name)
    : Material(pDevice, name, MaterialType::Bush)
{
    // Three texture slots. The Specular and Emissive slots stay disabled: the
    // Normal map already packs specular and roughness, and foliage does not emit.
    //
    // BaseColor must be the albedo/opacity map: Material::updateTextureHandle()
    // special-cases that slot to also publish the alpha texture handle in the
    // material header, which is what MaterialSystem::alphaTest() samples.
    mTextureSlotInfo[(uint32_t)TextureSlot::BaseColor] = { "baseColor", TextureChannelFlags::RGBA, true };
    // RG = normal XY (Z reconstructed), B = dielectric specular, A = roughness.
    mTextureSlotInfo[(uint32_t)TextureSlot::Normal] = { "normalSpecRough", TextureChannelFlags::RGBA, false };
    // RGB = subsurface color, A = opacity. sRGB like BaseColor since it is a color.
    mTextureSlotInfo[(uint32_t)TextureSlot::Transmission] = { "subsurfaceColor", TextureChannelFlags::RGBA, true };

    // Foliage is double-sided and alpha-masked. The threshold is the one and only
    // alpha cutoff; both the GBuffer clip and MaterialSystem::alphaTest read it
    // from the header. `thinSurface` is deliberately left alone: it only affects
    // StandardBSDF's transmission albedo, so setting it here would be noise.
    mHeader.setDoubleSided(true);
    mHeader.setAlphaMode(AlphaMode::Mask);
    mHeader.setAlphaThreshold((float16_t)0.333f);

    markUpdates(UpdateFlags::DataChanged);
}

ProgramDesc::ShaderModuleList BushMaterial::getShaderModules() const
{
    return { ProgramDesc::ShaderModule::fromFile(kShaderFile) };
}

TypeConformanceList BushMaterial::getTypeConformances() const
{
    return { { { "BushMaterial", "IMaterial" }, (uint32_t)MaterialType::Bush } };
}

Material::UpdateFlags BushMaterial::update(MaterialSystem* pOwner)
{
    FALCOR_ASSERT(pOwner);

    if (mUpdates != Material::UpdateFlags::None)
    {
        // Publishing the BaseColor handle also sets the header's alpha texture
        // handle, which is how alpha testing finds the opacity channel.
        updateTextureHandle(pOwner, TextureSlot::BaseColor, mData.texBaseColor);
        updateTextureHandle(pOwner, TextureSlot::Normal, mData.texNormalSpecRough);
        updateTextureHandle(pOwner, TextureSlot::Transmission, mData.texSubsurfaceColor);

        updateDefaultTextureSamplerID(pOwner, mpDefaultSampler);
    }

    auto flags = mUpdates;
    mUpdates = Material::UpdateFlags::None;
    return flags;
}

bool BushMaterial::isEqual(const ref<Material>& pOther) const
{
    auto other = dynamic_ref_cast<BushMaterial>(pOther);
    if (!other) return false;

    if (!isBaseEqual(*other)) return false;

#define compare_field(_a) if (mData._a != other->mData._a) return false
#define compare_vec_field(_a) if (any(mData._a != other->mData._a)) return false
    compare_field(flags);
    compare_vec_field(colorVariabilityTilling);
    compare_vec_field(baseColorMultiplier);
    compare_vec_field(subsurfaceColorMultiplier);
    compare_vec_field(colorVariabilityTint);
    compare_field(baseColorBrightness);
    compare_field(baseColorContrast);
    compare_field(baseColorSaturation);
    compare_field(subsurfaceColorBrightness);
    compare_field(subsurfaceColorContrast);
    compare_field(subsurfaceColorSaturation);
    compare_field(normalStrength);
    compare_field(roughnessContrast);
    compare_field(roughnessScale);
    compare_field(specularValue);
    compare_field(opacityValue);
    compare_field(colorVariabilityContrast);
#undef compare_field
#undef compare_vec_field

    return true;
}

void BushMaterial::setDefaultTextureSampler(const ref<Sampler>& pSampler)
{
    if (pSampler != mpDefaultSampler)
    {
        mpDefaultSampler = pSampler;
        markUpdates(UpdateFlags::ResourcesChanged);
    }
}

void BushMaterial::setAlphaMode(AlphaMode alphaMode)
{
    if (mHeader.getAlphaMode() != alphaMode)
    {
        mHeader.setAlphaMode(alphaMode);
        markUpdates(UpdateFlags::DataChanged);
    }
}

void BushMaterial::setAlphaThreshold(float alphaThreshold)
{
    const float16_t v = (float16_t)std::clamp(alphaThreshold, 0.f, 1.f);
    if (mHeader.getAlphaThreshold() != v)
    {
        mHeader.setAlphaThreshold(v);
        markUpdates(UpdateFlags::DataChanged);
    }
}

// ---- UI ----

bool BushMaterial::renderUI(Gui::Widgets& widget)
{
    // Material::renderUI() re-uses the update flags to detect changes and ORs the
    // incoming flags back in when it returns, so clearing them here lets both its
    // edits and ours accumulate in mUpdates. The caller's pending flags are
    // restored at the end.
    UpdateFlags prevUpdates = mUpdates;
    mUpdates = UpdateFlags::None;

    Material::renderUI(widget);

    // The alpha threshold is editable here (Material::renderUI only displays it),
    // because it is the single cutoff shared by the GBuffer clip and the shadow
    // alpha test.
    if (getAlphaMode() == AlphaMode::Mask)
    {
        float alphaThreshold = getAlphaThreshold();
        if (widget.var("Alpha clip", alphaThreshold, 0.f, 1.f, 0.01f)) setAlphaThreshold(alphaThreshold);
        widget.tooltip("Shared by the GBuffer clip and MaterialSystem::alphaTest (ShadowPass / VBufferRT),\n"
                       "so the shadow silhouette always matches the geometry silhouette.");
    }

    // Groups mirror the //#group annotations in foliage_bush_vertex_color.hlsl.
    if (auto group = widget.group("Albedo", true))
    {
        float4 v = getBaseColorMultiplier();
        if (group.var("Multiplier", v, 0.f, 10.f, 0.01f)) setBaseColorMultiplier(v);

        float f = getBaseColorBrightness();
        if (group.var("Brightness", f, 0.f, 10.f, 0.01f)) setBaseColorBrightness(f);

        f = getBaseColorContrast();
        if (group.var("Contrast", f, 0.01f, 10.f, 0.01f)) setBaseColorContrast(f);

        f = getBaseColorSaturation();
        if (group.var("Saturation", f, 0.f, 10.f, 0.01f)) setBaseColorSaturation(f);
    }

    if (auto group = widget.group("Normal", true))
    {
        float f = getNormalStrength();
        if (group.var("Strength", f, 0.f, 10.f, 0.01f)) setNormalStrength(f);
    }

    if (auto group = widget.group("OrsMask", true))
    {
        group.tooltip(
            "Roughness and dielectric specular come from the A and B channels of the packed normal map.\n"
            "Occlusion is not a material parameter: the packed map has no free channel and Falcor's vertex\n"
            "layout has no color channel, so occlusion comes from SSAOPass/GTAOPass instead.");

        float f = getRoughnessScale();
        if (group.var("Roughness scale", f, 0.f, 10.f, 0.01f)) setRoughnessScale(f);

        f = getRoughnessContrast();
        if (group.var("Roughness contrast", f, 0.01f, 10.f, 0.01f)) setRoughnessContrast(f);

        f = getSpecularValue();
        if (group.var("Specular", f, 0.f, 1.f, 0.01f)) setSpecularValue(f);
    }

    if (auto group = widget.group("SubSurface", true))
    {
        float4 v = getSubsurfaceColorMultiplier();
        if (group.var("Color multiplier", v, 0.f, 10.f, 0.01f)) setSubsurfaceColorMultiplier(v);

        float f = getSubsurfaceColorBrightness();
        if (group.var("Color brightness", f, 0.f, 10.f, 0.01f)) setSubsurfaceColorBrightness(f);

        f = getSubsurfaceColorContrast();
        if (group.var("Color contrast", f, 0.01f, 10.f, 0.01f)) setSubsurfaceColorContrast(f);

        f = getSubsurfaceColorSaturation();
        if (group.var("Color saturation", f, 0.f, 10.f, 0.01f)) setSubsurfaceColorSaturation(f);

        f = getOpacityValue();
        if (group.var("Opacity", f, 0.f, 1.f, 0.01f)) setOpacityValue(f);
    }

    if (auto group = widget.group("ColorVariability"))
    {
        group.tooltip(
            "World-space color variation. The mask texture is a scene-wide detail map that a render\n"
            "pass would bind (it is not a material texture slot), and no pass binds it today -- so\n"
            "enabling this samples an unbound texture. Rasterization only; the RT path skips it.");

        bool enabled = isColorVariabilityEnabled();
        if (group.checkbox("Enabled", enabled)) setColorVariabilityEnabled(enabled);

        float4 v = getColorVariabilityTint();
        if (group.var("Tint", v, 0.f, 10.f, 0.01f)) setColorVariabilityTint(v);

        v = getColorVariabilityTilling();
        if (group.var("Tilling (xy=scale, zw=offset)", v, -1000.f, 1000.f, 0.01f)) setColorVariabilityTilling(v);

        float f = getColorVariabilityContrast();
        if (group.var("Contrast", f, 0.01f, 10.f, 0.01f)) setColorVariabilityContrast(f);
    }

    bool changed = (mUpdates != UpdateFlags::None);
    markUpdates(prevUpdates | mUpdates);

    return changed;
}

// ---- Setters ----

void BushMaterial::setScalar(float16_t& dst, float value, float lo, float hi)
{
    const float16_t v = (float16_t)std::clamp(value, lo, hi);
    if (dst != v)
    {
        dst = v;
        markUpdates(UpdateFlags::DataChanged);
    }
}

void BushMaterial::setColor(float16_t4& dst, const float4& value, float lo, float hi)
{
    const float16_t4 v = float16_t4(clamp(value, float4(lo), float4(hi)));
    if (any(dst != v))
    {
        dst = v;
        markUpdates(UpdateFlags::DataChanged);
    }
}

void BushMaterial::setBaseColorMultiplier(const float4& v) { setColor(mData.baseColorMultiplier, v, 0.f, 10.f); }
void BushMaterial::setBaseColorBrightness(float v) { setScalar(mData.baseColorBrightness, v, 0.f, 10.f); }
void BushMaterial::setBaseColorContrast(float v) { setScalar(mData.baseColorContrast, v, 0.01f, 10.f); }
void BushMaterial::setBaseColorSaturation(float v) { setScalar(mData.baseColorSaturation, v, 0.f, 10.f); }

void BushMaterial::setNormalStrength(float v) { setScalar(mData.normalStrength, v, 0.f, 10.f); }

void BushMaterial::setRoughnessScale(float v) { setScalar(mData.roughnessScale, v, 0.f, 10.f); }
void BushMaterial::setRoughnessContrast(float v) { setScalar(mData.roughnessContrast, v, 0.01f, 10.f); }
void BushMaterial::setSpecularValue(float v) { setScalar(mData.specularValue, v, 0.f, 1.f); }

void BushMaterial::setSubsurfaceColorMultiplier(const float4& v) { setColor(mData.subsurfaceColorMultiplier, v, 0.f, 10.f); }
void BushMaterial::setSubsurfaceColorBrightness(float v) { setScalar(mData.subsurfaceColorBrightness, v, 0.f, 10.f); }
void BushMaterial::setSubsurfaceColorContrast(float v) { setScalar(mData.subsurfaceColorContrast, v, 0.01f, 10.f); }
void BushMaterial::setSubsurfaceColorSaturation(float v) { setScalar(mData.subsurfaceColorSaturation, v, 0.f, 10.f); }
void BushMaterial::setOpacityValue(float v) { setScalar(mData.opacityValue, v, 0.f, 1.f); }

void BushMaterial::setColorVariabilityTint(const float4& v) { setColor(mData.colorVariabilityTint, v, 0.f, 10.f); }
void BushMaterial::setColorVariabilityContrast(float v) { setScalar(mData.colorVariabilityContrast, v, 0.01f, 10.f); }

void BushMaterial::setColorVariabilityEnabled(bool enabled)
{
    if (mData.isColorVariabilityEnabled() != enabled)
    {
        mData.setColorVariabilityEnabled(enabled);
        markUpdates(UpdateFlags::DataChanged);
    }
}

void BushMaterial::setColorVariabilityTilling(const float4& v)
{
    // xy is a divisor in the shader, so zero would produce a division by zero.
    float4 clamped = v;
    clamped.x = (std::abs(clamped.x) < 1e-6f) ? 1.f : clamped.x;
    clamped.y = (std::abs(clamped.y) < 1e-6f) ? 1.f : clamped.y;

    if (any(mData.colorVariabilityTilling != clamped))
    {
        mData.colorVariabilityTilling = clamped;
        markUpdates(UpdateFlags::DataChanged);
    }
}
