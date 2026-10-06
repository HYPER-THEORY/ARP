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
#include "Utils/HostDeviceShared.slangh"

#ifdef HOST_CODE
#include "Scene/Material/TextureHandle.slang"
#include "Scene/Material/MaterialData.slang"
#else
__exported import Scene.Material.TextureHandle;
__exported import Scene.Material.MaterialData;
#endif

BEGIN_NAMESPACE_FALCOR

/** Host/device structure holding the parameters of a bush (foliage) material.

    This replaces the earlier reuse of `BasicMaterialData`, which forced the
    material to smuggle its parameters through unrelated fields (the subsurface
    color map went into `texTransmission`, the subsurface weight into
    `diffuseTransmission`, occlusion/roughness into `specular.rg`). Every field
    here has exactly one meaning and exactly one consumer.

    Three textures, matching the chaos asset naming convention (see the
    `*_d_a` / `*_n_s_r` / `*_surface` suffixes):

        texBaseColor        (slot BaseColor,    sRGB)
            RGB - Base color (albedo)
            A   - Opacity mask
        texNormalSpecRough  (slot Normal,       linear)
            RG  - Tangent-space normal XY; Z is reconstructed as
                  sqrt(1 - x^2 - y^2). Note this is *not* a BC5 two-channel
                  texture: it is a four-channel map whose B and A channels
                  carry unrelated data, so the Z channel is unavailable and
                  must be reconstructed.
            B   - Dielectric specular
            A   - Roughness
        texSubsurfaceColor  (slot Transmission, sRGB)
            RGB - Subsurface color
            A   - Opacity (UE material Opacity input: subsurface thickness)

    There are deliberately NO uniform fallbacks for the texture values. If a
    texture is not bound, `sampleTexture` returns the float4(0) default and the
    surface reads black / zero roughness, which makes an unbound texture
    immediately obvious instead of silently substituting a plausible-looking
    constant.

    Layout: vector types are kept at 8B/16B aligned offsets, same convention as
    `BasicMaterialData`. Scalars are float16 because they are all either [0,1] weights or
    small multipliers/exponents where half precision is well below the
    perceptual threshold.
*/
struct BushMaterialData
{
    // offset 0 (16B) - xy = world-space tiling divisor, zw = offset. Kept at
    // full precision because the offset is in world units.
    float4      colorVariabilityTilling     = float4(1.f, 1.f, 0.f, 0.f);

    // offset 16 (8B each) - .a is unused; float16_t4 is used over float16_t3
    // to keep the following members 8B aligned.
    float16_t4  baseColorMultiplier         = float16_t4(1.h);
    float16_t4  subsurfaceColorMultiplier   = float16_t4(1.h);
    float16_t4  colorVariabilityTint        = float16_t4(1.h);

    // offset 40 (2B each) - UE ColorCorrect parameters, see primaryColorCorrection().
    float16_t   baseColorBrightness         = 1.h;      ///< Gain. Range [0,inf).
    float16_t   baseColorContrast           = 1.h;      ///< Exponent about the 0.18 pivot. Range (0,inf).
    float16_t   baseColorSaturation         = 1.h;      ///< 0 = luma only, 1 = unchanged. Range [0,inf).
    float16_t   subsurfaceColorBrightness   = 1.h;
    float16_t   subsurfaceColorContrast     = 1.h;
    float16_t   subsurfaceColorSaturation   = 1.h;

    float16_t   normalStrength              = 1.h;      ///< Scales the decoded normal XY. Range [0,inf).

    float16_t   roughnessContrast           = 1.h;      ///< Exponent on roughness. Range (0,inf).
    float16_t   roughnessScale              = 1.h;      ///< Multiplier on the roughness channel. Range [0,inf).
    float16_t   specularValue               = 1.h;      ///< Multiplier on the dielectric specular channel. Range [0,1].
    float16_t   opacityValue                = 1.h;      ///< Multiplier on the opacity channel. Range [0,1].
    float16_t   colorVariabilityContrast    = 1.h;      ///< Exponent on the variability mask. Range (0,inf).

    // offset 64 (4B)
    uint        flags                       = 0;        ///< See accessors below.

    // offset 68 (4B each)
    TextureHandle texBaseColor;
    TextureHandle texNormalSpecRough;
    TextureHandle texSubsurfaceColor;

    static constexpr uint kColorVariabilityEnabledBits = 1;
    static constexpr uint kColorVariabilityEnabledOffset = 0;

    /** Enable/disable the world-space color variability modulation.
     */
    SETTER_DECL void setColorVariabilityEnabled(bool enabled)
    {
        flags = PACK_BITS(kColorVariabilityEnabledBits, kColorVariabilityEnabledOffset, flags, enabled ? 1 : 0);
    }

    bool isColorVariabilityEnabled() CONST_FUNCTION
    {
        return bool(EXTRACT_BITS(kColorVariabilityEnabledBits, kColorVariabilityEnabledOffset, flags));
    }
};

END_NAMESPACE_FALCOR
