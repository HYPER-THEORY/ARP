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
#pragma once
#include "Falcor.h"
#include "Core/AssetResolver.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"

#include "ProsailLidf.h"

using namespace Falcor;

/** Canopy multiple-scattering indirect lighting, via PROSAIL's 4SAIL.

    The transport model is a direct port of ref/prosail/prosail/FourSAIL.py,
    evaluated per pixel in FourSail.slang. See that file for the model, the
    interior-flux composition and the bounce accounting.

    SPLIT OF WORK. This class performs no arithmetic on the model. It owns
    parameters, resolves the sun out of the scene, and uploads raw values; the
    shader does every division, trigonometric call and exponential. The only
    computation on this side is ProsailLidf's scene-independent precompute --
    the 18-bin leaf inclination distribution, its bf moment and a G(tts) table
    -- rebuilt only when lidfa, lidfb or typelidf change, never per frame.

    That split is why the G table exists at all: ks = G(tts)/cos(tts) depends on
    the sun and so cannot be precomputed, but G itself depends only on the LIDF.
    Tabulating G over all solar zeniths keeps the sun-dependent part to one
    interpolation plus one divide in the shader.

    PARAMETERS. The model inputs are PROSAIL's own: lai, lidfa, lidfb, typelidf,
    rho (leaf reflectance), tau (leaf transmittance) and rsoil0 (soil albedo).
    PROSAIL's hotspot, tto and psi are absent by construction -- they drive only
    view-direction radiance, which this pass does not compute. Sky radiance and
    the canopy z range are not PROSAIL parameters but are unavoidable: the former
    is SAIL's upper boundary condition, the latter maps world space onto optical
    depth. The vertical slope and the lateral density map are rendering
    extensions on top; slope is exact within SAIL's cumulative-LAI coordinates,
    the density map is an independent-column approximation.

    Bounce accounting, unchanged from before: this pass emits only paths with two
    or more bounces, since single-bounce sun is DIPass and single-bounce sky is
    AmbientPass. The output DOES include base colour and subsurface colour,
    because the field solve stops at the incident fluxes and the leaf's own
    scattering is applied in the shader. Still not multiplied by AO, which models
    occlusion of the uncollided sky and would darken the very term meant to lift
    the shadow side.

    Outputs:
      - indirect   (RGBA32Float): >= 2-bounce radiance, additive
      - fieldDebug (RGBA32Float): r = lateral density, g = LAI above, b = column LAI
*/
class IndirectApproxPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(
        IndirectApproxPass,
        "IndirectApproxPass",
        "Canopy multiple-scattering indirect lighting (PROSAIL 4SAIL, evaluated in the shader)."
    );

    static ref<IndirectApproxPass> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<IndirectApproxPass>(pDevice, props);
    }

    IndirectApproxPass(ref<Device> pDevice, const Properties& props);

    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    Properties getProperties() const override;

private:
    void rebuildPass();
    void loadXYField();
    void updateFoliageBounds();

    /// Rebuild the LIDF, bf and the G table. Scene independent; depends only on
    /// lidfa, lidfb and typelidf.
    void updateLidf();

    /// Pull sun travel direction and intensity from the scene's first
    /// DirectionalLight. Returns false if there is none.
    bool fetchSun(float3& travelDir, float3& intensity) const;

    void bindFieldData(const ShaderVar& rootVar);

    ref<Scene> mpScene;
    ref<ComputePass> mpPass;

    ref<Texture> mpXYField;
    ref<Sampler> mpXYSampler;

    // ---- Field geometry ----
    bool mAutoBounds = true; ///< Derive zMin/zMax from the foliage geometry bbox.
    float mZMin = 0.f;
    float mZMax = 1.f;
    float mLai = 5.f;    ///< PROSAIL's lai, at lateral density 1. Wheat is 4-6.
    float mSlope = 0.f; ///< Vertical profile tilt; > 0 = top heavy (ears on top).

    // ---- Lateral density modulation (rendering extension, not PROSAIL) ----
    float mTileSize = 30.f; ///< Metres per XYField tile.
    float mDensityContrast = 1.f;
    float mDensityMax = 2.f;

    // ---- Leaf inclination distribution (PROSAIL parameterisation) ----
    uint32_t mTypeLidf = 1; ///< 1 = Verhoef bimodal, 2 = Campbell mean leaf angle.
    float mLidfA = -1.0f;  ///< Verhoef a, or the mean leaf angle in degrees if type 2.
    float mLidfB = -0.0f;  ///< Verhoef b. Unused for Campbell. Needs |a| + |b| < 1.

    // ---- Radiometry (dry/golden wheat straw literature values) ----
    float3 mRLeaf = float3(0.45f, 0.35f, 0.13f); ///< PROSAIL's rho.
    float3 mTLeaf = float3(0.18f, 0.13f, 0.05f); ///< PROSAIL's tau.
    float3 mSkyRadiance = float3(1.0f);          ///< SAIL's upper boundary condition.
    float3 mSoilAlbedo = float3(0.20f, 0.16f, 0.11f); ///< PROSAIL's rsoil0.

    // ---- Debug ----
    uint32_t mDebugView = 0;

    // ---- Cached scene-independent precompute ----
    float mLidf[ProsailLidf::kNumLidfBins] = {};
    float mGLut[ProsailLidf::kGLutSize] = {};
    float mBf = 0.f;
    bool mLidfDirty = true; ///< LIDF parameters changed.

    bool mDirty = true; ///< Shader program needs rebuilding.
};
