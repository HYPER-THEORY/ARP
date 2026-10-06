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
#include "Falcor.h"
#include "IndirectApproxPass.h"
#include "../Materials/Bush/BushMaterial.h"

namespace
{
const std::string kProgramFile = "Arctic/IndirectApproxPass/IndirectApproxPass.cs.slang";

const char kXYFieldTexturePath[] = "wheat/textures/XYField.jpg";

const ChannelList kInputs = {
    // clang-format off
    { "sceneDepth", "gSceneDepth", "GBuffer scene depth (D32Float)",                    true, ResourceFormat::Unknown },
    { "gBufferA",   "gGBufferA",   "World normal (encoded) + per-object",               true, ResourceFormat::Unknown },
    { "gBufferB",   "gGBufferB",   "Metallic/Specular/Roughness/ShadingModelID",        true, ResourceFormat::Unknown },
    { "gBufferC",   "gGBufferC",   "Base colour + AO",                                  true, ResourceFormat::Unknown },
    { "gBufferD",   "gGBufferD",   "Custom data (subsurface colour for foliage)",       true, ResourceFormat::Unknown },
    // clang-format on
};
const ChannelList kOutputs = {
    // clang-format off
    { "indirect",   "gIndirect",   ">= 2-bounce indirect radiance (HDR, additive)",     false, ResourceFormat::RGBA32Float },
    { "fieldDebug", "gFieldDebug", "r=lateral density g=LAI above b=column LAI",        false, ResourceFormat::RGBA32Float },
    // clang-format on
};

const char kLai[] = "lai";
const char kSlope[] = "slope";
const char kTileSize[] = "tileSize";
const char kDensityContrast[] = "densityContrast";
const char kDensityMax[] = "densityMax";
const char kTypeLidf[] = "typelidf";
const char kLidfA[] = "lidfa";
const char kLidfB[] = "lidfb";
const char kRLeaf[] = "rLeaf";
const char kTLeaf[] = "tLeaf";
const char kSkyRadiance[] = "skyRadiance";
const char kSoilAlbedo[] = "soilAlbedo";
const char kDebugView[] = "debugView";
} // namespace

IndirectApproxPass::IndirectApproxPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kLai) mLai = value;
        else if (key == kSlope) mSlope = value;
        else if (key == kTileSize) mTileSize = value;
        else if (key == kDensityContrast) mDensityContrast = value;
        else if (key == kDensityMax) mDensityMax = value;
        else if (key == kTypeLidf) mTypeLidf = value;
        else if (key == kLidfA) mLidfA = value;
        else if (key == kLidfB) mLidfB = value;
        else if (key == kRLeaf) mRLeaf = value;
        else if (key == kTLeaf) mTLeaf = value;
        else if (key == kSkyRadiance) mSkyRadiance = value;
        else if (key == kSoilAlbedo) mSoilAlbedo = value;
        else if (key == kDebugView) mDebugView = value;
    }

    // Repeat addressing realises the "infinite in XY" assumption.
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
    samplerDesc.setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Wrap);
    mpXYSampler = mpDevice->createSampler(samplerDesc);

    loadXYField();

    // Build the precompute up front so renderUI() and the port-check export
    // never read an uninitialised table. Needs no scene.
    updateLidf();
}

Properties IndirectApproxPass::getProperties() const
{
    Properties p;
    p[kLai] = mLai;
    p[kSlope] = mSlope;
    p[kTileSize] = mTileSize;
    p[kDensityContrast] = mDensityContrast;
    p[kDensityMax] = mDensityMax;
    p[kTypeLidf] = mTypeLidf;
    p[kLidfA] = mLidfA;
    p[kLidfB] = mLidfB;
    p[kRLeaf] = mRLeaf;
    p[kTLeaf] = mTLeaf;
    p[kSkyRadiance] = mSkyRadiance;
    p[kSoilAlbedo] = mSoilAlbedo;
    p[kDebugView] = mDebugView;
    return p;
}

RenderPassReflection IndirectApproxPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = compileData.defaultTexDims;
    addRenderPassInputs(reflector, kInputs, ResourceBindFlags::ShaderResource, uint2(0));
    addRenderPassOutputs(reflector, kOutputs, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void IndirectApproxPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpPass = nullptr;
    mDirty = true;
    updateFoliageBounds();
}

void IndirectApproxPass::rebuildPass()
{
    if (!mpScene) return;
    if (!mpPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kProgramFile).csEntry("csMain");
        desc.addTypeConformances(mpScene->getTypeConformances());
        mpPass = ComputePass::create(mpDevice, desc, mpScene->getSceneDefines());
    }
    mDirty = false;
}

void IndirectApproxPass::loadXYField()
{
    auto resolved = AssetResolver::getDefaultResolver().resolvePath(kXYFieldTexturePath);
    if (resolved.empty())
    {
        logWarning("IndirectApproxPass: could not resolve '{}'; the field will be laterally uniform.", kXYFieldTexturePath);
        return;
    }

    // Loaded as linear data, not colour: the mapping from photographic grey level
    // to vegetation density is arbitrary anyway, and treating it as data keeps the
    // histogram the artist filtered.
    mpXYField = Texture::createFromFile(mpDevice, resolved, false /* generateMips */, false /* loadAsSrgb */);
    if (!mpXYField) logWarning("IndirectApproxPass: failed to load '{}'.", resolved.string());
}

void IndirectApproxPass::updateFoliageBounds()
{
    if (!mAutoBounds || !mpScene) return;

    // Find the foliage material, then union the world bounds of every instance
    // using it. getSceneBounds() would include non-foliage geometry and is useless here.
    MaterialID foliageMatID{};
    bool foundMaterial = false;
    for (uint32_t i = 0; i < mpScene->getMaterialCount(); ++i)
    {
        MaterialID id{i};
        if (dynamic_ref_cast<BushMaterial>(mpScene->getMaterial(id)))
        {
            foliageMatID = id;
            foundMaterial = true;
            break;
        }
    }
    if (!foundMaterial)
    {
        logWarning("IndirectApproxPass: no BushMaterial in the scene; keeping the manual Z range.");
        return;
    }

    const auto* pAnim = mpScene->getAnimationController();
    if (!pAnim) return;
    const auto& globalMatrices = pAnim->getGlobalMatrices();

    AABB bounds;
    bounds.invalidate();
    for (uint32_t i = 0; i < mpScene->getGeometryInstanceCount(); ++i)
    {
        const auto& inst = mpScene->getGeometryInstance(i);
        if (inst.getType() != GeometryType::TriangleMesh) continue;
        if (inst.materialID != foliageMatID.getSlang()) continue;
        if (inst.globalMatrixID >= globalMatrices.size()) continue;

        const AABB& objBB = mpScene->getMeshBounds(inst.geometryID);
        if (!objBB.valid()) continue;
        bounds.include(objBB.transform(globalMatrices[inst.globalMatrixID]));
    }

    if (!bounds.valid() || bounds.extent().z <= 0.f)
    {
        logWarning("IndirectApproxPass: foliage bounds are degenerate; keeping the manual Z range.");
        return;
    }

    // The lower boundary doubles as the soil plane, which sits at z = 0.
    mZMin = std::min(bounds.minPoint.z, 0.f);
    mZMax = bounds.maxPoint.z;
    logInfo("IndirectApproxPass: foliage Z range = [{:.3f}, {:.3f}] m", mZMin, mZMax);
}

// -----------------------------------------------------------------------------
// Scene-independent precompute
// -----------------------------------------------------------------------------

void IndirectApproxPass::updateLidf()
{
    using namespace ProsailLidf;

    if (mTypeLidf == 2)
        computeLidfCampbell(mLidfA, mLidf);
    else
        computeLidfVerhoef(mLidfA, mLidfB, mLidf);

    mBf = computeBf(mLidf);
    computeGLut(mLidf, mGLut);

    // The distribution is a probability mass function, so this is an exact
    // identity for any valid (lidfa, lidfb). Drifting off it means the Verhoef
    // fixed-point iteration failed to converge, which happens outside
    // |lidfa| + |lidfb| < 1.
    float sum = 0.f;
    for (float v : mLidf) sum += v;
    if (std::abs(sum - 1.f) > 1e-3f)
    {
        logWarning(
            "IndirectApproxPass: LIDF sums to {:.6f}, not 1. Check that |lidfa| + |lidfb| < 1 "
            "(lidfa={}, lidfb={}, typelidf={}).",
            sum, mLidfA, mLidfB, mTypeLidf
        );
    }

    mLidfDirty = false;
}

bool IndirectApproxPass::fetchSun(float3& travelDir, float3& intensity) const
{
    if (!mpScene) return false;
    for (const auto& l : mpScene->getLights())
    {
        if (!l || l->getType() != LightType::Directional) continue;
        auto pDir = dynamic_cast<DirectionalLight*>(l.get());
        if (!pDir) continue;
        // Falcor's dirW is the direction the light TRAVELS, so it points downward.
        travelDir = normalize(pDir->getWorldDirection());
        intensity = pDir->getIntensity();
        return true;
    }
    return false;
}

// -----------------------------------------------------------------------------
// Binding. Raw values only: every division, trig call and exponential is the
// shader's job, so that nothing about the model is computed on this side.
// -----------------------------------------------------------------------------

void IndirectApproxPass::bindFieldData(const ShaderVar& rootVar)
{
    auto cb = rootVar["FieldCB"];

    cb["gZMin"] = mZMin;
    cb["gZMax"] = mZMax;
    cb["gLai"] = mLai;
    cb["gSlope"] = mSlope;

    cb["gTileSize"] = mTileSize;
    cb["gDensityContrast"] = mpXYField ? mDensityContrast : 0.f;
    cb["gDensityMax"] = mDensityMax;

    cb["gRLeaf"] = mRLeaf;
    cb["gTLeaf"] = mTLeaf;

    float3 sunDir = float3(0.f, 0.f, -1.f);
    float3 sunIntensity = float3(0.f);
    if (!fetchSun(sunDir, sunIntensity)) sunIntensity = float3(0.f);
    cb["gSunDirW"] = sunDir;
    cb["gSunIntensity"] = sunIntensity;

    cb["gSkyRadiance"] = mSkyRadiance;
    cb["gSoilAlbedo"] = mSoilAlbedo;

    cb["gBf"] = mBf;
    // Element-wise: HLSL pads constant-buffer array elements to a 16-byte
    // stride, so a flat blob copy would land in the wrong places.
    for (int i = 0; i < ProsailLidf::kGLutSize; ++i) cb["gGLut"][i] = mGLut[i];

    rootVar["gXYField"] = mpXYField;
    rootVar["gXYSampler"] = mpXYSampler;
}

void IndirectApproxPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pSceneDepth = renderData.getTexture("sceneDepth");
    auto pIndirect = renderData.getTexture("indirect");
    auto pFieldDebug = renderData.getTexture("fieldDebug");
    if (!pSceneDepth || !pIndirect || !pFieldDebug || !mpScene) return;

    if (mDirty) rebuildPass();
    if (!mpPass) return;

    if (mLidfDirty) updateLidf();

    const uint32_t w = pIndirect->getWidth();
    const uint32_t h = pIndirect->getHeight();

    auto var = mpPass->getVars()->getRootVar();
    mpScene->bindShaderData(var["gScene"]);
    bindFieldData(var);

    var["gSceneDepth"] = pSceneDepth;
    var["gGBufferA"] = renderData.getTexture("gBufferA");
    var["gGBufferB"] = renderData.getTexture("gBufferB");
    var["gGBufferC"] = renderData.getTexture("gBufferC");
    var["gGBufferD"] = renderData.getTexture("gBufferD");
    var["gIndirect"] = pIndirect;
    var["gFieldDebug"] = pFieldDebug;

    auto cb = var["PerFrameCB"];
    cb["gFrameDim"] = uint2(w, h);
    cb["gDebugView"] = mDebugView;

    mpPass->execute(pRenderContext, uint3(w, h, 1));
}

// -----------------------------------------------------------------------------
// UI
// -----------------------------------------------------------------------------

void IndirectApproxPass::renderUI(Gui::Widgets& widget)
{
    if (auto g = widget.group("Field geometry", true))
    {
        bool bounds = g.checkbox("Auto Z bounds", mAutoBounds);
        if (bounds) updateFoliageBounds();
        if (!mAutoBounds)
        {
            g.var("Z min", mZMin, -1000.f, 1000.f, 0.01f);
            g.var("Z max", mZMax, -1000.f, 1000.f, 0.01f);
        }
        else
        {
            g.text(fmt::format("Z range: [{:.3f}, {:.3f}] m", mZMin, mZMax));
        }
        g.var("LAI", mLai, 0.f, 40.f, 0.05f);
        g.tooltip("PROSAIL's lai, at lateral density 1. Wheat is typically 4-6.");
        g.var("Profile slope", mSlope, -1.f, 1.f, 0.01f);
        g.tooltip(
            "Vertical leaf area profile tilt. > 0 puts more leaf area near the top (ears on top).\n"
            "Orthogonal to LAI: the profile integrates to LAI for any slope. Not a PROSAIL\n"
            "parameter, but not a departure either: SAIL's equations are in cumulative-LAI\n"
            "coordinates, so a depth-varying leaf area density is handled exactly."
        );
    }

    if (auto g = widget.group("Lateral density", true))
    {
        g.text("Rendering extension, not part of PROSAIL. Each pixel is solved as");
        g.text("its own uniform column (independent-column approximation).");
        g.var("Tile size (m)", mTileSize, 0.5f, 200.f, 0.1f);
        g.var("Density contrast", mDensityContrast, 0.f, 10.f, 0.01f);
        g.tooltip("0 = laterally uniform, which is the configuration PROSAIL itself describes.");
        g.var("Density max", mDensityMax, 1.f, 10.f, 0.01f);
    }

    if (auto g = widget.group("Leaf angle distribution", true))
    {
        uint32_t type = mTypeLidf;
        Gui::DropdownList typeList = {{1, "Verhoef bimodal (lidfa, lidfb)"}, {2, "Campbell (mean leaf angle)"}};
        if (g.dropdown("LIDF type", typeList, type))
        {
            mTypeLidf = type;
            mLidfDirty = true;
        }
        if (mTypeLidf == 2)
        {
            if (g.var("Mean leaf angle (deg)", mLidfA, 1.f, 89.f, 0.1f)) mLidfDirty = true;
            g.tooltip("57.3 is spherical. Lower is more horizontal (planophile), higher more vertical.");
        }
        else
        {
            if (g.var("lidfa", mLidfA, -1.f, 1.f, 0.005f)) mLidfDirty = true;
            if (g.var("lidfb", mLidfB, -1.f, 1.f, 0.005f)) mLidfDirty = true;
            g.tooltip(
                "PROSAIL Verhoef bimodal, requires |a| + |b| < 1.\n"
                "spherical (-0.35, -0.15), planophile (1, 0), erectophile (-1, 0),\n"
                "plagiophile (0, -1), extremophile (0, 1), uniform (0, 0)."
            );
        }
        g.text(fmt::format("bf = {:.4f}   G(0 deg) = {:.4f}   G(60 deg) = {:.4f}", mBf, mGLut[0], mGLut[60]));
        g.tooltip(
            "bf is PROSAIL's cos^2 moment of the LIDF; G is the Ross function, tabulated\n"
            "against solar zenith. These three numbers are the only precompute this pass\n"
            "does on the CPU. Everything else is evaluated in the shader."
        );
    }

    if (auto g = widget.group("Radiometry", true))
    {
        g.var("Leaf reflectance (rho)", mRLeaf, 0.f, 1.f, 0.005f);
        g.var("Leaf transmittance (tau)", mTLeaf, 0.f, 1.f, 0.005f);
        g.tooltip(
            "PROSAIL's rho and tau. Bulk volume properties, used for the field solve. The\n"
            "per-pixel gather uses the GBuffer base colour and subsurface colour instead, so\n"
            "texture variation shows through. Note the deep field is much more saturated than\n"
            "a single leaf: each bounce loses more in the low channels."
        );
        g.var("Sky radiance", mSkyRadiance, 0.f, 20.f, 0.01f);
        g.var("Soil albedo (rsoil0)", mSoilAlbedo, 0.f, 1.f, 0.005f);
        g.tooltip("PROSAIL's rsoil0. Deliberately not a bright ground, which would swamp the\n"
                  "inter-leaf term.");
    }

    if (auto g = widget.group("Debug", false))
    {
        Gui::DropdownList views = {
            {0, "Off"},
            {1, "E down (collided)"},
            {2, "E up"},
            {3, "Lateral density"},
            {4, "LAI above"},
            {5, "Column LAI"},
            {6, "Collided share of down flux"},
        };
        g.dropdown("Debug view", views, mDebugView);
        g.text("Validation lives outside the app, see the header comments in:");
        g.text("  tools/check_prosail_port.py    precompute vs ref/prosail");
        g.text("  tools/check_sail_interior.py   interior solve vs a direct ODE solve");
        g.text("  tools/verify_indirect_approx.py  this shader vs that reference");
    }
}
