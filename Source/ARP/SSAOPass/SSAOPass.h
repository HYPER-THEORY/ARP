/// Copyright (C) 2026-2027 ARP (Auto Render Pipeline). All Rights Reserved.

#pragma once

#include "Falcor.h"
#include "RenderGraph/RenderPass.h"

using namespace Falcor;

class SSAOPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(SSAOPass, "SSAOPass", "Screen Space Ambient Occulsion");

    static ref<SSAOPass> create(ref<Device> pDevice, const Properties& props);

    SSAOPass(ref<Device> pDevice, const Properties& props);

    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;

    virtual void setProperties(const Properties& props) override;
    virtual Properties getProperties() const override;

    virtual void renderUI(RenderContext* pRenderContext, Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;

protected:
    void compilePasses();

    void initRandomNormalTexture();

    uint32_t mFrameIndex = 0;
    uint2 mFrameDim = {0, 0};

    float mRadius = 200.0f;
    float mBias = 3.0f;
    float mPower = 2.0f;
    float mIntensity = 0.5f;
    float mFadeDistance = 8000.f;
    float mFadeRadius = 5000.f;
    float mMipLevelFactor = 0.5f;

    bool mRadiusInWorldSpace = false;
    bool mEnableSmooth = false;
    bool mTemporal = true;
    bool mDifferentiable = false;

    ref<Scene> mpScene;

    ref<ComputePass> mpSSAOPass;
    ref<ComputePass> mpSSAOSmoothPass;

    ref<Texture> mpAOMap;
    ref<Texture> mpRandomNormalMap;

    ref<Sampler> mpZFurthestSampler;
    ref<Sampler> mpRandomNormalSampler;
};
