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
#include "MotionVectorPass.h"

namespace
{
const std::string kProgramFile = "Arctic/MotionVectorPass/MotionVectorPass.cs.slang";

const ChannelList kInputs = {
    { "sceneDepth", "gSceneDepth", "GBuffer scene depth (D32Float)", true, ResourceFormat::Unknown },
};
const ChannelList kOutputs = {
    { "motionVecs", "gMotionVecs", "Per-pixel motion vector (RG16Float)", false, ResourceFormat::RG16Float },
};
} // namespace

MotionVectorPass::MotionVectorPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {}

RenderPassReflection MotionVectorPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = compileData.defaultTexDims;
    addRenderPassInputs(reflector, kInputs, ResourceBindFlags::ShaderResource, uint2(0));
    addRenderPassOutputs(reflector, kOutputs, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void MotionVectorPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpPass.reset();
    mDirty = true;
}

void MotionVectorPass::rebuildPass()
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

void MotionVectorPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pSceneDepth = renderData.getTexture("sceneDepth");
    auto pMotionVecs = renderData.getTexture("motionVecs");
    if (!pSceneDepth || !pMotionVecs || !mpScene) return;

    if (mDirty) rebuildPass();
    if (!mpPass) return;

    const uint32_t w = pMotionVecs->getWidth();
    const uint32_t h = pMotionVecs->getHeight();

    auto var = mpPass->getVars()->getRootVar();
    mpScene->bindShaderData(var["gScene"]);
    var["gSceneDepth"] = pSceneDepth;
    var["gMotionVecs"] = pMotionVecs;
    var["PerFrameCB"]["gFrameDim"] = uint2(w, h);

    mpPass->execute(pRenderContext, uint3(w, h, 1));
}
