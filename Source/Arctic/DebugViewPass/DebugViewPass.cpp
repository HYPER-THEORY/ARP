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
 # PURPOSE ARE DISCLAIMED.  IN EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Falcor.h"
#include "DebugViewPass.h"

namespace
{
const std::string kProgramFile = "Arctic/DebugViewPass/DebugViewPass.3d.slang";

// All inputs are auto-sized (uint2(0)) so they inherit the source pass's
// dimensions. ResourceFormat::Unknown lets the graph pick up the source
// format from the upstream edge.
const ChannelList kInputs = {
    // clang-format off
    { "final",        "gFinal",        "Post-tonemap LDR",                      true, ResourceFormat::Unknown },
    { "gBufferA",     "gGBufferA",     "World normal / per-object",             true, ResourceFormat::Unknown },
    { "gBufferB",     "gGBufferB",     "Metallic/Specular/Roughness/ShadingID", true, ResourceFormat::Unknown },
    { "gBufferC",     "gGBufferC",     "Base color (sRGB) / GBufferAO",         true, ResourceFormat::Unknown },
    { "gBufferD",     "gGBufferD",     "Custom data",                           true, ResourceFormat::Unknown },
    { "gBufferE",     "gGBufferE",     "Precomputed shadow factors",            true, ResourceFormat::Unknown },
    { "depth",        "gDepth",        "GBuffer scene depth",                   true, ResourceFormat::Unknown },
    { "emissive",     "gEmissive",     "Emissive",                              true, ResourceFormat::Unknown },
    { "shadowMask",   "gShadowMask",   "Shadow (surface / SSS transmission)",   true, ResourceFormat::Unknown },
    { "aoMap",        "gAOMap",        "Screen-space AO",                       true, ResourceFormat::Unknown },
    { "diColor",      "gDIColor",      "Direct lighting",                       true, ResourceFormat::Unknown },
    { "ambient",      "gAmbient",      "Ambient lighting",                      true, ResourceFormat::Unknown },
    { "indirect",     "gIndirect",     "Indirect lighting",                     true, ResourceFormat::Unknown },
    { "composite",    "gComposite",    "Composited lighting",                   true, ResourceFormat::Unknown },
    { "fieldDebug",   "gFieldDebug",   "FieldDebug (XY density/LAD/LAI/G)",     true, ResourceFormat::Unknown },
    // clang-format on
};
const ChannelList kOutputs = {
    { "output", "gOutput", "Debug view output (RGBA16Float)", false, ResourceFormat::RGBA16Float },
};

// Buffer selection indices — must match DebugViewPass.3d.slang's switch.
// Channel selection: 0=RGB, 1=R, 2=G, 3=B, 4=A.
} // namespace

DebugViewPass::DebugViewPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    ProgramDesc desc;
    desc.addShaderLibrary(kProgramFile).csEntry("csMain");
    mpPass = ComputePass::create(mpDevice, desc);
}

RenderPassReflection DebugViewPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputs, ResourceBindFlags::ShaderResource, uint2(0));
    const uint2 sz = compileData.defaultTexDims;
    addRenderPassOutputs(reflector, kOutputs, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void DebugViewPass::renderUI(Gui::Widgets& widget)
{
    Gui::DropdownList bufferList;
    bufferList.push_back({0, "Final result"});
    bufferList.push_back({1, "GBufferA (normal/per-object)"});
    bufferList.push_back({2, "GBufferB (metallic/spec/rough/ID)"});
    bufferList.push_back({3, "GBufferC (baseColor/AO)"});
    bufferList.push_back({4, "GBufferD (custom data)"});
    bufferList.push_back({5, "GBufferE (shadow factors)"});
    bufferList.push_back({6, "Depth"});
    bufferList.push_back({7, "Emissive"});
    bufferList.push_back({8, "ShadowMask (surface/SSS trans)"});
    bufferList.push_back({9, "AO Map"});
    bufferList.push_back({10, "Direct Lighting"});
    bufferList.push_back({11, "Ambient Lighting"});
    bufferList.push_back({12, "Indirect Lighting"});
    bufferList.push_back({13, "Composited Lighting"});
    bufferList.push_back({14, "Field Debug (XY/LAD/LAI/G)"});
    bufferList.push_back({15, "Shading Model"});
    widget.dropdown("Buffer", bufferList, mBufferSelection);

    Gui::DropdownList channelList;
    channelList.push_back({0, "RGB"});
    channelList.push_back({1, "R"});
    channelList.push_back({2, "G"});
    channelList.push_back({3, "B"});
    channelList.push_back({4, "A"});
    widget.dropdown("Channel", channelList, mChannelSelection);
}

void DebugViewPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pOutput = renderData.getTexture("output");
    if (!pOutput || !mpPass) return;

    auto var = mpPass->getVars()->getRootVar();
    var["gFinal"]        = renderData.getTexture("final");
    var["gGBufferA"]     = renderData.getTexture("gBufferA");
    var["gGBufferB"]     = renderData.getTexture("gBufferB");
    var["gGBufferC"]     = renderData.getTexture("gBufferC");
    var["gGBufferD"]     = renderData.getTexture("gBufferD");
    var["gGBufferE"]     = renderData.getTexture("gBufferE");
    var["gDepth"]        = renderData.getTexture("depth");
    var["gEmissive"]     = renderData.getTexture("emissive");
    var["gShadowMask"]   = renderData.getTexture("shadowMask");
    var["gAOMap"]        = renderData.getTexture("aoMap");
    var["gDIColor"]      = renderData.getTexture("diColor");
    var["gAmbient"]      = renderData.getTexture("ambient");
    var["gIndirect"]     = renderData.getTexture("indirect");
    var["gComposite"]    = renderData.getTexture("composite");
    var["gFieldDebug"]   = renderData.getTexture("fieldDebug");
    var["gOutput"]       = pOutput;

    var["PerFrameCB"]["gSelection"] = mBufferSelection;
    var["PerFrameCB"]["gChannel"]   = mChannelSelection;
    var["PerFrameCB"]["gFrameDim"]  = uint2(pOutput->getWidth(), pOutput->getHeight());

    mpPass->execute(pRenderContext, uint3(pOutput->getWidth(), pOutput->getHeight(), 1));
}
