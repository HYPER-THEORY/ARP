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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"

using namespace Falcor;

/** Debug buffer viewer pass.

    Takes all deferred-graph intermediate buffers as inputs and forwards the
    selected one (optionally restricted to a single R/G/B/A channel) to its
    output. Switching selection is a uniform update, so no graph rebuild is
    needed. Only used in the deferred graph; the path-tracer graph bypasses it.
*/
class DebugViewPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(DebugViewPass, "DebugViewPass", "Debug buffer viewer pass.");

    static ref<DebugViewPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<DebugViewPass>(pDevice, props); }

    DebugViewPass(ref<Device> pDevice, const Properties& props);

    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;

    // Programmatic selection (used by --capture CLI). Indices must match the
    // shader.
    // Channel selection: 0=RGB, 1=R, 2=G, 3=B, 4=A.
    void setSelection(uint32_t buffer, uint32_t channel)
    {
        mBufferSelection = buffer;
        mChannelSelection = channel;
    }

private:
    ref<ComputePass> mpPass;
    uint32_t mBufferSelection = 0;
    uint32_t mChannelSelection = 0;
};
