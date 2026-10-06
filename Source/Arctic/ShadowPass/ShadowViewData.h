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
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "Falcor.h"

using namespace Falcor;

/** Light view parameters produced by ShadowPass and consumed by
    ShadowProjectionPass.

    These MUST be identical on both sides: ShadowPass rasterizes light-space
    linear depth using `lightViewProj` / `lightPos` / `lightDir`, and
    ShadowProjectionPass re-projects screen pixels with the same transform to
    look the depth up. Previously each pass recomputed the frustum from the
    scene bounds with duplicated code; any divergence between the two copies
    shows up as a whole-screen shadow offset, which is nearly impossible to
    tell apart from a bias or filter-centering problem.

    ShadowPass::execute publishes an instance into the RenderData dictionary
    under kShadowViewDataKey (Falcor's Dictionary is backed by std::any, so a
    POD struct round-trips fine). ShadowProjectionPass::execute reads it back.
*/
struct ShadowViewData
{
    /// proj * view for the light. Falcor is column-vector: clip = M * pos.
    float4x4 lightViewProj = float4x4::identity();
    /// Reference point for the light-space linear depth metric. Placed just in
    /// front of the nearest geometry so depths start near 0 (better float
    /// precision than the scene-radius-sized offsets an outside-the-scene
    /// light position would produce).
    float3 lightPos = float3(0.f);
    /// Light travel direction (normalized). Depth = dot(lightDir, p - lightPos).
    float3 lightDir = float3(0.f, -1.f, 0.f);
    /// Upper bound on any depth ShadowPass can write, including the maximum
    /// possible depth bias. Sampled values above this mean "no geometry was
    /// written here" (the shadowmap is cleared far beyond it).
    float maxSubjectDepth = 1.f;
    /// Half-extent of the orthographic frustum in world units along the
    /// light's right/up axes. Lets the projection pass convert a world-space
    /// filter radius (meters) into shadowmap UV: uvPerMeter = 0.5 / halfExtent.
    float2 orthoHalfExtent = float2(1.f);
    /// False until a directional light + valid scene bounds have been found.
    /// ShadowProjectionPass outputs "fully lit" rather than garbage when unset.
    bool valid = false;
};

/// RenderData dictionary key for ShadowViewData.
inline const char* kShadowViewDataKey = "arctic.shadowViewData";
