/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the copyright
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

/** GTAO upsample: half res -> full res.

    Direct port of UnrealEngine's GTAOUpsamplePSAndCS
    (Private/PostProcessAmbientOcclusion.usf).

    Worth being explicit about, because it is not what one would guess: UE's GTAO
    upsample has *no* bilateral term. It takes four bilinear taps a quarter of a
    half-res texel apart and keeps the **minimum**. Biasing toward occlusion is
    what suppresses light leaking across silhouettes; a depth/normal-weighted
    average would instead produce a bright halo wherever the weights collapse.
    (The bilateral ComputeUpsampleContribution in
    PostProcessAmbientOcclusionCommon.ush belongs to the SSAO path, not this
    one.) UE also reads a centre tap, AOC, and then never uses it — dropped here.

    Alignment: half-res texel h holds the AO of full-res pixel 2h, so the
    half-res sample grid sits a quarter of a texel off the centres the hardware
    bilinear unit assumes. Sampling in texel-coordinate space, full-res pixel p
    maps to half-res texel coordinate t = p * 0.5 — exact for even p, exactly
    halfway for odd p — which absorbs that offset with no magic epsilon.
*/

cbuffer PerFrameCB
{
    uint2 gFullFrameDim;
    uint2 gHalfFrameDim;
};

Texture2D<float> gAOMapHalf;   // temporally filtered half-res AO
SamplerState     gAOSampler;   // linear, clamp

RWTexture2D<float> gAOMap;

// t is a half-res texel coordinate (texel centres at integers, see header).
float SampleAO(float2 t)
{
    float2 uv = (t + 0.5f) / float2(gHalfFrameDim);
    return saturate(gAOMapHalf.SampleLevel(gAOSampler, uv, 0).r);
}

[numthreads(8, 8, 1)]
void csMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFullFrameDim)) return;

    float2 t = float2(pixel) * 0.5f;
    const float o = 0.25f;

    float AO0 = SampleAO(t + float2(-o, -o));
    float AO1 = SampleAO(t + float2( o, -o));
    float AO2 = SampleAO(t + float2(-o,  o));
    float AO3 = SampleAO(t + float2( o,  o));

    gAOMap[pixel] = min(min(AO0, AO1), min(AO2, AO3));
}
