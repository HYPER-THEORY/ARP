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

/** Lighting composite pass.

    Sums the deferred lighting terms, each of which is now produced in isolation
    by its own pass:

      geometry   = direct (DIPass) + ambient (AmbientPass) + indirect (IndirectPass)
                   + emissive (GBufferPass)
      background = skybox (SkyboxPass)

    Emissive is added here because UE's base pass writes it straight into
    SceneColor, before any light pass runs. Arctic has no base-pass color target,
    so the emissive is parked in a dedicated HDR MRT and it rejoins the sum here.
    This is also what makes SHADINGMODELID_UNLIT visible at all: DIPass outputs
    0 for those pixels.

    Nothing here is scene-dependent, so the pass is pure screen space.

    The per-term intensities double as enable switches: the C++ side passes 0 for
    a term that is disabled or whose input edge is absent, so the shader needs no
    separate gHas* flags.
*/

cbuffer PerFrameCB
{
    uint2 gFrameDim;
    float gDirectIntensity;
    float gAmbientIntensity;
    float gIndirectIntensity;
    float gSkyboxIntensity;
    float gEmissiveIntensity;
};

Texture2D<float>  gSceneDepth;   // device Z: 0 = near, 1 = far (>= 1 means background)
Texture2D<float4> gSkyboxColor;  // SkyboxPass.skyboxColor (0 on geometry)
Texture2D<float4> gDIColor;      // DIPass.color (0 on background)
Texture2D<float4> gAmbient;      // AmbientPass.ambient (0 on background)
Texture2D<float4> gIndirect;     // IndirectPass.indirect (0 on background)
Texture2D<float4> gEmissive;     // GBufferPass.emissive (0 where nothing was rasterized)

RWTexture2D<float4> gColor;

[numthreads(8, 8, 1)]
void csMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFrameDim)) return;

    float deviceZ = gSceneDepth.Load(int3(pixel, 0)).r;

    // Background: the skybox alone. The lighting terms are undefined here (their
    // passes early-out), so they must not be folded in.
    if (deviceZ >= 1.f)
    {
        float3 sky = gSkyboxColor.Load(int3(pixel, 0)).rgb * gSkyboxIntensity;
        gColor[pixel] = float4(sky, 1.f);
        return;
    }

    float3 direct = gDIColor.Load(int3(pixel, 0)).rgb * gDirectIntensity;
    float3 ambient = gAmbient.Load(int3(pixel, 0)).rgb * gAmbientIntensity;
    float3 indirect = gIndirect.Load(int3(pixel, 0)).rgb * gIndirectIntensity;
    float3 emissive = gEmissive.Load(int3(pixel, 0)).rgb * gEmissiveIntensity;

    gColor[pixel] = float4(direct + ambient + indirect + emissive, 1.f);
}
