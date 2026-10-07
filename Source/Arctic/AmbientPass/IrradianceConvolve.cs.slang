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
 #    from this software without specific prior written permission.
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

/** Builds the cosine-convolved irradiance map the diffuse IBL lookup samples.

    UnrealEngine convolves the cubemap mips at or below
    `GDiffuseConvolveMipLevel` with a full-hemisphere cone (see
    TextureCompressorModule.cpp:GenerateAngularFilteredMips, where the cone angle
    is clamped to PI/2 for exactly those mips). The resulting mip is *irradiance*,
    not a blurred copy of the source, and AmbientCubemapComposite.usf samples it
    at `AmbientCubemapMipAdjust.z` regardless of roughness.

    Falcor's lat-long mip chain is a plain box-filtered blit
    (Texture.cpp:generateMips), which is not a cosine kernel at any level, so it
    cannot be used for this. This pass does the convolution explicitly.

    What is stored is the *average incoming radiance over the cosine-weighted
    hemisphere*:

        DiffuseLookup = (1/N) * sum(L_i)

    which is what UE's DiffuseIBL accumulates (`SampleColor * DiffuseColor` summed
    over cosine-distributed samples, divided by N). Because the cosine-weighted
    pdf is cos(theta)/PI, this mean is related to irradiance E by
    E = PI * mean(L), and the Lambertian outgoing radiance is
    (albedo/PI) * E = albedo * mean(L) -- so the caller keeps multiplying by
    albedo directly, exactly as before.

    The convolution runs in the env map's *local* space: rotation is applied at
    lookup time (EnvMap::eval applies toLocal before the lat-long projection), so
    rotating the environment never invalidates this map and only a change of env
    map texture or resolution triggers a rebuild.
*/

import Utils.Math.MathHelpers;

cbuffer PerFrameCB
{
    uint2 gIrradianceDim;   // Output dimensions (lat-long), e.g. 64x32.
    uint  gNumSamples;      // Cosine-weighted samples per output texel.
    float gSrcLod;          // Source mip used to prefilter, suppressing fireflies.
};

Texture2D<float4> gEnvMap;
SamplerState      gEnvSampler;
RWTexture2D<float4> gIrradiance;

/** Van der Corput radical inverse (bit reversal). MathHelpers has no low
    discrepancy sequence, so the second Hammersley dimension comes from here.
*/
float radicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;   // / 0x100000000
}

[numthreads(8, 8, 1)]
void csMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gIrradianceDim)) return;

    // Output texel centre -> direction in the env map's local frame.
    float2 uv = (float2(pixel) + 0.5f) / float2(gIrradianceDim);
    float3 N = latlong_map_to_world(uv);

    float3 b1, b2;
    branchlessONB(N, b1, b2);

    float3 sum = float3(0.f);
    for (uint i = 0; i < gNumSamples; i++)
    {
        float2 u = float2((float)i / float(gNumSamples), radicalInverse_VdC(i));

        float pdf;
        float3 localDir = sample_cosine_hemisphere_concentric(u, pdf);
        float3 dir = localDir.x * b1 + localDir.y * b2 + localDir.z * N;

        // The source is sampled at a mip chosen to match one sample's footprint,
        // which trades a little bias for a large variance reduction: without it a
        // small bright sun contributes single-texel spikes that survive the mean
        // as salt-and-pepper noise on the diffuse term.
        sum += gEnvMap.SampleLevel(gEnvSampler, world_to_latlong_map(dir), gSrcLod).rgb;
    }

    gIrradiance[pixel] = float4(sum / float(gNumSamples), 1.f);
}
