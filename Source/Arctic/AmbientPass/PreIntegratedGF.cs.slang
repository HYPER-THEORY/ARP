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

/** Generates UE's PreintegratedGF split-sum table.

    Port of SystemTextures.cpp:532-669 (the CPU-side builder that fills
    `GSystemTextures.PreintegratedGF`). It is the first half of the Karis 2013
    split-sum approximation:

        EnvBRDF(SpecularColor, Roughness, NoV)
            = AB.x * SpecularColor + AB.y * saturate(50 * SpecularColor.g)

    where A and B are the GGX-importance-sampled integrals of the environment
    BRDF over the half-vector space.

    Layout, matching UE exactly:
      - 128 x 32, RG16_UNORM (`PF_G16R16` in UE; the `bReference` 128x128 path is
        off by default). UE's own comment: "for testing, with 128x128 R8G8 we are
        very close to the reference".
      - x is NoV, y is roughness.
      - .r = A, .g = B.
      - Both are clamped to [0,1] before write, which the UNORM format would do
        anyway; UE clamps explicitly on the CPU side.

    The result depends only on (NoV, Roughness), so it is generated once and kept
    resident -- UE likewise builds it once at system-texture init.
*/

// MathConstants is included rather than imported: macro definitions are not
// exported from a Slang module (see MathConstants.slangh).
#include "Utils/Math/MathConstants.slangh"

cbuffer PerFrameCB
{
    uint2 gTableDim;      // UE's 128 x 32.
    uint  gNumSamples;    // UE's 128.
};

RWTexture2D<float2> gTable;

/** Van der Corput radical inverse. UE uses ReverseBits(i) / 0x100000000 for the
    second Hammersley dimension (SystemTextures.cpp:589).
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
    if (any(pixel >= gTableDim)) return;

    // x is NoV, y is roughness, both at texel centres (SystemTextures.cpp:568,574).
    float Roughness = (float(pixel.y) + 0.5f) / float(gTableDim.y);
    float NoV = (float(pixel.x) + 0.5f) / float(gTableDim.x);

    float m = Roughness * Roughness;
    float m2 = m * m;

    float3 V = float3(sqrt(max(0.f, 1.f - NoV * NoV)), 0.f, NoV);

    float A = 0.f;
    float B = 0.f;

    for (uint i = 0; i < gNumSamples; i++)
    {
        float E1 = (float)i / float(gNumSamples);
        float E2 = radicalInverse_VdC(i);

        // GGX importance sample of the half vector (SystemTextures.cpp:592-599).
        float Phi = 2.f * M_PI * E1;
        float CosTheta = sqrt(max(0.f, (1.f - E2) / (1.f + (m2 - 1.f) * E2)));
        float SinTheta = sqrt(max(0.f, 1.f - CosTheta * CosTheta));

        float3 H = float3(SinTheta * cos(Phi), SinTheta * sin(Phi), CosTheta);
        float3 L = 2.f * dot(V, H) * H - V;

        float NoL = max(L.z, 0.f);
        float NoH = max(H.z, 0.f);
        float VoH = max(dot(V, H), 0.f);

        if (NoL > 0.f)
        {
            // UE passes `m` here, which is sqrt(a2) for a2 = Pow4(Roughness) --
            // i.e. the same quantity Vis_SmithJointApprox derives internally.
            float Vis_SmithV = NoL * (NoV * (1.f - m) + m);
            float Vis_SmithL = NoV * (NoL * (1.f - m) + m);
            float Vis = 0.5f / (Vis_SmithV + Vis_SmithL);

            // pdf = D * NoH / (4 * VoH), so NoL * Vis / pdf = NoL * Vis * 4 * VoH / NoH.
            float NoL_Vis_PDF = NoL * Vis * (4.f * VoH / NoH);

            float Fc = 1.f - VoH;
            Fc *= Fc * Fc * Fc * Fc;      // Pow5

            A += NoL_Vis_PDF * (1.f - Fc);
            B += NoL_Vis_PDF * Fc;
        }
    }

    A /= float(gNumSamples);
    B /= float(gNumSamples);

    // UE clamps both to [0,1] on write.
    gTable[pixel] = float2(saturate(A), saturate(B));
}
