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
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.ShadingData;
import Utils.Color.ColorHelpers;
import Utils.Math.MathHelpers;
__exported import Rendering.Materials.IBSDF;

/** Bush BSDF: a direct transcription of UE's TwoSidedBxDF.

    This is the ray-tracing counterpart of what the deferred path computes in
    `DIPass.3d.slang::twoSidedBxDF` (which is itself the `#else` branch of UE's
    ShadingModels.ush TwoSidedBxDF). The two are aligned term for term:

        diffuse       = baseColor / PI                          * saturate(NoL)
        specular      = D_GGX * Vis_SmithJointApprox * F_Schlick * saturate(NoL)
        transmission  = wrapNoL * scatter * subsurfaceColor      * saturate(-NoL)

    with `a2 = roughness^4` (UE Pow4), `specularColor = 0.08 * dielectricSpecular`
    (metallic is always 0 for foliage), `wrap = 0.5`, and
    `scatter = D_GGX(0.6^2, saturate(-VoL))`.

    Three consequences of matching UE exactly, all inherited rather than
    introduced here:

    1. The lobes are defined by the sign of `wo.z` in the shading frame, not
       relative to `wi`. The upper hemisphere gets diffuse+specular, the lower
       gets transmission. This is why the shading normal is NOT flipped for
       backfacing hits (see BushMaterial.slang) — the deferred path stores the
       unflipped normal too, so flipping here would make RT and raster disagree.
    2. `NoV` is taken as `abs(dot(N, V))`, exactly as DIPass does. Combined with
       (1) this makes the model non-reciprocal: it is an artist-facing
       approximation of light through a thin leaf, not a physical BSDF.
    3. There is no separate "subsurface weight" scalar. The transmission lobe is
       weighted by `subsurfaceColor` alone, matching DIPass. The earlier
       implementation invented a weight by borrowing `BasicMaterialData::
       diffuseTransmission`, which no shading path ever set.

    Importance sampling is cosine-weighted per hemisphere, with the hemisphere
    chosen by the relative luminance of the two lobes. The specular lobe is
    folded into the cosine-sampled upper hemisphere rather than getting its own
    GGX sampling routine, so variance grows as roughness falls. That is
    acceptable for foliage (authored roughness is high) and keeps the sampling
    consistent with the eval above; it would need revisiting for a glossy
    material.
*/
struct BushBRDF : IBSDF
{
    float3 diffuseColor;        ///< Base color. Foliage is dielectric so this is the full diffuse albedo.
    float3 specularColor;       ///< 0.08 * dielectric specular (UE convention).
    float3 subsurfaceColor;     ///< Transmission tint; also acts as the transmission lobe weight.
    float  roughness;           ///< Linear roughness in [0,1]. a2 = roughness^4.

    // ---- IBSDF interface ----

    float3 eval<S : ISampleGenerator>(const float3 wi, const float3 wo, inout S sg, BSDFContext bc)
    {
        // Both hemispheres are valid: the transmission lobe lives in the one
        // opposite the frame normal. Only grazing directions are rejected.
        if (min(abs(wi.z), abs(wo.z)) < kMinCosTheta) return float3(0.f);

        const float NoL = wo.z;
        float3 result = float3(0.f);

        if (NoL > 0.f)
        {
            result += (evalDiffuse() + evalSpecular(wi, wo)) * NoL;
        }
        else
        {
            result += evalTransmission(wi, wo) * (-NoL);
        }

        return result;
    }

    bool sample<S : ISampleGenerator>(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobeType, inout S sg, BSDFContext bc)
    {
        wo = {};
        pdf = 0.f;
        weight = {};
        lobeType = {};

        const float pTransmission = getTransmissionProbability();
        const bool sampleTransmission = pTransmission > 0.f && sampleNext1D(sg) < pTransmission;

        float cosPdf = 0.f;
        wo = sample_cosine_hemisphere_concentric(sampleNext2D(sg), cosPdf); // +z hemisphere, pdf = cos / pi

        if (sampleTransmission)
        {
            wo.z = -wo.z;
            pdf = cosPdf * pTransmission;
            lobeType = (uint)LobeType::DiffuseTransmission;
        }
        else
        {
            pdf = cosPdf * (1.f - pTransmission);
            lobeType = (uint)LobeType::DiffuseReflection;
        }

        if (min(abs(wi.z), abs(wo.z)) < kMinCosTheta || pdf == 0.f) return false;

        weight = eval(wi, wo, sg, bc) / pdf;
        return true;
    }

    float evalPdf(const float3 wi, const float3 wo, BSDFContext bc)
    {
        if (min(abs(wi.z), abs(wo.z)) < kMinCosTheta) return 0.f;

        const float pTransmission = getTransmissionProbability();
        const float cosPdf = abs(wo.z) * M_1_PI;

        return (wo.z > 0.f) ? (1.f - pTransmission) * cosPdf : pTransmission * cosPdf;
    }

    AlbedoContributions evalAlbedo(const float3 wi, const LobeType lobetype)
    {
        return AlbedoContributions(diffuseColor, 1.f - diffuseColor, 0.f, subsurfaceColor);
    }

    float3 getIorAsReflectance()
    {
        return specularColor;
    }

    RoughnessInformation getRoughnessInformation(const float3 wi)
    {
        RoughnessInformation r;
        r.roughnessBSDFNotation = float2(roughness * roughness);
        return r;
    }

    // ---- UE BRDF terms (ported from BRDF.ush; identical to DIPass.3d.slang) ----

    float3 evalDiffuse()
    {
        return diffuseColor * M_1_PI;
    }

    float3 evalSpecular(const float3 wi, const float3 wo)
    {
        const float3 h = normalize(wi + wo);

        // abs() on NoV mirrors DIPass: the shading normal is not flipped for
        // backfacing hits, so dot(N, V) may be negative.
        const float NoV = saturate(abs(wi.z) + 1e-5f);
        const float NoL = saturate(wo.z);
        const float NoH = saturate(h.z);
        const float VoH = saturate(dot(wi, h));

        float a2 = roughness * roughness;
        a2 = a2 * a2; // UE Pow4

        return D_GGX(a2, NoH) * Vis_SmithJointApprox(a2, NoV, NoL) * F_Schlick(specularColor, VoH);
    }

    float3 evalTransmission(const float3 wi, const float3 wo)
    {
        // Energy-conserving wrapped diffuse.
        // http://blog.stevemcauley.com/2011/12/03/energy-conserving-wrapped-diffuse/
        const float wrap = 0.5f;
        const float wrapNoL = saturate((-wo.z + wrap) / ((1.f + wrap) * (1.f + wrap)));

        const float VoL = dot(wi, wo);
        const float scatter = D_GGX(0.6f * 0.6f, saturate(-VoL));

        return wrapNoL * scatter * subsurfaceColor;
    }

    float D_GGX(float a2, float NoH)
    {
        float d = (NoH * a2 - NoH) * NoH + 1.f;
        return a2 / (M_PI * d * d);
    }

    float Vis_SmithJointApprox(float a2, float NoV, float NoL)
    {
        float Vis_SmithV = NoV * sqrt((-NoL * a2 + NoL) * NoL + a2);
        float Vis_SmithL = NoL * sqrt((-NoV * a2 + NoV) * NoV + a2);
        return 0.5f / (Vis_SmithV + Vis_SmithL + 1e-5f);
    }

    float3 F_Schlick(float3 specColor, float VoH)
    {
        float f = 1.f - VoH;
        float f2 = f * f;
        float Fc = f2 * f2 * f;
        return saturate(50.f * specColor.g) * Fc + (1.f - Fc) * specColor;
    }

    /** Probability of picking the transmission (lower) hemisphere, from the
        relative luminance of the two lobes. Clamped away from 0 and 1 so
        neither lobe can be starved of samples while still contributing.
    */
    float getTransmissionProbability()
    {
        const float lumTransmission = luminance(subsurfaceColor);
        if (lumTransmission <= 0.f) return 0.f;

        const float lumReflection = luminance(diffuseColor + specularColor);
        return clamp(lumTransmission / (lumReflection + lumTransmission), 0.05f, 0.95f);
    }
};
