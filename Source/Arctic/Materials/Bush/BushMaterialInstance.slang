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

__exported import Rendering.Materials.IMaterialInstance;
__exported import Arctic.Materials.Bush.BushBRDF;

/** Bush material instance: wraps BushBRDF as an IMaterialInstance.

    The shading frame is used as-is, without the usual backface flip. See the
    header comment in BushBRDF.slang for why: the deferred path stores an
    unflipped normal for two-sided foliage, and the BRDF's lobes are defined by
    the sign of `wo.z` in that frame.
*/
struct BushMaterialInstance : MaterialInstanceBase, IMaterialInstance
{
    ShadingFrame sf;            ///< Shading frame in world space.
    BushBRDF brdf;

    __init(const ShadingFrame sf, const BushBRDF brdf)
    {
        this.sf = sf;
        this.brdf = brdf;
    }

    float3 eval<S : ISampleGenerator>(const ShadingData sd, const float3 wo, inout S sg)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        return brdf.eval(wiLocal, woLocal, sg, BSDFContext());
    }

    [Differentiable]
    float3 evalAD<S : ISampleGenerator>(const DiffMaterialData diffData, const ShadingData sd, const float3 wo, inout S sg)
    {
        return float3(0.f);
    }

    bool sample<S : ISampleGenerator>(const ShadingData sd, inout S sg, out BSDFSample result, bool useImportanceSampling = true)
    {
        result = BSDFSample();

        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = {};

        if (!brdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobeType, sg, BSDFContext()))
        {
            result = BSDFSample();
            return false;
        }

        result.wo = sf.fromLocal(woLocal);
        return true;
    }

    float evalPdf(const ShadingData sd, const float3 wo, bool useImportanceSampling = true)
    {
        float3 wiLocal = sf.toLocal(sd.V);
        float3 woLocal = sf.toLocal(wo);

        return brdf.evalPdf(wiLocal, woLocal, BSDFContext());
    }

    BSDFProperties getProperties(const ShadingData sd)
    {
        BSDFProperties p = {};

        p.guideNormal = sf.N;
        p.roughness = brdf.roughness;
        p.diffuseReflectionAlbedo = brdf.diffuseColor;
        p.diffuseTransmissionAlbedo = brdf.subsurfaceColor;
        p.specularReflectionAlbedo = brdf.specularColor;

        if (any(brdf.subsurfaceColor > 0.f))
        {
            p.flags |= (uint)BSDFProperties.Flags::IsTransmissive;
        }

        return p;
    }

    uint getLobeTypes(const ShadingData sd)
    {
        // SpecularReflection is deliberately absent: the specular term is folded
        // into the cosine-sampled upper hemisphere, so `sample()` never returns a
        // SpecularReflection lobe. Advertising a lobe we never emit would mislead
        // callers that filter samples by lobe type. The specular energy is still
        // reported through `specularReflectionAlbedo` for denoiser guides.
        uint lobes = (uint)LobeType::DiffuseReflection;
        if (any(brdf.subsurfaceColor > 0.f))
        {
            lobes |= (uint)LobeType::DiffuseTransmission;
        }
        return lobes;
    }
};
