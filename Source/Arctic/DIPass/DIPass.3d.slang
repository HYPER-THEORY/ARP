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

import Scene.Scene;
import Scene.Camera.Camera;
import Utils.Math.MathHelpers;
import Arctic.GBufferPass.GBufferHelpers;

cbuffer PerFrameCB
{
    uint2  gFrameDim;
    float3 gLightDir;        // DirectionalLight travel direction (normalized)
    float3 gLightIntensity;  // Falcor LightData.intensity (premultiplied radiance)
    uint   gHasLight;
    float  gMinRoughness;    // UE View.MinRoughness
    float  gSubsurfaceTransmittanceDistance; // UE View.SubSurfaceColorAsTransmittanceAtDistanceInMeters
};

Texture2D<float4> gGBufferA;
Texture2D<float4> gGBufferB;
Texture2D<float4> gGBufferC;
Texture2D<float4> gGBufferD;
Texture2D<float4> gGBufferE;
Texture2D<float>  gSceneDepth;
Texture2D<float2> gShadowMask;  // .r = surface shadow, .g = SSS transmission shadow
RWTexture2D<float4> gColor;

static const float PI = 3.14159265358979f;

// ---- UE BRDF helpers (ported from BRDF.ush) ----

float pow5(float x) { float x2 = x * x; return x2 * x2 * x; }
float square(float x) { return x * x; }

float D_GGX(float a2, float NoH)
{
    float d = (NoH * a2 - NoH) * NoH + 1.f;
    return a2 / (PI * d * d);
}

// BRDF.ush. This is UE's *approximation*, which is what DefaultLit
// actually uses via SpecularGGX (ShadingModels.ush). The exact
// Vis_SmithJoint (BRDF.ush) is a visibly different function -- brighter
// at mid roughness -- so using it here would silently diverge from UE.
float Vis_SmithJointApprox(float a2, float NoV, float NoL)
{
    float a = sqrt(a2);
    float Vis_SmithV = NoL * (NoV * (1.f - a) + a);
    float Vis_SmithL = NoV * (NoL * (1.f - a) + a);
    return 0.5f * rcp(Vis_SmithV + Vis_SmithL);
}

float3 F_Schlick(float3 SpecularColor, float VoH)
{
    float Fc = pow5(1.f - VoH);
    return saturate(50.f * SpecularColor.g) * Fc + (1.f - Fc) * SpecularColor;
}

// ---- UE color space helpers (ColorSpace.ush), used by SubsurfaceBxDF ----

float3 HUE_2_LinearRGB(float H)
{
    float R = abs(H * 6.f - 3.f) - 1.f;
    float G = 2.f - abs(H * 6.f - 2.f);
    float B = 2.f - abs(H * 6.f - 4.f);
    return saturate(float3(R, G, B));
}

float3 HSV_2_LinearRGB(float3 HSV)
{
    float3 RGB = HUE_2_LinearRGB(HSV.x);
    return ((RGB - 1.f) * HSV.y + 1.f) * HSV.z;
}

float3 RGB_2_HCV(float3 RGB)
{
    float4 P = (RGB.g < RGB.b) ? float4(RGB.bg, -1.f, 2.f / 3.f) : float4(RGB.gb, 0.f, -1.f / 3.f);
    float4 Q = (RGB.r < P.x) ? float4(P.xyw, RGB.r) : float4(RGB.r, P.yzx);
    float chroma = Q.x - min(Q.w, Q.y);
    float hue = abs((Q.w - Q.y) / (6.f * chroma + 1e-10f) + Q.z);
    return float3(hue, chroma, Q.x);
}

float3 LinearRGB_2_HSV(float3 RGB)
{
    float3 HCV = RGB_2_HCV(RGB);
    float s = HCV.y / (HCV.z + 1e-10f);
    return float3(HCV.x, s, HCV.z);
}

// ---- UE participating-media helpers (ParticipatingMediaCommon.ush) ----

static const float kMinTransmittance = 1e-4f;  // PARTICIPATING_MEDIA_MIN_TRANSMITTANCE
static const float kMinMfpMeter      = 1e-4f;  // PARTICIPATING_MEDIA_MIN_MFP_METER

// TransmittanceColor = exp(-Extinction * Thickness)  =>  Extinction = -log(T) / Thickness
float3 transmittanceToExtinction(float3 transmittanceColor, float thicknessMeters)
{
    return -log(clamp(transmittanceColor, kMinTransmittance, 1.f)) / max(kMinMfpMeter, thicknessMeters);
}

float3 extinctionToTransmittance(float3 extinction, float thicknessMeters)
{
    return exp(-extinction * thicknessMeters);
}

// ---- GBuffer decode ----

struct DIGBuffer
{
    float3 worldNormal;
    float  metallic;
    float  specular;        // UE dielectric specular (0..1)
    float  roughness;
    float3 baseColor;
    float  gBufferAO;
    float3 subsurfaceColor; // Decoded GBufferD.rgb (Subsurface / TwoSidedFoliage)
    float  opacity;         // GBufferD.a, UE's material Opacity input
    uint   shadingModelID;
};

DIGBuffer decodeGBuffer(uint2 pixel)
{
    DIGBuffer g;
    float4 a = gGBufferA.Load(int3(pixel, 0));
    float4 b = gGBufferB.Load(int3(pixel, 0));
    float4 c = gGBufferC.Load(int3(pixel, 0));
    float4 d = gGBufferD.Load(int3(pixel, 0));

    g.worldNormal = decodeNormal(a.rgb);
    g.metallic = b.r;
    g.specular = b.g;
    // UE clamps roughness at the bottom for every shading model, inside
    // IntegrateBxDF (CapsuleLightIntegrate.ush). Without it a roughness of 0
    // collapses D_GGX to a delta and the highlight vanishes entirely.
    g.roughness = max(b.b, gMinRoughness);
    g.shadingModelID = decodeShadingModelId(b.a);
    g.baseColor = c.rgb;
    g.gBufferAO = c.a;
    // GBufferD.rgb is sqrt-encoded to buy precision in the darks on the 8-bit
    // target (GBufferHelpers::encodeSubsurfaceColor). Undo it here, as UE's
    // ExtractSubsurfaceColor does.
    g.subsurfaceColor = extractSubsurfaceColor(d.rgb);
    g.opacity = d.a;
    return g;
}

// Reconstruct world position from screen pixel + device depth.
float3 reconstructWorldPos(uint2 pixel)
{
    float deviceZ = gSceneDepth.Load(int3(pixel, 0)).r;
    float2 ndc;
    ndc.x = (float(pixel.x) + 0.5f) / float(gFrameDim.x) * 2.f - 1.f;
    ndc.y = 1.f - (float(pixel.y) + 0.5f) / float(gFrameDim.y) * 2.f;
    float4 clip = float4(ndc, deviceZ, 1.f);
    // Falcor matrices are column-vector: v' = M * v. Use mul(M, v), not mul(v, M).
    float4 worldH = mul(gScene.camera.data.invViewProj, clip);
    return worldH.xyz / worldH.w;
}

struct DirectLighting
{
    float3 diffuse;
    float3 specular;
    float3 transmission;
};

// UE DefaultLitBxDF (ShadingModels.ush), reduced to a punctual,
// isotropic light. Energy conservation is deliberately absent: UE's
// ComputeEnergyPreservation/ComputeEnergyConservation both return exactly 1.0
// on the legacy non-Substrate path, where USE_ENERGY_CONSERVATION is 0
// (ShadingEnergyConservation.ush), so omitting them *is* UE's default.
DirectLighting defaultLitBxDF(DIGBuffer g, float3 N, float3 V, float3 L, float NoL)
{
    DirectLighting lighting;
    lighting.diffuse = float3(0.f);
    lighting.specular = float3(0.f);
    lighting.transmission = float3(0.f);

    if (NoL <= 0.f) return lighting;

    // Derived colors (UE DeferredShadingCommon.ush:GetPrecomputedIndirectLighting / SetGBufferForShading).
    float3 diffuseColor = g.baseColor * (1.f - g.metallic);
    float3 specularColor = lerp(0.08f * g.specular, g.baseColor, g.metallic);

    // H via normalize(V+L) is algebraically identical to UE's InvLenH =
    // rsqrt(2 + 2*VoL) form (BRDF.ush), since |V+L| = sqrt(2 + 2*VoL).
    float3 H = normalize(V + L);
    float NoV = saturate(abs(dot(N, V)) + 1e-5f);
    float NoH = saturate(dot(N, H));
    float VoH = saturate(dot(V, H));

    float a2 = g.roughness * g.roughness;
    a2 = a2 * a2; // roughness^4 (UE Pow4)

    float Vis = Vis_SmithJointApprox(a2, NoV, NoL);
    float D = D_GGX(a2, NoH);
    float3 F = F_Schlick(specularColor, VoH);

    lighting.diffuse = diffuseColor / PI;  // Diffuse_Lambert
    lighting.specular = D * Vis * F;

    return lighting;
}

// UE TwoSidedBxDF (#else branch in ShadingModels.ush): DefaultLit + wrap
// transmission.
DirectLighting twoSidedBxDF(DIGBuffer g, float3 N, float3 V, float3 L, float NoL)
{
    DirectLighting lighting = defaultLitBxDF(g, N, V, L, NoL);

    // http://blog.stevemcauley.com/2011/12/03/energy-conserving-wrapped-diffuse/
    float wrap = 0.5f;
    float wrapNoL = saturate((-dot(N, L) + wrap) / square(1.f + wrap));

    float VoL = dot(V, L);
    float scatter = D_GGX(0.6f * 0.6f, saturate(-VoL));

    // wrapNoL IS the cosine term for the transmitted lobe -- it is non-zero all
    // the way from NoL = +wrap down to NoL = -1. UE applies no further cosine
    // (ShadingModels.ush), so neither may we.
    lighting.transmission = wrapNoL * scatter * g.subsurfaceColor;
    return lighting;
}

// UE SubsurfaceBxDF (ShadingModels.ush): DefaultLit + a wrap/in-scatter
// back-scatter term modulated by a Beer-Lambert transmitted color.
DirectLighting subsurfaceBxDF(DIGBuffer g, float3 N, float3 V, float3 L, float NoL, float transmissionThickness)
{
    DirectLighting lighting = defaultLitBxDF(g, N, V, L, NoL);

    float3 subsurfaceColor = g.subsurfaceColor;
    float opacity = g.opacity;

    // To get an effect when you see through the material. Hard coded pow constant.
    float inScatter = pow(saturate(dot(L, -V)), 12.f) * lerp(3.f, 0.1f, opacity);

    // Wrap-around lighting: /(PI*2) to be energy consistent. Opacity 0 gives no
    // normal-dependent lighting, 1 gives a strong normal contribution.
    // Simplified form of the w=0.5, n=1.5 wrapped diffuse.
    float wrappedDiffuse = pow(saturate(dot(N, L) * (1.f / 1.5f) + (0.5f / 1.5f)), 1.5f) * (2.5f / 1.5f);
    float normalContribution = lerp(1.f, wrappedDiffuse, opacity);
    float backScatter = g.gBufferAO * normalContribution / (PI * 2.f);

    // Emulate Beer-Lambert absorption by reading an extinction coefficient out of
    // SubsurfaceColor, interpreted as a transmittance color at a normalized
    // distance. TransmittedColor is evaluated at 1 normalized unit, then pushed
    // through HSV to keep the original color's value while picking up its hue
    // shift.
    float3 extinctionCoefficients = transmittanceToExtinction(subsurfaceColor, gSubsurfaceTransmittanceDistance);
    float3 rawTransmittedColor = extinctionToTransmittance(extinctionCoefficients, 1.f);
    float3 transmittedColor =
        HSV_2_LinearRGB(float3(LinearRGB_2_HSV(rawTransmittedColor).xy, LinearRGB_2_HSV(subsurfaceColor).z));

    // lerp to never exceed 1 (energy conserving).
    lighting.transmission =
        lerp(backScatter, 1.f, inScatter) * lerp(transmittedColor, subsurfaceColor, transmissionThickness);

    return lighting;
}

[numthreads(8, 8, 1)]
void csMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFrameDim)) return;

    float deviceZ = gSceneDepth.Load(int3(pixel, 0)).r;

    // Background pixel: no direct lighting. CompositePass fills these from
    // SkyboxPass, so this pass must not write the sky itself.
    if (deviceZ >= 1.f)
    {
        gColor[pixel] = float4(0.f, 0.f, 0.f, 1.f);
        return;
    }

    // No directional light in scene: output black (background already handled).
    if (gHasLight == 0u)
    {
        gColor[pixel] = float4(0.f, 0.f, 0.f, 1.f);
        return;
    }

    DIGBuffer g = decodeGBuffer(pixel);

    // Unlit receives no direct lighting at all. UE skips the whole deferred light
    // pass for these pixels (DeferredLightPixelShaders.usf, bMaterialIsLit)
    // and IntegrateBxDF's default case also returns 0 (ShadingModels.ush).
    // Their radiance is entirely the emissive term, which CompositePass adds.
    if (g.shadingModelID == SHADINGMODELID_UNLIT)
    {
        gColor[pixel] = float4(0.f, 0.f, 0.f, 1.f);
        return;
    }

    float3 worldPos = reconstructWorldPos(pixel);
    float3 N = normalize(g.worldNormal);
    float3 V = normalize(gScene.camera.getPosition() - worldPos);
    // Falcor DirectionalLight.dirW is the light travel direction. The shading
    // vector L (from surface toward light) is -dirW.
    float3 L = -normalize(gLightDir);
    float NoL = dot(N, L);

    float2 shadow = gShadowMask.Load(int3(pixel, 0)).rg;
    // UE ShadowTerms: .r = SurfaceShadow (diffuse/specular), .g = TransmissionShadow
    // (subsurface transmission). Splitting these fixes the previous bug where the
    // surface shadow killed back-lit transmission (transmission should be lit by
    // light penetrating the leaf, not gated by the front-face surface shadow).
    float surfaceShadow = shadow.r;
    float transmissionShadow = shadow.g;
    // UE: TransmissionThickness = min(LightAttenuation.y, LightAttenuation.w)
    // (DeferredLightingCommon.ush). With no static lighting and no per-object
    // shadows, LightAttenuation.w is 1 and this reduces exactly to the
    // transmission shadow -- so no third shadowMask channel is needed.
    float transmissionThickness = transmissionShadow;

    // Shading model dispatch, mirroring UE IntegrateBxDF (ShadingModels.ush).
    DirectLighting lighting;
    switch (g.shadingModelID)
    {
    case SHADINGMODELID_SUBSURFACE:
        lighting = subsurfaceBxDF(g, N, V, L, NoL, transmissionThickness);
        break;
    case SHADINGMODELID_TWOSIDED_FOLIAGE:
        lighting = twoSidedBxDF(g, N, V, L, NoL);
        break;
    default: // SHADINGMODELID_DEFAULT_LIT and any model we do not produce.
        lighting = defaultLitBxDF(g, N, V, L, NoL);
        break;
    }

    // UE applies NoL and the falloff inside the BxDF; for a directional light the
    // falloff is 1, so folding NoL in here is equivalent. Transmission carries its
    // own cosine (wrapNoL / backScatter) and must not be multiplied by NoL.
    float NoLclamped = max(NoL, 0.f);
    float3 color = (lighting.diffuse + lighting.specular) * NoLclamped * surfaceShadow;
    color += lighting.transmission * transmissionShadow;

    color *= gLightIntensity;

    gColor[pixel] = float4(color, 1.f);
}
