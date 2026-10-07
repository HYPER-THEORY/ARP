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

/** Ambient direct lighting pass.

    Computes the sky's direct (single-bounce) contribution only. It does not
    composite anything.

    Two modes:
      - Fixed color: a uniform gAmbientColor stands in
        for the envmap lookup in diffuse/specular/foliage subsurface.
      - EnvMap: ports UnrealEngine's AmbientCubemapComposite.usf:MainPS to Slang.

    IBL sources, each matching UE's, differ per lobe:

      - Diffuse reads the cosine-convolved irradiance map built by
        IrradianceConvolve.cs.slang. UE does the same convolution offline at
        texture build time, into the mips at and below GDiffuseConvolveMipLevel
        (TextureCompressorModule.cpp:GenerateAngularFilteredMips, cone angle
        clamped to PI/2). The lookup is *independent of roughness*; it is
        AmbientCubemapMipAdjust.z, a single absolute mip for every pixel. Falcor's
        box-filtered lat-long mips cannot stand in for it, hence the dedicated map.

      - Specular reads the env map's roughness-derived mip, matching
        ComputeCubemapMipFromRoughness (CubemapCommon.ush).

      - The F*G term comes from a PreintegratedGF table built once by
        PreIntegratedGF.cs.slang, matching UE's EnvBRDF (BRDF.ush) rather than the
        Lazarov analytic fit -- the fit is a different curve, not merely a cheaper
        way to evaluate the same one.

    In both modes we read the AO pass's aoMap, combine it with the material AO in
    GBufferC.a the way UE does, and apply specular occlusion via
    GetSpecularOcclusion (ReflectionEnvironmentShared.ush) as UE's reflection
    path does.
*/

import Scene.Scene;
import Scene.Camera.Camera;
import Utils.Math.MathHelpers;
import Arctic.GBufferPass.GBufferHelpers;

cbuffer PerFrameCB
{
    uint2  gFrameDim;
    float3 gTint;             // AmbientCubemapColor.rgb
    float  gIntensity;        // additional user multiplier
    float3 gAmbientColor;     // fixed ambient IBL color (used when gUseFixedColor != 0)
    uint   gUseFixedColor;    // 1 = use gAmbientColor, 0 = sample envMap
    float  gMipCount;         // AmbientCubemapMipAdjust.w (envMap mode only)
};

Texture2D<float4> gGBufferA;
Texture2D<float4> gGBufferB;
Texture2D<float4> gGBufferC;
Texture2D<float4> gGBufferD;
Texture2D<float>  gSceneDepth;
Texture2D<float4> gAOMap;        // AO in .r

// Cosine-convolved irradiance (lat-long), built by IrradianceConvolve.cs.slang.
Texture2D<float4> gIrradiance;
SamplerState      gIrradianceSampler;

// UE's PreintegratedGF split-sum table: x = NoV, y = roughness, .r = A, .g = B.
Texture2D<float2> gPreIntegratedGF;
SamplerState      gPreIntegratedGFSampler;

RWTexture2D<float4> gAmbient;

static const float PI = 3.14159265358979f;

// ---- UE BRDF helpers ----

float pow5(float x) { float x2 = x * x; return x2 * x2 * x; }

/** UE CubemapCommon.ush -- mip from roughness. Only the *specular* lookup uses
    this; the diffuse lookup is a fixed mip (see csMain).
*/
float ComputeCubemapMipFromRoughness(float Roughness, float MipCount)
{
    float Level = 3.0f - 1.15f * log2(max(Roughness, 1e-4f));
    return MipCount - 1.0f - Level;
}

/** UE AmbientCubemapMipAdjust.z == MipCount - GDiffuseConvolveMipLevel.

    Computed here rather than uploaded: both terms are already available
    (gMipCount) and GDiffuseConvolveMipLevel is a fixed engine constant
    (GlobalRenderResources.cpp: `const uint32 GDiffuseConvolveMipLevel = 4`), so
    uploading the result separately would just be a second source of truth.

    Note the constant is a *count back from the coarsest mip*, not a mip index:
    AbsoluteDiffuseMip = MipCount - 4 means mip 6 for a 512-wide cube (MipCount
    10) and mip 4 for 128 (MipCount 8).
*/
float GetAbsoluteDiffuseMip()
{
    const float kGDiffuseConvolveMipLevel = 4.0f;
    return max(gMipCount - kGDiffuseConvolveMipLevel, 0.0f);
}

/** UE BRDF.ush -- preintegrated G*F from the split-sum table.

    Anything less than 2% is physically impossible and is instead considered to
    be shadowing, hence the F90 clamp.
*/
float3 EnvBRDF(float3 SpecularColor, float Roughness, float NoV)
{
    float2 AB = gPreIntegratedGF.SampleLevel(gPreIntegratedGFSampler, float2(NoV, Roughness), 0).rg;
    return SpecularColor * AB.x + saturate(50.0f * SpecularColor.g) * AB.y;
}

/** UE ReflectionEnvironmentShared.ush -- AO applied to the specular lobe.

    Not a plain multiplication: a rough surface occluded from the side is barely
    darker, because the lobe still sees plenty of unoccluded directions.
*/
float GetSpecularOcclusion(float NoV, float RoughnessSq, float AO)
{
    return saturate(pow(NoV + AO, RoughnessSq) - 1.0f + AO);
}

// ---- GBuffer decode ----

struct AmbientGBuffer
{
    float3 worldNormal;
    float  metallic;
    float  specular;     // UE dielectric specular (0..1)
    float  roughness;
    float3 baseColor;
    float  gbufferAO;    // UE GBuffer.GBufferAO (material AO, GBufferC.a)
    float3 subsurfaceColor; // Decoded GBufferD.rgb (Subsurface / TwoSidedFoliage)
    uint   shadingModelID;
};

AmbientGBuffer decodeGBuffer(uint2 pixel)
{
    AmbientGBuffer g;
    float4 a = gGBufferA.Load(int3(pixel, 0));
    float4 b = gGBufferB.Load(int3(pixel, 0));
    float4 c = gGBufferC.Load(int3(pixel, 0));
    float4 d = gGBufferD.Load(int3(pixel, 0));

    g.worldNormal = decodeNormal(a.rgb);
    g.metallic = b.r;
    g.specular = b.g;
    g.roughness = b.b;
    g.shadingModelID = decodeShadingModelId(b.a);
    g.baseColor = c.rgb;
    g.gbufferAO = c.a;
    // GBufferD.rgb is sqrt-encoded on the 8-bit target; undo it as UE's
    // ExtractSubsurfaceColor does.
    g.subsurfaceColor = extractSubsurfaceColor(d.rgb);
    return g;
}

float3 reconstructWorldPos(uint2 pixel)
{
    float deviceZ = gSceneDepth.Load(int3(pixel, 0)).r;
    float2 ndc;
    ndc.x = (float(pixel.x) + 0.5f) / float(gFrameDim.x) * 2.f - 1.f;
    ndc.y = 1.f - (float(pixel.y) + 0.5f) / float(gFrameDim.y) * 2.f;
    float4 clip = float4(ndc, deviceZ, 1.f);
    float4 worldH = mul(gScene.camera.data.invViewProj, clip);
    return worldH.xyz / worldH.w;
}

[numthreads(8, 8, 1)]
void csMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFrameDim)) return;

    float deviceZ = gSceneDepth.Load(int3(pixel, 0)).r;

    // Background pixel: no ambient.
    if (deviceZ >= 1.f)
    {
        gAmbient[pixel] = float4(0.f, 0.f, 0.f, 1.f);
        return;
    }

    AmbientGBuffer g = decodeGBuffer(pixel);

    float3 worldPos = reconstructWorldPos(pixel);
    float3 N = normalize(g.worldNormal);
    float3 V = normalize(gScene.camera.getPosition() - worldPos);

    // UE combines the screen-space AO with the material AO baked into GBufferC.a
    // (AmbientCubemapComposite.usf -- GBuffer.GBufferAO * AmbientOcclusionTexture).
    float ao = g.gbufferAO * gAOMap.Load(int3(pixel, 0)).r;

    // Derived colors (UE SetGBufferForShading convention).
    float3 diffuseColor = g.baseColor * (1.f - g.metallic);
    float3 specularColor = lerp(0.08f * g.specular, g.baseColor, g.metallic);

    float NoV = saturate(dot(N, V));

    // Off-specular peak direction (UE AmbientCubemapComposite.usf).
    float a = g.roughness * g.roughness;
    float3 R0 = reflect(-V, N);
    float3 R = lerp(N, R0, (1.f - a) * (sqrt(1.f - a) + a));

    // Resolve the ambient IBL lookups. In fixed-color mode we use gAmbientColor
    // for all three (diffuse, specular, foliage subsurface), bypassing envMap.
    float3 diffuseLookup;
    float3 specularLookup;
    float3 subsurfLookup;
    if (gUseFixedColor != 0u)
    {
        diffuseLookup   = gAmbientColor;
        specularLookup  = gAmbientColor;
        subsurfLookup   = gAmbientColor;
    }
    else
    {
        // ---- Diffuse: cosine-convolved irradiance, roughness-independent ----
        // envMap.eval() bundles intensity+tint into its result, so the
        // irradiance equivalent must too -- otherwise diffuse and specular would
        // disagree about the environment's brightness.
        float3 envIntensity = gScene.envMap.getIntensity();
        float2 diffuseUV = gScene.envMap.worldToUv(N);
        diffuseLookup = gIrradiance.SampleLevel(gIrradianceSampler, diffuseUV, 0).rgb * envIntensity;

        // ---- Specular: roughness-derived mip ----
        float specMip = ComputeCubemapMipFromRoughness(g.roughness, gMipCount);
        specMip = clamp(specMip, 0.f, max(gMipCount - 1.f, 0.f));
        specularLookup = gScene.envMap.eval(R, specMip);

        // UE samples this along ScreenVector (AmbientCubemapComposite.usf) --
        // the camera->pixel direction, i.e. the sky *behind* the surface, which is
        // where light transmitted toward the viewer comes from. Not V.
        float3 screenVector = normalize(worldPos - gScene.camera.getPosition());
        // And UE *subtracts* 2.5 mips: transmission is more directional than
        // diffuse, so this lookup is sharper than the diffuse one, not blurrier.
        float subsurfMip = clamp(GetAbsoluteDiffuseMip() - 2.5f, 0.f, max(gMipCount - 1.f, 0.f));
        subsurfLookup = gScene.envMap.eval(screenVector, subsurfMip);
    }

    // ---- Diffuse IBL (surface normal direction) ----
    float3 nonSpecular = diffuseColor * diffuseLookup;

    // ---- Specular IBL ----
    float3 specular = specularLookup * EnvBRDF(specularColor, g.roughness, NoV);

    // ---- Subsurface tint ----
    // UE AmbientCubemapComposite.usf. UE gates this on
    // SUBSURFACE || PREINTEGRATED_SKIN; we widen it to every model whose
    // GBufferD actually holds a subsurface color, which adds TWOSIDED_FOLIAGE.
    if (hasSubsurfaceColorCustomData(g.shadingModelID))
    {
        float dependentSplit = 0.5f;
        // View-independent part: shared diffuse lookup, weighted by subsurface color.
        nonSpecular += diffuseLookup * g.subsurfaceColor * dependentSplit;
        // View-dependent part: sharper lookup along the screen vector.
        specular += subsurfLookup * g.subsurfaceColor * (ao * (1.f - dependentSplit));
    }

    // AO is applied here, *after* the subsurface terms are folded in, because
    // that is where UE applies it: a single `OutColor *= AmbientOcclusion` at
    // AmbientCubemapComposite.usf. The ordering is load-bearing -- it gives
    // the view-independent subsurface term one factor of AO and the
    // view-dependent term two (it already carries the inline `ao` above).
    //
    // The specular term is the exception: UE's reflection path scales it by
    // specular occlusion instead (ReflectionEnvironmentPixelShader.usf:209-212),
    // which folds AO and the view/roughness dependence into one factor. A plain
    // `ao` multiply would over-darken rough surfaces at grazing angles.
    nonSpecular *= ao;
    specular *= GetSpecularOcclusion(NoV, g.roughness * g.roughness, ao);

    // Final ambient = tint * intensity * (diffuse + specular).
    float3 ambient = gTint * gIntensity * (nonSpecular + specular);

    gAmbient[pixel] = float4(ambient, 1.f);
}
