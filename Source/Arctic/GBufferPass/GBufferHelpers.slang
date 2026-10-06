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
 #    contributors may be used to endorse or promote products derived from
 #    this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 # LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

/** Helpers for encoding UE-style GBuffer data.

    Ported from UnrealEngine's DeferredShadingCommon.ush (PC path). The GBuffer
    layout is byte-for-byte UE's, so a value written here means the same thing it
    would mean in UE and the shading code in DIPass/AmbientPass can be a direct
    port of UE's.

    GBufferPass produces Unlit, DefaultLit and TwoSidedFoliage; DIPass
    additionally decodes Subsurface. The remaining IDs reserve their values so the
    packed byte stays UE-compatible, but nothing produces or consumes them.
*/

// ShadingModelID values (from ShadingCommon.ush).
static const uint SHADINGMODELID_UNLIT               = 0;
static const uint SHADINGMODELID_DEFAULT_LIT         = 1;
static const uint SHADINGMODELID_SUBSURFACE          = 2;
static const uint SHADINGMODELID_PREINTEGRATED_SKIN  = 3;
static const uint SHADINGMODELID_CLEAR_COAT          = 4;
static const uint SHADINGMODELID_SUBSURFACE_PROFILE  = 5;
static const uint SHADINGMODELID_TWOSIDED_FOLIAGE    = 6;
static const uint SHADINGMODELID_HAIR                = 7;
static const uint SHADINGMODELID_CLOTH               = 8;
static const uint SHADINGMODELID_EYE                 = 9;

// Mask for the ShadingModelID bits in the packed byte (from ShadingCommon.ush).
static const uint SHADINGMODELID_MASK = 0xF;

// SelectiveOutputMask bits (from ShadingCommon.ush).
static const uint HAS_ANISOTROPY_MASK      = (1 << 4);
static const uint SKIP_PRECSHADOW_MASK     = (1 << 5);
static const uint ZERO_PRECSHADOW_MASK     = (1 << 6);
static const uint IS_FIRST_PERSON_MASK     = (1 << 6);

/** Encode normal for GBufferA.rgb (DeferredShadingCommon.ush).
    Default implementation is N*0.5+0.5 (octahedral variant is commented out in UE).
*/
float3 encodeNormal(float3 N)
{
    return N * 0.5f + 0.5f;
}

/** Decode normal from GBufferA.rgb (DeferredShadingCommon.ush).
*/
float3 decodeNormal(float3 N)
{
    return N * 2.f - 1.f;
}

/** Encode ShadingModelID and SelectiveOutputMask into a single byte stored as
    float in [0,1] (DeferredShadingCommon.ush).
*/
float encodeShadingModelIdAndSelectiveOutputMask(uint shadingModelId, uint selectiveOutputMask)
{
    uint value = (shadingModelId & SHADINGMODELID_MASK) | selectiveOutputMask;
    return (float)value / (float)0xFF;
}

/** Decode ShadingModelID from the packed byte (DeferredShadingCommon.ush).
*/
uint decodeShadingModelId(float packedChannel)
{
    return ((uint)round(packedChannel * (float)0xFF)) & SHADINGMODELID_MASK;
}

/** Decode SelectiveOutputMask from the packed byte (DeferredShadingCommon.ush).
*/
uint decodeSelectiveOutputMask(float packedChannel)
{
    return ((uint)round(packedChannel * (float)0xFF)) & ~SHADINGMODELID_MASK;
}

/** EncodeBaseColor is identity (DeferredShadingCommon.ush).
    Precision comes from the sRGB render target format.
*/
float3 encodeBaseColor(float3 baseColor)
{
    return baseColor;
}

/** Encode SubsurfaceColor for GBufferD.rgb (DeferredShadingCommon.ush).

    The sqrt is not cosmetic: GBufferD is an 8-bit UNORM target, and storing the
    square root spends more of the 256 codes on the darks where subsurface tints
    actually live. Must be paired with extractSubsurfaceColor on read.
*/
float3 encodeSubsurfaceColor(float3 subsurfaceColor)
{
    return sqrt(saturate(subsurfaceColor));
}

/** Decode SubsurfaceColor from GBufferD.rgb (DeferredShadingCommon.ush,
    UE's ExtractSubsurfaceColor). Inverse of encodeSubsurfaceColor.
*/
float3 extractSubsurfaceColor(float3 encoded)
{
    return encoded * encoded;
}

/** UE IsSubsurfaceModel (DeferredShadingCommon.ush).

    Note this is the *lighting* classification: it includes models whose
    CustomData layout is not (SubsurfaceColor, Opacity) — SUBSURFACE_PROFILE
    stores a profile id, HAIR stores an octahedral normal. Use
    hasSubsurfaceColorCustomData when the question is "does GBufferD.rgb hold a
    subsurface color".
*/
bool isSubsurfaceModel(uint shadingModelId)
{
    return shadingModelId == SHADINGMODELID_SUBSURFACE
        || shadingModelId == SHADINGMODELID_PREINTEGRATED_SKIN
        || shadingModelId == SHADINGMODELID_SUBSURFACE_PROFILE
        || shadingModelId == SHADINGMODELID_TWOSIDED_FOLIAGE
        || shadingModelId == SHADINGMODELID_HAIR
        || shadingModelId == SHADINGMODELID_EYE;
}

/** True for the shading models whose GBufferD is (encodeSubsurfaceColor(c), Opacity).

    These are exactly the three branches in UE's SetGBufferForShadingModel that
    write `EncodeSubsurfaceColor(SubsurfaceColor)` + `Opacity`
    (ShadingModelsMaterial.ush). CLOTH also writes an encoded color
    but its .a is a cloth mask, not an opacity, so it is deliberately excluded.
*/
bool hasSubsurfaceColorCustomData(uint shadingModelId)
{
    return shadingModelId == SHADINGMODELID_SUBSURFACE
        || shadingModelId == SHADINGMODELID_PREINTEGRATED_SKIN
        || shadingModelId == SHADINGMODELID_TWOSIDED_FOLIAGE;
}

/** Debug palette for a ShadingModelID (ShadingCommon.ush).
*/
float3 getShadingModelColor(uint shadingModelId)
{
    switch (shadingModelId)
    {
        case SHADINGMODELID_UNLIT:              return float3(0.1f, 0.1f, 0.2f); // Dark blue
        case SHADINGMODELID_DEFAULT_LIT:        return float3(0.1f, 1.0f, 0.1f); // Green
        case SHADINGMODELID_SUBSURFACE:         return float3(1.0f, 0.1f, 0.1f); // Red
        case SHADINGMODELID_PREINTEGRATED_SKIN: return float3(0.6f, 0.4f, 0.1f); // Brown
        case SHADINGMODELID_CLEAR_COAT:         return float3(0.1f, 0.4f, 0.4f);
        case SHADINGMODELID_SUBSURFACE_PROFILE: return float3(0.2f, 0.6f, 0.5f); // Cyan
        case SHADINGMODELID_TWOSIDED_FOLIAGE:   return float3(0.2f, 0.2f, 0.8f); // Blue
        case SHADINGMODELID_HAIR:               return float3(0.6f, 0.1f, 0.5f);
        case SHADINGMODELID_CLOTH:              return float3(0.7f, 1.0f, 1.0f);
        case SHADINGMODELID_EYE:                return float3(0.3f, 1.0f, 1.0f);
        default:                                return float3(1.0f, 1.0f, 1.0f); // White
    }
}
