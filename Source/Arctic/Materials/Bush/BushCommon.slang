/** Pure math helpers shared by the bush material's two shading paths.

    The rasterization path (`BushGBuffer.slang`) and the ray-tracing path
    (`BushMaterial.slang`) each assemble the surface independently — they differ
    structurally (screen-space derivatives, GBuffer encoding vs. BSDF
    construction), so they do not share an assembly function.

    What lives here is only stateless math that must not drift between the two:
    the UE ColorCorrect transform and the normal decode. If these two were
    copy-pasted, a fix to one path would silently leave the other wrong, and the
    RT and raster images would disagree for reasons that look like a lighting bug.

    No texture sampling, no frame logic, no material data access.
*/

/** UE ColorCorrect (ref/unrealengine ColorCorrect, Gamma=1, Offset=0).

    \param[in] color Input linear color.
    \param[in] saturation 0 = luma only, 1 = unchanged.
    \param[in] multiplier Per-channel gain.
    \param[in] brightness Scalar gain applied on top of the multiplier.
    \param[in] contrast Exponent about the 0.18 mid-gray pivot.
*/
float3 bushColorCorrection(float3 color, float saturation, float3 multiplier, float brightness, float contrast)
{
    // AP1_RGB2Y (ref/unrealengine/Shaders/Private/ACES/ACESCommon.ush).
    static const float3 AP1_RGB2Y = float3(0.2722287168, 0.6740817658, 0.0536895174);

    // Saturation first: lerp towards luma (UE ColorCorrect line 97).
    float luma = dot(color, AP1_RGB2Y);
    color = max(0, lerp(luma.xxx, color, saturation));

    // Contrast around the 0.18 mid-gray pivot (UE ColorCorrect line 98).
    color = pow(color * (1.0 / 0.18), contrast) * 0.18;

    // Gain: multiplier * brightness (UE ColorCorrect line 100, Gamma=1, Offset=0).
    color = color * (multiplier * brightness);
    return color;
}

/** Decode a tangent-space normal from the RG channels of the packed
    normal/specular/roughness map, reconstructing Z.

    The map is four-channel (B = dielectric specular, A = roughness), so the
    normal's Z channel is not stored and must be reconstructed from XY the same
    way a two-channel BC5 normal map would be.

    \param[in] normalXY The RG channels, in [0,1].
    \param[in] strength Scales the tangent-space XY before Z reconstruction is
               *not* redone — matching the reference shader, which scales XY
               after decode and leaves Z alone.
*/
float3 bushDecodeNormal(float2 normalXY, float strength)
{
    float2 xy = normalXY * 2.0 - 1.0;
    float z = sqrt(saturate(1.0 - dot(xy, xy)));
    return float3(xy * strength, z);
}
