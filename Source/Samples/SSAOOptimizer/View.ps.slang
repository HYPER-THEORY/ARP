/// Copyright (C) 2026-2027 ARP (Auto Render Pipeline). All Rights Reserved.

/** Background view pixel shader for the optimizer window.

    One pass serves all three backgrounds the GUI offers, selected by `viewMode`:

        0  AO         - the current primal AO, as greyscale
        1  Reference  - the frozen target AO, as greyscale
        2  Error      - |AO - reference|, white at zero to red at maxError

    Greyscale rather than a blit
    ---------------------------
    The AO image is R32Float and carries a single channel. Blitting it straight
    into the RGBA8 frame buffer writes r only, leaving g and b at zero, which
    reads as pure red — the same colour the error map uses for its worst value,
    so a blit makes the two views ambiguous. This pass replicates the value
    across rgb instead, which is what makes the AO view read as grey.

    The AO is ASSAOPass' aoMap texture, which every graph run refreshes, so the
    view follows the displayed iteration; the torch-facing AO buffer is only
    refreshed when the script calls forward(). The reference has no texture and
    is read from the buffer captureTarget() fills.
*/

Texture2D<float>          gAOMap;        // Current AO (ASSAOPass forward).
RWStructuredBuffer<float> gAOReference;  // Frozen target AO.

cbuffer CB
{
    uint2 frameDim;
    uint  viewMode;     // 0 = AO, 1 = reference, 2 = error.
    float invMaxError;  // 1 / maxError; error mode saturates at this.
};

float4 main(float2 texC : TEXCOORD) : SV_TARGET
{
    // Map tex coords to a pixel index in row-major order.
    uint2 pixelPos = uint2(texC * float2(frameDim));
    pixelPos = clamp(pixelPos, uint2(0, 0), frameDim - uint2(1, 1));
    uint index = pixelPos.y * frameDim.x + pixelPos.x;

    float ao = gAOMap[pixelPos];

    if (viewMode == 0)
        return float4(ao, ao, ao, 1.0);

    if (viewMode == 1)
    {
        float ref = gAOReference[index];
        return float4(ref, ref, ref, 1.0);
    }

    float t = saturate(abs(ao - gAOReference[index]) * invMaxError);

    // White (1,1,1) at t=0 to red (1,0,0) at t=1.
    return float4(1.0, 1.0 - t, 1.0 - t, 1.0);
}
