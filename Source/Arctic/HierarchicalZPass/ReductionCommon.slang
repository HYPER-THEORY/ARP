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

/** Port of UnrealEngine's Private/ReductionCommon.ush — the subset HZB.usf uses.

    The bit-interleaved thread -> pixel mapping is what makes the recursive 2x2
    block reduction read its four source values from LDS at a stride of
    ReduceBankSize without bank conflicts, and — more importantly here — makes
    those four reads land on a genuine 2x2 pixel block of the tile rather than
    on four unrelated texels.
*/

uint signedRightShift(uint x, int bitshift)
{
    // UE's SignedRightShift: a POSITIVE shift is a left shift. The name
    // describes where the call site wants the bits to come from, not which
    // operator to use.
    if (bitshift > 0)
    {
        return x << uint(bitshift);
    }
    else if (bitshift < 0)
    {
        return x >> uint(-bitshift);
    }
    return x;
}

/** Returns the pixel position inside a 2^tileSizeLog2 x 2^tileSizeLog2 tile for
    the group thread whose index is sharedArrayId.

    The index bits are interleaved so that the four LDS entries a thread reads
    during one 2x2 reduction step — sharedArrayId + i * reduceBankSize for
    i = 0..3 — correspond to a 2x2 block of pixels. For tileSizeLog2 == 3 the
    mapping is x = 4*b0 + 2*b2 + b4 and y = 4*b1 + 2*b3 + b5, which also spreads
    consecutive thread indices across different LDS banks.
*/
uint2 initialTilePixelPositionForReduction2x2(uint tileSizeLog2, uint sharedArrayId)
{
    uint x = 0;
    uint y = 0;

    [unroll]
    for (uint i = 0; i < tileSizeLog2; i++)
    {
        const uint destBitId = tileSizeLog2 - 1 - i;
        const uint destBitMask = 1u << destBitId;
        x |= destBitMask & signedRightShift(sharedArrayId, int(destBitId) - int(i * 2 + 0));
        y |= destBitMask & signedRightShift(sharedArrayId, int(destBitId) - int(i * 2 + 1));
    }

    return uint2(x, y);
}
