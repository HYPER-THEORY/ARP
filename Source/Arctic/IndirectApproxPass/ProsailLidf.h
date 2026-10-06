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
#pragma once

/** Scene-independent PROSAIL precomputation for IndirectApproxPass.

    Everything in here is a function of the leaf inclination distribution alone,
    so it is rebuilt only when lidfa / lidfb / typelidf change and never per
    frame. The rest of 4SAIL -- every quantity that depends on LAI, depth, the
    sun or the leaf optics -- is evaluated in FourSail.slang.

    Ported from ref/prosail/prosail/FourSAIL.py. Cross-checked against that file
    by tools/check_prosail_port.py, which loads PROSAIL's own source and diffs
    the values this produces.

    Why these three and nothing else: a flux-only solve needs exactly two
    angular quantities out of 4SAIL, bf and ks. bf depends only on the LIDF, so
    it is exact to precompute. ks = G(tts)/cos(tts) additionally depends on the
    solar zenith, so what is tabulated is G, which is bounded; the shader does
    the division. Every other angular quantity in 4SAIL (ko, sob, sof, and the
    vb/vf/w set) feeds only view-direction radiance, which this pass does not
    compute -- the leaf BSDF is applied per pixel from the GBuffer instead.
*/

namespace ProsailLidf
{

/// Leaf inclination bins. 18 is PROSAIL's own n_elements, i.e. 5 degree bins.
constexpr int kNumLidfBins = 18;

/// Entries in the G(tts) table: one per degree of solar zenith over [0, 90].
/// Linear interpolation at this spacing costs at most 1.5e-4 relative error
/// against the exact sum, measured by tools/check_prosail_port.py.
constexpr int kGLutSize = 91;

/** 18-bin LIDF via PROSAIL's Verhoef two-parameter bimodal form (typelidf = 1).

    Requires |lidfa| + |lidfb| < 1. Canonical pairs: spherical (-0.35, -0.15),
    planophile (1, 0), erectophile (-1, 0), plagiophile (0, -1),
    extremophile (0, 1), uniform (0, 0).
    Ported from FourSAIL.py verhoef_bimodal().
*/
void computeLidfVerhoef(float lidfa, float lidfb, float lidf[kNumLidfBins]);

/** 18-bin LIDF via PROSAIL's Campbell ellipsoidal form (typelidf = 2).

    lidfa is the mean leaf inclination in degrees; ~57.3 is spherical.
    Ported from FourSAIL.py campbell().
*/
void computeLidfCampbell(float lidfa, float lidf[kNumLidfBins]);

/** PROSAIL's bf accumulator: sum_i lidf_i * cos^2(litab_i).

    This is the `bf += bfli * lidf[i]` line of weighted_sum_over_lidf(). It is
    the only output of that function with no dependence on tts, tto or psi,
    which is why hoisting it out of the per-frame path is exact rather than an
    approximation. Drives ddb/ddf (diffuse-diffuse) and sdb/sdf (beam-diffuse).
*/
float computeBf(const float lidf[kNumLidfBins]);

/** Ross G function against solar zenith, tabulated at 1 degree steps.

    gLut[d] = sum_i lidf_i * chi_s(d degrees, litab_i), with chi_s the solar
    interception function from volscatt(). PROSAIL forms ks = chi_s/cts inside
    the same loop; splitting it here and dividing in the shader keeps the table
    finite, since ks diverges as the sun approaches the horizon while G does not.
*/
void computeGLut(const float lidf[kNumLidfBins], float gLut[kGLutSize]);

/** Solar interception function chi_s for one leaf inclination.

    The first return value of FourSAIL.py's volscatt(), which for chi_s depends
    only on tts and ttl -- the view arguments tto and psi affect chi_o, frho and
    ftau but never chi_s. Both angles in degrees. Exposed for the port check.
*/
float volscattChiS(float tts, float ttl);

/// Centre of inclination bin i, in degrees. PROSAIL's
/// litab = arange(n)*step + step/2 with step = 90/n.
float litabCentre(int bin);

} // namespace ProsailLidf
