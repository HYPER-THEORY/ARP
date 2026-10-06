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
#include "ProsailLidf.h"

#include <algorithm>
#include <cmath>

namespace ProsailLidf
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

inline double rad(double deg)
{
    return deg * kPi / 180.0;
}

/// 90 / n, PROSAIL's `step`.
constexpr double kStep = 90.0 / double(kNumLidfBins);
} // namespace

float litabCentre(int bin)
{
    return float(double(bin) * kStep + kStep * 0.5);
}

// -----------------------------------------------------------------------------
// Leaf inclination distributions
// -----------------------------------------------------------------------------

void computeLidfVerhoef(float lidfa, float lidfb, float lidf[kNumLidfBins])
{
    const double a = double(lidfa);
    const double b = double(lidfb);

    double freq = 1.0;
    double tmp[kNumLidfBins];

    // PROSAIL walks the inclinations from high to low and reverses at the end.
    for (int i = 0; i < kNumLidfBins; ++i)
    {
        const double tl1 = rad(double(kNumLidfBins - 1 - i) * kStep);
        double f;
        if (a > 1.0)
        {
            f = 1.0 - std::cos(tl1);
        }
        else
        {
            const double eps = 1e-8;
            double delx = 1.0;
            double x = 2.0 * tl1;
            const double p = x;
            double y = 0.0;
            // Fixed-point iteration for Verhoef's cumulative distribution. It
            // converges in roughly 20 steps; the cap only exists so that a
            // pathological (a,b) outside |a| + |b| < 1 cannot hang the caller.
            for (int it = 0; it < 100 && delx >= eps; ++it)
            {
                y = a * std::sin(x) + 0.5 * b * std::sin(2.0 * x);
                const double dx = 0.5 * (y - x + p);
                x += dx;
                delx = std::abs(dx);
            }
            f = (2.0 * y + p) / kPi;
        }
        freq -= f;
        tmp[i] = freq;
        freq = f;
    }

    for (int i = 0; i < kNumLidfBins; ++i) lidf[i] = float(tmp[kNumLidfBins - 1 - i]);
}

void computeLidfCampbell(float lidfa, float lidf[kNumLidfBins])
{
    const double alpha = double(lidfa);
    const double excent =
        std::exp(-1.6184e-5 * alpha * alpha * alpha + 2.1145e-3 * alpha * alpha - 1.2390e-1 * alpha + 3.2491);

    // tan(90 deg) is infinite and the limit of x is 0; guard explicitly rather
    // than rely on an infinity propagating cleanly through the sqrt.
    auto xOf = [&](double tl)
    {
        if (tl >= 0.5 * kPi - 1e-9) return 0.0;
        const double t = std::tan(tl);
        return excent / std::sqrt(1.0 + excent * excent * t * t);
    };

    double freq[kNumLidfBins];
    double sum0 = 0.0;

    // Note these are bin EDGES, unlike litab which is bin centres. PROSAIL
    // integrates the ellipsoidal density across each bin.
    for (int i = 0; i < kNumLidfBins; ++i)
    {
        const double tl1 = rad(double(i) * kStep);
        const double tl2 = rad(double(i + 1) * kStep);
        const double x1 = xOf(tl1);
        const double x2 = xOf(tl2);

        if (std::abs(excent - 1.0) < 1e-9)
        {
            freq[i] = std::abs(std::cos(tl1) - std::cos(tl2));
        }
        else
        {
            const double alph = excent / std::sqrt(std::abs(1.0 - excent * excent));
            const double alph2 = alph * alph;
            const double x12 = x1 * x1;
            const double x22 = x2 * x2;
            if (excent > 1.0)
            {
                const double alpx1 = std::sqrt(alph2 + x12);
                const double alpx2 = std::sqrt(alph2 + x22);
                const double dum = x1 * alpx1 + alph2 * std::log(x1 + alpx1);
                freq[i] = std::abs(dum - (x2 * alpx2 + alph2 * std::log(x2 + alpx2)));
            }
            else
            {
                const double almx1 = std::sqrt(std::max(0.0, alph2 - x12));
                const double almx2 = std::sqrt(std::max(0.0, alph2 - x22));
                const double dum = x1 * almx1 + alph2 * std::asin(std::min(1.0, x1 / alph));
                freq[i] = std::abs(dum - (x2 * almx2 + alph2 * std::asin(std::min(1.0, x2 / alph))));
            }
        }
        sum0 += freq[i];
    }

    for (int i = 0; i < kNumLidfBins; ++i) lidf[i] = float(freq[i] / std::max(sum0, 1e-12));
}

// -----------------------------------------------------------------------------
// Derived angular quantities
// -----------------------------------------------------------------------------

float computeBf(const float lidf[kNumLidfBins])
{
    double bf = 0.0;
    for (int i = 0; i < kNumLidfBins; ++i)
    {
        const double cttl = std::cos(rad(double(litabCentre(i))));
        bf += cttl * cttl * double(lidf[i]);
    }
    return float(bf);
}

float volscattChiS(float tts, float ttl)
{
    const double cts = std::cos(rad(double(tts)));
    const double sts = std::sin(rad(double(tts)));
    const double cttl = std::cos(rad(double(ttl)));
    const double sttl = std::sin(rad(double(ttl)));
    const double cs = cttl * cts;
    const double ss = sttl * sts;

    double cosbts = 5.0;
    if (std::abs(ss) > 1e-6) cosbts = -cs / ss;

    // |cosbts| >= 1 means the leaf normal cone never crosses the solar
    // direction, so the whole leaf is lit from one side and bts collapses to pi.
    const double bts = (std::abs(cosbts) < 1.0) ? std::acos(cosbts) : kPi;

    return float(2.0 / kPi * ((bts - kPi * 0.5) * cs + std::sin(bts) * ss));
}

void computeGLut(const float lidf[kNumLidfBins], float gLut[kGLutSize])
{
    for (int d = 0; d < kGLutSize; ++d)
    {
        const float tts = float(d);
        double g = 0.0;
        for (int i = 0; i < kNumLidfBins; ++i)
        {
            g += double(volscattChiS(tts, litabCentre(i))) * double(lidf[i]);
        }
        gLut[d] = float(g);
    }
}

} // namespace ProsailLidf
