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

/** 4SAIL canopy transport, evaluated inside the canopy.

    A direct port of ref/prosail/prosail/FourSAIL.py. Verhoef's four streams are
    Es (direct solar), E- (diffuse down), E+ (diffuse up) and Eo (radiance toward
    the sensor); this file solves the first three. Eo and everything feeding it
    -- ko, sob, sof, vb, vf, w and the hotspot -- is deliberately absent: a
    deferred renderer wants the field arriving at a shading point, then applies
    the material BSDF per pixel from the GBuffer. Directional single-scattering
    effects such as the hot spot are DIPass' business.

    HOW THE INTERIOR IS REACHED. PROSAIL publishes reflectance and transmittance
    for a whole slab, not fluxes at depth. Those fluxes come from composing
    PROSAIL's OWN layer operators at partial thickness: split the canopy at tau
    into an upper layer A of thickness tau and a lower layer B of thickness
    (LAI - tau), collapse B plus the soil into an effective lower boundary using
    PROSAIL's rddt/rsdt formulas, then solve the 2x2 interface system between A
    and that boundary.

        E-(tau) = [tdd_A Ed0 + tsd_A Es0 + rdd_A R_sd Es(tau)] / (1 - rdd_A R_dd)
        E+(tau) = R_dd E-(tau) + R_sd Es(tau)

    This is not an approximation layered on top of SAIL, it is SAIL evaluated at
    depth. Checked three ways in tools/check_sail_interior.py, over LAI 0.5-12,
    planophile through erectophile, and sun-only / sky-only / black-soil limits:
      * against a direct numerical BVP solve of SAIL's own ODE system: ~1e-10
      * at tau = 0, against PROSAIL's published rddt and rsdt: 0.000e+00
      * at tau = LAI, against the soil boundary condition: ~1e-16

    ACCURACY. PROSAIL's diffuse closure is two-stream, so interior fluxes carry
    its angular error: measured against a 32-ordinate discrete-ordinates solver,
    13.8% median and 36% worst case on the collided down-flux. That is a property
    of the published model, not of this port, and it is the deliberate trade for
    having no CPU-side solve at all. tools/check_sail_vs_do.py measures it.

    BOUNCE ACCOUNTING -- the part that is easy to get wrong:
      sun -> leaf -> eye            1 bounce, DIPass
      sky -> leaf -> eye            1 bounce, AmbientPass
      sun -> leaf -> leaf -> eye    2 bounces, here
      sky -> leaf -> leaf -> eye    2 bounces, here
      sun -> ground -> leaf -> eye  2 bounces, here
    So the fluxes returned here have already scattered at least once, and the
    shading point's own scattering is applied on top by gatherTwoSidedFoliage().
    Two consequences: the uncollided sky is subtracted, or AmbientPass would be
    double counted; and the direct beam Es is only ever a source term, never
    added to E-, so DIPass cannot leak in either.

    WHAT THE CPU SUPPLIES. Only the leaf inclination distribution's three
    derived quantities: bf and a G(tts) table (see ProsailLidf.h). Everything
    else below is computed here, per pixel, from raw uploaded parameters.
*/

static const float kPi = 3.14159265358979323846f;
static const int kGLutSize = 91; ///< Must match ProsailLidf::kGLutSize.

cbuffer FieldCB
{
    // ---- Slab geometry and the vertical leaf area profile ----
    float gZMin;  ///< Canopy bottom, also the soil plane [m].
    float gZMax;  ///< Canopy top [m].
    float gLai;   ///< Column LAI where the lateral density is 1. PROSAIL's `lai`.
    float gSlope; ///< Vertical profile tilt in [-1,1]; > 0 is top heavy.

    // ---- Lateral density modulation ----
    float gTileSize;        ///< Metres per XYField tile.
    float gDensityContrast; ///< 0 = laterally uniform.
    float gDensityMax;      ///< Upper clamp on the density multiplier.

    // ---- Leaf optics. PROSAIL's rho and tau, per channel. ----
    float3 gRLeaf;
    float3 gTLeaf;

    // ---- Boundary conditions ----
    float3 gSunDirW;     ///< Direction the beam TRAVELS, so it points downward.
    float3 gSunIntensity; ///< Directional light intensity, per channel.
    float3 gSkyRadiance;  ///< Uniform sky radiance at the canopy top.
    float3 gSoilAlbedo;   ///< Lambertian soil albedo, PROSAIL's rsoil0.

    // ---- Precomputed from the LIDF on the CPU (scene independent) ----
    float gBf;              ///< sum_i lidf_i cos^2(litab_i).
    float gGLut[kGLutSize]; ///< G(tts) at 1 degree steps over [0, 90].
};

Texture2D<float4> gXYField;
SamplerState gXYSampler;

/// Floor on cos(solar zenith). ks = G/cos(tts) diverges at the horizon, where
/// there is no meaningful direct beam anyway.
static const float kMinMuS = 0.01f;

// -----------------------------------------------------------------------------
// Field geometry
// -----------------------------------------------------------------------------

/** Accumulated LAI from the canopy top down to z, at lateral density 1.

    The implied leaf area density is u(zeta) = LAI(1 - s + 2 s zeta), which
    integrates to exactly LAI for any slope s, so LAI and slope stay orthogonal.
    Note this is not a departure from PROSAIL: SAIL's equations are written in
    cumulative-LAI coordinates, so any depth-varying leaf area density is handled
    exactly as long as rho, tau and the LIDF are depth constant, which they are.
*/
float laiAbove(float z)
{
    float zeta = saturate((z - gZMin) / max(gZMax - gZMin, 1e-4f));
    return gLai * (1.f - zeta) * (1.f + gSlope * zeta);
}

/// Lateral density multiplier, normalised so that mean(D) == 1 regardless of the
/// texture's overall brightness, which keeps contrast and total LAI independent.
float xyDensityScale(float2 xy)
{
    float t = gXYField.SampleLevel(gXYSampler, xy / max(gTileSize, 1e-3f), 0.f).r;
    float d = 1.f + gDensityContrast * (t * 2.f - 1.f);
    return clamp(d, 0.f, max(gDensityMax, 1.f));
}

// -----------------------------------------------------------------------------
// 4SAIL coefficients
// -----------------------------------------------------------------------------

/// Ross G for the solar direction, by linear interpolation of the CPU table.
float sampleG(float ttsDeg)
{
    float t = clamp(ttsDeg, 0.f, 90.f);
    int i0 = int(t);
    int i1 = min(i0 + 1, kGLutSize - 1);
    return lerp(gGLut[i0], gGLut[i1], t - float(i0));
}

/// Everything in FourSAIL that depends on the material and the sun but not on
/// LAI or depth. One per colour channel; the angular part is shared.
struct SailCoeffs
{
    float ks;   ///< Beam extinction per unit LAI, G(tts)/cos(tts).
    float att;  ///< 1 - sigf, the diffuse attenuation.
    float m;    ///< sqrt(att^2 - sigb^2), the diffuse decay rate.
    float sigb; ///< Backward diffuse-diffuse scattering.
    float sb;   ///< Beam to upward diffuse.
    float sf;   ///< Beam to downward diffuse.
    float rinf; ///< (att - m)/sigb, the semi-infinite reflectance.
};

/** Build the coefficients for one channel. Mirrors FourSAIL.foursail()'s
    preamble exactly, with ks and bf coming from the CPU-side LIDF work.
*/
SailCoeffs makeCoeffs(float rho, float tau, float ks)
{
    // Geometric factors to be used later with rho and tau.
    float sdb = 0.5f * (ks + gBf);
    float sdf = 0.5f * (ks - gBf);
    float ddb = 0.5f * (1.f + gBf);
    float ddf = 0.5f * (1.f - gBf);

    float sigb = ddb * rho + ddf * tau;
    float sigf = ddf * rho + ddb * tau;
    // PROSAIL clamps both away from zero; sigb in particular divides into rinf.
    sigb = max(sigb, 1e-36f);
    sigf = max(sigf, 1e-36f);

    SailCoeffs c;
    c.ks = ks;
    c.att = 1.f - sigf;
    c.m = sqrt(max(c.att * c.att - sigb * sigb, 0.f));
    c.sigb = sigb;
    c.sb = sdb * rho + sdf * tau;
    c.sf = sdf * rho + sdb * tau;
    c.rinf = (c.att - c.m) / sigb;
    return c;
}

/// PROSAIL's J1, with its singularity avoided when k and l nearly coincide.
float Jfunc1(float k, float l, float t)
{
    float del = (k - l) * t;
    if (abs(del) > 1e-3f) return (exp(-l * t) - exp(-k * t)) / (k - l);
    return 0.5f * t * (exp(-k * t) + exp(-l * t)) * (1.f - del * del / 12.f);
}

/// PROSAIL's J2.
float Jfunc2(float k, float l, float t)
{
    return (1.f - exp(-(k + l) * t)) / (k + l);
}

/// Transmittance and reflectance operators for a slab of thickness L.
struct LayerOps
{
    float tss; ///< Direct beam transmittance.
    float tdd; ///< Diffuse-diffuse transmittance.
    float rdd; ///< Diffuse-diffuse reflectance.
    float tsd; ///< Direct to downward-diffuse transmittance.
    float rsd; ///< Direct to upward-diffuse reflectance.
};

/** PROSAIL's tss/tdd/rdd/tsd/rsd, verbatim, for an arbitrary thickness L.

    In FourSAIL these are only ever evaluated at L = lai. Evaluating them at
    partial thickness is what makes the interior solve possible, and is exact:
    the derivation never assumes anything about L.
*/
LayerOps layerOps(SailCoeffs c, float L)
{
    LayerOps o;
    if (L <= 0.f)
    {
        // Vanishing layer: perfectly transmitting, non-reflecting. Handled
        // explicitly because denom -> 0 and rdd/tdd become 0/0.
        o.tss = 1.f;
        o.tdd = 1.f;
        o.rdd = 0.f;
        o.tsd = 0.f;
        o.rsd = 0.f;
        return o;
    }

    float e1 = exp(-c.m * L);
    float e2 = e1 * e1;
    float rinf2 = c.rinf * c.rinf;
    float re = c.rinf * e1;
    float denom = 1.f - rinf2 * e2;

    float J1ks = Jfunc1(c.ks, c.m, L);
    float J2ks = Jfunc2(c.ks, c.m, L);
    float Pss = (c.sf + c.sb * c.rinf) * J1ks;
    float Qss = (c.sf * c.rinf + c.sb) * J2ks;

    o.tss = exp(-c.ks * L);
    o.tdd = (1.f - rinf2) * e1 / denom;
    o.rdd = c.rinf * (1.f - e2) / denom;
    o.tsd = (Pss - re * Qss) / denom;
    o.rsd = (Qss - re * Pss) / denom;
    return o;
}

// -----------------------------------------------------------------------------
// Interior solve
// -----------------------------------------------------------------------------

/// Hemispherical fluxes at the query point. Both have already scattered at
/// least once, so applying a leaf BSDF on top yields >= 2 bounces.
struct CanopyFluxes
{
    float3 eDown; ///< Downward diffuse flux, uncollided sky removed.
    float3 eUp;   ///< Upward diffuse flux. Collided by construction: reaching an
                  ///< upward direction requires a leaf or soil interaction.
};

/** Diffuse fluxes at optical depth tau, for one channel.

    \param[in] Es0 Direct solar irradiance on a horizontal plane at the top.
    \param[in] Ed0 Diffuse sky irradiance at the top (pi * sky radiance).
*/
void solveChannel(
    SailCoeffs c, float lai, float tau, float rsoil, float Es0, float Ed0, out float eDown, out float eUp
)
{
    LayerOps A = layerOps(c, tau);            // canopy top down to the query point
    LayerOps B = layerOps(c, lai - tau);      // query point down to the soil

    // Layer B closed from below by a Lambertian soil. These two lines are
    // PROSAIL's rddt and rsdt with lai replaced by (lai - tau).
    float dn = max(1.f - rsoil * B.rdd, 1e-36f);
    float R_dd = B.rdd + B.tdd * rsoil * B.tdd / dn;
    float R_sd = B.rsd + (B.tsd + B.tss) * rsoil * B.tdd / dn;

    // Beam reaching the query point, then the 2x2 interface system between A
    // and the effective lower boundary.
    float esTau = Es0 * A.tss;
    float num = A.tdd * Ed0 + A.tsd * Es0 + A.rdd * R_sd * esTau;
    float eDnTotal = num / max(1.f - A.rdd * R_dd, 1e-36f);
    float eUpTotal = R_dd * eDnTotal + R_sd * esTau;

    // Uncollided sky, in SAIL's own normalisation. With rho = tau = 0 the model
    // gives att = m = 1, so the never-scattered part of the downward diffuse
    // flux is exactly Ed0 exp(-tau). It must be subtracted in the SAME
    // discretisation as the total: in the blue channel the collided part is a
    // small difference of two much larger numbers, and mixing two forms for the
    // uncollided term would dominate the result.
    float eDnUncollided = Ed0 * exp(-tau);

    eDown = max(eDnTotal - eDnUncollided, 0.f);
    eUp = max(eUpTotal, 0.f);
}

/** Solve the canopy field at a shading point.

    \param[in] worldPos Shading point in world space. World up is +Z.
*/
CanopyFluxes solveCanopyField(float3 worldPos)
{
    CanopyFluxes f;
    f.eDown = float3(0.f);
    f.eUp = float3(0.f);

    // Solar zenith. The beam travels downward, so |dirW.z| is its cosine.
    float muS = max(abs(gSunDirW.z), kMinMuS);
    float ttsDeg = degrees(acos(clamp(muS, 0.f, 1.f)));
    float ks = sampleG(ttsDeg) / muS;

    // Irradiance on a horizontal plane. The cosine is signed here, unlike in ks
    // above: abs() would mirror a set sun back above the horizon and light the
    // canopy at night. ks may keep the mirrored value because every term it feeds
    // (tsd Es0, esTau, R_sd esTau) is multiplied by Es0 and vanishes with it.
    float3 Es0 = gSunIntensity * max(-gSunDirW.z, 0.f);
    float3 Ed0 = kPi * gSkyRadiance;    // hemispherical sky irradiance

    float density = xyDensityScale(worldPos.xy);
    float lai = gLai * density;
    float tau = clamp(laiAbove(worldPos.z) * density, 0.f, lai);

    if (lai < 1e-3f)
    {
        // Degenerate column: nothing to scatter in, so the soil simply reflects
        // whatever arrives. Handled explicitly because the layer operators go
        // singular as the slab thickness vanishes.
        f.eUp = gSoilAlbedo * (Ed0 + Es0);
        return f;
    }

    [ForceUnroll]
    for (uint ch = 0; ch < 3; ++ch)
    {
        SailCoeffs c = makeCoeffs(gRLeaf[ch], gTLeaf[ch], ks);
        float eDown, eUp;
        solveChannel(c, lai, tau, gSoilAlbedo[ch], Es0[ch], Ed0[ch], eDown, eUp);
        f.eDown[ch] = eDown;
        f.eUp[ch] = eUp;
    }
    return f;
}

// -----------------------------------------------------------------------------
// Gather
// -----------------------------------------------------------------------------

/** Irradiance on a surface of normal n, from a hemisphere-isotropic field.

    A downward flux E arriving isotropically from the upper hemisphere carries
    radiance E/pi, and the irradiance it deposits on a surface of normal n is
    E*(1 + n.z)/2. Check the ends: n = +Z gives E, n horizontal gives E/2, and
    n = -Z gives 0, since a downward-facing surface sees none of the upper
    hemisphere. The upward flux is the mirror image, with (1 - n.z)/2.

    Multiple scattering is close to isotropic within each hemisphere, which is
    why two hemispheres suffice and why the field carries no azimuthal or view
    dependence.
*/
float3 hemisphereIrradiance(CanopyFluxes f, float3 n)
{
    return f.eDown * (0.5f * (1.f + n.z)) + f.eUp * (0.5f * (1.f - n.z));
}

/** Indirect radiance leaving a two-sided foliage surface.

    \param[in] n Shading normal.
    \param[in] rho Leaf reflectance at this pixel, i.e. GBuffer base colour.
    \param[in] tauLeaf Leaf transmittance at this pixel, i.e. GBuffer subsurface colour.

    The per-pixel rho and tauLeaf come from the GBuffer rather than from the
    field's global gRLeaf/gTLeaf so that texture variation shows through. The
    field solve uses the global pair because it describes a bulk volume property.
*/
float3 gatherTwoSidedFoliage(CanopyFluxes f, float3 n, float3 rho, float3 tauLeaf)
{
    float3 eFront = hemisphereIrradiance(f, n);
    float3 eBack = hemisphereIrradiance(f, -n);
    return (rho * eFront + tauLeaf * eBack) / kPi;
}
