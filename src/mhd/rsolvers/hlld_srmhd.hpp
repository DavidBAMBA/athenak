#ifndef MHD_RSOLVERS_HLLD_SRMHD_HPP_
#define MHD_RSOLVERS_HLLD_SRMHD_HPP_
//========================================================================================
// Athena++ (Kokkos version) astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlld_srmhd.hpp
//! \brief HLLD Riemann solver for special relativistic MHD.  Per-face implementation
//! for the split-kernel path (see hlle_srmhd.hpp).
//!
//! Implements the 5-wave HLLD solver of Mignone, Ugliano & Bodo (2009), MNRAS 393, 1141
//! ("MUB").  Ported from the Athena++ implementation in
//! src/hydro/rsolvers/mhd/hlld_rel.cpp, restricted to the special-relativistic path (no
//! metric transformations), and restructured for Kokkos: the residual evaluation that
//! Athena++ writes out three times (for SIMD reasons) is factored here into a single
//! device lambda.
//!
//! Algorithm outline, with the MUB equation numbers used in the comments below:
//!   1. Build the L and R conserved states and fluxes                        (MUB 8, 15)
//!   2. Form the jump vectors R = lambda*U - F across each fast wave         (MUB 12)
//!   3. Form the HLL average state and flux                                  (MB 29, 31)
//!   4. Invert the HLL state to get a total pressure estimate p_hll, using the
//!      Mignone & McKinney (2007) energy residual with Newton-Raphson        (MM A1-A18)
//!   5. Take an initial guess for the total pressure in the fan              (MUB 53, 55)
//!   6. Secant-iterate on the residual (MUB 48) to find the contact pressure
//!   7. Assemble the aL, aR and c states and their fluxes             (MUB 21-34, 45-52)
//!   8. Pick the state the interface sits in; fall back to HLLE on any failure
//!
//! The fallback to HLLE is essential: the pressure iteration can fail in strongly
//! magnetized or near-degenerate states, and MUB itself recommends reverting there.
//!
//! ROBUSTNESS (added for 3D KSI): the original Athena++ port only reverted to HLLE on
//! non-finite values.  That is not enough: with a strong NORMAL field through low-beta
//! gas (the x3 sweep of the 3D KSI problem, bx = B3 ~ 3.2, beta ~ 2e-5) the secant can
//! stall, or converge to a wrong root, and still return a finite positive ptot_c whose
//! fan is unphysical -- superluminal a/c-state velocities, broken wave ordering, or
//! negative densities/energies -- which the isfinite() guard never catches.  Measured
//! in a randomized ensemble of that regime, ~20% of fan-region faces produced such
//! silent garbage, which in the 3D KSI run pumps grid noise until rho O(1) errors.
//! Three changes (validated against the MUB 1-4 shock tubes, see build_test_hlld/):
//!   1. The secant runs up to 12 iterations with an early convergence break instead
//!      of exactly 4 blind ones.  Hard-jump faces (MUB tubes) converge superlinearly
//!      from |res| ~ 1e-5 at iter 4 to ~1e-12 by iter 7, and ~40% of the faces that
//!      used to emit garbage now converge to a physical fan.  Smooth faces break after
//!      2 residual evaluations instead of always paying for 6.
//!   2. Convergence gate: if the final |residual| > tol_gate the fan is not trusted.
//!   3. Physicality gates on the assembled fan: subluminal aL/aR/c velocities, wave
//!      ordering lambda_L <= lambda_aL <= vx_c <= lambda_aR <= lambda_R, and positive
//!      fan densities/energies.  Any violation (or NaN, since the checks are written
//!      NaN-safe) falls back to HLLE on that face only.
//!
//! NOTE ON COST: this solver is far heavier than HLLE (an inner Newton solve plus a
//! secant loop per interface) and uses many registers, so expect a substantial slowdown
//! on GPUs relative to hlle.  It is worth it when numerical resistivity matters -- e.g.
//! reconnection-rate studies -- because HLLE badly over-diffuses the rotational and
//! contact discontinuities that HLLD resolves.

#include <math.h>

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

namespace mhd {

//----------------------------------------------------------------------------------------
//! \struct HLLDFan
//! \brief The intermediate-state quantities that depend on the trial total pressure.
//! Recomputed by SRHLLDResidual() at each secant iterate and reused afterwards to build
//! the aL/aR/c states, so that the converged values are not thrown away.

struct HLLDFan {
  Real vx_al, vy_al, vz_al;   // velocity in the aL region      (MUB 23, 24)
  Real vx_ar, vy_ar, vz_ar;   // velocity in the aR region
  Real by_al, bz_al;          // transverse B in the aL region  (MUB 21)
  Real by_ar, bz_ar;
  Real kx_l, ky_l, kz_l;      // K vector, left                 (MUB 43)
  Real kx_r, ky_r, kz_r;      // K vector, right
  Real eta_l, eta_r;          // MUB 35
  Real lambda_al, lambda_ar;  // Alfven speeds == kx_l, kx_r
  Real delta_kx;              // kx_r - kx_l (+ regularizer)    (MUB 45)
};

//----------------------------------------------------------------------------------------
//! \fn Real SRHLLDResidual
//! \brief Evaluate the MUB (48) residual for a trial total pressure, filling `s` with the
//! intermediate-state quantities implied by that pressure.  A root of this function is the
//! total pressure in the Riemann fan.

KOKKOS_INLINE_FUNCTION
Real SRHLLDResidual(const Real ptot, const Real bbx,
                    const Real lambda_l, const Real lambda_r,
                    const MHDCons1D &rl, const MHDCons1D &rr,
                    const Real delta_kx_aug, HLLDFan &s) {
  // ---- velocities in the aL region (MUB 23, 24) with the auxiliaries MUB 26-30
  Real al = rl.mx - lambda_l*rl.e + ptot*(1.0 - SQR(lambda_l));          // (MUB 26)
  Real gl = SQR(rl.by) + SQR(rl.bz);                                     // (MUB 27)
  Real cl = rl.my*rl.by + rl.mz*rl.bz;                                   // (MUB 28)
  Real ql = -al - gl + SQR(bbx)*(1.0 - SQR(lambda_l));                   // (MUB 29)
  Real xl_inv = 1.0/(bbx*(al*lambda_l*bbx + cl)
                     - (al + gl)*(lambda_l*ptot + rl.e));                // (MUB 30)
  s.vx_al = (bbx*(al*bbx + lambda_l*cl) - (al + gl)*(ptot + rl.mx))*xl_inv;
  s.vy_al = (ql*rl.my + rl.by*(cl + bbx*(lambda_l*rl.mx - rl.e)))*xl_inv;
  s.vz_al = (ql*rl.mz + rl.bz*(cl + bbx*(lambda_l*rl.mx - rl.e)))*xl_inv;

  // ---- velocities in the aR region
  Real ar = rr.mx - lambda_r*rr.e + ptot*(1.0 - SQR(lambda_r));          // (MUB 26)
  Real gr = SQR(rr.by) + SQR(rr.bz);                                     // (MUB 27)
  Real cr = rr.my*rr.by + rr.mz*rr.bz;                                   // (MUB 28)
  Real qr = -ar - gr + SQR(bbx)*(1.0 - SQR(lambda_r));                   // (MUB 29)
  Real xr_inv = 1.0/(bbx*(ar*lambda_r*bbx + cr)
                     - (ar + gr)*(lambda_r*ptot + rr.e));                // (MUB 30)
  s.vx_ar = (bbx*(ar*bbx + lambda_r*cr) - (ar + gr)*(ptot + rr.mx))*xr_inv;
  s.vy_ar = (qr*rr.my + rr.by*(cr + bbx*(lambda_r*rr.mx - rr.e)))*xr_inv;
  s.vz_ar = (qr*rr.mz + rr.bz*(cr + bbx*(lambda_r*rr.mx - rr.e)))*xr_inv;

  // ---- transverse magnetic field in the a regions (MUB 21)
  s.by_al = (rl.by - bbx*s.vy_al)/(lambda_l - s.vx_al);
  s.bz_al = (rl.bz - bbx*s.vz_al)/(lambda_l - s.vx_al);
  s.by_ar = (rr.by - bbx*s.vy_ar)/(lambda_r - s.vx_ar);
  s.bz_ar = (rr.bz - bbx*s.vz_ar)/(lambda_r - s.vx_ar);

  // ---- total enthalpy in the a regions (MUB 31)
  Real v_rm_l  = s.vx_al*rl.mx + s.vy_al*rl.my + s.vz_al*rl.mz;
  Real wtot_al = ptot + (rl.e - v_rm_l)/(lambda_l - s.vx_al);
  Real v_rm_r  = s.vx_ar*rr.mx + s.vy_ar*rr.my + s.vz_ar*rr.mz;
  Real wtot_ar = ptot + (rr.e - v_rm_r)/(lambda_r - s.vx_ar);

  // ---- eta (MUB 35)
  s.eta_l = -copysign(sqrt(wtot_al), bbx);
  s.eta_r =  copysign(sqrt(wtot_ar), bbx);

  // ---- K vectors (MUB 43); note R_{B^x} = lambda*B^x
  Real denom_al_inv = 1.0/(lambda_l*ptot + rl.e + bbx*s.eta_l);
  s.kx_l = (rl.mx + ptot + lambda_l*bbx*s.eta_l)*denom_al_inv;
  s.ky_l = (rl.my + rl.by*s.eta_l)*denom_al_inv;
  s.kz_l = (rl.mz + rl.bz*s.eta_l)*denom_al_inv;
  Real denom_ar_inv = 1.0/(lambda_r*ptot + rr.e + bbx*s.eta_r);
  s.kx_r = (rr.mx + ptot + lambda_r*bbx*s.eta_r)*denom_ar_inv;
  s.ky_r = (rr.my + rr.by*s.eta_r)*denom_ar_inv;
  s.kz_r = (rr.mz + rr.bz*s.eta_r)*denom_ar_inv;

  // The x-component of K *is* the Alfven speed of that rotational discontinuity
  s.lambda_al = s.kx_l;
  s.lambda_ar = s.kx_r;

  // ---- B_c times delta_Kx (MUB 45)
  s.delta_kx = s.kx_r - s.kx_l + delta_kx_aug;
  Real bbx_c_dkx = bbx*s.delta_kx;
  Real by_c_dkx  = s.by_ar*(s.lambda_ar - s.vx_ar)
                   - s.by_al*(s.lambda_al - s.vx_al) + bbx*(s.vy_ar - s.vy_al);
  Real bz_c_dkx  = s.bz_ar*(s.lambda_ar - s.vx_ar)
                   - s.bz_al*(s.lambda_al - s.vx_al) + bbx*(s.vz_ar - s.vz_al);

  // ---- residual (MUB 48) built from Y_L, Y_R (MUB 49)
  Real k_sq_l = SQR(s.kx_l) + SQR(s.ky_l) + SQR(s.kz_l);
  Real kdb    = s.kx_l*bbx_c_dkx + s.ky_l*by_c_dkx + s.kz_l*bz_c_dkx;
  Real y_l    = (1.0 - k_sq_l)/(s.eta_l*s.delta_kx - kdb);
  Real k_sq_r = SQR(s.kx_r) + SQR(s.ky_r) + SQR(s.kz_r);
  kdb         = s.kx_r*bbx_c_dkx + s.ky_r*by_c_dkx + s.kz_r*bz_c_dkx;
  Real y_r    = (1.0 - k_sq_r)/(s.eta_r*s.delta_kx - kdb);
  return s.delta_kx*(1.0 - bbx*(y_r - y_l));
}

//----------------------------------------------------------------------------------------
//! \fn Real SREResidual
//! \brief Mignone & McKinney (2007) energy residual: vanishes at the correct total
//! enthalpy W for the given conserved state.  Used only to invert the HLL average state
//! for an initial total-pressure estimate.

KOKKOS_INLINE_FUNCTION
Real SREResidual(const Real w, const Real dd, const Real ee, const Real m_sq,
                 const Real bb_sq, const Real ss_sq, const Real gamma_prime) {
  Real v_sq = (m_sq + ss_sq/SQR(w)*(2.0*w + bb_sq))/SQR(w + bb_sq);      // (cf. MM A3)
  Real lor  = sqrt(1.0/(1.0 - v_sq));
  Real chi  = (1.0 - v_sq)*(w - lor*dd);                                 // (cf. MM A11)
  Real pgas = chi/gamma_prime;                                           // (MM A17)
  return (w - pgas + 0.5*bb_sq*(1.0 + v_sq) - ss_sq/(2.0*SQR(w))) - ee;  // (MM A1)
}

//----------------------------------------------------------------------------------------
//! \fn Real SREResidualPrime
//! \brief Derivative of SREResidual() with respect to W (MM A14-A18).

KOKKOS_INLINE_FUNCTION
Real SREResidualPrime(const Real w, const Real dd, const Real m_sq,
                      const Real bb_sq, const Real ss_sq, const Real gamma_prime) {
  Real v_sq = (m_sq + ss_sq/SQR(w)*(2.0*w + bb_sq))/SQR(w + bb_sq);      // (cf. MM A3)
  Real lor  = sqrt(1.0/(1.0 - v_sq));
  Real chi  = (1.0 - v_sq)*(w - lor*dd);                                 // (cf. MM A11)
  Real w_cu   = SQR(w)*w;
  Real w_b_cu = SQR(w + bb_sq)*(w + bb_sq);
  Real dv_sq_dw = -2.0/(w_cu*w_b_cu)
                  *(ss_sq*(3.0*w*(w + bb_sq) + SQR(bb_sq)) + m_sq*w_cu);  // (MM A16)
  Real dchi_dw  = 1.0 - v_sq - 0.5*lor*(dd + 2.0*lor*chi)*dv_sq_dw;       // (cf. MM A14)
  Real dpgas_dw = dchi_dw/gamma_prime;                                    // (MM A18)
  return 1.0 - dpgas_dw + 0.5*bb_sq*dv_sq_dw + ss_sq/w_cu;
}

//----------------------------------------------------------------------------------------
//! \fn void HLLD_SR
//! \brief The HLLD Riemann solver for SR MHD (MUB 2009).

template <int ivx>
KOKKOS_INLINE_FUNCTION
void HLLD_SR(const EOS_Data &eos,
             const int m, const int k, const int j, const int i,
             const int is, const int js, const int ks,
             const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
             const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
             const DvceArray4D<Real> &bx,
             const DvceArray5D<Real> &flx,
             const DvceArray4D<Real> &ey, const DvceArray4D<Real> &ez) {
  constexpr int ivy = IVX + ((ivx-IVX) + 1)%3;
  constexpr int ivz = IVX + ((ivx-IVX) + 2)%3;
  constexpr int iby = ((ivx-IVX) + 1)%3;
  constexpr int ibz = ((ivx-IVX) + 2)%3;
  const Real gm1 = (eos.gamma - 1.0);
  const Real gamma_prime = eos.gamma/gm1;

  // Solver parameters, following Athena++ hlld_rel.cpp
  const Real p_transition  = 0.01;    // B^2/p_tot below which the field is "weak"
  const Real vc_extension  = 1.0e-6;  // treat as contact region if Alfven speeds closer
  const Real delta_kx_aug  = 1.0e-12; // regularizer added to Delta K^x
  const Real initial_offset= 1.0e-5;  // relative offset for the second secant point
  const Real tol_res       = 1.0e-12;
  const int  num_secant    = 12;      // max iterations; early break once converged
  const int  num_nr        = 2;
  // Robustness gates (see header comment).  tol_gate: healthy faces converge to
  // |res| <~ 1e-12 with the extended secant while stalled ones sit at >~ 1e-3, so any
  // value in between works; the residual is O(1) in c=1 units (it is Delta K^x times a
  // dimensionless factor).  ord_slack: the B_perp -> 0 degeneracy makes the fast and
  // Alfven speeds coincide analytically, so roundoff can put lambda_a* an epsilon
  // outside [lambda_L, lambda_R]; measured healthy margins are >= -1e-15, measured
  // garbage violations are O(0.1-1).
  const Real tol_gate      = 1.0e-4;
  const Real ord_slack     = 1.0e-9;
  // Degeneracy switch (the actual 3D KSI cure; the gates above only catch broken fans,
  // not this).  On a COLD, magnetically dominated face (b^2 >> gamma*p, so cs << ca)
  // whose field is aligned with a coordinate direction, HLLD's extra waves carry ~zero
  // jumps and it degenerates into a zero-dissipation resolver for the remaining modes:
  //  - B ~ normal (B_perp ~ 0):  fast = Alfven along the normal; contact and rotational
  //    modes get no dissipation.  This is the 3D-KSI x3 sweep in the far field
  //    (bx = B3 ~ 3.2, B_perp ~ roundoff, p = 1e-4).  Measured there: HLLD pumps
  //    field-aligned grid noise from u3_rms ~ 4e-7 (HLLE) to 2e-3 by t = 0.25 and rho
  //    O(1) caustics by t = 0.5, while the gated fan stays formally "physical" face by
  //    face.
  //  - B ~ transverse (bx ~ 0): the fan collapses to a tangential discontinuity whose
  //    (rho, u_z, B_z) jumps are resolved exactly.  Grid-scale u3/rho shear noise then
  //    has zero damping in EVERY sweep (the x1/x2 faces of the 3D-KSI far field), and
  //    the wall-settling transient keeps feeding it: measured std(rho) 5.9e-2 vs
  //    HLLE 3.1e-4 at t = 0.25 near the walls with the normal-channel switch alone.
  // In both channels HLLE loses nothing -- the waves HLLD would add resolve jumps that
  // are ~ 0 -- so revert to HLLE there.  Faces where the KSI physics lives never
  // trigger: the sheet and its flanks have p ~ 0.6-5 (not cold), drip surfaces have
  // p ~ 5 on one side, reconnection regions and bent field lines have mixed B
  // components.  2D KSI keeps HLLD on every face that has structure; its quiescent far
  // field now gets HLLE-like fluxes, which changes nothing dynamically (u3 == 0 there
  // by symmetry).  All MUB 1-4 tube faces: either not cold (1, 2, 4) or mixed-B (3);
  // verified bit-for-bit unchanged.
  const Real degen_ratio   = 1.0e-4;  // threshold on min(bx^2,B_perp^2)/max(...)
  const Real degen_prs     = 10.0;    // threshold on b^2 / (gamma*pgas)

  // ------------------------------------------------------------------ L and R states
  Real rho_l = wl(m,IDN,k,j,i);
  Real u_l[4];
  u_l[1] = wl(m,ivx,k,j,i);
  u_l[2] = wl(m,ivy,k,j,i);
  u_l[3] = wl(m,ivz,k,j,i);
  u_l[0] = sqrt(1.0 + SQR(u_l[1]) + SQR(u_l[2]) + SQR(u_l[3]));
  Real bb2_l = bl(m,iby,k,j,i);
  Real bb3_l = bl(m,ibz,k,j,i);

  Real rho_r = wr(m,IDN,k,j,i);
  Real u_r[4];
  u_r[1] = wr(m,ivx,k,j,i);
  u_r[2] = wr(m,ivy,k,j,i);
  u_r[3] = wr(m,ivz,k,j,i);
  u_r[0] = sqrt(1.0 + SQR(u_r[1]) + SQR(u_r[2]) + SQR(u_r[3]));
  Real bb2_r = br(m,iby,k,j,i);
  Real bb3_r = br(m,ibz,k,j,i);

  Real pgas_l = eos.IdealGasPressure(wl(m,IEN,k,j,i));
  Real pgas_r = eos.IdealGasPressure(wr(m,IEN,k,j,i));

  // Normal magnetic field (continuous across the interface)
  Real bbx = bx(m,k,j,i);

  // 4-magnetic field, left
  Real b_l[4];
  b_l[0] = bbx*u_l[1] + bb2_l*u_l[2] + bb3_l*u_l[3];
  b_l[1] = (bbx   + b_l[0]*u_l[1])/u_l[0];
  b_l[2] = (bb2_l + b_l[0]*u_l[2])/u_l[0];
  b_l[3] = (bb3_l + b_l[0]*u_l[3])/u_l[0];
  Real b_sq_l = -SQR(b_l[0]) + SQR(b_l[1]) + SQR(b_l[2]) + SQR(b_l[3]);

  // 4-magnetic field, right
  Real b_r[4];
  b_r[0] = bbx*u_r[1] + bb2_r*u_r[2] + bb3_r*u_r[3];
  b_r[1] = (bbx   + b_r[0]*u_r[1])/u_r[0];
  b_r[2] = (bb2_r + b_r[0]*u_r[2])/u_r[0];
  b_r[3] = (bb3_r + b_r[0]*u_r[3])/u_r[0];
  Real b_sq_r = -SQR(b_r[0]) + SQR(b_r[1]) + SQR(b_r[2]) + SQR(b_r[3]);

  // Fast-magnetosonic speeds and the extremal wavespeeds (MB 55)
  Real lm_l, lp_l, lm_r, lp_r;
  eos.IdealSRMHDFastSpeeds(rho_l, pgas_l, u_l[1], u_l[0], b_sq_l, lp_l, lm_l);
  eos.IdealSRMHDFastSpeeds(rho_r, pgas_r, u_r[1], u_r[0], b_sq_r, lp_r, lm_r);
  Real lambda_l = fmin(lm_l, lm_r);
  Real lambda_r = fmax(lp_l, lp_r);

  // Conserved state and flux, L region (MUB 8, 15).  Note `.e` is the FULL energy E
  // (rest mass included); the conversion to tau = E - D happens at the very end.
  MHDCons1D consl, fl;
  Real wtot_l = rho_l + gamma_prime*pgas_l + b_sq_l;
  Real ptot_l = pgas_l + 0.5*b_sq_l;
  consl.d  = rho_l*u_l[0];
  consl.e  = wtot_l*u_l[0]*u_l[0] - b_l[0]*b_l[0] - ptot_l;
  consl.mx = wtot_l*u_l[1]*u_l[0] - b_l[1]*b_l[0];
  consl.my = wtot_l*u_l[2]*u_l[0] - b_l[2]*b_l[0];
  consl.mz = wtot_l*u_l[3]*u_l[0] - b_l[3]*b_l[0];
  consl.by = b_l[2]*u_l[0] - b_l[0]*u_l[2];
  consl.bz = b_l[3]*u_l[0] - b_l[0]*u_l[3];
  fl.d  = rho_l*u_l[1];
  fl.e  = wtot_l*u_l[0]*u_l[1] - b_l[0]*b_l[1];
  fl.mx = wtot_l*u_l[1]*u_l[1] - b_l[1]*b_l[1] + ptot_l;
  fl.my = wtot_l*u_l[2]*u_l[1] - b_l[2]*b_l[1];
  fl.mz = wtot_l*u_l[3]*u_l[1] - b_l[3]*b_l[1];
  fl.by = b_l[2]*u_l[1] - b_l[1]*u_l[2];
  fl.bz = b_l[3]*u_l[1] - b_l[1]*u_l[3];

  // Conserved state and flux, R region (MUB 8, 15)
  MHDCons1D consr, fr;
  Real wtot_r = rho_r + gamma_prime*pgas_r + b_sq_r;
  Real ptot_r = pgas_r + 0.5*b_sq_r;
  consr.d  = rho_r*u_r[0];
  consr.e  = wtot_r*u_r[0]*u_r[0] - b_r[0]*b_r[0] - ptot_r;
  consr.mx = wtot_r*u_r[1]*u_r[0] - b_r[1]*b_r[0];
  consr.my = wtot_r*u_r[2]*u_r[0] - b_r[2]*b_r[0];
  consr.mz = wtot_r*u_r[3]*u_r[0] - b_r[3]*b_r[0];
  consr.by = b_r[2]*u_r[0] - b_r[0]*u_r[2];
  consr.bz = b_r[3]*u_r[0] - b_r[0]*u_r[3];
  fr.d  = rho_r*u_r[1];
  fr.e  = wtot_r*u_r[0]*u_r[1] - b_r[0]*b_r[1];
  fr.mx = wtot_r*u_r[1]*u_r[1] - b_r[1]*b_r[1] + ptot_r;
  fr.my = wtot_r*u_r[2]*u_r[1] - b_r[2]*b_r[1];
  fr.mz = wtot_r*u_r[3]*u_r[1] - b_r[3]*b_r[1];
  fr.by = b_r[2]*u_r[1] - b_r[1]*u_r[2];
  fr.bz = b_r[3]*u_r[1] - b_r[1]*u_r[3];

  // Jump vectors across each fast wave, R = lambda*U - F (MUB 12)
  MHDCons1D rl, rr;
  rl.d  = lambda_l*consl.d  - fl.d;
  rl.e  = lambda_l*consl.e  - fl.e;
  rl.mx = lambda_l*consl.mx - fl.mx;
  rl.my = lambda_l*consl.my - fl.my;
  rl.mz = lambda_l*consl.mz - fl.mz;
  rl.by = lambda_l*consl.by - fl.by;
  rl.bz = lambda_l*consl.bz - fl.bz;
  rr.d  = lambda_r*consr.d  - fr.d;
  rr.e  = lambda_r*consr.e  - fr.e;
  rr.mx = lambda_r*consr.mx - fr.mx;
  rr.my = lambda_r*consr.my - fr.my;
  rr.mz = lambda_r*consr.mz - fr.mz;
  rr.by = lambda_r*consr.by - fr.by;
  rr.bz = lambda_r*consr.bz - fr.bz;

  // HLL average state (MB 29) and flux (MB 31)
  MHDCons1D cons_hll, flux_hll;
  Real ldiff_inv = 1.0/(lambda_r - lambda_l);
  cons_hll.d  = (rr.d  - rl.d )*ldiff_inv;
  cons_hll.e  = (rr.e  - rl.e )*ldiff_inv;
  cons_hll.mx = (rr.mx - rl.mx)*ldiff_inv;
  cons_hll.my = (rr.my - rl.my)*ldiff_inv;
  cons_hll.mz = (rr.mz - rl.mz)*ldiff_inv;
  cons_hll.by = (rr.by - rl.by)*ldiff_inv;
  cons_hll.bz = (rr.bz - rl.bz)*ldiff_inv;
  flux_hll.d  = (lambda_l*rr.d  - lambda_r*rl.d )*ldiff_inv;
  flux_hll.e  = (lambda_l*rr.e  - lambda_r*rl.e )*ldiff_inv;
  flux_hll.mx = (lambda_l*rr.mx - lambda_r*rl.mx)*ldiff_inv;
  flux_hll.my = (lambda_l*rr.my - lambda_r*rl.my)*ldiff_inv;
  flux_hll.mz = (lambda_l*rr.mz - lambda_r*rl.mz)*ldiff_inv;
  flux_hll.by = (lambda_l*rr.by - lambda_r*rl.by)*ldiff_inv;
  flux_hll.bz = (lambda_l*rr.bz - lambda_r*rl.bz)*ldiff_inv;

  bool use_hlle = false;

  // ------------------------------------------------------------ degeneracy switch
  // Cold magnetically dominated face with the field aligned to a coordinate axis:
  // HLLD adds only zero-strength waves here and is numerically unstable (see header).
  {
    Real bperp_sq_max = fmax(SQR(bb2_l) + SQR(bb3_l), SQR(bb2_r) + SQR(bb3_r));
    Real b_sq_max     = fmax(b_sq_l, b_sq_r);
    if (b_sq_max > degen_prs*eos.gamma*fmax(pgas_l, pgas_r) &&
        (bperp_sq_max < degen_ratio*SQR(bbx) ||     // B ~ normal: fast = Alfven
         SQR(bbx) < degen_ratio*bperp_sq_max)) {    // B ~ transverse: tangential disc.
      use_hlle = true;
    }
  }

  // ----------------------------------------- total pressure of the HLL average state
  Real ptot_hll = 0.0;
  if (!use_hlle) {
    Real m_sq  = SQR(cons_hll.mx) + SQR(cons_hll.my) + SQR(cons_hll.mz);
    Real bb_sq = SQR(bbx) + SQR(cons_hll.by) + SQR(cons_hll.bz);
    Real m_dot_bb = cons_hll.mx*bbx + cons_hll.my*cons_hll.by
                    + cons_hll.mz*cons_hll.bz;
    Real ss_sq = SQR(m_dot_bb);

    // Initial guess for the total enthalpy W (MM A26-A27)
    Real a1 = 4.0/3.0*(bb_sq - cons_hll.e);
    Real a0 = (1.0/3.0)*(m_sq + bb_sq*(bb_sq - 2.0*cons_hll.e));
    Real s2 = SQR(a1) - 4.0*a0;
    Real ss = (s2 < 0.0) ? 0.0 : sqrt(s2);
    Real w  = (s2 >= 0.0 && a1 >= 0.0) ? -2.0*a0/(a1 + ss) : 0.5*(-a1 + ss);

    // A couple of Newton-Raphson steps are enough for an initial estimate
    Real res = SREResidual(w, cons_hll.d, cons_hll.e, m_sq, bb_sq, ss_sq, gamma_prime);
    for (int n = 0; n < num_nr; ++n) {
      Real deriv = SREResidualPrime(w, cons_hll.d, m_sq, bb_sq, ss_sq, gamma_prime);
      w  -= res/deriv;
      res = SREResidual(w, cons_hll.d, cons_hll.e, m_sq, bb_sq, ss_sq, gamma_prime);
    }

    // Primitives from W (MM A3, A10, A11, A17), then the total pressure (MM 2)
    Real v_sq   = (m_sq + ss_sq/SQR(w)*(2.0*w + bb_sq))/SQR(w + bb_sq);
    Real lor_sq = 1.0/(1.0 - v_sq);
    Real chi    = (1.0 - v_sq)*(w - sqrt(lor_sq)*cons_hll.d);
    Real pgas   = chi/gamma_prime;
    Real vx = (cons_hll.mx + m_dot_bb/w*bbx)/(w + bb_sq);
    Real vy = (cons_hll.my + m_dot_bb/w*cons_hll.by)/(w + bb_sq);
    Real vz = (cons_hll.mz + m_dot_bb/w*cons_hll.bz)/(w + bb_sq);
    Real v_bb = vx*bbx + vy*cons_hll.by + vz*cons_hll.bz;
    ptot_hll = pgas + 0.5*(bb_sq/lor_sq + SQR(v_bb));
  }

  // ------------------------------------------- initial guess for the fan pressure
  Real ptot_init = 0.0;
  if (!use_hlle) {
    if (SQR(bbx)/ptot_hll < p_transition) { // weak field: quadratic estimate (MUB 55)
      Real a1 = cons_hll.e - flux_hll.mx;
      Real a0 = cons_hll.mx*flux_hll.e - flux_hll.mx*cons_hll.e;
      Real s2 = SQR(a1) - 4.0*a0;
      Real ss = (s2 < 0.0) ? 0.0 : sqrt(s2);
      ptot_init = (s2 >= 0.0 && a1 >= 0.0) ? -2.0*a0/(a1 + ss) : 0.5*(-a1 + ss);
    } else {                                // strong field (MUB 53)
      ptot_init = ptot_hll;
    }
    if (!Kokkos::isfinite(ptot_init) || ptot_init <= 0.0) use_hlle = true;
  }

  // ------------------------------------------------ secant iteration on MUB (48)
  // NOTE: `s` is filled by every SRHLLDResidual() call, so after the loop it holds the
  // fan implied by the last evaluated pressure p_n == ptot_c.
  HLLDFan s;
  Real ptot_c = ptot_init;
  if (!use_hlle) {
    Real tol_ptot = 1.0e-8*ptot_init;
    Real p_last = ptot_init;
    Real r_last = SRHLLDResidual(p_last, bbx, lambda_l, lambda_r, rl, rr,
                                 delta_kx_aug, s);
    Real p_n = ptot_init*(1.0 + initial_offset);
    Real r_n = SRHLLDResidual(p_n, bbx, lambda_l, lambda_r, rl, rr, delta_kx_aug, s);

    for (int n = 0; n < num_secant; ++n) {
      // converged (or stalled to a fixed point / non-finite): stop
      if (!(fabs(r_n) > tol_res) || !(fabs(p_n - p_last) > tol_ptot) ||
          !(r_n != r_last) || !Kokkos::isfinite(p_n)) {
        break;
      }
      Real p_old = p_last;
      Real r_old = r_last;
      p_last = p_n;
      r_last = r_n;
      p_n = (r_last*p_old - r_old*p_last)/(r_last - r_old);
      r_n = SRHLLDResidual(p_n, bbx, lambda_l, lambda_r, rl, rr, delta_kx_aug, s);
    }
    ptot_c = p_n;

    // Convergence gate + the original finiteness checks.  Conditions are written so
    // that a NaN anywhere trips the fallback.
    if (!(fabs(r_n) < tol_gate) ||
        !Kokkos::isfinite(s.lambda_al) || !Kokkos::isfinite(s.lambda_ar) ||
        !Kokkos::isfinite(ptot_c) || ptot_c <= 0.0) {
      use_hlle = true;
    }
  }

  // ------------------------------------------------------- assemble the wave fan
  MHDCons1D cons_al, cons_ar, flux_al, flux_ar, cons_c, flux_c;
  Real vx_c = 0.0;
  if (!use_hlle) {
    // aL region (MUB 32, 33, 34)
    Real v_bb_al = s.vx_al*bbx + s.vy_al*s.by_al + s.vz_al*s.bz_al;
    cons_al.by = s.by_al;
    cons_al.bz = s.bz_al;
    cons_al.d  = rl.d/(lambda_l - s.vx_al);
    cons_al.e  = (rl.e + ptot_c*s.vx_al - v_bb_al*bbx)/(lambda_l - s.vx_al);
    cons_al.mx = (cons_al.e + ptot_c)*s.vx_al - v_bb_al*bbx;
    cons_al.my = (cons_al.e + ptot_c)*s.vy_al - v_bb_al*cons_al.by;
    cons_al.mz = (cons_al.e + ptot_c)*s.vz_al - v_bb_al*cons_al.bz;

    // aR region
    Real v_bb_ar = s.vx_ar*bbx + s.vy_ar*s.by_ar + s.vz_ar*s.bz_ar;
    cons_ar.by = s.by_ar;
    cons_ar.bz = s.bz_ar;
    cons_ar.d  = rr.d/(lambda_r - s.vx_ar);
    cons_ar.e  = (rr.e + ptot_c*s.vx_ar - v_bb_ar*bbx)/(lambda_r - s.vx_ar);
    cons_ar.mx = (cons_ar.e + ptot_c)*s.vx_ar - v_bb_ar*bbx;
    cons_ar.my = (cons_ar.e + ptot_c)*s.vy_ar - v_bb_ar*cons_ar.by;
    cons_ar.mz = (cons_ar.e + ptot_c)*s.vz_ar - v_bb_ar*cons_ar.bz;

    // Fluxes in the a regions (MUB 11, 12)
    flux_al.d  = lambda_l*cons_al.d  - rl.d;
    flux_al.e  = lambda_l*cons_al.e  - rl.e;
    flux_al.mx = lambda_l*cons_al.mx - rl.mx;
    flux_al.my = lambda_l*cons_al.my - rl.my;
    flux_al.mz = lambda_l*cons_al.mz - rl.mz;
    flux_al.by = lambda_l*cons_al.by - rl.by;
    flux_al.bz = lambda_l*cons_al.bz - rl.bz;
    flux_ar.d  = lambda_r*cons_ar.d  - rr.d;
    flux_ar.e  = lambda_r*cons_ar.e  - rr.e;
    flux_ar.mx = lambda_r*cons_ar.mx - rr.mx;
    flux_ar.my = lambda_r*cons_ar.my - rr.my;
    flux_ar.mz = lambda_r*cons_ar.mz - rr.mz;
    flux_ar.by = lambda_r*cons_ar.by - rr.by;
    flux_ar.bz = lambda_r*cons_ar.bz - rr.bz;

    // Transverse field in the contact region (MUB 45)
    Real denom_c_inv = 1.0/(s.lambda_ar - s.lambda_al + delta_kx_aug);
    Real numer_al = cons_al.by*(s.lambda_al - s.vx_al) + bbx*s.vy_al;
    Real numer_ar = cons_ar.by*(s.lambda_ar - s.vx_ar) + bbx*s.vy_ar;
    cons_c.by = (numer_ar - numer_al)*denom_c_inv;
    numer_al = cons_al.bz*(s.lambda_al - s.vx_al) + bbx*s.vz_al;
    numer_ar = cons_ar.bz*(s.lambda_ar - s.vx_ar) + bbx*s.vz_ar;
    cons_c.bz = (numer_ar - numer_al)*denom_c_inv;

    // Contact velocity (MUB 47), averaging the two one-sided estimates
    Real k_sq_l = SQR(s.kx_l) + SQR(s.ky_l) + SQR(s.kz_l);
    Real k_bc_l = s.kx_l*bbx + s.ky_l*cons_c.by + s.kz_l*cons_c.bz;
    Real f_l    = (1.0 - k_sq_l)/(s.eta_l - k_bc_l);
    Real k_sq_r = SQR(s.kx_r) + SQR(s.ky_r) + SQR(s.kz_r);
    Real k_bc_r = s.kx_r*bbx + s.ky_r*cons_c.by + s.kz_r*cons_c.bz;
    Real f_r    = (1.0 - k_sq_r)/(s.eta_r - k_bc_r);
    vx_c = 0.5*((s.kx_l - bbx*f_l)       + (s.kx_r - bbx*f_r));
    Real vy_c = 0.5*((s.ky_l - cons_c.by*f_l) + (s.ky_r - cons_c.by*f_r));
    Real vz_c = 0.5*((s.kz_l - cons_c.bz*f_l) + (s.kz_r - cons_c.bz*f_r));

    // Remaining conserved quantities in the contact region (MUB 50, 51, 52)
    Real v_bb_c = vx_c*bbx + vy_c*cons_c.by + vz_c*cons_c.bz;
    if (vx_c >= 0.0) {   // cL
      cons_c.d = cons_al.d*(s.lambda_al - s.vx_al)/(s.lambda_al - vx_c);
      cons_c.e = (s.lambda_al*cons_al.e - cons_al.mx + ptot_c*vx_c - v_bb_c*bbx)
                 /(s.lambda_al - vx_c);
    } else {             // cR
      cons_c.d = cons_ar.d*(s.lambda_ar - s.vx_ar)/(s.lambda_ar - vx_c);
      cons_c.e = (s.lambda_ar*cons_ar.e - cons_ar.mx + ptot_c*vx_c - v_bb_c*bbx)
                 /(s.lambda_ar - vx_c);
    }
    cons_c.mx = (cons_c.e + ptot_c)*vx_c - v_bb_c*bbx;
    cons_c.my = (cons_c.e + ptot_c)*vy_c - v_bb_c*cons_c.by;
    cons_c.mz = (cons_c.e + ptot_c)*vz_c - v_bb_c*cons_c.bz;

    // Fluxes in the contact region (MUB 11)
    if (vx_c >= 0.0) {
      flux_c.d  = flux_al.d  + s.lambda_al*(cons_c.d  - cons_al.d );
      flux_c.e  = flux_al.e  + s.lambda_al*(cons_c.e  - cons_al.e );
      flux_c.mx = flux_al.mx + s.lambda_al*(cons_c.mx - cons_al.mx);
      flux_c.my = flux_al.my + s.lambda_al*(cons_c.my - cons_al.my);
      flux_c.mz = flux_al.mz + s.lambda_al*(cons_c.mz - cons_al.mz);
      flux_c.by = flux_al.by + s.lambda_al*(cons_c.by - cons_al.by);
      flux_c.bz = flux_al.bz + s.lambda_al*(cons_c.bz - cons_al.bz);
    } else {
      flux_c.d  = flux_ar.d  + s.lambda_ar*(cons_c.d  - cons_ar.d );
      flux_c.e  = flux_ar.e  + s.lambda_ar*(cons_c.e  - cons_ar.e );
      flux_c.mx = flux_ar.mx + s.lambda_ar*(cons_c.mx - cons_ar.mx);
      flux_c.my = flux_ar.my + s.lambda_ar*(cons_c.my - cons_ar.my);
      flux_c.mz = flux_ar.mz + s.lambda_ar*(cons_c.mz - cons_ar.mz);
      flux_c.by = flux_ar.by + s.lambda_ar*(cons_c.by - cons_ar.by);
      flux_c.bz = flux_ar.bz + s.lambda_ar*(cons_c.bz - cons_ar.bz);
    }

    // -------------------------------------------- physicality gates on the fan
    // A converged-looking pressure can still imply an unphysical fan (wrong root).
    // Every condition is written as !(good) so that a NaN also trips the fallback.
    Real v_sq_al = SQR(s.vx_al) + SQR(s.vy_al) + SQR(s.vz_al);
    Real v_sq_ar = SQR(s.vx_ar) + SQR(s.vy_ar) + SQR(s.vz_ar);
    Real v_sq_c  = SQR(vx_c) + SQR(vy_c) + SQR(vz_c);
    if (!(v_sq_al < 1.0) || !(v_sq_ar < 1.0) || !(v_sq_c < 1.0) ||       // subluminal
        !(s.lambda_al >= lambda_l - ord_slack) ||                       // ordering
        !(s.lambda_ar <= lambda_r + ord_slack) ||
        !(s.lambda_al <= s.lambda_ar + ord_slack) ||
        !(vx_c >= s.lambda_al - ord_slack) ||
        !(vx_c <= s.lambda_ar + ord_slack) ||
        !(cons_al.d > 0.0) || !(cons_ar.d > 0.0) || !(cons_c.d > 0.0) || // positivity
        !(cons_al.e > 0.0) || !(cons_ar.e > 0.0) || !(cons_c.e > 0.0)) {
      use_hlle = true;
    }
  }

  // ------------------------------------------------------------- select the region
  MHDCons1D *fi;
  if (lambda_l >= 0.0) {                          // L
    fi = &fl;
  } else if (lambda_r <= 0.0) {                   // R
    fi = &fr;
  } else if (use_hlle) {                          // HLL fallback
    fi = &flux_hll;
  } else if (s.lambda_al >= -vc_extension) {      // aL
    fi = &flux_al;
  } else if (s.lambda_ar <= vc_extension) {       // aR
    fi = &flux_ar;
  } else {                                        // c
    fi = &flux_c;
  }

  // Guard against any non-finite value produced by the HLLD branch
  if (!Kokkos::isfinite(fi->d) || !Kokkos::isfinite(fi->e)  ||
      !Kokkos::isfinite(fi->mx)|| !Kokkos::isfinite(fi->my) ||
      !Kokkos::isfinite(fi->mz)|| !Kokkos::isfinite(fi->by) ||
      !Kokkos::isfinite(fi->bz)) {
    fi = &flux_hll;
  }

  flx(m,IDN,k,j,i) = fi->d;
  flx(m,ivx,k,j,i) = fi->mx;
  flx(m,ivy,k,j,i) = fi->my;
  flx(m,ivz,k,j,i) = fi->mz;
  flx(m,IEN,k,j,i) = fi->e;

  ey(m,k,j,i) = -(fi->by);
  ez(m,k,j,i) =  (fi->bz);

  // We evolve tau = E - D
  flx(m,IEN,k,j,i) -= flx(m,IDN,k,j,i);
  return;
}
} // namespace mhd
#endif // MHD_RSOLVERS_HLLD_SRMHD_HPP_
