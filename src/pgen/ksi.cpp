//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ksi.cpp
//! \brief Problem generator for the Kruskal-Schwarzschild instability (KSI)
//!        in special relativistic MHD.
//!
//! Supports both 2D and 3D configurations (auto-detected from the mesh, exactly as
//! rt.cpp does: `three_d = (indcs.nx3 > 1)` selects the branch).
//!
//! GEOMETRY (identical in 2D and 3D, following Athena-C KSI.c):
//!   x1  transverse to the sheet-normal, PERIODIC        (Gill's y)
//!   x2  sheet normal, gravity direction, REFLECTING     (Gill's z)
//!   x3  out of the 2D plane, PERIODIC; also the initial B direction
//! The current sheet lies in the x1-x3 plane at x2 = 0 and gravity is along -x2 in BOTH
//! 2D and 3D.  This differs from rt.cpp, which rotates gravity from x2 (2D) to x3 (3D);
//! here x3 must stay the field direction, so the walls remain on x2 in 3D too.  That
//! keeps KSIBoundaries() dimension-independent -- it only ever touches x2.
//!
//! ON HYDROSTATIC BALANCE (the hard part of this setup):
//!   An exactly balanced initial state DOES NOT EXIST for these profiles.  Relativistic
//!   hydrostatic equilibrium requires d(p_gas + B^2/2)/dx2 = -w*g with w = rho*h + b^2.
//!   Integrating that ODE downward from the sheet with sigma=10 drives p_gas negative at
//!   x2 = -0.0168 (only 3.4a below the sheet): at this magnetization the field already
//!   accounts for the entire pressure budget, so there is no gas pressure left to hold
//!   up the weight.  Gill, Granot & Lyubarsky (2018) therefore drop the stratification
//!   explicitly (their eqs. 8-10), keeping p_tot uniform and relying on
//!     Delta << c^2/g == L_dyn,   here 0.01 << 10.
//!   The residual imbalance is O(g*L_x2) ~ 2-4%; the slab settles against the reflecting
//!   walls on an Alfven crossing time t_A = L_x2/v_A ~ 0.21, well before the instability
//!   becomes nonlinear.  The hydrostatic pressure correction in KSIBoundaries() is what
//!   keeps that settling from launching spurious acoustic waves off the walls.
//!
//! Equilibrium profiles (smooth tanh/sech^2, following Athena-C KSI.c):
//!   B3     = -b0 * tanh(x2/a)        (reversed-field current sheet)
//!   rho    = d0 + 2*d0 * sech^2(x2/a)
//!   pgas   = p0 * sech^2(x2/a)       where p0 = b0^2/2
//!
//! Total pressure balance: pgas + B^2/2 = p0 everywhere
//! (uses the identity sech^2(xi) + tanh^2(xi) = 1, and p0 = b0^2/2).
//!
//! Velocity perturbation in the x2-direction only, selected by problem/iprob:
//!   iprob = 1 (single mode, the Gill+2018 / Athena-C fiducial)
//!     2D: v2 = amp * 0.5  * sin(2*pi*nx*x1/lx) * (1 + cos(2*pi*ny*x2/ly))
//!     3D: v2 = amp * 0.25 * sin(2*pi*nx*x1/lx) * cos(2*pi*nz*x3/lz)
//!                         * (1 + cos(2*pi*ny*x2/ly))
//!   iprob = 2 (multimode, white noise seeded by the same x2 envelope)
//!     v2 = amp * (rand - 0.5) * (1 + cos(2*pi*ny*x2/ly))
//!   In 3D a single mode forces one artificially coherent sheet of fingers; iprob=2 is
//!   the physical choice there, and is what rt.cpp defaults to for its 3D runs.  Use
//!   iprob=1 when validating the linear growth rate against the dispersion relation.
//!
//! Effective gravity is in -x2 direction, modeling the frame acceleration of a
//! relativistic jet slab. In the rest frame of the slab, this manifests as a
//! Rayleigh-Taylor-like instability (the KSI).
//!
//! User reflecting BCs with hydrostatic pressure correction prevent spurious
//! acoustic wave emission due to gravitational stratification at the x2 walls.
//!
//! REFERENCE: Adapted from Athena-C KSI.c and Athena++ 2dksi.cpp

// C++ headers
#include <cmath>
#include <iostream>

// AthenaK headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "bvals/bvals.hpp"
#include "pgen.hpp"

#include <Kokkos_Random.hpp>

// Namespace for parameters shared across user-defined functions
namespace {
  Real grav_acc;    // effective gravitational acceleration (negative = -x2 direction)
  Real refine_thr;  // AMR refinement threshold on |grad(p)|/p
  Real gam_gas;     // adiabatic index, needed by the source term
  bool grav_on_w;   // couple gravity to enthalpy w (paper) instead of rest mass D (Athena-C)
}

// Forward declarations for user-defined functions
void KSIBoundaries(Mesh *pm);
void KSIGravitySourceTerm(Mesh *pm, const Real bdt);
void KSIRefinement(MeshBlockPack *pmbp);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Problem Generator for the Kruskal-Schwarzschild instability in SR MHD.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // Read problem parameters
  Real amp    = pin->GetReal("problem", "amp");
  Real sigma  = pin->GetReal("problem", "sigma");
  Real delta  = pin->GetReal("problem", "delta");
  Real a      = pin->GetOrAddReal("problem", "a", delta);
  int  nx_m   = pin->GetOrAddInteger("problem", "nx", 1);
  int  ny_m   = pin->GetOrAddInteger("problem", "ny", 1);
  int  nz_m   = pin->GetOrAddInteger("problem", "nz", 1);
  // 1 = single mode (Gill+2018 eq. 15), 2 = multimode white noise. See file header.
  int  iprob  = pin->GetOrAddInteger("problem", "iprob", 1);

  // Store shared parameters for user BC and source term functions
  grav_acc   = pin->GetOrAddReal("problem", "grav", -0.1);
  refine_thr = pin->GetOrAddReal("problem", "thr",  0.25);
  gam_gas    = pin->GetOrAddReal("mhd", "gamma", 4.0/3.0);
  // Which inertia does the effective gravity couple to?  See KSIGravitySourceTerm.
  //   false (default) -> f = D*g,  reproducing Athena-C's StaticGravPot machinery
  //   true            -> f = w*g,  reproducing Gill, Granot & Lyubarsky (2018) eq. (5)
  grav_on_w  = pin->GetOrAddBoolean("problem", "grav_on_enthalpy", false);

  // Enroll user-defined boundary condition function (x2 walls only)
  user_bcs_func = KSIBoundaries;

  // Enroll user-defined gravity source term
  user_srcs_func = KSIGravitySourceTerm;

  // Enroll AMR refinement condition if adaptive mesh is enabled
  if (pmy_mesh_->adaptive) {
    user_ref_func = KSIRefinement;
  }

  // Return early on restarts; function pointers have been enrolled above
  if (restart) return;

  // Get mesh/block index information
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  // Verify MHD is enabled
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "KSI problem requires MHD to be enabled." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // The sheet needs at least one transverse direction to corrugate in
  if (pmy_mesh_->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "KSI problem only works in 2D/3D." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Derived equilibrium parameters
  Real d0  = 1.0;
  Real b0_ = std::sqrt(sigma * d0);  // field amplitude: b0 = sqrt(sigma)
  Real p0  = 0.5 * b0_ * b0_;        // pressure balance: p0 = b0^2/2

  // AthenaK stores the INTERNAL ENERGY DENSITY e = p_gas/(gamma-1) in the w0(IEN)==w0(IPR)
  // slot, not the gas pressure.  See SingleP2C_IdealSRMHD() in eos/ideal_c2p_mhd.hpp, where
  // the total enthalpy is (w.d + gamma*w.e + b^2) and the gas pressure is (gamma-1)*w.e.
  Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;

  // Domain extents (for perturbation wavenumbers)
  Real lx = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real ly = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  Real lz = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;

  // Detect 2D vs 3D from number of active x3 cells
  bool three_d = (indcs.nx3 > 1);

  // Arrays
  auto &w0   = pmbp->pmhd->w0;    // primitive variables
  auto &b0   = pmbp->pmhd->b0;    // face-centered magnetic field
  auto &bcc0 = pmbp->pmhd->bcc0;  // cell-centered magnetic field

  // -----------------------------------------------------------------------
  // 1. Initialize primitive variables (density, velocity, pressure)
  //
  //    The equilibrium (rho, p_gas, B) is identical in 2D and 3D -- it depends on x2
  //    only.  What changes is the seed perturbation, so the 2D/3D split is confined to
  //    the v2 expression, mirroring rt.cpp's rt2d/rt3d kernels.
  // -----------------------------------------------------------------------
  Real two_pi = 2.0 * M_PI;

  // Random pool seeded per-MeshBlock-pack so multimode runs are reproducible for a
  // given decomposition (same convention as rt.cpp).
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);

  par_for("ksi_init_w", DevExeSpace(), 0, (pmbp->nmb_thispack - 1),
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // Smooth equilibrium profiles (x2 only, same in 2D and 3D)
    Real cosh_val = cosh(x2v / a);
    Real sech2    = 1.0 / (cosh_val * cosh_val);

    Real rho  = d0 + 2.0 * d0 * sech2;
    Real pgas = p0 * sech2;

    // x2-envelope: vanishes at both walls so the seed never pushes on the boundary
    Real env = 1.0 + cos(two_pi * ny_m * x2v / ly);

    // ---- Velocity perturbation in the x2-direction -----------------------
    Real v2;
    if (iprob == 1) {
      // Single mode.  The 3D form carries an extra cos(k_z x3) factor and the
      // amplitude is halved again so that max|v2| stays equal to amp in both cases.
      if (three_d) {
        v2 = amp * 0.25 * sin(two_pi * nx_m * x1v / lx)
                        * cos(two_pi * nz_m * x3v / lz) * env;
      } else {
        v2 = amp * 0.5  * sin(two_pi * nx_m * x1v / lx) * env;
      }
    } else {
      // Multimode: white noise with the same envelope. In 3D this seeds all
      // transverse wavenumbers at once instead of forcing a single coherent sheet.
      auto rand_gen = rand_pool64.get_state();
      Real r = rand_gen.frand() - 0.5;
      rand_pool64.free_state(rand_gen);
      v2 = amp * r * env;
    }

    w0(m, IDN, k, j, i) = rho;
    w0(m, IVX, k, j, i) = 0.0;
    // SR primitives are the spatial 4-velocity u^i = gamma*v^i, not the 3-velocity
    w0(m, IVY, k, j, i) = v2 / sqrt(1.0 - v2*v2);
    w0(m, IVZ, k, j, i) = 0.0;
    // w0(IEN) is the internal energy density e = p_gas/(gamma-1), NOT p_gas
    w0(m, IEN, k, j, i) = pgas / gm1;
  });

  // -----------------------------------------------------------------------
  // 2. Initialize face-centered magnetic fields
  //    B1 = 0, B2 = 0, B3 = -b0 * tanh(x2/a)
  //    Extra faces on upper boundaries handled inside loop (blast.cpp pattern)
  // -----------------------------------------------------------------------
  par_for("ksi_init_b", DevExeSpace(), 0, (pmbp->nmb_thispack - 1),
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real B3val = -b0_ * tanh(x2v / a);

    // B1 face (x1-direction): zero
    b0.x1f(m, k, j, i) = 0.0;
    if (i == ie) b0.x1f(m, k, j, ie+1) = 0.0;

    // B2 face (x2-direction): zero
    b0.x2f(m, k, j, i) = 0.0;
    if (j == je) b0.x2f(m, k, je+1, i) = 0.0;

    // B3 face (x3-direction): tanh profile (B3 depends on x2, not x3)
    // x3f face is at same (x1,x2) cell center position as the active cell
    b0.x3f(m, k, j, i) = B3val;
    if (k == ke) b0.x3f(m, ke+1, j, i) = B3val;
  });

  // -----------------------------------------------------------------------
  // 3. Compute cell-centered B as average of face values (bcc0 = avg(b0 faces))
  // -----------------------------------------------------------------------
  par_for("ksi_bcc", DevExeSpace(), 0, (pmbp->nmb_thispack - 1),
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bcc0(m, IBX, k, j, i) = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    bcc0(m, IBY, k, j, i) = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    bcc0(m, IBZ, k, j, i) = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
  });

  // -----------------------------------------------------------------------
  // 4. Convert primitives to conserved variables (SR MHD aware)
  // -----------------------------------------------------------------------
  pmbp->pmhd->peos->PrimToCons(w0, bcc0, pmbp->pmhd->u0,
                                is, ie, js, je, ks, ke);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void KSIGravitySourceTerm()
//! \brief Constant effective gravity source term in -x2 direction (SR MHD version).
//!
//! For an external force density f_i acting on the rest mass, the SR conservation laws are
//!   d(S_i)/dt = f_i          with f_2 = D*g,  D = rho*W the conserved baryon density
//!   d(tau)/dt = f_i v^i      = D*g*v^2       (the rate at which that force does work)
//!
//! The work term MUST use the 3-velocity v^2, NOT the conserved momentum S_2.  In SR MHD
//! S_2 = (rho*h + b^2)*W^2*v^2 - b^0 b^2, which exceeds D*v^2 by the factor (rho*h+b^2)/rho
//! -- about 11x in the magnetically dominated region at sigma=10 and about 8x inside the
//! sheet.  Using S_2*g (as AthenaK's built-in SourceTerms::ConstantAccel does for SR)
//! over-injects energy by that factor and destroys the hydrostatic response.
//!
//! Athena-C does the same thing correctly: integrate_2d_vl_sr.c sources the energy with the
//! conserved MASS flux x2Flux[].d = D*v^2, not with the momentum.
//!
//! WHICH INERTIA DOES GRAVITY COUPLE TO?  (problem/grav_on_enthalpy)
//!
//! This effective gravity is an INERTIAL force from the frame acceleration of the jet, so
//! physically it couples to the total inertia density w = rho*h + b^2, not to the rest mass.
//! Gill, Granot & Lyubarsky (2018) eq. (5) writes w*dv/dt = -c^2 grad(p) + (B.grad)B + w*g,
//! and their Atwood number eq. (21) is built from w_h = rho_h + 4 p_h = 23 and
//! w_c = rho_c + b0^2 = 11, giving A = 12/34 = 0.353 and eta = 1.342 (e-fold 0.745).
//!
//! Athena-C's StaticGravPot machinery can only source D*g (integrate_2d_vl_sr.c:432 uses
//! pG->U[].d, which under SPECIAL_RELATIVITY is D = rho*W).  That replaces the numerator of
//! the Atwood number by rho_h - rho_c = 2 instead of w_h - w_c = 12, so the drive is 6x
//! weaker in eta^2, i.e. sqrt(6) = 2.45x slower growth (eta ~ 0.61, e-fold ~ 1.6).
//!
//! DEFAULT is false = D*g, which reproduces the Athena-C run this port is being validated
//! against.  Set problem/grav_on_enthalpy = true to recover the paper's linear growth rate.
//!
//! D*v^2 is recovered from the primitives without a Lorentz factor: D*v^2 = (rho*W)(u^2/W)
//! = rho * u^2 = w0(IDN)*w0(IVY).  w0 holds the beginning-of-stage state (the same state the
//! fluxes were reconstructed from), which matches Athena-C's use of U^n / U^{n+1/2}.

void KSIGravitySourceTerm(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;

  Real g     = grav_acc;
  Real gamma = gam_gas;
  bool on_w  = grav_on_w;
  auto &u0   = pmbp->pmhd->u0;
  auto &w0   = pmbp->pmhd->w0;
  auto &bcc0 = pmbp->pmhd->bcc0;

  par_for("ksi_grav", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real rho = w0(m, IDN, k, j, i);
    Real ux  = w0(m, IVX, k, j, i);
    Real uy  = w0(m, IVY, k, j, i);
    Real uz  = w0(m, IVZ, k, j, i);
    Real lor = sqrt(1.0 + ux*ux + uy*uy + uz*uz);   // Lorentz factor W

    // Inertia density the force couples to, per unit coordinate volume.
    Real finert;
    if (on_w) {
      // w = rho*h + b^2 = rho + gamma*e + b^2, with the fluid-frame b^2 from the
      // cell-centered field:  b^2 = (B^2 + (B.u)^2/W^2)/W^2 * W^2 ... evaluated as
      // b^2 = B^2/W^2 + (B.v)^2, using v^i = u^i/W.
      Real bx = bcc0(m, IBX, k, j, i);
      Real by = bcc0(m, IBY, k, j, i);
      Real bz = bcc0(m, IBZ, k, j, i);
      Real bdotv = (bx*ux + by*uy + bz*uz) / lor;
      Real bsq   = (bx*bx + by*by + bz*bz) / (lor*lor) + bdotv*bdotv;
      Real wtot  = rho + gamma*w0(m, IEN, k, j, i) + bsq;
      finert = wtot * lor * lor;   // w*W^2, the coordinate-volume inertia density
    } else {
      finert = rho * lor;          // D = rho*W  (Athena-C StaticGravPot convention)
    }

    // v^2 = u^2/W, so the work rate is (inertia)*g*v^2
    Real v2 = uy / lor;

    u0(m, IM2, k, j, i) += bdt * g * finert;      // momentum: dS_2 = f*g*dt
    u0(m, IEN, k, j, i) += bdt * g * finert * v2; // energy:   dtau = f*g*v^2*dt
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void KSIBoundaries()
//! \brief Reflecting BCs in x2 with hydrostatic pressure correction.
//!
//! Applied at both inner and outer x2 boundaries (ix2_bc = ox2_bc = user).
//! Prevents spurious acoustic wave emission from gravitational stratification.
//!
//! For velocity:  v2 is reflected (antisymmetric: v2_ghost = -v2_mirror)
//! For pressure:  hydrostatic correction applied using mirror-cell density
//!                p_inner_ghost = p_mirror - rho_mirror * g * dist  (g<0 -> p increases)
//!                p_outer_ghost = p_mirror + rho_mirror * g * dist  (g<0 -> p decreases)
//! For density/v1/v3: copied from mirror cell
//! For B field:   B2 reflected (antisymmetric), B1 and B3 copied
//!
//! DIMENSIONALITY: this function is written once and works unchanged in 2D and 3D,
//! because the walls are always on x2.  n3 = nx3 + 2*ng in 3D and 1 in 2D, and the k
//! loop plus the `k == n3-1` guard covers the n3+1 x3-faces in both cases (in 2D
//! b0.x3f has exactly 2 entries in k, so the guard fills k=1).  Nothing here is
//! specialized per dimension.
//!
//! Flow follows NoInflowTorus pattern in gr_torus.cpp:
//!   1. Set ghost-zone B field (b0 face-centered)
//!   2. ConsToPrim in ghost zones (computes w0 and bcc0 from u0, b0)
//!   3. Override w0 in ghost zones (reflection + pressure correction)
//!   4. PrimToCons in ghost zones (recomputes u0 from modified w0, bcc0)

void KSIBoundaries(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int &js = indcs.js, &je = indcs.je;
  int nx2 = indcs.nx2;
  int nmb = pmbp->nmb_thispack;
  int nvar = pmbp->pmhd->u0.extent_int(1);

  auto &mb_bcs  = pmbp->pmb->mb_bcs;
  auto &size    = pmbp->pmb->mb_size;
  auto &u0_     = pmbp->pmhd->u0;
  auto &w0_     = pmbp->pmhd->w0;
  auto &b0_     = pmbp->pmhd->b0;
  auto &bcc_    = pmbp->pmhd->bcc0;

  Real g   = grav_acc;
  // w0(IEN) holds the internal energy density e = p_gas/(gamma-1), so a hydrostatic
  // PRESSURE increment dp must be applied to e as dp/(gamma-1).
  Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;
  Real efloor = pmbp->pmhd->peos->eos_data.pfloor / gm1;

  // -----------------------------------------------------------------------
  // Step 1: Set face-centered B in x2 ghost zones
  //   B1 (x1f), B3 (x3f): copy from active-zone boundary cell
  //   B2 (x2f): reflect with sign flip (antisymmetric about x2 wall)
  //
  // Index convention (inner x2 boundary):
  //   Physical boundary face: b0.x2f(js)        — NOT modified
  //   Ghost face jg=0:        b0.x2f(js-1)  <-> mirror b0.x2f(js+1)
  //   Ghost face jg=1:        b0.x2f(js-2)  <-> mirror b0.x2f(js+2)
  //
  // Index convention (outer x2 boundary):
  //   Physical boundary face: b0.x2f(je+1)       — NOT modified
  //   Ghost face jg=0:        b0.x2f(je+2)  <-> mirror b0.x2f(je)
  //   Ghost face jg=1:        b0.x2f(je+3)  <-> mirror b0.x2f(je-1)
  // -----------------------------------------------------------------------
  par_for("ksi_bc_b_x2", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int i) {
    // --- Inner X2 boundary ---
    if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int jg = 0; jg < ng; ++jg) {
        // B1 (x1-face): tangential -> symmetric MIRROR about the wall, matching
        // Athena-C reflect_ix2 (B1i[js-j] = B1i[js+j-1]).  A flat copy from the single
        // row js would zero the gradient instead of reflecting it.
        b0_.x1f(m, k, js-jg-1, i) = b0_.x1f(m, k, js+jg, i);
        if (i == n1-1) {
          b0_.x1f(m, k, js-jg-1, i+1) = b0_.x1f(m, k, js+jg, i+1);
        }
        // B2 (x2-face): normal -> antisymmetric about the inner wall (b0.x2f(js) = 0)
        // ghost face js-jg-1  mirrors  active face js+jg+1
        b0_.x2f(m, k, js-jg-1, i) = -b0_.x2f(m, k, js+jg+1, i);
        // B3 (x3-face): tangential -> symmetric mirror (Athena-C: B3i[js-j] = B3i[js+j-1])
        b0_.x3f(m, k, js-jg-1, i) = b0_.x3f(m, k, js+jg, i);
        if (k == n3-1) {
          b0_.x3f(m, k+1, js-jg-1, i) = b0_.x3f(m, k+1, js+jg, i);
        }
      }
    }

    // --- Outer X2 boundary ---
    if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int jg = 0; jg < ng; ++jg) {
        // B1 (x1-face): tangential -> symmetric MIRROR (Athena-C: B1i[je+j] = B1i[je-j+1])
        b0_.x1f(m, k, je+jg+1, i) = b0_.x1f(m, k, je-jg, i);
        if (i == n1-1) {
          b0_.x1f(m, k, je+jg+1, i+1) = b0_.x1f(m, k, je-jg, i+1);
        }
        // B2 (x2-face): normal -> antisymmetric about the outer wall (b0.x2f(je+1) = 0)
        // ghost face je+jg+2  mirrors  active face je-jg
        b0_.x2f(m, k, je+jg+2, i) = -b0_.x2f(m, k, je-jg, i);
        // B3 (x3-face): tangential -> symmetric mirror
        b0_.x3f(m, k, je+jg+1, i) = b0_.x3f(m, k, je-jg, i);
        if (k == n3-1) {
          b0_.x3f(m, k+1, je+jg+1, i) = b0_.x3f(m, k+1, je-jg, i);
        }
      }
    }
  });

  // -----------------------------------------------------------------------
  // Step 2: ConsToPrim in x2 ghost zones + the mirror active cells
  //   Converts u0 and the updated b0 into w0 (primitives) and bcc0 (cell-centered B).
  //
  //   The j-range MUST span every mirror cell that Step 3 reads, i.e. js..js+ng-1 and
  //   je-ng+1..je -- not just the single row js/je.  ApplyPhysicalBCs runs BEFORE the
  //   global ConToPrim task (see mhd_tasks.cpp: ... rkupdt -> srctrms -> ... -> bcs ->
  //   prol -> c2p), so on entry w0 still holds the PREVIOUS stage's primitives.  Reading
  //   un-refreshed mirror rows would build the ghost zones from stale data and degrade the
  //   wall to first order in time.
  // -----------------------------------------------------------------------
  pmbp->pmhd->peos->ConsToPrim(u0_, b0_, w0_, bcc_,
                                false, 0, n1-1, js-ng,    js+ng-1, 0, n3-1);
  pmbp->pmhd->peos->ConsToPrim(u0_, b0_, w0_, bcc_,
                                false, 0, n1-1, je-ng+1,  je+ng,   0, n3-1);

  // -----------------------------------------------------------------------
  // Step 3: Override primitives in x2 ghost zones
  //   - IVY (x2-velocity): reflect (antisymmetric about wall)
  //   - IPR (gas pressure): hydrostatic correction from mirror-cell density
  //   - All other vars (IDN, IVX, IVZ): copy from mirror cell
  //
  // The hydrostatic correction prevents spurious acoustic waves:
  //   Inner: p_ghost = p_mirror - rho_mirror * g * dist   (g<0 → p increases inward)
  //   Outer: p_ghost = p_mirror + rho_mirror * g * dist   (g<0 → p decreases outward)
  //   where dist = (2*jg + 1) * dx2 is the distance between mirror and ghost centers
  // -----------------------------------------------------------------------
  par_for("ksi_bc_w_x2", DevExeSpace(), 0, (nmb-1), 0, (nvar-1), 0, (n3-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int i) {
    // --- Inner X2 boundary ---
    if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::user) {
      Real dx2 = (size.d_view(m).x2max - size.d_view(m).x2min) / nx2;
      for (int jg = 0; jg < ng; ++jg) {
        int j_ghost  = js - jg - 1;   // ghost cell index
        int j_mirror = js + jg;        // mirror active cell index
        if (n == IVY) {
          // Reflect x2-velocity
          w0_(m, n, k, j_ghost, i) = -w0_(m, n, k, j_mirror, i);
        } else if (n == IEN) {
          // Hydrostatic correction (uses mirror density, not ghost density).
          // Athena-C KSI.c reflect_ix2: W.P += W.d * |g| * (2j-1) * dx2
          Real rho_mirror = w0_(m, IDN, k, j_mirror, i);
          Real e_mirror   = w0_(m, IEN, k, j_mirror, i);
          Real dist       = (2.0*jg + 1.0) * dx2;
          // g < 0 (downward), dist > 0 -> p_ghost > p_mirror (pressure increases inward)
          Real dp         = -rho_mirror * g * dist;
          w0_(m, n, k, j_ghost, i) = fmax(e_mirror + dp/gm1, efloor);
        } else {
          // Copy all other variables (IDN, IVX, IVZ)
          w0_(m, n, k, j_ghost, i) = w0_(m, n, k, j_mirror, i);
        }
      }
    }

    // --- Outer X2 boundary ---
    if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) {
      Real dx2 = (size.d_view(m).x2max - size.d_view(m).x2min) / nx2;
      for (int jg = 0; jg < ng; ++jg) {
        int j_ghost  = je + jg + 1;   // ghost cell index
        int j_mirror = je - jg;        // mirror active cell index
        if (n == IVY) {
          // Reflect x2-velocity
          w0_(m, n, k, j_ghost, i) = -w0_(m, n, k, j_mirror, i);
        } else if (n == IEN) {
          // Hydrostatic correction; Athena-C reflect_ox2: W.P -= W.d * |g| * (2j-1) * dx2
          Real rho_mirror = w0_(m, IDN, k, j_mirror, i);
          Real e_mirror   = w0_(m, IEN, k, j_mirror, i);
          Real dist       = (2.0*jg + 1.0) * dx2;
          // g < 0 (downward), dist > 0 -> p_ghost < p_mirror (pressure decreases outward)
          Real dp         = rho_mirror * g * dist;
          w0_(m, n, k, j_ghost, i) = fmax(e_mirror + dp/gm1, efloor);
        } else {
          // Copy all other variables
          w0_(m, n, k, j_ghost, i) = w0_(m, n, k, j_mirror, i);
        }
      }
    }
  });

  // -----------------------------------------------------------------------
  // Step 4: PrimToCons in x2 ghost zones
  //   Recomputes u0 (conserved) from the modified w0 and bcc0 in ghost zones.
  //   bcc0 in ghost zones was computed by ConsToPrim in Step 2.
  // -----------------------------------------------------------------------
  pmbp->pmhd->peos->PrimToCons(w0_, bcc_, u0_,
                                0, n1-1, js-ng, js-1,   0, n3-1);
  pmbp->pmhd->peos->PrimToCons(w0_, bcc_, u0_,
                                0, n1-1, je+1,  je+ng,  0, n3-1);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void KSIRefinement()
//! \brief AMR refinement condition based on normalized pressure gradient |grad(p)|/p.
//!
//! Refines where |grad(p)|/p > thr (tracks current sheet and developing structures).
//! Derefinement where |grad(p)|/p < thr/4.
//!
//! Follows AthenaK refinement pattern: flags stored in pmesh->pmr->refine_flag,
//! indexed by global MeshBlock ID (m + gids_offset).

void KSIRefinement(MeshBlockPack *pmbp) {
  auto &indcs  = pmbp->pmesh->mb_indcs;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmbp->nmb_thispack;
  bool three_d = (nx3 > 1);

  // Global MeshBlock ID offset for this MPI rank
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];

  Real thr = refine_thr;
  auto &w0_          = pmbp->pmhd->w0;
  auto &refine_flag_ = pmbp->pmesh->pmr->refine_flag;

  // Use team-level parallelism: outer loop over meshblocks, inner reduction over cells.
  // The inner range covers nx3*nx2*nx1 cells; in 2D nx3 == 1 so this reduces to the
  // planar loop and the x3 gradient term is skipped.
  par_for_outer("ksi_refine", DevExeSpace(), 0, 0, 0, (nmb-1),
  KOKKOS_LAMBDA(TeamMember_t tmember, int m) {
    const int nkji = nx3 * nx2 * nx1;
    const int nji  = nx2 * nx1;

    Real team_maxeps = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
    [=](const int idx, Real &maxeps) {
      int krel = idx / nji;
      int jrel = (idx - krel*nji) / nx1;
      int irel = idx - krel*nji - jrel*nx1;
      int k = krel + ks;
      int j = jrel + js;
      int i = irel + is;

      Real p    = w0_(m, IPR, k, j, i);
      // Centered pressure gradient (uses ghost cells, which must be valid)
      Real epsx = 0.5 * fabs(w0_(m,IPR,k,j,i+1) - w0_(m,IPR,k,j,i-1));
      Real epsy = 0.5 * fabs(w0_(m,IPR,k,j+1,i) - w0_(m,IPR,k,j-1,i));
      Real epsz = 0.0;
      if (three_d) {
        epsz = 0.5 * fabs(w0_(m,IPR,k+1,j,i) - w0_(m,IPR,k-1,j,i));
      }
      Real eps  = sqrt(epsx*epsx + epsy*epsy + epsz*epsz) / p;
      maxeps    = fmax(maxeps, eps);
    }, Kokkos::Max<Real>(team_maxeps));

    // Set refinement flag:  1 = refine, -1 = derefine, 0 = keep
    int &flag = refine_flag_.d_view(m + mbs);
    if (team_maxeps > thr) {
      flag = 1;
    } else if (team_maxeps < 0.25*thr) {
      flag = -1;
    } else {
      flag = 0;
    }
  });

  // Sync device data back to host (required after device modification of DualArray)
  refine_flag_.template modify<DevExeSpace>();
  refine_flag_.template sync<HostMemSpace>();

  return;
}
