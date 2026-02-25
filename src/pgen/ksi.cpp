//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ksi.cpp
//! \brief Problem generator for the Kruskal-Schwarzschild instability (KSI)
//!        in special relativistic MHD.
//!
//! Ported from Athena++ 2dksi.cpp to the GPU-capable AthenaK version.
//! Supports both 2D and 3D configurations (auto-detected from mesh dimensions).
//!
//! Equilibrium profiles (smooth tanh/sech^2, following Athena-C KSI.c):
//!   B3     = -b0 * tanh(x2/a)        (reversed-field current sheet)
//!   rho    = d0 + 2*d0 * sech^2(x2/a)
//!   pgas   = p0 * sech^2(x2/a)       where p0 = b0^2/2
//!
//! Total pressure balance: pgas + B^2/2 = p0 everywhere
//! (uses the identity sech^2(xi) + tanh^2(xi) = 1, and p0 = b0^2/2).
//!
//! Velocity perturbation in x2-direction only:
//!   2D: v2 = amp * 0.5 * sin(2*pi*nx*x1/lx) * (1 + cos(2*pi*ny*x2/ly))
//!   3D: v2 = amp * 0.25 * sin(2*pi*nx*x1/lx) * cos(2*pi*nz*x3/lz)
//!                       * (1 + cos(2*pi*ny*x2/ly))
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

// Namespace for parameters shared across user-defined functions
namespace {
  Real grav_acc;    // effective gravitational acceleration (negative = -x2 direction)
  Real refine_thr;  // AMR refinement threshold on |grad(p)|/p
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

  // Store shared parameters for user BC and source term functions
  grav_acc   = pin->GetOrAddReal("problem", "grav", -0.1);
  refine_thr = pin->GetOrAddReal("problem", "thr",  0.25);

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

  // Derived equilibrium parameters
  Real d0  = 1.0;
  Real b0_ = std::sqrt(sigma * d0);  // field amplitude: b0 = sqrt(sigma)
  Real p0  = 0.5 * b0_ * b0_;        // pressure balance: p0 = b0^2/2

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
  // -----------------------------------------------------------------------
  Real two_pi = 2.0 * M_PI;
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

    // Smooth equilibrium profiles
    Real cosh_val = cosh(x2v / a);
    Real sech2    = 1.0 / (cosh_val * cosh_val);

    Real rho  = d0 + 2.0 * d0 * sech2;
    Real pgas = p0 * sech2;

    // Velocity perturbation in x2-direction
    // For small amp, spatial 4-velocity u^2 ≈ 3-velocity v^2
    Real v2;
    if (three_d) {
      v2 = amp * 0.25
           * sin(two_pi * nx_m * x1v / lx)
           * cos(two_pi * nz_m * x3v / lz)
           * (1.0 + cos(two_pi * ny_m * x2v / ly));
    } else {
      v2 = amp * 0.5
           * sin(two_pi * nx_m * x1v / lx)
           * (1.0 + cos(two_pi * ny_m * x2v / ly));
    }

    w0(m, IDN, k, j, i) = rho;
    w0(m, IVX, k, j, i) = 0.0;
    w0(m, IVY, k, j, i) = v2;
    w0(m, IVZ, k, j, i) = 0.0;
    w0(m, IPR, k, j, i) = pgas;
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
//! Adds to conserved variables:
//!   Momentum: dS_2/dt = D * g    where D = rho*gamma (conserved baryon density)
//!   Energy:   dtau/dt = S_2 * g  (work done by gravity on existing momentum)
//!
//! This is the AthenaK-native SR convention, equivalent to the Athena++ flux-averaged
//! formulation in the continuous limit.

void KSIGravitySourceTerm(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;

  Real g = grav_acc;
  auto &u0 = pmbp->pmhd->u0;

  par_for("ksi_grav", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real D  = u0(m, IDN, k, j, i);   // conserved density D = rho * Lorentz factor
    Real S2 = u0(m, IM2, k, j, i);   // x2-momentum before gravity update
    u0(m, IM2, k, j, i) += bdt * g * D;   // momentum source: S_M2 += D*g*dt
    u0(m, IEN, k, j, i) += bdt * g * S2;  // energy source:   tau  += S2*g*dt
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

  Real g = grav_acc;

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
        // B1 (x1-face): copy from first active cell row
        b0_.x1f(m, k, js-jg-1, i) = b0_.x1f(m, k, js, i);
        if (i == n1-1) {
          b0_.x1f(m, k, js-jg-1, i+1) = b0_.x1f(m, k, js, i+1);
        }
        // B2 (x2-face): antisymmetric about the inner wall (b0.x2f(js) = 0)
        // ghost face js-jg-1  mirrors  active face js+jg+1
        b0_.x2f(m, k, js-jg-1, i) = -b0_.x2f(m, k, js+jg+1, i);
        // B3 (x3-face): copy from first active cell row
        b0_.x3f(m, k, js-jg-1, i) = b0_.x3f(m, k, js, i);
        if (k == n3-1) {
          b0_.x3f(m, k+1, js-jg-1, i) = b0_.x3f(m, k+1, js, i);
        }
      }
    }

    // --- Outer X2 boundary ---
    if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int jg = 0; jg < ng; ++jg) {
        // B1 (x1-face): copy from last active cell row
        b0_.x1f(m, k, je+jg+1, i) = b0_.x1f(m, k, je, i);
        if (i == n1-1) {
          b0_.x1f(m, k, je+jg+1, i+1) = b0_.x1f(m, k, je, i+1);
        }
        // B2 (x2-face): antisymmetric about the outer wall (b0.x2f(je+1) = 0)
        // ghost face je+jg+2  mirrors  active face je-jg
        b0_.x2f(m, k, je+jg+2, i) = -b0_.x2f(m, k, je-jg, i);
        // B3 (x3-face): copy from last active cell row
        b0_.x3f(m, k, je+jg+1, i) = b0_.x3f(m, k, je, i);
        if (k == n3-1) {
          b0_.x3f(m, k+1, je+jg+1, i) = b0_.x3f(m, k+1, je, i);
        }
      }
    }
  });

  // -----------------------------------------------------------------------
  // Step 2: ConsToPrim in x2 ghost zones + boundary active cells
  //   This converts u0 (unchanged in ghost zones) and the updated b0 into
  //   w0 (primitives) and bcc0 (cell-centered B). For SR MHD, this involves
  //   an iterative inversion.
  // -----------------------------------------------------------------------
  pmbp->pmhd->peos->ConsToPrim(u0_, b0_, w0_, bcc_,
                                false, 0, n1-1, js-ng, js,   0, n3-1);
  pmbp->pmhd->peos->ConsToPrim(u0_, b0_, w0_, bcc_,
                                false, 0, n1-1, je,    je+ng, 0, n3-1);

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
        } else if (n == IPR) {
          // Hydrostatic pressure correction (uses mirror density, not ghost density)
          Real rho_mirror = w0_(m, IDN, k, j_mirror, i);
          Real p_mirror   = w0_(m, IPR, k, j_mirror, i);
          Real dist       = (2.0*jg + 1.0) * dx2;
          // g < 0 (downward), dist > 0 → p_ghost > p_mirror (pressure increases inward)
          w0_(m, n, k, j_ghost, i) = fmax(p_mirror - rho_mirror*g*dist, 1.0e-15);
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
        } else if (n == IPR) {
          // Hydrostatic pressure correction
          Real rho_mirror = w0_(m, IDN, k, j_mirror, i);
          Real p_mirror   = w0_(m, IPR, k, j_mirror, i);
          Real dist       = (2.0*jg + 1.0) * dx2;
          // g < 0 (downward), dist > 0 → p_ghost < p_mirror (pressure decreases outward)
          w0_(m, n, k, j_ghost, i) = fmax(p_mirror + rho_mirror*g*dist, 1.0e-15);
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
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2;
  int nmb = pmbp->nmb_thispack;

  // Global MeshBlock ID offset for this MPI rank
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];

  Real thr = refine_thr;
  auto &w0_          = pmbp->pmhd->w0;
  auto &refine_flag_ = pmbp->pmesh->pmr->refine_flag;

  // Use team-level parallelism: outer loop over meshblocks, inner reduction over cells
  par_for_outer("ksi_refine", DevExeSpace(), 0, 0, 0, (nmb-1),
  KOKKOS_LAMBDA(TeamMember_t tmember, int m) {
    const int nkji = nx2 * nx1;

    Real team_maxeps = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
    [=](const int idx, Real &maxeps) {
      int jrel = idx / nx1;
      int irel = idx - jrel * nx1;
      int j = jrel + js;
      int i = irel + is;

      Real p    = w0_(m, IPR, ks, j, i);
      // Centered pressure gradient (uses ghost cells, which must be valid)
      Real epsx = 0.5 * fabs(w0_(m,IPR,ks,j,i+1) - w0_(m,IPR,ks,j,i-1));
      Real epsy = 0.5 * fabs(w0_(m,IPR,ks,j+1,i) - w0_(m,IPR,ks,j-1,i));
      Real eps  = sqrt(epsx*epsx + epsy*epsy) / p;
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
