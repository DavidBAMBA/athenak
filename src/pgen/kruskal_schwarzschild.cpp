//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kruskal_schwarzschild.cpp
//! \brief Problem generator for 2D Kruskal-Schwarzschild instability in relativistic MHD.
//!
//! This problem models a relativistic current sheet with antiparallel magnetic field
//! and a hot central layer. In the co-moving frame of a relativistic jet slab,
//! there is an effective gravity due to frame acceleration that drives the instability.
//!
//! Configuration:
//!   - Central hot layer (|x2| < delta/2): density 3*d0, no B-field
//!   - Upper/lower magnetized layers: density d0, B3 = ±b0 (antiparallel)
//!   - Effective gravity in -x2 direction (from frame acceleration)
//!   - Small velocity perturbation in x2 direction
//!
//! Parameters in <problem> block:
//!   - amp:   velocity perturbation amplitude
//!   - sigma: magnetization parameter (B^2/rho)
//!   - delta: interface thickness (total hot layer width)
//!   - nx:    mode number in x1-direction
//!   - ny:    mode number in x2-direction
//!   - drat:  density ratio of hot layer to cold layer (default 3.0)
//!
//! Adapted from Athena-C version: src/prob/2Dks.c
//!
//! REFERENCE: Kruskal & Schwarzschild 1954, Proc. R. Soc. London A 223, 348

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Problem Generator for Kruskal-Schwarzschild instability

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  // Check that MHD is enabled
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Kruskal-Schwarzschild problem requires MHD to be enabled" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Check for 2D problem
  if (pmy_mesh_->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Kruskal-Schwarzschild problem requires 2D or 3D" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Read problem parameters from input file
  Real amp   = pin->GetReal("problem", "amp");
  Real sigma = pin->GetReal("problem", "sigma");
  Real delta = pin->GetReal("problem", "delta");
  int nxm    = pin->GetOrAddInteger("problem", "nx", 1);
  int nym    = pin->GetOrAddInteger("problem", "ny", 1);
  Real drat  = pin->GetOrAddReal("problem", "drat", 3.0);

  // Domain size
  Real x1min = pmy_mesh_->mesh_size.x1min;
  Real x1max = pmy_mesh_->mesh_size.x1max;
  Real x2min = pmy_mesh_->mesh_size.x2min;
  Real x2max = pmy_mesh_->mesh_size.x2max;
  Real lx = x1max - x1min;
  Real ly = x2max - x2min;

  // Capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  // Get EOS data
  Real gamma = pmbp->pmhd->peos->eos_data.gamma;
  Real gm1 = gamma - 1.0;

  // Physical parameters (following Athena-C convention)
  Real d0 = 1.0;                      // base density
  Real b0 = std::sqrt(sigma * d0);    // magnetic field strength
  Real p0 = 0.5 * b0 * b0;            // pressure (magnetic pressure balance)

  // Get MHD arrays
  auto &w0_ = pmbp->pmhd->w0;
  auto &b0_ = pmbp->pmhd->b0;
  auto &bcc0_ = pmbp->pmhd->bcc0;

  // Initialize primitive variables and face-centered magnetic fields
  par_for("ks_pgen", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Get cell-center coordinates
    Real &x1min_mb = size.d_view(m).x1min;
    Real &x1max_mb = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min_mb, x1max_mb);

    Real &x2min_mb = size.d_view(m).x2min;
    Real &x2max_mb = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min_mb, x2max_mb);

    // Velocity perturbation in x2-direction (zero net momentum form)
    // Same formula as Athena-C: v2 = amp * 0.5 * sin(2*PI*nx*x1/lx) * (1 + cos(2*PI*ny*x2/ly))
    Real v2 = amp * 0.5 * sin(2.0 * M_PI * nxm * x1v / lx)
                       * (1.0 + cos(2.0 * M_PI * nym * x2v / ly));

    // Default: hot central layer (rho = drat*d0, no magnetic field)
    Real rho = drat * d0;
    Real pgas = p0;
    Real bz = 0.0;

    // Lower magnetized layer (x2 <= -0.5*delta)
    if (x2v <= -0.5 * delta) {
      rho = d0;
      pgas = p0;
      bz = b0;   // +B in lower layer
    }

    // Upper magnetized layer (x2 >= 0.5*delta)
    if (x2v >= 0.5 * delta) {
      rho = d0;
      pgas = p0;
      bz = -b0;  // -B in upper layer (antiparallel)
    }

    // Set primitive variables
    w0_(m, IDN, k, j, i) = rho;
    w0_(m, IVX, k, j, i) = 0.0;
    w0_(m, IVY, k, j, i) = v2;
    w0_(m, IVZ, k, j, i) = 0.0;
    w0_(m, IEN, k, j, i) = pgas / gm1;

    // Set face-centered magnetic fields
    b0_.x1f(m, k, j, i) = 0.0;
    b0_.x2f(m, k, j, i) = 0.0;
    b0_.x3f(m, k, j, i) = bz;

    // Set edge values for face-centered fields
    if (i == ie) b0_.x1f(m, k, j, i+1) = 0.0;
    if (j == je) b0_.x2f(m, k, j+1, i) = 0.0;
    if (k == ke) b0_.x3f(m, k+1, j, i) = bz;
  });

  // Compute cell-centered magnetic fields from face-centered fields
  par_for("ks_bcc", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bcc0_(m, IBX, k, j, i) = 0.5 * (b0_.x1f(m, k, j, i) + b0_.x1f(m, k, j, i+1));
    bcc0_(m, IBY, k, j, i) = 0.5 * (b0_.x2f(m, k, j, i) + b0_.x2f(m, k, j+1, i));
    bcc0_(m, IBZ, k, j, i) = 0.5 * (b0_.x3f(m, k, j, i) + b0_.x3f(m, k+1, j, i));
  });

  // Convert primitives to conserved variables
  auto &u0_ = pmbp->pmhd->u0;
  pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);

  return;
}
