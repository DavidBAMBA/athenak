#!/usr/bin/env python3
# ==============================================================================
#  analysis_ksi_1.py -- analisis Gill+2018 (Figs. 4, 6 y 10) para UNA corrida
#                       2D de la KSI en AthenaK
# ==============================================================================
#
#  Consolida en un solo script el analisis que antes estaba repartido en
#  Analisis-fourier.py / plot-analisis-fourier.py / volume-averaged.py /
#  gamma-analisis.py / analisis-velocidad.py / Reconection-rate-analisis.py
#  (la version Athena-C + pyvista de 2024), ahora leyendo los VTK legacy que
#  escribe AthenaK directamente con numpy (sin pyvista ni VTK instalados).
#
#  FIGURAS (numeracion del paper Gill, Granot & Lyubarsky 2018, MNRAS 474, 3535)
#
#   fig4_growth.png
#     izquierda: ln<f_1>(t) para regiones de mezcla Dz = 0.02, 0.05, 0.1
#                centradas en x2=0, mas la recta de la teoria lineal
#                eta = (g/2)^(1/2) k^(3/4) delta^(1/4)   (rama k*delta < 1,
#                Gill eq. 11; identica a la usada en el analisis viejo)
#     derecha:   espectro <f_m>(m) en t = 2, 3, 4, 5 (Dz = 0.05)
#
#   fig6_averages.png
#     izquierda: ln<f_1>(t) (Dz = 0.1) + teoria + ajuste medido
#     derecha:   <U_b> = <B^2/2>,  <U_th> = <e_int>,  <gamma>-1  vs t
#
#   fig10_vz_dissipation.png
#     izquierda: Max[v2](x2) y Min[v2](x2) (extremos sobre x1) al tiempo
#                --tvz (default: el ultimo dump)
#     derecha:   |dE_b/dt| vs t (Savitzky-Golay sobre E_b = integral de B^2/2)
#
#  La amplitud de Fourier sigue la definicion del paper (eq. del texto de la
#  Fig. 4): en cada altura x2,
#      f_m(x2) = (1/Ly) int [rho/rho_bar(x2) - 1] e^{-i 2 pi m x1 / Ly} dx1
#  y <f_m> es el promedio de |f_m| sobre la region de mezcla Dz.
#
#  USO
#    python3 analysis_ksi_1.py /ruta/al/run_dir [opciones]
#
#  donde run_dir contiene vtk/ksi.mhd_w.NNNNN.vtk (y ksi.mhd_bcc.NNNNN.vtk
#  para U_b y la disipacion) y, opcionalmente, el athinput de la corrida
#  (athinput.* o in.athinput) del que se leen g, a, delta y amp.
#
#    --outdir DIR   default: <run_dir>/figs_analysis
#    --fit T0 T1    ventana del ajuste lineal de ln<f_1> (default 1.0 3.0)
#    --tvz T        tiempo del perfil de v2 de la fig 10 (default: ultimo)
#    --mmax M       modos del espectro (default 20)
#    --csv          ademas escribe series de tiempo en analysis_ksi_1.csv
#
#  Ejemplo (la corrida 2D de produccion en GPU):
#    python3 vis/python/analysis_ksi_1.py build_ksi_gpu/src/run_ksi --csv
# ==============================================================================

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ------------------------------------------------------------- lector VTK ---
def read_vtk(filename):
    """VTK legacy STRUCTURED_POINTS + CELL_DATA de AthenaK (un solo rank).

    Devuelve dict: 'time', 'x1', 'x2' (centros de celda), y cada variable como
    array (nx2, nx1) float32.  Asume nx3 = 1 (dumps 2D).
    """
    with open(filename, 'rb') as fp:
        raw = fp.read()

    # header de texto hasta CELL_DATA
    head_end = raw.index(b'CELL_DATA')
    head = raw[:head_end].decode('ascii', 'ignore')
    m = re.search(r'time=\s*([0-9eE.+-]+)', head)
    time = float(m.group(1)) if m else np.nan
    dims = [int(v) for v in re.search(r'DIMENSIONS\s+(\d+)\s+(\d+)\s+(\d+)', head).groups()]
    orig = [float(v) for v in re.search(
        r'ORIGIN\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)', head).groups()]
    spac = [float(v) for v in re.search(
        r'SPACING\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)', head).groups()]
    nx1, nx2 = dims[0] - 1, dims[1] - 1          # DIMENSIONS son puntos, no celdas
    ncell = nx1 * nx2 * max(dims[2] - 1, 1)

    out = {'time': time,
           'x1': orig[0] + spac[0]*(np.arange(nx1) + 0.5),
           'x2': orig[1] + spac[1]*(np.arange(nx2) + 0.5),
           'dx1': spac[0], 'dx2': spac[1]}

    # bloques SCALARS <name> float / LOOKUP_TABLE default / <binario big-endian>
    pos = head_end
    while True:
        m = re.compile(rb'SCALARS\s+(\S+)\s+float\s*\nLOOKUP_TABLE\s+default\s*\n')\
              .search(raw, pos)
        if m is None:
            break
        name = m.group(1).decode()
        a0 = m.end()
        data = np.frombuffer(raw, dtype='>f4', count=ncell, offset=a0)
        out[name] = data.reshape(nx2, nx1).astype(np.float32)
        pos = a0 + 4*ncell
    return out


def dump_index(path):
    return int(re.search(r'\.(\d+)\.vtk$', path).group(1))


# --------------------------------------------------------------- parametros --
def read_params(rundir):
    """g, a, delta, amp, gamma_ad desde el athinput de la corrida (si existe)."""
    p = dict(g=0.1, a=0.005, delta=0.01, amp=9.5346e-7, gamma_ad=4.0/3.0)
    cands = (glob.glob(os.path.join(rundir, 'athinput.*'))
             + glob.glob(os.path.join(rundir, 'in.athinput'))
             + glob.glob(os.path.join(rundir, '*.athinput')))
    if not cands:
        print('      AVISO: no encontre athinput en el run_dir; uso defaults', p)
        return p
    txt = open(cands[0]).read()
    def get(key, default):
        m = re.search(rf'^{key}\s*=\s*([0-9eE.+-]+)', txt, re.M)
        return float(m.group(1)) if m else default
    p['g'] = abs(get('grav', -p['g']))
    p['a'] = get('a', p['a'])
    p['delta'] = get('delta', p['delta'])
    p['amp'] = get('amp', p['amp'])
    p['gamma_ad'] = get('gamma', p['gamma_ad'])
    return p


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description='Figs. 4, 6 y 10 de Gill+2018 para una corrida 2D KSI de AthenaK')
    ap.add_argument('rundir')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--fit', nargs=2, type=float, default=[1.0, 3.0], metavar=('T0','T1'))
    ap.add_argument('--tvz', type=float, default=-1.0, help='tiempo del perfil v2 (fig10); <0 = ultimo dump')
    ap.add_argument('--mmax', type=int, default=20)
    ap.add_argument('--spectrum-times', nargs='+', type=float, default=[2.0, 3.0, 4.0, 5.0])
    ap.add_argument('--csv', action='store_true')
    args = ap.parse_args()

    rundir = args.rundir.rstrip('/')
    outdir = args.outdir or os.path.join(rundir, 'figs_analysis')
    os.makedirs(outdir, exist_ok=True)

    wfiles = sorted(glob.glob(os.path.join(rundir, 'vtk', '*.mhd_w.*.vtk')), key=dump_index)
    if not wfiles:
        wfiles = sorted(glob.glob(os.path.join(rundir, '*.mhd_w.*.vtk')), key=dump_index)
    if not wfiles:
        sys.exit(f'ERROR: no hay *.mhd_w.*.vtk en {rundir}(/vtk)')
    P = read_params(rundir)
    print(f'[1/4] {len(wfiles)} dumps mhd_w    params: g={P["g"]}, delta={P["delta"]}, '
          f'a={P["a"]}, amp={P["amp"]:.3e}')

    # geometria del primer dump
    d0 = read_vtk(wfiles[0])
    x1, x2 = d0['x1'], d0['x2']
    Ly = x1[-1] - x1[0] + d0['dx1']
    DZ = (0.02, 0.05, 0.10)
    masks = {dz: np.abs(x2) <= dz/2 for dz in DZ}
    have_bcc = os.path.exists(wfiles[0].replace('mhd_w', 'mhd_bcc'))

    # ------------------------------------------------ pasada sobre los dumps
    times, f1 = [], {dz: [] for dz in DZ}
    spec = {}                                    # t_pedido -> <f_m>(m), Dz=0.05
    Ub, Uth, gam1 = [], [], []
    spectrum_left = sorted(args.spectrum_times)
    vz_prof, vz_time = None, None
    tvz_target = args.tvz

    for n, f in enumerate(wfiles):
        d = read_vtk(f)
        t = d['time']
        times.append(t)
        rho = d['dens']

        # ---- amplitud de Fourier de la perturbacion de densidad (Fig 4)
        rho_bar = rho.mean(axis=1, keepdims=True)            # media horizontal
        delta_rho = rho/rho_bar - 1.0
        fm = np.abs(np.fft.rfft(delta_rho, axis=1))/rho.shape[1]   # |f_m|(x2)
        for dz in DZ:
            f1[dz].append(fm[masks[dz], 1].mean())
        while spectrum_left and t >= spectrum_left[0] - 1e-9:
            tt = spectrum_left.pop(0)
            spec[tt] = fm[masks[0.05], 1:args.mmax+1].mean(axis=0)

        # ---- promedios de volumen (Fig 6 derecha)
        u1, u2v, u3 = d['velx'], d['vely'], d['velz']
        W = np.sqrt(1.0 + u1*u1 + u2v*u2v + u3*u3)
        gam1.append(W.mean() - 1.0)
        Uth.append(d['eint'].mean())
        if have_bcc:
            b = read_vtk(f.replace('mhd_w', 'mhd_bcc'))
            Ub.append(0.5*(b['bcc1']**2 + b['bcc2']**2 + b['bcc3']**2).mean())

        # ---- perfil de v2 (Fig 10 izquierda)
        is_last = (n == len(wfiles) - 1)
        if (tvz_target >= 0 and vz_prof is None and t >= tvz_target - 1e-9) or \
           (tvz_target < 0 and is_last):
            v2 = u2v/W
            vz_prof = (v2.max(axis=1), v2.min(axis=1))
            vz_time = t
        if n % 25 == 0:
            print(f'      dump {n}/{len(wfiles)}  t={t:.2f}')

    times = np.array(times)
    for dz in DZ:
        f1[dz] = np.array(f1[dz])

    # ------------------------------------------------ teoria lineal + ajuste
    lam = Ly                                     # modo m=1: lambda_0 = Ly
    k = 2*np.pi/lam
    eta_th = np.sqrt(P['g']/2.0) * k**0.75 * P['delta']**0.25   # rama k*delta<1
    t0, t1 = args.fit
    w = (times >= t0) & (times <= t1) & (f1[0.10] > 0)
    eta_fit, lnf0 = np.polyfit(times[w], np.log(f1[0.10][w]), 1)
    print(f'[2/4] eta teoria (k*delta={k*P["delta"]:.2f}<1) = {eta_th:.3f}   '
          f'eta medido (fit {t0}-{t1}, Dz=0.1) = {eta_fit:.3f}   '
          f'razon = {eta_fit/eta_th:.2f}')

    # ------------------------------------------------------------- figura 4
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.5))
    cols = {0.02: 'r', 0.05: 'b', 0.10: 'g'}
    for dz in DZ:
        axL.plot(times, np.log(np.maximum(f1[dz], 1e-30)), cols[dz],
                 label=fr'$\Delta z = {dz:g}$')
    tt = np.linspace(0, times.max(), 50)
    for dz in DZ:                                # teoria anclada a cada curva
        ww = (times >= t0) & (times <= t1)
        off = np.median(np.log(np.maximum(f1[dz][ww], 1e-30)) - eta_th*times[ww])
        axL.plot(tt, eta_th*tt + off, 'k--', lw=1)
    axL.set_xlabel('$t$'); axL.set_ylabel(r'$\ln\langle f_1\rangle$')
    axL.set_title(fr'$\kappa_v={P["amp"]:.2g}$, $g={P["g"]:g}$, $\lambda_0={lam:g}$'
                  fr'   ($\eta_{{teo}}={eta_th:.2f}$, $\eta_{{fit}}={eta_fit:.2f}$)')
    axL.set_xlim(0, min(times.max(), 6.0)); axL.set_ylim(-14, -1)
    axL.legend(loc='lower right', fontsize=9)
    for tt_, fm_ in sorted(spec.items()):
        axR.plot(np.arange(1, len(fm_)+1), np.log(np.maximum(fm_, 1e-30)),
                 label=f'$t={tt_:g}$')
    axR.set_xlabel('$m$'); axR.set_ylabel(r'$\ln\langle f_m\rangle$')
    axR.set_title(r'espectro de modos ($\Delta z=0.05$)')
    axR.legend(fontsize=9)
    fig.suptitle('Fig. 4 de Gill+2018 -- crecimiento lineal y espectro', y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig4_growth.png'), dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------- figura 6
    fig = plt.figure(figsize=(11, 6))
    axL = fig.add_subplot(1, 2, 1)
    axL.plot(times, np.log(np.maximum(f1[0.10], 1e-30)), 'b', label=r'$\Delta z=0.1$')
    off = lnf0
    axL.plot(tt, eta_fit*tt + off, 'b--', lw=1, label=fr'fit $\eta={eta_fit:.2f}$')
    off2 = np.median(np.log(np.maximum(f1[0.10][w], 1e-30)) - eta_th*times[w])
    axL.plot(tt, eta_th*tt + off2, 'k:', lw=1.2, label=fr'teoria $\eta={eta_th:.2f}$')
    axL.set_xlabel('$t$'); axL.set_ylabel(r'$\ln\langle f_1\rangle$')
    axL.set_xlim(0, min(times.max(), 6.0)); axL.set_ylim(-14, -1)
    axL.legend(fontsize=9)
    gs = fig.add_gridspec(3, 2)
    axs = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    if Ub:
        axs[0].plot(times, Ub, 'b'); axs[0].set_ylabel(r'$\langle U_b\rangle$')
    else:
        axs[0].text(0.5, 0.5, 'sin dumps mhd_bcc', ha='center', transform=axs[0].transAxes)
    axs[1].plot(times, Uth, 'b'); axs[1].set_ylabel(r'$\langle U_{th}\rangle$')
    axs[2].semilogy(times, np.maximum(gam1, 1e-12), 'b')
    axs[2].set_ylabel(r'$\langle\gamma\rangle-1$'); axs[2].set_xlabel('$t$')
    for ax in axs[:2]:
        ax.tick_params(labelbottom=False)
    fig.suptitle('Fig. 6 de Gill+2018 -- crecimiento vs teoria y promedios de volumen')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig6_averages.png'), dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------ figura 10
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    if vz_prof is not None:
        axL.plot(x2, vz_prof[0], 'r', label=r'Max$[v_2]$')
        axL.plot(x2, vz_prof[1], 'b', label=r'Min$[v_2]$')
        axL.set_title(f'perfil de $v_2$ en $t={vz_time:.2f}$')
    axL.set_xlabel('$x_2$'); axL.set_ylabel('$v_2$'); axL.legend(fontsize=9)
    if Ub:
        Eb = np.array(Ub)                        # <B^2/2>; volumen cte -> prop. a E_b
        try:
            from scipy.signal import savgol_filter
            nwin = min(21, (len(Eb)//2)*2 - 1)
            dt = np.median(np.diff(times))
            Ebdot = savgol_filter(Eb, nwin, 3, deriv=1, delta=dt)
        except Exception:
            Ebdot = np.gradient(Eb, times)
        axR.semilogy(times, np.abs(Ebdot), 'k')
        axR.set_xlabel('$t$'); axR.set_ylabel(r'$|d\langle U_b\rangle/dt|$')
        axR.set_title('tasa de disipacion magnetica')
    fig.suptitle('Fig. 10 de Gill+2018 -- velocidad pico y disipacion')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig10_vz_dissipation.png'), dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------------ csv
    if args.csv:
        import csv as _csv
        with open(os.path.join(outdir, 'analysis_ksi_1.csv'), 'w', newline='') as fh:
            wcsv = _csv.writer(fh)
            hdr = ['time'] + [f'f1_dz{dz:g}' for dz in DZ] + ['Uth', 'gamma_m1']
            if Ub:
                hdr.append('Ub')
            wcsv.writerow(hdr)
            for i, t in enumerate(times):
                row = [t] + [f1[dz][i] for dz in DZ] + [Uth[i], gam1[i]]
                if Ub:
                    row.append(Ub[i])
                wcsv.writerow(row)

    print(f'[4/4] listo -> {outdir}/fig4_growth.png, fig6_averages.png, '
          f'fig10_vz_dissipation.png' + (', analysis_ksi_1.csv' if args.csv else ''))


if __name__ == '__main__':
    main()
