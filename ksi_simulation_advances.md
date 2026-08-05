# KSI en AthenaK — estado, hallazgos y siguientes pasos

Documento de traspaso. Resume todo lo aprendido portando la Kruskal-Schwarzschild
Instability (KSI) de Athena C-version a AthenaK, en 2D y 3D.

---

## 1. El problema y las referencias

**KSI**: inestabilidad tipo Rayleigh-Taylor en una lámina de corriente (current sheet)
magnetizada, en SR-MHD. Modela la aceleración de marco de un jet relativista: en el
marco comóvil del slab aparece una gravedad efectiva que hace flotar el plasma denso
sobre las capas magnetizadas ligeras.

**Papers en el repo:**

| Archivo | Es | Rol |
|---|---|---|
| `hydro-ksistx3000.pdf` | **Gill, Granot & Lyubarsky 2018, MNRAS 474, 3535** | **La fuente real del setup.** Todo parámetro sale de aquí |
| `kinetic_ksi2412.09541v1.pdf` | Gröger, Hakobyan & Sironi 2024 (PIC cinético) | *No* es la fuente del setup MHD (usa doble Harris periódico). Sí útil para recetas numéricas |

**Referencia de comparación**: `/home/davidbamba/repositories/Athena-Cversion/` —
`src/prob/KSI.c` + `tst/2D-sr-mhd/athinput.2Dks`. Solo sobreviven artefactos
post-procesados (`run_2Dks/density_evolution.mp4`, 193 frames, t=0→19.2); los dumps
crudos 2D se borraron. Su corrida 3D (`run_3Dks`) solo llegó a t=0.06 — nunca corrió
de verdad.

### Equilibrio (idéntico en 2D y 3D, función solo de x2)

```
B3   = -b0·tanh(x2/a)          b0 = sqrt(sigma) = sqrt(10)
rho  = 1 + 2·sech²(x2/a)       (rho: 1 fuera, 3 en la lámina)
pgas = p0·sech²(x2/a)          p0 = b0²/2 = 5
```

`p_gas + B²/2 = p0 = 5` exacto (identidad sech²+tanh²=1). Gravedad efectiva en −x2.

Parámetros: `sigma=10`, `Delta=0.01` (ancho **completo** de la capa caliente),
`a = Delta/2 = 0.005` (longitud de escala del tanh), `gamma=4/3`, caja
`(Lx1, Lx2) = (0.1, 0.2)`, x1 periódico, x2 paredes reflectantes.
`amp = 9.5346e-7 = kappa_v·v_A` con `kappa_v = 1e-6` (fiducial de Gill, ec. 15).

### Geometría en AthenaK

```
x1  transversal, PERIÓDICO           (la 'y' de Gill)
x2  normal a la lámina, gravedad, PAREDES REFLECTANTES   (la 'z' de Gill)
x3  fuera del plano 2D, PERIÓDICO; dirección de B
```

**En 3D la gravedad y las paredes siguen en x2** (a diferencia de `rt.cpp`, que rota la
gravedad a x3 en 3D). x3 debe seguir siendo la dirección del campo. Esto hace que
`KSIBoundaries()` funcione sin cambios en 2D y 3D.

---

## 2. Estado del repositorio

Fork `DavidBAMBA/athenak`, rama `main`, sincronizado con upstream
`IAS-Astrophysics/athenak` (merge de 59 commits ya hecho).

**Archivos propios:**

| Archivo | Qué es |
|---|---|
| `src/pgen/ksi.cpp` | Problem generator, 2D y 3D |
| `src/mhd/rsolvers/hlld_srmhd.hpp` | **SR HLLD que no existía en AthenaK** (port de Athena++ `hlld_rel.cpp`, algoritmo MUB 2009) |
| `inputs/srmhd/athinput.2dksi` | Config 2D validada |
| `inputs/srmhd/athinput.3dksi` | Config 3D producción |
| `build_ksi_gpu/src/plot_density3d.py` | Lector `.bin` + video 3 paneles |
| `build_ksi_gpu/src/analyze_ksi.py` | Diagnósticos de crecimiento |
| `build_test_hlld/` | Suite de validación del HLLD (tubos de choque MUB 1–4) |

**Build GPU local** (laptop RTX 5060):
```
NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/g++-13 make -j8
```
(el gcc de conda es 15.2 y CUDA 12.8 lo rechaza)

---

## 3. Bugs encontrados y corregidos en el port 2D

La versión inicial no reproducía nada. Cuatro defectos, verificados con los datos:

### 3.1 CRÍTICO — `w0(IPR)` es energía interna, no presión

En AthenaK `IPR == IEN == 4`, y ese slot guarda **e = p/(γ−1)**, no p. El pgen escribía
`w0(IPR) = pgas` sin `/gm1`, así que la presión real era **3× menor** (γ=4/3).

Consecuencia medida del VTK: el balance de presión total se rompía — `p_tot` iba de
**1.667 en la lámina a 5.01 afuera** (contraste 3×) en vez de 5.0 uniforme. El
desbalance era **3.5 millones de veces** la perturbación semilla. La lámina implotaba
en t<0.05.

Firma: la prueba definitiva fue que el piso de presión en el output valía exactamente
`0.03 = pfloor/(γ−1)`, no `pfloor`.

### 3.2 CRÍTICO — `pfloor = 1e-2`

La presión de fondo física es `5·sech²(20) = 8.7e-17`. El piso estaba **14 órdenes por
encima** y fijaba el 83.8% de las celdas. Athena-C usa `TINY_NUMBER = 1e-20`.

### 3.3 MAYOR — término de energía de la gravedad

Tenía `u0(IEN) += bdt·g·S2` con S2 el momento SR. Athena-C
(`integrate_2d_vl_sr.c:432`) usa el **flujo de masa** `D·v²`. Como
`S2 = (ρh+b²)W²v²`, sobre-inyectaba energía **8–11×**.

**Nota**: el `SourceTerms::ConstantAccel` nativo de AthenaK tiene el mismo error para SR.

### 3.4 MENOR — BCs

`w0` está *stale* cuando corre `user_bcs_func` (el `ConsToPrim` global va después,
`mhd_tasks.cpp:73-75`), y solo se refrescaba 1 de las 4 filas espejo. Además B1/B3 se
copiaban en vez de reflejarse.

### Resultado tras corregir

| | t=0 tot-E | t=0.05 2-KE | t=0.05 3-ME |
|---|---|---|---|
| Antes | 1.00487e-2 | 3.02e-4 | 9.3882e-3 (−1.19%) |
| **Después** | **1.10000e-2** | **1.98e-8** | 9.50001e-3 (+1e-6) |

`tot-E = 1.10e-2` coincide exactamente con ⟨U_b⟩+⟨U_th⟩ = 4.75+0.75 = 5.5 de la Fig. 6
de Gill (×V=0.002).

---

## 4. Equivalencias numéricas Athena-C ↔ AthenaK

Verificadas leyendo el código de Athena-C, **no** por inferencia:

| | Athena-C | AthenaK | Nota |
|---|---|---|---|
| Reconstrucción | `--with-order=3p` → `lr_states_prim3.c` | **`ppmx`** | Ambos usan limitadores **Colella-Sekora** extremum-preserving. `ppm4` sería el limitador clásico y **no** coincide |
| CFL | `cour_no=0.5` | **`cfl_number=0.5`** (2D) | Convención **idéntica**: ambos hacen `dt = cfl·min_i(dx_i/c_i)`. `new_dt.c:168-174` usa MAX sobre direcciones, no suma |
| Integrador | VL2 | `rk2` | AthenaK no tiene vl2; rk2 es lo más cercano |
| Riemann | `hlld_sr.c` | **`hlld`** (ver §5) | |
| Piso presión | `TINY_NUMBER=1e-20` | `pfloor=1e-18` | debe estar bajo `p0·sech²(x2max/a)=8.66e-17` |
| Gravedad | `U.d` = D | `D·g` | ambos sobre masa en reposo, no entalpía |

**Trampa**: `delta = 0.01` es el ancho **completo** de Gill, y `a = delta/2 = 0.005` es
la escala del tanh. `ksi.cpp` hace `a = pin->GetOrAddReal("a", delta)` — si omites `a`,
duplicas el grosor de la lámina.

---

## 5. SR HLLD: implementado desde cero

AthenaK **no tenía** HLLD para SR-MHD (solo `llf` y `hlle`). Lo porté de Athena++
`src/hydro/rsolvers/mhd/hlld_rel.cpp` (algoritmo Mignone, Ugliano & Bodo 2009).

- Archivo nuevo: `src/mhd/rsolvers/hlld_srmhd.hpp`
- Cableado: `mhd.hpp` (enum), `mhd.cpp` (registro), `mhd_fluxes.cpp`, `mhd_tasks.cpp`
- Reestructurado: Athena++ escribe el bloque del residuo 3× por SIMD; lo factoricé en
  una función `SRHLLDResidual()` + struct `HLLDFan`

**Validación** (`build_test_hlld/`, tubos de choque MUB 1–4 del propio paper):
- Convergencia: a 8× resolución HLLE y HLLD coinciden a 1e-4–1e-3 → mismo solución débil
- MUB1: HLLD tiene 15% menos error L1, `vely` **28% mejor** (la discontinuidad
  rotacional que existe para resolver)
- Costo: 1.23–1.54× HLLE en CPU

**Tras el merge con upstream** el PR #757 cambió la API de los solvers (de rango con
`ScrArray2D` a por-cara con template `<ivx>` y `DvceArray5D`). Porté HLLD a la nueva
API y verifiqué **bit a bit idéntico** pre/post merge.

### ✅ HLLD en 3D: ARREGLADO (2026-08-05)

**Historia.** HLLD destruía el fondo en 3D. Comparación controlada (96×192×48
isótropa, cfl=0.3, iprob=1, p_bg=1e-4), std(ρ) en |x2|>4a:

| t | HLLE | HLLD original | HLLD arreglado |
|---|---|---|---|
| 0.25 | 1.06e-2 | 6.7e-2 (max\|δρ\| = 2.9) | **1.06e-2** |
| 0.50 | 2.56e-3 | 7.5e-1 (ρ ∈ [1.7e-4, 9.4]) | **2.57e-3** |

**Diagnóstico** (tres mecanismos, encontrados por transcripción NumPy del solver +
ensayo aleatorio de 4000 estados en el régimen KSI + corridas controladas):

1. *Abanicos rotos silenciosos*: con campo normal fuerte y gas frío, la secante de
   MUB (4 iteraciones fijas) se estanca con residuo O(1) o converge a una raíz
   equivocada, y devuelve ptot_c finito y positivo cuyo abanico es no-físico
   (velocidades superlumínicas |v|² hasta 9, densidades negativas, ordenamiento de
   ondas roto). El único guard existente era `isfinite` — nunca dispara. ~20% de las
   caras del ensayo producían esta basura.
2. *Canal degenerado normal* (barrido x3 del fondo: bx=B3≈3.2, B⊥≈0, cs≪vA):
   fast ≡ Alfvén, las ondas extra de HLLD llevan saltos ~0 y el solver queda con
   **disipación cero** para los modos de contacto/Alfvén → bombea ruido de malla
   alineado con B (u3 rms: 4e-7 HLLE vs 1.9e-3 HLLD a t=0.25), con abanico
   formalmente "físico" cara a cara.
3. *Canal degenerado transversal* (caras x1/x2 del fondo: bx≈0, B3 enorme, p
   diminuta): el abanico colapsa a una discontinuidad tangencial resuelta
   exactamente → el ruido de cizalla (ρ, u3) a escala de malla no tiene disipación
   en NINGÚN barrido, y el transitorio de asentamiento contra las paredes lo
   alimenta. En 2D este canal existe pero es inofensivo: u3 ≡ 0 por simetría
   (la fuerza de Lorentz no tiene componente z con ∂z=0 y B=B3 ẑ).

**Arreglo** (todo en `hlld_srmhd.hpp`, por-cara, cae a HLLE solo donde toca):

1. Secante extendida: hasta 12 iteraciones con salida temprana al converger (los
   tubos MUB convergen superlinealmente de |res|~1e-5 en la iter 4 a ~1e-12 en la 7;
   las caras suaves ahora salen tras 2 evaluaciones en vez de pagar siempre 6).
2. Puerta de convergencia (|res| < 1e-4) + puertas de fisicalidad del abanico
   (sublumínico, ordenamiento λL ≤ λaL ≤ vx_c ≤ λaR ≤ λR, positividad de d y E),
   escritas NaN-safe.
3. **El arreglo decisivo — interruptor de degeneración**: si la cara es fría y
   magnéticamente dominada (b² > 10γp) Y el campo está alineado con un eje
   (min(bx², B⊥²) < 1e-4·max(...)), usar HLLE. Ahí las ondas extra de HLLD llevan
   saltos ~0, así que no se pierde nada.

**Dónde queda activo HLLD en la KSI 3D**: la lámina y sus flancos (p ~ 0.6–5, no es
"frío"), las superficies de los drips (p~5 de un lado), toda región con líneas
dobladas o reconexión (componentes B mezcladas). El interruptor solo apaga HLLD en
los rieles magnéticos sin estructura. En cuanto B⊥ supera el 1% de Bx, HLLD se
reactiva solo.

**Validación**:
- MUB 1–4: el interruptor es bit-a-bit inerte; la secante extendida cambia el error
  L1 vs la referencia 8× en <3% (mub1 mejora, mub2 idéntico bit a bit).
- 3D KSI t=0.5: fondo idéntico a HLLE a 3 cifras (tabla arriba); crecimiento 2-KE
  sigue a HLLE al 0.1%.
- 3D KSI **fase no lineal** (amp=1e-3, t=0→3, 96×192×48): estable sin NaN ni ruido;
  el fondo bajo los dedos queda limpio en ambos solvers. Y se ve el beneficio de
  HLLD: a t=3 la interfaz muestra **dedos más delgados y numerosos** con corrugación
  de escala fina (censo a x2=−0.01: 2 dedos de ~13.5 celdas vs 1 de ~16 con HLLE;
  ρ_max 3.11 vs 2.98), mientras HLLE alisa la superficie en un solo lóbulo grueso.
  Figura: `fingers_hlle_vs_hlld.png` (scratchpad de la sesión).
- p_bg=0 (β~2e-17, el régimen más duro): fondo limpio también (std 2.6e-3 a t=0.5).
- 2D 336×672 a t=2 con el binario nuevo: 2-KE idéntico a 5 cifras entre hlld y hlle,
  fondo ρ ∈ [0.998, 1.003] — el 2D validado no cambia.
- Throughput GPU: igual a HLLE en fase lineal (4.39e6 vs 4.38e6 zc/s; el interruptor
  ahorra la iteración en las caras degeneradas) y solo 7% más lento en la no lineal
  (4.28e6 vs 4.58e6).

**Nota 2D**: el interruptor también dispara en el fondo quiescente 2D (mismo canal
transversal frío), así que las corridas 2D nuevas no son bit-a-bit con las viejas en
el fondo — dinámicamente idéntico (u3≡0, ρ fondo ∈ [0.998,1.001]).

---

## 6. Resultados 2D (validados)

Corrida producción: **336×672**, HLLD, cfl=0.5, g=0.1, amp fiducial, t=0→20.
(La malla original 1024×2048 son ~50 h en la laptop; 336×672 son 2.4 h.)

Comparación contra Athena-C (reconstruida invirtiendo el colormap de su .mp4, ya que
no hay datos crudos):

| ventana | Athena-C | AthenaK | dif |
|---|---|---|---|
| [4, 7] | 0.837 | 0.801 | 4% |
| [3, 9] | 0.718 | 0.758 | 5% |
| [4, 9] | 0.687 | 0.619 | 10% |

**Ambas caen sobre la predicción de `D·g` (η≈0.61), no sobre la de `w·g` (η≈1.34)** —
consistente, porque ambos códigos acoplan la gravedad a la masa en reposo.

Envolvente superior de la capa de mezcla se superpone casi exactamente hasta t≈13.
Diferencias restantes (7 vs 15 dedos, penetración más rápida) son efecto de resolución:
AthenaK a 336×672 tiene 16.8 celdas/`a`, Athena-C a 1024×2048 tiene 51.

---

## 7. Punto de física: no existe equilibrio hidrostático

Integrando `d(p_gas+B²/2)/dx2 = −w·g` (w = ρh + b²) hacia abajo desde la lámina,
**p_gas se vuelve negativa en x2 = −0.0168** (solo 3.4a bajo la lámina). A σ=10 la
presión magnética consume todo el presupuesto; no queda gas que sostenga el peso.

Gill lo descarta explícitamente (sus ec. 8–10) justificándolo con `Delta ≪ c²/g`. El
desbalance residual es O(2–4%) y se asienta contra las paredes en un tiempo de cruce de
Alfvén `t_A = 0.21`. La corrección hidrostática en las BCs de usuario es lo que evita
que ese asentamiento lance ondas acústicas espurias.

**No intentes "arreglarlo"** — es una restricción física, no un descuido.

---

## 8. Bugs 3D encontrados y corregidos

### 8.1 `cfl_number` — límite unsplit en 3D

`dt = cfl·min_i(dx_i/c_i)`, así que el Courant efectivo es la **suma** sobre
direcciones ≈ cfl·ndim. El límite es 1:

| config | Σ | resultado |
|---|---|---|
| 2D, cfl=0.5 | 1.0 | marginal ✓ |
| 3D isótropo, cfl=0.4 | 1.2 | **NaN en t≈0.03** |
| 3D isótropo, cfl=0.3 | 0.9 | ✓ |

**Trampa**: celdas anisótropas (dx3 = 2·dx1) sobreviven cfl=0.4 porque Σ = 0.4×2.5 = 1.0
exacto. Eso enmascara la violación — subir resolución en x3 "rompía" la simulación.

### 8.2 `nz=1` — el modo sembrado estaba magnéticamente estabilizado

El factor `cos(2π·nz·x3/lz)` con nz=1 da **k_z = 125.7** a lo largo de B. Ese modo
dobla líneas de campo y la tensión lo estabiliza:

| | |
|---|---|
| Impulso KSI (interchange) | η² = 0.30 |
| Tensión magnética | (k_z·v_A)² = 14 356 |
| **Razón** | **4.8×10⁴** |

Lo que se sembraba era una **onda de Alfvén de corte estable**, no la KSI. La lámina
solo crecía del ruido de rejilla. En 2D nunca pasó porque k_z ≡ 0 por construcción.

**Fix: `nz = 0`** → interchange puro, el mismo modo inestable de la 2D, pero libre de
desarrollar estructura 3D desde el ruido.

Heredado de `KSI.c` de Athena-C, cuya corrida 3D nunca ejerció ese camino.

### 8.3 Ruido en el campo lejano — presión bajo la precisión de máquina

En el fondo `p_gas ≈ 1e-16` y `B²/2 = 5`, o sea **β ≈ 2×10⁻¹⁷**, por debajo del épsilon
de doble precisión (2.2×10⁻¹⁶) relativo a la energía total. `ConsToPrim` no puede
recuperar la presión.

Evidencia medida (192×384×96, t=18, campo lejano):

| | |
|---|---|
| ρ | 1.1e-3 → 13.5 (4 órdenes) |
| **B²** | **10.000 ± 0.3%** — no se comprime |
| **corr(ρ, B²)** | **+0.001** |
| u3 (a lo largo de B) | rms **20×** mayor que u1, u2 |
| Long. correlación x3 | 3 celdas |

En MHD ideal comprimir ⊥ B **obliga** a que B² suba. Que no lo haga prueba que el
movimiento es **paralelo** a B. El fondo es polvo sin presión deslizándose sobre rieles
magnéticos, apilándose en cáusticas de escala de rejilla.

**La 2D es inmune** — sin extensión en x3 no hay a lo largo de qué deslizarse.
Verificado: 2D a t=5 tiene ρ ∈ [0.998, 1.001] contra 3D [0.67, 1.52].

**Fix: `p_bg = 1.0e-4`** sumado uniformemente a `p_gas`. Al ser uniforme no tiene
gradiente → `p_tot = p0 + p_bg` sigue constante → **el equilibrio no se toca**.

Valor tomado de Gröger+2024 (su Tabla 1, `T0 = 1e-4 mₑc²`): β = 2e-5, nueve órdenes
sobre el límite de máquina, cinco bajo equipartición, `c_s = 0.0129c` — suficiente para
borrar cáusticas.

**Confirmado que Gröger sí usa fondo tibio**; Gill **no** (su capa fría tiene presión
exactamente cero, es un setup de losas uniformes, no Harris).

### 8.4 `iprob=2` (multimodo) no sirve con amplitud fiducial

El ruido blanco por celda deja potencia ~`amp/√N` en los modos de escala de caja. Con
amp=9.5e-7 queda **6 órdenes** por detrás de iprob=1 y nunca satura. Si quieres
multimodo, sube amp a ~1e-4.

---

## 9. Gravedad: subirla acelera, con límite

`eta ∝ √g` **exactamente** (Gill ec. 11/12, confirmado por Gröger Fig. 2).

Restricciones de Gill para esta caja (Lz=0.2): `Delta ≪ c²/g` y `√(Lz/L_dyn) < 1`.

| g | Δ/L_dyn | √(Lz/L_dyn) | ¿Válido? |
|---|---|---|---|
| 0.1 (fiducial Gill) | 0.001 | 0.141 | ✓ |
| **1.0** | 0.010 | 0.447 | ✓ (marginal en Lz≪L_dyn) |
| 10 | 0.100 | 1.414 | ✗ **viola** |

**g=1 es el máximo defendible.** También es la gravedad más alta que Gill corrió en la
caja original (su Fig. 11); su punto en g=10³ usó una caja 1000× más chica.

Valores que Gill realmente corrió: g = 0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e3.
**No hay g=10 ni g=100.**

Con g=1 y acoplamiento `D·g`: η ≈ 1.73, e-folding 0.577, saturación en t≈6.7 en vez de
t≈21.

**Advertencia honesta**: a g=1, un e-folding (0.577) ya es comparable al tiempo de cruce
de Alfvén (t_A = 0.21). Gill advierte que en ese régimen la fase lineal deja de estar
limpiamente separada de la respuesta global de la caja. Si el ajuste de σ sale ruidoso,
esa es la causa probable, no un bug.

### Ramp de gravedad (de Gröger)

Gröger no enciende la gravedad de golpe: la sube con coseno alzado sobre `t_g = Lx/c`
"to avoid artificial transients". Implementado como `grav_ramp` (default 0).
Usamos 0.2 ≈ t_A, que es 1/3 de un e-folding — no cuesta nada en crecimiento.

---

## 10. Estado actual: la corrida 3D funciona

**Job 247889**, 4 GPUs, 192×384×96 isótropo (9.6 celdas/`a`), cfl=0.3, HLLE,
`nz=0`, `p_bg=1e-4`, `grav=-1.0`, `grav_ramp=0.2`, t=0→10.

### Salud numérica

| | |
|---|---|
| NaN | **0** |
| masa | **deriva exactamente 0** |
| tot-E | +5.7e-4 |
| 3-ME | **−24%** ← esto es reconexión real, la lámina se rompe |

### Crecimiento — coincide con la teoría lineal

| ventana | σ medido |
|---|---|
| [2, 5] | **1.890** |
| [3, 5] | 1.854 |
| [2, 6] | 1.638 |
| [3, 6] | 1.460 |

**Predicho para g=1 con `D·g`: η = 1.73** → acuerdo del **9%** en la ventana temprana.
Es la primera vez que la 3D reproduce la teoría lineal.

### Comparación antes/después de los fixes

| | Antes (g=0.1, nz=1, sin p_bg) | Ahora |
|---|---|---|
| 1-KE final | 1.97e-9 (t=20) | **8.76e-6** (t=10) |
| 3-KE final | 1.27e-9 | **1.36e-4** |
| 3-ME | −4.6e-5 (nada pasaba) | **−24%** |
| Ruido fondo (std) | 0.870 | **0.160** |
| ρ_max fondo | 13.5 | 2.4–3.7 |

**3-KE creció 5 órdenes** — eso es dinámica genuinamente 3D (flujos a lo largo de B por
reconexión), imposible en 2D.

### Lo que queda: ruido residual

El fondo mejoró **5.4×** en std pero no desapareció. A t=10 sigue teniendo
ρ ∈ [0.054, 2.43], std=0.16. Posibles causas, sin distinguir todavía:

1. `p_bg=1e-4` amortigua pero no elimina — podría subirse a 1e-3 (β=2e-4, aún 4 órdenes
   bajo equipartición)
2. Material real de dedos penetrando el fondo (a t=10 hay reconexión al 24%, así que
   parte de eso **es física**)
3. Resolución: 9.6 celdas/`a` sigue bajo las 16.8 de la 2D validada

**Discriminante barato**: ρ_max en el fondo. Si es < 3 (densidad de la lámina), es
material de dedos → físico. Si supera 3, son cáusticas → numérico. A t=8.5 llegó a
3.70, ligeramente por encima → probablemente mezcla de ambos.

---

## 11. Siguientes pasos sugeridos

1. **Barrido de resolución — obligatorio.** En MHD ideal *toda* la reconexión es
   numérica, así que una sola resolución no es confiable por construcción (Gill lo
   muestra en su Fig. 10). Correr `nx2 = 192 / 384 / 576` (4.8 / 9.6 / 14.4 celdas/`a`,
   celdas isótropas) y verificar que σ y la tasa de reconexión convergen.

2. **Separar ruido residual de física**: correr con `p_bg=1e-3` y comparar. Si el ruido
   baja proporcionalmente era numérico; si no, es material de dedos.

3. **Comparar contra el paper**: Gill mide la tasa de reconexión `v_in`, que satura en
   ~5e-3. Ese es el número publicable.

4. **Considerar `grav_on_enthalpy=true`**: el acoplamiento físicamente correcto es
   `w·g` (Gill ec. 5), no `D·g`. Athena-C solo podía hacer `D·g` por limitación de su
   `StaticGravPot`. Con `w·g` el crecimiento es √6 = 2.45× más rápido y coincidiría con
   la teoría del paper en vez de con una versión debilitada. Está implementado como
   opción; default `false` para reproducir Athena-C.

5. **Verificar el path MPI de `KSIBoundaries`** — llama `ConsToPrim`/`PrimToCons` sobre
   todos los MeshBlocks del pack. Con varios ranks cada uno tiene solo los suyos y el
   filtro `mb_bcs == user` es por bloque, así que *debería* estar bien, pero comparar
   1 rank vs 2 ranks en corrida corta lo confirmaría. (La corrida a 4 GPUs funcionó, lo
   que es evidencia indirecta buena.)

---

## 12. Trampas de herramientas

### El lector `.bin` de AthenaK tiene dos bugs

`vis/python/bin_convert.py` **no sirve tal cual**:

1. `get_from_header` hace `line.split("=")` sin `maxsplit`. El header lleva el input
   embebido y valores como `configure = -D PROBLEM=ksi` tienen dos `=` → `ValueError`.
   Fix: `split("=", 1)` y quitar comentarios antes.

2. **`mb_index` NO da la posición global del bloque** — son índices *locales*
   (idénticos en todos los bloques). Ensamblar con ellos escribe todos los MeshBlocks
   encima en la misma esquina (se llena ~1/54 del dominio, plots todos negros). La
   posición viene de **`mb_logical = (lx1, lx2, lx3, level)`**:
   ```python
   i0, j0, k0 = lx1*n1, lx2*n2, lx3*n3
   ```

`build_ksi_gpu/src/plot_density3d.py` ya tiene ambos fixes.

### Plots: escala de color

Los videos 2D y 3D usan `log10(rho)` con `vmin=0.0, vmax=0.5` (cubre ρ∈[1, 3.16]).
**Si dejas que matplotlib autoescale, los videos dejan de ser comparables.**
`origin='lower'` obligatorio o x2 sale invertido.

### MeshBlock en x3: usar 3 bloques, no 2 ni 4

La respuesta de densidad de la semilla va como `cos²(2πx3/lz)`, período lz/2. Con 2 o 4
bloques el período de MeshBlock coincide y **no puedes distinguir un armónico físico de
un artefacto de frontera de bloque**. Con 3 bloques (`meshblock nx3=32` para
`nx3=96`) sí.

---

## 13. Resumen de parámetros de la config 3D actual

```
malla       192 x 384 x 96, isótropa (dx=5.208e-4, 9.6 celdas/a)
x3          [-0.025, 0.025]   (mitad de x1, para abaratar)
meshblock   32 x 32 x 32      (3 bloques en x3, deliberado)
cfl_number  0.3               (<= 1/3 obligatorio en 3D)
rsolver     hlld              (seguro en 3D desde el arreglo 2026-08-05, ver §5)
reconstruct ppmx              (= 3p de Athena-C)
pfloor      1e-18             (debe estar bajo 8.66e-17)
grav        -1.0              (máximo permitido por Gill en esta caja)
grav_ramp   0.2
p_bg        1e-4              (solo 3D; en 2D debe ser 0)
nz          0                 (CRÍTICO)
iprob       1                 (modo único, comparable a 2D)
amp         9.5346e-7         (fiducial de Gill)
grav_on_enthalpy  false       (convención Athena-C)
tlim        10.0
```

Costo: ~1.1 GB de output útil, corrió en 4 GPUs.
