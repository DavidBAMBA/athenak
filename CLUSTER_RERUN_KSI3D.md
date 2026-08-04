# KSI 3D — instrucciones para relanzar en el cluster

## Contexto para el agente

Estamos corriendo la Kruskal-Schwarzschild Instability (KSI) en SR-MHD con AthenaK,
reproduciendo Gill, Granot & Lyubarsky 2018, MNRAS 474, 3535 (`hydro-ksistx3000.pdf` en
el repo). La corrida 3D anterior en el cluster (`export_2gpu`, 192×384×96, g=0.1,
tlim=20) terminó **sin NaN y con conservación excelente** (masa exacta, 3-ME con deriva
de solo −4.6e-5 en 20 unidades de tiempo), pero:

1. La lámina apenas empezaba a ondularse a t=20 — mucho más lento que la 2D.
2. Fuera de la lámina aparecía ruido creciente: a t=18 la densidad de fondo iba de
   1e-3 a 13.5 (la lámina en sí tiene ρ=3), con la media exactamente en 1.0 —
   **redistribución, no una fuente real**.

Diagnosticamos ambos problemas y los corregimos. Ya están commiteados en el repo
(`src/pgen/ksi.cpp` e `inputs/srmhd/athinput.3dksi`). Este documento explica qué
cambió, por qué, y qué hacer en el cluster.

## 1. Por qué la lámina crecía tan lento: `nz` mal puesto

El input anterior tenía `nz = 1`, lo que le da a la perturbación semilla un factor
`cos(2π·x3/lz)`, es decir número de onda **k_z = 125.7** a lo largo de B (el campo
apunta en x3). Un modo con k_z ≠ 0 dobla las líneas de campo, y la tensión magnética
lo estabiliza:

- Impulso de la KSI (modo interchange puro): η² ≈ 0.30
- Tensión magnética del modo sembrado: (k_z·v_A)² ≈ 14 356
- **Razón: 4.8×10⁴** — el modo sembrado está establemente amortiguado, no crece.

Lo que en realidad sembrábamos era una onda de Alfvén de corte estable, no la KSI. La
lámina solo podía crecer del ruido de rejilla (medimos σ≈0.35 desde el ruido puro), por
eso tardaba tanto. En 2D esto nunca pasó porque no hay extensión en x3 → k_z ≡ 0 por
construcción, siempre es interchange puro.

**Fix:** `nz = 0`. Con `cos(0)=1` la perturbación queda uniforme en x3, k_z=0,
interchange puro — el mismo modo inestable que ya validamos en 2D, mientras la corrida
sigue libre de desarrollar estructura 3D genuina desde el ruido.

## 2. Por qué aparecía ruido fuera de la lámina: presión del fondo por debajo de la precisión de máquina

En el campo lejano (lejos de la lámina), p_gas ≈ p0·sech²(x2/a) → 1e-16, mientras
B²/2 = 5. Eso da β ≈ 2×10⁻¹⁷, **por debajo del épsilon de doble precisión (2.2×10⁻¹⁶)**
relativo a la energía total. El inversor `ConsToPrim` no puede recuperar esa presión —
literalmente no hay información suficiente en el float. El fondo se comporta como polvo
sin presión ensartado en el campo magnético: se desliza libremente **a lo largo** de las
líneas de B (que es el único movimiento que un fluido sin presión no puede resistir) y
se apila en cáusticas a escala de rejilla.

Evidencia medida en la corrida del cluster (t=18, campo lejano):
- ρ va de 1.1e-3 a 13.5 (cuatro órdenes de magnitud)
- **B² se mantiene en 10.000 ± 0.3%** — el campo casi no se comprime
- **corr(ρ, B²) = +0.001** — prácticamente cero. En MHC ideal, comprimir ⊥ B obliga a
  que B² suba (flujo congelado); que no lo haga es la firma de que el movimiento es
  **paralelo** a B, no perpendicular.
- u3 (velocidad a lo largo de B) tiene rms 20× mayor que u1, u2
- longitud de correlación en x3: 3 celdas — ruido de rejilla puro

En 2D esto no pasa porque no hay extensión en x3 para deslizarse.

**Fix:** `p_bg = 1.0e-4`, sumado uniformemente a `p_gas` en todo el dominio. Al ser
uniforme no tiene gradiente, así que **no perturba el balance de presión total**
(p_tot sigue siendo constante = p0 + p_bg). Este valor no es arbitrario: es exactamente
la presión de fondo que usa el estudio cinético (PIC) de la misma instabilidad,
**Gröger, Hakobyan & Sironi 2024** (`kinetic_ksi2412.09541v1.pdf`, Tabla 1,
T0 = 1e-4 mₑc²). Con nuestra normalización eso da β = 2×10⁻⁵ — nueve órdenes por
encima del límite de máquina, cinco órdenes por debajo de equipartición, y da al fondo
una velocidad del sonido real (c_s ≈ 0.0129c) capaz de borrar las cáusticas.

**Importante: la 2D NO necesita esto y debe seguir con `p_bg=0`.** Verificamos que en
2D el campo lejano se mantiene prístino (ρ = 0.998–1.001 a t=5) porque sin extensión en
x3 el fluido no tiene a lo largo de qué deslizarse. Añadir `p_bg` en 2D cambiaría un
setup ya validado bit-a-bit contra Athena-C sin necesidad.

## 3. Cambio pedido por el usuario: subir la gravedad un orden de magnitud

Se quiere saturar más rápido para gastar menos horas de cluster. `g: -0.1 → -1.0`.

Verificado contra Gill+2018 (relectura completa del paper, incluida la ecuación de
dispersión exacta, no solo las asíntotas):

- La tasa de crecimiento lineal escala como **η ∝ √g exactamente** (su ec. 11/12,
  confirmado también en el paper cinético de Gröger, su Fig. 2).
- Para nuestros parámetros (k=2π/0.1=62.83, Δ=0.01) k·Δ=0.63 está en el régimen
  **intermedio** — ni la asíntota de onda corta ni la de onda larga aplica limpio, hay
  que usar la ec. (11) completa.
- Restricciones de Gill para que g siga siendo válido en esta caja (Lz=0.2):
  - `Δ ≪ c²/g` (su ec. 10)
  - `√(Lz/L_dyn) < 1`, con L_dyn = c²/g

| g | Δ/L_dyn | √(Lz/L_dyn) | ¿Válido? |
|---|---|---|---|
| 0.1 (original) | 0.001 | 0.141 | ✅ |
| **1.0 (nuevo)** | **0.010** | **0.447** | ✅ (ver nota abajo) |
| 10.0 | 0.100 | 1.414 | ❌ viola la segunda condición |

**g=1 es el máximo defendible para esta caja** — g=10 ya rompe la restricción de Gill.
Y de hecho **g=1 es el valor de gravedad más alto que Gill corrió en su caja original**
(su Fig. 11 solo llega a g=10⁰ sin achicar la caja; el único punto en g=10³ que
mencionan lo corrieron en una caja 1000× más chica).

Con el acoplamiento que usamos por defecto (`grav_on_enthalpy=false`, gravedad sobre la
densidad en reposo D, convención de Athena-C — más débil por √6 que el `w·g` físico del
paper):
- η(g=1) ≈ 1.73, e-folding ≈ 0.58
- η(g=0.1) ≈ 0.55, e-folding ≈ 1.83
- Saturación (desde amp=9.5346e-7 hasta O(1), ~11.6 e-foldings): t≈6.7 en vez de t≈21

**Nota honesta, del propio Gill (p. 3539–3540):** con la ec. (11) exacta, a g=1 un solo
e-folding (0.577, o 0.236 si usaran su acoplamiento `w·g` más fuerte) ya es comparable
al tiempo de cruce de Alfvén de la caja (t_A = Lz/v_A = 0.21). El propio paper advierte
que en ese régimen la fase lineal deja de estar limpiamente separada de la respuesta
dinámica global de la caja. No es motivo para no correrlo — es el valor máximo que las
propias restricciones de Gill permiten, y con `grav_on_enthalpy=false` el crecimiento es
más lento que con su convención, lo que da algo más de margen — pero si el ajuste de la
tasa de crecimiento sale ruidoso, esa es la razón más probable, no un bug.

También bajamos `tlim: 20.0 → 10.0` (con g=1 la saturación llega ~3× más rápido, no
hace falta correr tanto) y añadimos `grav_ramp = 0.2`: en vez de encender la gravedad de
golpe, sube con un coseno alzado en 0.2 unidades de tiempo (≈ un tiempo de cruce de
Alfvén, solo 1/3 de un e-folding — no cuesta nada en crecimiento). Esto sigue la
receta de Gröger+2024 (p. 3), que hacen exactamente esto "to avoid artificial
transients" — apagar de golpe una gravedad que desbalancea 2-4% la presión lanza un
repique acústico que antes dominaba 2-KE en los primeros pasos.

## 4. Qué hacer en el cluster

```bash
git pull
```

Trae dos archivos modificados: `src/pgen/ksi.cpp` (código nuevo: `nz=0` no requiere
cambio de código, pero `p_bg` y `grav_ramp` sí son parámetros nuevos que el pgen debe
soportar) y `inputs/srmhd/athinput.3dksi` (todos los valores ya actualizados).

**Hay que recompilar** — cambió `ksi.cpp`:

```bash
cmake --build build_cluster -j
```

(Usa el mismo `build_cluster` que ya tenías configurado con MPI+CUDA; no cambié nada
del CMakeLists ni de las flags de Kokkos.)

Luego lanzar igual que la vez anterior:

```bash
mpirun -n <N_GPUs> ./build_cluster/src/athena -i inputs/srmhd/athinput.3dksi
```

### Verifica antes de dejarlo correr las 8 horas

El input trae `ndiag=100` así que el log es corto. Revisa los primeros ~200 ciclos:

- **Sin NaN** en el log ni en el `.hst`.
- `dt` estable (no debe colapsar — si colapsa, algo del ramp de gravedad o `p_bg`
  interactuó mal, avísanos).
- A los pocos pasos, `2-KE` en el `.hst` no debería tener el pico instantáneo que
  tenía antes (el ramp de gravedad debería suavizarlo).

### Qué traer de vuelta cuando termine (no los 18 GB completos)

Con `p_bg` y `nz=0` esperamos que el campo lejano se quede limpio, así que probablemente
**sí** valga la pena traer más esta vez para confirmarlo, pero como orden de prioridad:

1. **`.hst` completo** (KB) — primero, siempre. Con eso solo ya se puede confirmar o
   descartar el diagnóstico completo (columnas 8/9/10 = 1-KE/2-KE/3-KE, columna 13 =
   3-ME).
2. **Slices `slice_x3=0`** (nuevo output, cadencia `dt=0.05`, ~1.5 MB cada uno) — dan
   el video del panel "x1-x2 @ x3=0" con buena cadencia temporal, y bastan para
   confirmar visualmente si el fondo se mantiene limpio.
3. Un puñado de volúmenes `mhd_w`/`mhd_bcc` (3–4 tiempos, ~230 MB cada uno) — solo si
   se quiere medir el campo lejano en 3D completo (lo que hicimos para diagnosticar
   esta vez) o generar los paneles "x3-x2" y "x1-x3" del video de 3 paneles.
4. **No hace falta traer los restarts** salvo que se planee continuar esta corrida
   específica más allá de tlim=10.

## 5. Resumen de todos los valores que cambiaron en `athinput.3dksi`

| Parámetro | Antes | Ahora | Motivo |
|---|---|---|---|
| `grav` | −0.1 | **−1.0** | saturar 3× más rápido (pedido explícito) |
| `grav_ramp` | (no existía) | **0.2** | evitar transitorio acústico al encender g |
| `nz` | 1 | **0** | el modo nz=1 estaba magnéticamente estabilizado por 4.8e4× |
| `p_bg` | (no existía) | **1.0e-4** | dar presión resoluble al fondo (β=2e-5, igual que Gröger+2024) |
| `tlim` | 20.0 | **10.0** | con g=1 la saturación llega ~3× antes |
| `<output2>/<output3> dt` | 0.5 | **0.25** | más cadencia ahora que la corrida es más corta |
| `<output4>` (slice x3=0) | (no existía) | **nuevo, dt=0.05** | video barato de alta cadencia |
| `<output5>` (rst) `dt` | 2.0 | **1.0** | igual, más cadencia con tlim menor |

Todo lo demás (malla 192×384×96, `cfl_number=0.3`, `rsolver=hlle`, `reconstruct=ppmx`,
`meshblock nx3=32` → 3 bloques en x3, `sigma=10`, `amp=9.5346e-7`, `iprob=1`,
`seed_confine=false`, `grav_on_enthalpy=false`) queda **exactamente igual** que la
corrida anterior — solo estamos aislando el efecto de estos cinco cambios.
