# Guía de monitoreo del healing (Gemma-Loop)

Esta guía explica qué observa el monitor de Stage 1, qué aspecto tiene una
señal sana, qué problema indica cada patrón malo y qué hace el sistema
automáticamente. El monitor **solo observa**: no elige la receta, no cambia el
optimizador, el orden de los datos, el RNG ni la pérdida. La única excepción son
las reglas de aborto registradas, que **dejan intacto el último checkpoint
COMPLETE** (nunca escriben uno nuevo) y terminan el proceso con el código
dedicado **76**. En el orquestador, 76 es terminal: no se reintenta; para
relanzar hace falta `--ack-abort <alert-id>`, que queda registrado en
`run.json` y limpia los flags persistidos del aborto.

## 1. Qué escribe el monitor

| Archivo | Contenido |
|---|---|
| `heal_metrics.jsonl` | Una línea por paso de optimizador (métricas baratas). |
| `heal_probes.jsonl` | Una línea por sonda de inferencia (cada 0,5 M tokens y al final). |
| `heal_probe_texts/` | Textos generados por las sondas de generación (para leerlos a mano). |
| `alerts.json` | Todas las alertas, con severidad, regla, paso y tokens. |
| `heal_monitor_state.json` | Contadores del monitor para sobrevivir a un reinicio. |
| `heal_monitor_split.json` | Filas exactas del split MONITOR y semilla de selección. |
| `heal_report.md`, `heal_report.json`, `*.png` | Informe final o de vigilancia. |

El split **MONITOR** es un corpus aparte, no una partición de las filas de
entrenamiento: son los **64 problemas del bucket de entrenamiento 0-49 con las
claves de orden `sha256(seed:problem_id)` más grandes** (1 muestra de O cada uno,
con el filtro registrado; no hace falta que las muestras estén verificadas).
El sampler aplica esta regla al target `train` (las claves más pequeñas excluyen
los 64 problemas MONITOR); las demás etapas deben usar la misma convención de
subconjunto de entrenamiento. El split de ajuste de temperatura del piloto
(57-58) no se toca. El sampler local lo produce con `--target monitor`
y escribe
`monitor/monitor_rows.jsonl` + `monitor/manifest.json`; train.py lo recibe con
`--monitor-rows <jsonl>`.

Para una corrida registrada (`--prereg` o `--pilot-arm`) `--monitor-rows` es
**obligatorio**: el monitor comprueba por id que los problemas del MONITOR no
aparecen en las filas de entrenamiento, en las trazas de evaluación del piloto,
en el split de ajuste de temperatura ni en los conjuntos confirmatorios. Los
ids reservados se leen del contrato `reserved_ids.json` = mapa
`nombre_de_split -> ids` con claves `monitor`, `pilot_eval`, `pilot_fit`,
`heal_train`, `confirmatory` (se permiten claves extra), más
`GEMMALOOP_HEAL_MONITOR_RESERVED` y `--pilot-manifest` si existen. El monitor
comprueba `monitor ∩ (cada otra clave) = ∅` y **nunca** trata su propia clave
`monitor` como solapamiento. Si el archivo falta, no se puede leer, no trae ids,
hay solapamiento, o el corpus MONITOR no coincide con la clave `monitor`, la
corrida registrada sale con código 2. Un corpus MONITOR corto (menos de 32
filas tokenizables) o ilegible también sale con código 2. En una corrida no
registrada (smoke) el monitor puede recortar un split de las propias filas de
entrenamiento, pero avisa de que no puede probar la disyunción.

`heal_monitor_split.json` registra el path de `--monitor-rows`, `problem_ids`
(ids de problema), `row_ids` (índices elegidos), la semilla, las fuentes
reservadas, el recuento, el estado de `teacher_fit` y los umbrales efectivos.

## 2. Señales por paso

### Pérdida total y componentes por salida

- **Qué mide:** la pérdida media por token de asistente del **paso de
  optimizador completo** (medias ponderadas por tokens sobre todos los
  micro-batches), y por cada salida supervisada `t`: `KL_teacher(t)`,
  `KL(parent || t)` (ancla a O), el valor del hinge monótono y la fracción de
  filas donde el hinge está activo. El histograma de `T` cuenta **cada sorteo**
  de micro-batch, no solo el último.
- **Sano:** la pérdida baja de forma suave; `KL_teacher` baja y se estabiliza;
  `KL_parent` se mantiene pequeño; el hinge se activa poco después de la primera
  fase.
- **Patrones malos:**
  - pérdida que sube o explota → divergencia (o un bug de la pérdida);
  - `KL_parent` creciendo mucho → el estudiante se aleja del original O;
  - hinge activo en casi todas las filas después de 1 M tokens → las pasadas
    posteriores son peores que las anteriores (el bucle no está ayudando);
  - componentes `NaN`/`inf` → bug numérico o datos rotos.
- **Acción automática:** pérdida no finita → **ABORT**; hinge > 50 % después de
  1 M tokens → **WARN**; pico de pérdida > 3× la mediana durante 20 pasos
  consecutivos → **WARN**.

### Normas de gradiente, clipping y ratio actualización/peso

- **Qué mide:** la norma antes del clipping, total y por grupo (LoRA, `bridge`,
  `loop_ple`), la fracción de pasos con clipping y `lr * ||grad|| / ||peso||`
  por grupo (estimación de primer orden del tamaño del update).
- **Sano:** normas estables, del mismo orden que su mediana; el puente no domina
  a LoRA; el clipping no es la norma.
- **Patrones malos:**
  - norma > 10× su mediana → inestabilidad o divergencia;
  - clipping casi siempre activo → learning rate o escala de pérdida mal;
  - grupos con gradiente exactamente cero → parámetros congelados por error.
- **Acción automática:** norma > 10× la mediana → **WARN**; valores no finitos →
  **ABORT**.

### Puente g y normas LoRA B

- **Qué mide:** media y máximo de `|g|`, fracción de canales con `|g| > 1e-3`
  (activos) y con `|g| > 0.5` (saturación), si `g` es todo cero, y la norma de
  Frobenius de cada matriz LoRA B por capa.
- **Sano:** `g` despierta desde cero y crece sin saturarse; las normas LoRA B
  crecen de forma gradual.
- **Patrones malos:**
  - `g` todo cero después de 1 M tokens → el puente no aprende (el bucle no se
    usa);
  - > 10 % de canales saturados → el puente está en un régimen degenerado;
  - LoRA B que no cambia → adaptadores sin señal.
- **Acción automática:** saturación > 10 % (en **cualquier** momento) o `g` todo
  cero después de 1 M tokens → **WARN**. Los umbrales de activo/saturación se
  leen de `MonitorConfig`.

### Learning rate, tokens y horizonte

- **Qué mide:** learning rate por grupo, tokens vistos, tokens supervisados del
  paso, histograma de `T` y fracción empírica frente a la registrada
  (`P(T=4)=0.70`, cola `0.30`).
- **Sano:** el histograma se parece al registrado; los tokens supervisados son
  positivos y estables; el LR sigue el coseno por tokens.
- **Patrones malos:** histograma desviado → el sampler no está haciendo lo que
  dice el prereg; LR plano o en cero → receta mal aplicada; tokens supervisados
  en cero → máscara rota.
- **Acción automática:** informativo; los valores quedan en el JSONL y en el
  informe.

### Cambio relativo de hidden y ratio período 2

- **Qué mide:** `||h_t − h_{t−1}|| / ||h_{t−1}||` y
  `||h_t − h_{t−2}|| / ||h_t − h_{t−1}||` por salida.
- **Sano:** el cambio relativo baja con las pasadas; el ratio período 2 se
  mantiene bajo (no hay oscilación de dos pasos).
- **Patrones malos:** cambio que no baja → las pasadas no convergen; ratio
  período 2 alto → dinámica oscilatoria o inestable.
- **Acción automática:** informativo en el informe (no hay regla registrada).

### Integridad de los datos del teacher

- **Qué mide, por micro-batch:** suma de top-k + cola antes y después de
  renormar en fp32, desviación máxima, distribución de la masa de cola, número
  de tokens enmascarados y apariciones del token glitch.
- **Sano:** la suma antes y después de renormar está dentro de 2e−3 de 1; las
  colas son pequeñas y estables. El layout real `[B,S,K]` se aplana antes de
  validar, igual que en el objetivo dial.
- **Patrones malos:** sumas fuera de tolerancia, shapes/ids mal, `NaN`, token
  glitch sin enmascarar → **los datos del teacher están rotos**.
- **Acción automática:** **ABORT** inmediato.

### Finitud, throughput, memoria y tiempo de paso

- **Qué mide:** contadores de pérdida/gradiente/logits no finitos, tokens/s por
  paso, memoria asignada/reservada actual y pico, y tiempo de pared del paso.
- **Sano:** todo finito; tokens/s estable después del calentamiento; memoria
  cómodamente por debajo del tope del allocator.
- **Patrones malos:**
  - memoria a menos de 0,3 GiB del tope → riesgo de spill/OOM;
  - tokens/s < 50 % de la mediana post-calentamiento → spill o contención;
  - no finitos → bug de precisión.
- **Acción automática:** no finitos → **ABORT**; memoria o throughput →
  **WARN**.

## 3. Sondas de inferencia

Se ejecutan cada 0,5 M tokens y al final, sobre ~32 filas del split MONITOR,
siempre en inferencia (eval, sin gradiente, sin consumir el RNG).

### Invariante: salida 1 == O

- **Qué mide:** compara, en un lote fijo, la salida multi-exit 1 (adaptadores
  apagados por `loop_only`) contra el forward de O con `disable_adapter()`.
- **Sano:** diferencia máxima exactamente 0 (bit a bit).
- **Patrón malo:** cualquier diferencia → bug de la política de adaptadores, del
  forward looped o del checkpoint.
- **Acción automática:** **ABORT** inmediato.

### Salud del ancla: KL(O || estudiante N)

- **Qué mide:** cuánto se aleja cada salida `N = 2, 3, 4, 6, 8` de la
  distribución original O, en nat/token, sobre tokens de asistente.
- **Sano:** valores pequeños y estables; `N=1` es 0 por construcción.
- **Patrón malo:** crecimiento sostenido, especialmente > 1,0 nat/token en el N
  primario (4) → el estudiante se escapa del original.
- **Acción automática:** > 1,0 nat/token en el N primario durante 3 sondas
  consecutivas → **ABORT** (el checkpoint completo queda guardado).

### Ajuste al teacher: KL(teacher || estudiante N), train vs monitor

- **Qué mide:** el KL forward al teacher en las filas del monitor y en el lote
  de entrenamiento actual; la diferencia es la señal de sobreajuste.
- **Sano:** ambos bajan juntos; la brecha monitor−train es pequeña y estable.
- **Patrón malo:** brecha creciendo durante 3 sondas → sobreajuste;
  KL estancado → el teacher no está guiando.
- **Acción automática:** brecha creciente 3 sondas → **WARN**; KL de teacher sin
  mejorar ≥ 5 % en 1 M tokens → **WARN**.
- **Nota:** el sampler escribe las filas MONITOR en formato heal (mensajes).
  Para que esta señal funcione, el corpus MONITOR debe pasar por el precompute
  del teacher y train.py debe recibir `--monitor-teacher-data <dir>` (mismo
  contrato que `--teacher-data`). Sin esos shards la sonda no tiene top-k del
  teacher: el informe muestra **“Teacher fit: unavailable”** y el WARN de
  sobreajuste no puede dispararse. El ancla, la NLL y la generación sí se miden
  siempre.

### NLL calibrada en N = 1..4

- **Qué mide:** NLL por token en el split MONITOR, por N. Las filas MONITOR se
  parten determinísticamente en dos mitades: la temperatura se ajusta por
  `(checkpoint, N)` en una mitad y se puntúa en la otra. **Nunca se toca el
  split de ajuste de temperatura del piloto.**
- **Sano:** tendencia descendente o plana con N.
- **Patrón malo:** empeorar al aumentar N → la segunda pasada no ayuda.
- **Acción automática:** solo descriptivo; **no es una regla de selección**.

### Sanidad de generación

- **Qué mide, en 8 prompts fijos y N = 1, 2, 4, greedy con tope de 512 tokens:**
  longitud, motivo de fin, canal de pensamiento cerrado, respuesta final
  presente, tasa de repetición de n-gramas, tasa de no-ASCII/basura. Los textos
  se guardan en `heal_probe_texts/` y se enlazan desde el informe.
- **Sano:** canal de pensamiento cerrado, respuesta no vacía, sin bucles de
  n-gramas, texto legible.
- **Patrones malos:** canal abierto, respuesta ausente, bucles de repetición o
  basura → el modelo se rompió al curar, aunque la pérdida parezca bien.
- **Acción automática:** cualquier sonda con bucles en N ≤ 4 → **WARN**; la
  respuesta ausente y el canal abierto solo cuentan si la generación terminó
  antes del tope de 512 tokens. Una traza de pensamiento de 8192 tokens
  truncada por el tope es lo esperado, no una anomalía (regla M1).

## 4. Reglas registradas y valores por defecto

Los valores viven en `gemmaloop.train.heal_monitor.MonitorConfig` y se pueden
citar desde el prereg. Los abortos **no escriben checkpoint** (el último
COMPLETE queda intacto) y salen con código 76; el orquestador los trata como
terminales y exige `--ack-abort`. Los avisos siguen entrenando y escriben
`alerts.json`.

| Regla | Condición | Acción |
|---|---|---|
| `nonfinite` | pérdida/gradiente/logit no finito | ABORT |
| `exit1_bit_exact` | salida 1 ≠ O bit a bit | ABORT |
| `teacher_integrity` | suma fuera de 2e−3, shapes/ids malos | ABORT |
| `parent_kl_runaway` | KL(O‖N primario) > 1,0 durante 3 sondas | ABORT |
| `loss_spike` | pérdida > 3× mediana durante 20 pasos | WARN |
| `grad_spike` | norma de gradiente > 10× mediana | WARN |
| `teacher_stall` | teacher KL mejora < 5 % en 1 M tokens | WARN |
| `hinge_active` | hinge activo > 50 % tras 1 M tokens | WARN |
| `overfitting_gap` | brecha monitor/train crece 3 sondas | WARN |
| `generation_sanity` | bucles en N ≤ 4, o respuesta/canal ausente si no se truncó | WARN |
| `bridge_saturated` | > 10 % canales con \|g\| > 0,5 (sin esperar 1 M) | WARN |
| `bridge_zero` | g todo cero tras 1 M tokens | WARN |
| `probe_error` | una sonda falló (p. ej. generación) | WARN |
| `memory_margin` | reservado a < 0,3 GiB del tope | WARN |
| `throughput_spill` | tokens/s < 50 % de la mediana | WARN |

## 5. Cómo leer el informe en 2 minutos

1. **Mira arriba del todo.** La tabla *Traffic lights* tiene una fila `Overall`:
   verde = todo bien; amarillo = hay avisos, el run sigue; rojo = una regla de
   aborto disparó. Si es rojo, ve a *Alerts* y busca la primera entrada
   `severity: abort`; el relanzamiento exige `--ack-abort <alert-id>`.
2. **Mira la pérdida y el teacher KL.** Deben bajar y estabilizarse. Un salto
   vertical con recuperación es normal; una subida sostenida no.
3. **Mira el ancla (`KL(O||N)`).** Si crece sin parar, el estudiante se está
   alejando de O; el monitor aborta si supera 1,0 en N=4 tres veces.
4. **Mira el puente g.** Si está todo cero o saturado, el bucle no está
   aprendiendo aunque la pérdida baje.
5. **Mira las sondas de generación.** Abre dos o tres textos: ¿canal cerrado?
   ¿respuesta final? ¿bucles? Si hay bucles en N ≤ 4, hay un aviso.
6. **Mira memoria y tokens/s al final.** Si la memoria está cerca del tope o el
   throughput se hunde, el problema es de spill, no del objetivo.
7. **Recuerda:** todo esto es diagnóstico y seguridad de ejecución. **Ninguna
   señal del monitor es una entrada de selección del piloto.**

## 6. Comandos

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:GEMMALOOP_HEAL_MONITOR_RESERVED = "C:\ruta\splits-reservados.json"

# Corpus MONITOR dedicado (64 claves más grandes del bucket 0-49)
uv run python scripts/sample_stage0_local.py --pool-dir C:\ruta\pool `
    --output-dir C:\ruta\stage0-monitor --target monitor

# Precompute del teacher sobre el corpus MONITOR (mismo teacher/runtime)
uv run python scripts/precompute_teacher_topk.py `
    --teacher e4b --data C:\ruta\stage0-monitor\monitor\monitor_rows.jsonl `
    --out C:\ruta\stage0-monitor\teacher-monitor

# Entrenamiento con el corpus MONITOR explicito
uv run python scripts/train.py ... `
    --monitor-rows C:\ruta\stage0-monitor\monitor\monitor_rows.jsonl `
    --monitor-teacher-data C:\ruta\stage0-monitor\teacher-monitor

# Informe final de una corrida
uv run python scripts/heal_report.py --run C:\ruta\run

# Vigilancia cada 5 minutos mientras la corrida está viva (baja prioridad)
uv run python scripts/heal_report.py --watch C:\ruta\run

# Comparar las cuatro armas del teacher (o cualquier conjunto de corridas)
uv run python scripts/heal_report.py --compare C:\ruta\run-a C:\ruta\run-b `
    --labels 25-29 16-18 --out C:\ruta\comparacion
```

`--watch` baja la prioridad del proceso para no competir con el entrenamiento.
El monitor se activa por defecto en runs con objetivo dial/heal (nunca en
`--smoke`); `GEMMALOOP_HEAL_MONITOR=0` lo apaga y `=1` lo fuerza. Apagarlo no
toca nada del entrenamiento: la corrida es bit a bit idéntica, pero **también
omite la prueba del contrato de splits**, así que úsalo solo en smoke o
diagnóstico. El ejecutor de producción es `scripts/orchestrate.py`; un
`--resume` directo de `train.py` no consulta `heal_monitor_abort.json` ni exige
`--ack-abort`.
