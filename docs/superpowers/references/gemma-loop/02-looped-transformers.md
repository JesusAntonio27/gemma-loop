# 02 — Transformers con loops: evidencia y receta de conversión

Fecha: 2026-11-03. Alcance: recurrencia en profundidad (loops), papers 2018–2026 y síntesis para Gemma 4 E2B/E4B con LoRA/QLoRA en 12 GB.
Convención: **[E]** = evidencia de una fuente abierta; **[I]** = inferencia mía, no medida.

## Resumen

- Dory 1.0 4B **sí existe** y es el caso más cercano: 8 pares de capas de entrada + 20 pares recurrentes + 8 de salida; el bloque de 20 se aplica `n_recurrent_loops` veces (1–4) y la entrada se reinyecta en cada loop.
- El `config.json` de Dory cuenta **sublayers** (atención o MLP, nunca ambas): 72 = 16+40+16; la model card cuenta **pares** atención+MLP: 8+20+8. No es una contradicción.
- Dory tiene dos modos de KV: `per_loop` (por defecto) y `last_loop` (un solo cache, sin overhead de memoria); el segundo es una aproximación que la card dice “cercana” pero no cuantifica.
- **No hay paper de Dory** ni receta pública de entrenamiento del número de loops. Sí hay código de inferencia: PR de vLLM #60149 fusionado el 2026-10-25 y PR de llama.cpp #30306 aún abierto.
- La familia de papers se divide en dos cirugías: **(a) partición** (prelude/bloque/coda sin comprimir capas; N=1 idéntico al original: Dory, Huginn, Ouro, Closing the Loop) y **(b) compresión** (bloque más pequeño que el tramo original; N=1 no idéntico: Relaxed Recursive, MoR, McLeish).
- Para nuestro diseño (bloque contiguo de 25 capas repetido) la cirugía es **(a)**: no hace falta promediar pesos y N=1 es exacto. Promediar/stepwise solo aplica si se decide comprimir el bloque.
- La evidencia convergente dice que **el loop 2 aporta casi todo** (Dory, Ouro) y que **más loops no es monótono**: Ouro con T=4 entrenado fijo empeora en T=5–8; Dory baja en GPQA/HLE a 4 loops.
- Para un solo checkpoint que sirva N=1..4 hay que **muestrear N durante el entrenamiento** (Huginn, McLeish) o hacer currículum; entrenar fijo a N=2 no garantiza extrapolación.
- Con 100–200 M de tokens estamos 150–600× por debajo de los presupuestos de conversión publicados (15 B mínimo en Relaxed, 50 B en McLeish, 60 B en Closing the Loop). La expectativa honesta es paridad en N=1 y una mejora pequeña o nula en N=2.

## Hechos verificados

### Dory 1.0 4B (NVIDIA, 2026)

- Repo exacto: `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`, creado 2026-10-16; también MXFP8, NVFP4, GGUF Q5_K_M. https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16
- Reparto: 8 pares de entrada + 20 pares recurrentes + 8 pares de salida = 36 pares. La card lo dice literalmente; el config lo desdobla en 72 sublayers (`n_input_layers:16`, `n_recurrent_layers:40`, `n_output_layers:16`). https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/raw/main/config.json
- `num_hidden_layers:72`, `hidden_size:2560`, 16 cabezas / 4 KV, `head_dim:256`, vocab 131072, contexto 262144, `sliding_window:1024`; patrón híbrido 3 sliding + 1 full por cada 4 atenciones (9 full en total). Mismo config.
- Loop por defecto: `n_recurrent_loops:2`. La card recomienda 2; 1 si prima el cómputo por token; 4 para tareas agénticas (Terminal-Bench, SWE-bench). https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16
- Reinyección: la card dice “the input is reinjected into every loop”. El código lo hace con `combined = justnorm(state + input_h)`, ejecuta el bloque y hace `state = justnorm(h)` al final de cada loop. https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/dory.py
- KV: `recurrent_kv_cache_mode` acepta `per_loop` (default; un cache por loop) y `last_loop` (un cache por capa recurrente, sobrescrito en cada loop; sin crecimiento con N). En `last_loop`, el token actual atiende al KV del loop actual y los tokens anteriores al último loop. Mismo código y card.
- Benchmark por número de loops (1 / 2 / 4): SciCode 27.3 / 34.94 / 35.11; SWE-bench Verified 50.3 / 59.5 / 61.8; Terminal-Bench 2.1 12.5 / 18.1 / 22.2; τ³-bench Banking 11.6 / 12.89 / 14.69; GDPval-AA-V2 Elo 815 / 946 / 967; GPQA Diamond 69.19 / 72.41 / 72.16; HLE 8.25 / 10.89 / 10.43; IFBench 74.29 / 77.94 / 79; AA-LCR 63 / 71 / 73; AA-Omniscience 10.03 / 10.08 / 10.03. Card, misma URL.
- Lectura de la tabla: el salto grande es 1→2; 2→4 mejora-coding/agentes y **empeora** GPQA, HLE y AA-Omniscience. No asumir monotonía.
- Licencia: **OpenMDW 1.1**, uso comercial y no comercial; no es Apache 2.0. https://openmdw.ai/license/1-1/
- Entrenamiento declarado: 6 T tokens (NVFP4 los primeros ~4.5 T, luego BF16), extensión de contexto 300 B tokens, Megatron-LM; SFT; RL con GRPO (NeMo RL / NeMo Gym). Nada sobre cuántos loops vio en entrenamiento. Card, misma URL.
- Código: vLLM tiene `DoryConfig` y `DoryForCausalLM`; el PR #60149 se fusionó en `main` el 2026-10-25 (aún no en release). https://github.com/vllm-project/vllm/pull/60149
- llama.cpp: PR #30306 “dory support for llama.cpp” sigue **abierto** (creado 2026-10-10). La card da un fork/commit fijo para probarlo. https://github.com/ggml-org/llama.cpp/pull/30306
- No hay paper: el repo HF no incluye `modeling_*.py` (solo config/tokenizer/pesos) y `transformers` no tiene `models/dory/` (404 en `modeling_dory.py`). La documentación es la model card + el código de vLLM.
- El loop es un **entero fijo de carga**, sin halting, router ni salida adaptativa por token (el código lee `config.n_recurrent_loops` una vez). Confirma que “elástico” es de despliegue, no adaptativo.

### Gemma 4 E2B/E4B: datos que condicionan la cirugía

- E2B: `num_hidden_layers:35`, patrón 4 sliding + 1 full (7 periodos), hidden 1536, 8 cabezas / 1 KV, `head_dim:256`, `global_head_dim:512`, vocab 262144, contexto 131072, `sliding_window:512`. https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json
- E4B: `num_hidden_layers:42`, patrón 5 sliding + 1 full (7 periodos), hidden 2560, 8 cabezas / 2 KV, resto igual. https://huggingface.co/google/gemma-4-E4B-it/raw/main/config.json
- Ambos tienen **PLE** (per-layer embeddings): `hidden_size_per_layer_input:256`, `vocab_size_per_layer_input:262144`; la tabla empaquetada tiene `num_hidden_layers * 256` columnas y se indexa por capa física. https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py
- Ambos tienen **KV sharing entre capas**: `num_kv_shared_layers:20` (E2B) y `18` (E4B). En E2B las capas 15–34 son consumidoras (reusan el KV del productor); en E4B, 24–41. Los consumidores no tienen `k_proj`/`v_proj`. https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/configuration_gemma4.py
- E2B activa `use_double_wide_mlp:true` **solo en capas consumidoras de KV**; E4B lo tiene en `false`. Es decir, en E2B el MLP de 15–34 tiene otra forma que el de 0–14. Mismos dos archivos.
- Licencia declarada en HF: `apache-2.0`, enlazada a los términos de Gemma 4. https://huggingface.co/api/models/google/gemma-4-E4B-it

### Relaxed Recursive Transformers (Bae et al., 2024/2025)

https://arxiv.org/abs/2410.20672

- Convierte un Transformer de N capas en uno recursivo con un bloque de K capas repetido N/K veces. Ejemplo literal: Gemma 2B de 18 capas → 9 capas guardadas y dos pasadas.
- Tres inicializaciones: **Stepwise** (fija primera y última capa y elige intermedias a intervalos regulares), **Average** (promedia las matrices de las capas tied) y **Lower** (usa las K primeras).
- Resultado: **Stepwise** gana en el Recursive puro; **Average** gana en el Relaxed (con LoRA) porque el residual `W_layer − W_shared` es de bajo rango y SVD lo aproxima bien.
- Relaxed: un LoRA **distinto por loop** en todas las matrices lineales; se inicializa por SVD truncada del residual (B = U_r·Σ_r, A = V_rᵀ), no a cero. Rango 64/128/256/512; con 512 iguala al modelo completo (58.4 vs 58.6 de media few-shot).
- En su setup se entrenan **todos** los parámetros (no solo LoRA): 15 B tokens bastan para que el Gemma recursivo de 1B supere a TinyLlama/Pythia; 60 B + destilación forward-KL + pérdida en las salidas intermedias lo acercan al Gemma 2B completo.
- Hiperparámetros: contexto 2048, batch ~2 M tokens, AdamW lr 2e-4 con cosine, warmup 200 (15 B) / 800 (60 B), 8×H100.

### Mixture-of-Recursions (Bae et al., 2025)

https://arxiv.org/abs/2507.10524

- Un stack compartido de L/N_r capas se aplica hasta N_r veces por token; un router ligero decide la profundidad por token (135 M–1.7 B, 20 B tokens o 16.5e18 FLOPs).
- Dos routers: **expert-choice** (top-k por paso, filtrado jerárquico, puede filtrar información del futuro y necesita auxiliary loss) y **token-choice** (top-1 al inicio, necesita balancing loss). Mejor configuración: expert-choice + sigmoid + router lineal + auxiliary loss.
- KV: “recursion-wise caching” (solo tokens activos por paso) o “recursive KV sharing” (cachea todo en la primera recursión y lo reusa). El sharing ahorra memoria y baja un poco la accuracy.
- Con N_r=2 a isoFLOPs: 43.1% vs 42.3% de media few-shot del vanilla, usando ~50% menos parámetros. A 20 B tokens fijos: −25% FLOPs, −19% tiempo de entrenamiento y −25% memoria pico.
- N_r=3 y 4 siguen siendo competitivos; el paper reporta también estrategias Cycle/Sequence y variantes Middle (preservan primera y última capa sin compartir).

### Huginn: recurrencia de profundidad a escala (Geiping et al., 2025)

https://arxiv.org/abs/2502.05171

- 3.5 B parámetros, 800 B tokens; split (2,4,2): 2 capas de prelude, **4 de bloque recurrente**, 2 de coda; ~1.5 B en el bloque y 1.5 B en prelude+cabeza.
- Estado inicial aleatorio `s0 ~ N(0, 2/5)`, embedding de entrada `e` **inyectado en cada paso**, RMSNorm “sandwich” y escala de embedding √h. Sin inyección, la recurrencia deriva.
- Entrenamiento con `r` muestreado de una **log-normal Poisson** de media 32 y σ=0.5; **truncated backpropagation** solo por los últimos k=8 pasos (memoria constante). El prelude recibe gradiente en todos los pasos porque `e` se inyecta siempre.
- En test llega a r=64 y mejora hasta un cómputo equivalente a un dense de ~50 B. El checkpoint evaluado a 1 loop deja de mejorar tras 180 B tokens: la mejora vive en las iteraciones, no en prelude/coda.
- La primera run a escala se estancó con RMSNorm sin parámetros, sin escala de embedding, adaptador aditivo `s+e` y lr 4e-4; la configuración final necesitó inicialización tipo Takase (varianza 2/(5h)) y out-projections pequeñas.

### Ouro: Looped Language Models (Zhu et al., 2025)

https://arxiv.org/abs/2510.25741

- 1.4 B y 2.6 B; **24 capas compartidas**, `Tmax=4`; 1.4 B con 4 loops iguala a densos de 4 B y 2.6 B a 8 B en la mayoría de benchmarks.
- Pipeline de 5 etapas y 7.7 T tokens: warmup 3 T, stable 3 T, CT annealing 1.4 T, LongCT 20 B, mid-training 300 B. El 2.6 B se “upcyclea” duplicando 24→48 capas y manteniendo 4 loops.
- Empieza con **8 loops** en la etapa 1a y baja a **4** por inestabilidad (spikes); con 4 el coste/beneficio es mejor.
- Exit gate por paso + regularización de entropía con prior uniforme (evita colapso al último paso); la etapa II congela el LM y solo entrena la gate.
- Entrenado fijo a T=4: el rendimiento **pica en 4 y decae en 5–8** (p. ej. ARC-C 60.92→58.96 en T=5; MMLU 65.77 en T=4 → 64.49 en la fila de extrapolación). No hay extrapolación cero-shot.
- El salto 1→2 es enorme (ARC-C 37.63→54.86; MMLU 41.21→66.71 en 1.4 B), coherente con Dory.

### Retrofitted Recurrence (McLeish et al., 2025)

https://arxiv.org/abs/2511.07384 · código https://github.com/mcleish7/retrofitting-recurrence

- Convierte TinyLlama-1.1B, OLMo-2-1B y Llama-3.2-1B: conserva capas tempranas como **prelude**, capas tardías como **bloque recurrente + coda** y **elimina las intermedias**. Ejemplo: (4,8,4) sobre 22 capas usa [0–3], [10–17], [18–21].
- Inyección de entrada: en cada recurrencia concatena el estado `s_i` con el embedding del prelude `e` y aplica un **linear adapter 2h→h**; en t=0 usa ruido aleatorio. Sin esto, el estado se sale de la manifold.
- Currículum de recurrencias: sube la **media** de la Poisson-lognormal con schedule lineal o `1-sqrt` (mejor) durante el 75% del entrenamiento y luego la deja constante. Programar la media es mejor en FLOPs que programar la profundidad máxima de backprop.
- Inicializar desde el modelo preentrenado bate a inicializar aleatorio: a igual FLOPs la curva es mejor; la extrapolación estima ~950 B tokens para que el aleatorio iguale.
- 50 B tokens de Nemotron-CC-Math: el recurrente supera al no recurrente a igual FLOPs y sigue mejorando con más recurrencias en inferencia. “Healing”: 26 B de FineWeb-Edu antes de 26 B de datos de alta calidad sube ARC-C ~5 puntos.
- Muon (lr 1e-3) es más estable que AdamW (5e-5), que hizo spike/NaN en sus runs. El repo incluye `convert_llama.py` y verifica que los hidden states del N=1 coincidan exactamente (imprime `True, 0.000`).

### Closing the Loop: receta práctica de conversión (2026)

https://arxiv.org/abs/2610.00673

- Receta from-scratch de 310 B tokens (vs 7.7 T de Ouro) y **conversión de Qwen3-1.7B-Base** aplicando el decoder completo de forma recurrente con T=3; se evalúa saliendo en el paso 2.
- Loop naive degrada: los pasos 2 y 3 quedan +0.01 y +0.02 nats por encima del paso 1. Una proyección aprendida por paso tampoco funciona (la gate abandona el paso intermedio).
- Lo que sí funciona: **un único escalar aprendido** `z_t = w·e + (1−w)·h_{t−1}`, `w = sigmoid(a)`, inicializado en **0.99**. Con eso los pasos 2 y 3 quedan ~0.02 nats por debajo del 1.
- Inestabilidad de la exit gate: si un paso recibe poca masa, aprende poco. Suavizan solo la distribución usada en la pérdida LM: `(1−ε)·Σ p_t L_t + ε/T·Σ L_t`, con **ε=0.05** y T=3 → cada paso conserva ≥0.0167 de peso. La gate y la inferencia usan las probabilidades originales.
- LoRA por paso **no aporta**: rank-32 independiente por paso da pérdidas casi idénticas y diferencias de benchmark dentro de ±2 puntos; el default es todo compartido.
- Presupuesto de conversión: **60 B tokens** (mezcla curada) y **84 B** (web), <1% del cómputo de pretraining de Qwen3-1.7B. Resultados con 60 B: GSM8K 66.64→70.96, MATH 51.14→53.34, MMLU 60.62→61.05; Qwen3-4B-Base sigue por delante (82.87 / 65.92).

### Retrofitting Recurrent Depth (2026)

https://arxiv.org/abs/2608.11233

- Qwen2.5-0.5B-Instruct (24 capas) partido en **Prelude 6 / bloque recurrente 12 / Coda 6**. El camino de 1 loop es **identidad exacta**: las adiciones recurrentes se desactivan en T=1.
- “Split bridge” solo en loops ≥2: proyecciones separadas del estado y de la salida del prelude (`W_s`, `W_p`, 896×896) con una gate sesgada a la identidad y normalización opcional. La reinyección es imprescindible: sin ella el estado se sale de la manifold en pocos loops.
- Dos presupuestos sobre la misma cirugía: **full-block 180.6 M** parámetros vs **rank-16 LoRA 6.01 M** (base congelada). El adaptador iguala en global (83.8% vs 84.0%), gana hasta profundidad 11 y pierde más allá.
- Supervisión intermedia exacta (un paso de la tarea por loop) instala el mecanismo; extrapola a **1.44–1.5×** la profundidad supervisada y luego hay techo.
- Al ser el bloque la mitad de las capas, el coste por token es **(1+T)/2** del forward base (1.0×, 2.5×, 4.5×, 6.5× para T=1,4,8,12). No hay atención entre loops ni crecimiento de KV con T.

### Training-Free Looped Transformers (2026)

https://arxiv.org/abs/2605.23872

- Envuelve un checkpoint congelado y repite un **bloque contiguo de 4 capas del medio**; sin pesos nuevos. Repetir “a lo bruto” `x←g(x)` degrada casi siempre: a K=6 cae a 37.89% (−17.71 pp) y el MSE empeora ~320× a K=8.
- La solución es tratar el bloque como un paso de Euler y **sub-dividir**: `x_{k+1} = (1−1/K)·x_k + (1/K)·g(x_k)`, o un Runge-Kutta de K etapas con peso ancla β que interpola entre la salida original (β=1) y el endpoint damped (β=0).
- Receta fija: RK de 3 etapas en las 4 capas del medio; **block-mode** para densos, **layer-mode** para MoE (el block-mode hace thrashing del router). Sin tuning por benchmark.
- La ventana óptima está centrada en profundidad fraccional 0.43–0.71 para modelos >1.7 B (moda ~0.50) y más temprano (0.25–0.56) para <1 B; con n≥6 capas hay un acantilado.
- Ganancias: Qwen3-4B-Instruct +2.64 pp en MMLU-Pro y +2.01 en GPQA; estable para K=1..24. `cache=none` es catastrófico en decode.

### Universal Transformers (Dehghani et al., 2018)

https://arxiv.org/abs/1807.03819

- Aplica la **misma** transición (self-attention + FFN/conv) a todas las posiciones en cada paso de profundidad; residual + LayerNorm alrededor de cada bloque.
- Añade **coordinate embeddings** de posición y de paso de tiempo (sumadas), algo que un decoder preentrenado no tiene y que hay que decidir si se reinyecta o no.
- Halting ACT por posición; el “ponder time” crece con la dificultad (2.3±0.8 pasos para 1 hecho, 3.8±2.2 para 3). El mejor UT fijo usó 6 pasos; el dinámico promedió 8.2.
- Un UT con pasos fijos equivale a un Transformer con pesos tied entre capas. Bajo ciertas asunciones es Turing-completo.
- Gana en bAbI, tareas algorítmicas y WMT14 En-De (+0.9 BLEU); es el origen conceptual de casi todo lo anterior.

### Herramientas y estado del arte

- vLLM `DoryForCausalLM`: el forward del bloque recurrente es explícito y corto; sirve como referencia de implementación. https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/dory.py
- `transformers` sí tiene Gemma 4 (`Gemma4ForConditionalGeneration`, `Gemma4TextModel`, `Gemma4TextDecoderLayer`), con PLE y KV sharing implementados. https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py
- Tool calling sobre loops: un estudio 2026 encuentra que la recurrencia ayuda sobre todo en tool use composicional/dependiente, y que la profundidad adaptativa da mejor compute que la fija. https://arxiv.org/abs/2608.18171

## Know-how

### A. Elegir la cirugía antes de tocar código

1. **[E]** Si el bloque compartido conserva **todas** las capas del tramo (partición), N=1 es exactamente el original y no hay que promediar nada. Es el caso de Dory (8/20/8), Ouro (24 capas), Huginn (2/4/2) y Closing the Loop (decoder completo).
2. **[E]** Si el bloque es **más pequeño** que el tramo (compresión), hay que inicializar: Stepwise para pesos tied puros; Average + LoRA SVD para Relaxed. N=1 ya no es idéntico y hay que recuperar calidad con entrenamiento.
3. **[I]** Para este proyecto, elegir partición: garantiza el criterio S1 (N=1 dentro del ruido de O) casi por construcción y evita entrenar 25 capas desde cero con 200 M tokens.
4. **[E]** El bloque debe contener un número entero de periodos local/global para no cambiar la densidad de atención full. E2B: periodos de 5 capas; E4B: de 6.
5. **[I]** Cortes seguros (respetan periodos y no mezclan roles si algún día se comprime):
   - E2B: prelude 1 periodo (0–4), bloque 2 periodos (5–14), coda 4 periodos (15–34). N=2 → 5+20+20=45 capas efectivas (1.29×); N=4 → 65 (1.86×).
   - E4B: prelude 1 periodo (0–5), bloque 3 periodos (6–23), coda 3 periodos (24–41). N=2 → 6+36+18=60 (1.43×); N=4 → 96 (2.29×).
   - El corte 1/5/1 del diseño (E2B 5/25/5) también es válido en partición, pero el bloque de 25 cruza la frontera de KV sharing (capa 15) y en E2B la de MLP double-wide. No rompe el forward (cada capa conserva su forma), pero complica el cache y cualquier variante comprimida. Declarar el corte por modelo está permitido por el diseño.
6. **[E]** Verificar la identidad de N=1 numéricamente antes de entrenar: comparar hidden states/logits contra el modelo original capa a capa; el repo de McLeish imprime `True, 0.000` cuando coincide. Un test que falle aquí invalida todo lo demás.
7. **[I]** Congelar el modelo base y entrenar solo LoRA sobre el bloque compartido + (opcional) un vector/puerta por loop. No congelar el bloque entero si se comprime; en partición sí se puede congelar.

### B. Inicialización del bloque (solo si se comprime)

1. **[E]** Average = media de las matrices de las capas que se tied. Es el centroide y minimiza el error cuadrático medio de reconstrucción; combinado con SVD de los residuales es lo mejor para Relaxed.
2. **[E]** Stepwise = primera y última fijas, intermedias a intervalos regulares; gana cuando se entrenan todos los pesos y no hay LoRA que absorba el residual.
3. **[E]** SVD del residual: para cada capa origen `W_i` y bloque `W'`, descomponer `W_i − W' = UΣVᵀ`, quedarse con r componentes y poner `B = U_r Σ_r^{1/2}`, `A = Σ_r^{1/2} V_rᵀ` (o `B=U_rΣ_r`, `A=V_rᵀ`). Inicializar a cero pierde varios puntos.
4. **[E]** El rango necesario para recuperar el modelo completo fue grande (512 en modelos de ~2 B). Con rank 16–64 solo se recupera una fracción; no esperar paridad por LoRA si se comprime mucho.
5. **[I]** Si se comprime E2B, el bloque no puede cruzar la capa 15 sin re-crear `k_proj`/`v_proj` y el MLP double-wide: las formas no coinciden. O se elige un bloque dentro de 0–14 / 15–34, o se añaden proyecciones nuevas (~0.8 M por capa) y se acepta el coste.

### C. LoRA/QLoRA: qué entrenar y qué no

1. **[E]** Closing the Loop: LoRA rank-32 **por loop** no mejora sobre un adaptador compartido; diferencias ≤2 puntos y sin ganador consistente. Default: un solo LoRA compartido.
2. **[E]** Relaxed RT: LoRA por loop sí ayuda, pero con rank hasta 512 y entrenando además todos los pesos; es otro régimen de cómputo, no el nuestro.
3. **[I]** Recomendación: LoRA compartido, rank 32–64, en `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, con `lora_alpha = 2·r` y dropout 0.05. Un adaptador compartido se aplica automáticamente en cada loop porque el módulo es el mismo; no duplicar adaptadores por loop (PEFT no lo soporta sin routing manual).
4. **[I]** QLoRA 4-bit NF4 con double quant para E4B (pesos ~2.5 GB); E2B cabe en bf16 LoRA. `gradient_checkpointing_kwargs={"use_reentrant": False}` es obligatorio con pesos compartidos y checkpointing.
5. **[I]** Con 12 GB: batch 1, grad-accum 16–32, seq 1024–2048, `optim="paged_adamw_8bit"`. El coste de activaciones crece ~N× en el bloque; empezar con N≤2 y subir a 3–4 solo si cabe.
6. **[I]** Si se usa per-loop LoRA, implementarlo con `set_adapter(f"loop{t}")` antes de cada pasada del bloque y guardar todos los adaptadores; es fragil y no lo recomiendo como default.
7. **[I]** Guardar solo el adaptador + un `config.json` con `n_recurrent_loops`, `loop_block=[a,b]`, `bridge` y el hash del checkpoint base. El merge debe hacerse una sola vez sobre el bloque.

### D. Muestreo de N, pérdida y truncation

1. **[E]** Huginn entrena con `r ~ LogNormalPoisson(mean=32, σ=0.5)` y backprop solo por los últimos 8 pasos; así puede evaluar en r=64 sin explotar memoria.
2. **[E]** McLeish programa la **media** de la distribución (lineal o `1-sqrt`) durante el 75% del entrenamiento; es más eficiente en FLOPs que programar la profundidad máxima.
3. **[E]** Ouro entrena fijo a T=4 y **no extrapola** a 5–8. Closing the Loop entrena a T=3 y evalúa a 2. Un checkpoint entrenado solo a N=2 no está garantizado para N=4.
4. **[I]** Para un checkpoint que sirva 1–4: muestrear N por batch de `{1,2,3,4}` con probabilidades ~[0.25, 0.40, 0.20, 0.15]. Mantener N=1 en al menos 1 de cada 4 batches para no perder la identidad; dar más masa a N=2, que es donde está la ganancia.
5. **[I]** Con N≤4 no hace falta truncated BPTT: backprop completo por los 4 loops. Si se sube a 6–8, backprop solo por los últimos 3–4 (Huginn k=8 sobre media 32 es mucho más agresivo que lo que necesitamos).
6. **[E]** Entrenar con pérdida LM en **cada** salida de loop, no solo en la última: Relaxed lo hace para sus exit points, Ouro pondera por la gate, McLeish supervisa pasos intermedios. Sin señal intermedia, los loops 2–4 no aprenden a refinar.
7. **[I]** Pérdida sugerida sin gate: `L = Σ_t w_t L_t` con `w = [0.25,0.35,0.25,0.15]` para N=4 (y renormalizar si N<4), o `0.7·L_final + 0.3·mean(L_t)`. Añadir el suavizado de Closing (`ε=0.05–0.1`) si se incorpora una gate.
8. **[E]** Si se añade exit gate: regularización de entropía con prior uniforme (Ouro) y dos etapas (entrenar gate con LM congelado). No es necesario para el objetivo N fijo; sí para decidir N en inferencia.

### E. Normalización, residual y puente entre loops

1. **[E]** Dory: `combined = justnorm(state + input_h)` antes del bloque y `state = justnorm(h)` después. La norma explícita evita que el residual crezca con N.
2. **[E]** Closing the Loop: un solo escalar `w` mezcla embedding y estado, `w=0.99` inicial; es lo más simple que funciona y no añade parámetros por paso.
3. **[E]** McLeish: adaptador lineal sobre `[s_i ; e]` (2h→h). Retrofitting: dos proyecciones separadas + gate identidad, solo en loops ≥2.
4. **[I]** Orden recomendado para probar: (1) escalar de Closing, (2) `RMSNorm(state + input_h)` de Dory, (3) adaptador concat de McLeish, (4) split bridge. Cada uno es una ablación pequeña de una tarde en E2B.
5. **[I]** La puerta debe ser **inactiva en el loop 1**; si se aplica también en t=1, N=1 deja de ser el original y se pierde el control S1.
6. **[E]** Universal Transformers suma embeddings de tiempo; Ouro/Huginn no los usan. **[I]** No añadir embeddings de paso al principio: rompen la identidad N=1 y no hay evidencia de que aporten en conversión.

### F. Cache KV y coste

1. **[E]** Dory: `per_loop` por defecto; `last_loop` mantiene un solo cache y la card dice que la accuracy queda “cerca”, sin números. El código comenta que en `last_loop` el token actual usa el KV del loop actual y los anteriores el del último loop.
2. **[E]** MoR: cachear solo tokens activos por recursión o compartir el KV de la primera recursión; el sharing ahorra memoria y baja algo la accuracy.
3. **[E]** En un modelo de profundidad recurrente por **estado** (Retrofitting), no hay atención entre loops y el KV no crece con T. En un modelo que **repite capas sobre la secuencia** (Dory), sí hay un cache por loop sobre las mismas posiciones.
4. **[I]** Para Gemma 4: el KV sharing entre capas reduce el número de productores, pero cada loop necesita su propio cache de productores. Implementar `past_key_values` como lista indexada por `(loop, layer)` o por `virtual_layer = loop*L + layer`; no reutilizar el `DynamicCache` estándar con índices físicos.
5. **[I]** En entrenamiento usar `use_cache=False`; el cache solo importa en generación. En evaluación, `per_loop` para la medición principal y `last_loop` como ablación de memoria.
6. **[I]** Presupuesto: el bloque crece N× en FLOPs y activaciones, no en parámetros. Reportar siempre FLOPs por token y por problema, no solo latencia.

### G. Presupuestos de tokens: qué reportan los papers

| Trabajo | Tokens de adaptación | Qué recupera |
|---|---|---|
| Relaxed RT (Gemma 2B→1B) | 15 B (0.5% del pretraining) | Supera a TinyLlama/Pythia del mismo tamaño |
| Relaxed RT + distill | 60 B (2%) | 58.4 vs 58.6 del Gemma 2B completo (rank 512) |
| McLeish (Llama-3.2-1B) | 50 B math; 120 B FineWeb en el ablación | Mejor que el no recurrente a igual FLOPs |
| McLeish healing | 26 B + 26 B | +5% ARC-C tras la cirugía |
| Closing the Loop (Qwen3-1.7B) | 60 B curado / 84 B web | GSM8K +4.3, MATH +2.2, MMLU +0.4 |
| Huginn (from scratch) | 800 B | 3.5 B con cómputo de ~50 B dense |
| Ouro (from scratch) | 7.7 T | 1.4 B ≈ 4 B dense |
| Dory | 6 T + 300 B | 4 B con 1/2/4 loops |

**[I]** Con 100–200 M tokens estamos 300–600× por debajo de la conversión más barata (60 B). La conclusión honesta: no esperar recuperar la calidad de un modelo comprimido; sí se puede medir si el loop 2 aporta con la arquitectura ya particionada y N=1 intacto. Priorizar datos de razonamiento/código de alta calidad y 1–2 épocas sobre 50–100 M tokens únicos.

## Trampas

1. **Confundir “layer” en Dory con par atención+MLP.** `n_input_layers:16` son 8 pares; `num_hidden_layers:72` son 72 sublayers. Contar mal desplaza todo el split.
2. **Asumir que más loops es mejor.** Dory a 4 loops baja en GPQA/HLE/AA-Omniscience; Ouro decae en 5–8. Medir N=1..4 y reportar la curva, no solo el mejor punto.
3. **Entrenar fijo a N=2 y evaluar a N=4.** Ouro demuestra que no extrapola. Si se quiere un checkpoint multi-N, muestrear N o hacer currículum.
4. **Poner la puerta de reinyección también en N=1.** Rompe la identidad y con ella el control S1. La puerta es solo para t≥2.
5. **Olvidar el PLE de Gemma 4.** `embed_tokens_per_layer` está indexado por capa física. En un loop, la misma capa recibe el mismo vector PLE en cada iteración; decidir explícitamente si se comparte o se añade un vector por loop. No hay evidencia de papers sobre esto: es ablación propia.
6. **Ignorar el KV sharing de Gemma 4.** Las capas 15–34 (E2B) / 24–41 (E4B) no tienen `k_proj`/`v_proj`. Cualquier compresión que las mezcle con productoras falla por shape. En partición no falla, pero el cache debe indexar productores por loop.
7. **Ignorar `use_double_wide_mlp` en E2B.** Solo las capas consumidoras tienen MLP ancho; mezclar roles al comprimir da error de dimensión.
8. **Aplicar LoRA dos veces.** PEFT sobre un módulo compartido ya se reutiliza en cada loop; crear un adaptador por loop multiplica parámetros y no lo soporta el guardado estándar.
9. **Checkpointing reentrante con pesos compartidos.** Usar `use_reentrant=False`; el reentrante puede fallar o duplicar gradientes con módulos reutilizados.
10. **Cache con índices físicos.** `DynamicCache` estándar sobrescribe el KV del loop anterior. Usar índices virtuales `(loop, layer)` o desactivar cache en training.
11. **Repetir capas “a lo bruto” en inferencia (variante R).** Training-Free demuestra que degrada (hasta −17.71 pp a K=6). Si se hace R, usar damped Euler `(1−1/K)x + (1/K)g(x)` o RK, y ventana del medio, no el bloque entero.
12. **Copiar el learning rate de un paper de full fine-tuning.** McLeish necesitó Muon 1e-3 porque AdamW 5e-5 hacía NaN; en LoRA el régimen es otro (1e-4–2e-4 típico). No mezclar recetas.
13. **Dar por hecho que el harness oficial reproduce.** Antes de comparar loops, verificar que el modelo original con el mismo harness cae a <3 puntos de la cifra oficial (ya está en el diseño).
14. **Nombrar el resultado “Gemma”.** La licencia de Gemma 4 permite derivados, pero la marca y los términos tienen condiciones; usar un nombre propio (p. ej. “Gemma-Loop”) y enlazar la licencia.

## Referencias

- https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 — model card de Dory: arquitectura 8/20/8, loops 1/2/4, benchmarks, KV `last_loop`, licencia OpenMDW, receta de entrenamiento.
- https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/raw/main/config.json — config exacto: 72 sublayers, 16/40/16, 2 loops, `recurrent_kv_cache_mode`, patrón de atención.
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/dory.py — forward real del loop, `justnorm(state + input_h)`, cache por loop.
- https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/dory.py — `DoryConfig`, validaciones y semántica de `last_loop`.
- https://github.com/vllm-project/vllm/pull/60149 — PR de soporte Dory, fusionado en main el 2026-10-25.
- https://github.com/ggml-org/llama.cpp/pull/30306 — PR de llama.cpp para Dory, aún abierto.
- https://openmdw.ai/license/1-1/ — licencia OpenMDW 1.1 de Dory.
- https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json — 35 capas, 4:1, PLE 256, `num_kv_shared_layers:20`, double-wide MLP.
- https://huggingface.co/google/gemma-4-E4B-it/raw/main/config.json — 42 capas, 5:1, PLE 256, `num_kv_shared_layers:18`.
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py — clases `Gemma4*`, PLE, KV sharing, forward del text tower.
- https://arxiv.org/abs/2410.20672 — Relaxed Recursive Transformers: Stepwise/Average/Lower, LoRA por loop con SVD, 15 B/60 B tokens.
- https://arxiv.org/abs/2507.10524 — Mixture-of-Recursions: routers por token, KV sharing, Pareto a isoFLOPs.
- https://arxiv.org/abs/2502.05171 — Huginn: recurrencia de profundidad, log-normal Poisson, truncated BPTT k=8, 800 B tokens.
- https://arxiv.org/abs/2511.07384 — Retrofitted Recurrence: prelude/bloque/coda, currículum 1-sqrt, Muon, código de conversión.
- https://github.com/mcleish7/retrofitting-recurrence — `convert_llama.py`, verificación de identidad N=1, scripts de entrenamiento.
- https://arxiv.org/abs/2510.25741 — Ouro: 24 capas, T=4, gate con entropía, 7.7 T tokens, extrapolación fallida.
- https://arxiv.org/abs/2610.00673 — Closing the Loop: escalar w=0.99, smoothed exit loss, LoRA por paso inútil, 60 B/84 B.
- https://arxiv.org/abs/2608.11233 — Retrofitting Recurrent Depth: split bridge, identidad T=1, 6 M vs 180 M, extrapolación 1.5×.
- https://arxiv.org/abs/2605.23872 — Training-Free Looped Transformers: damped Euler/RK, ventana del medio, colapso del loop naive.
- https://arxiv.org/abs/1807.03819 — Universal Transformers: transición compartida, ACT, coordinate embeddings.
- https://arxiv.org/abs/2608.18171 — Looped Language Models Improve Compositional Tool Calling: loops y tool use agéntico.

## Sin verificar

- **Receta de entrenamiento de Dory**: cuántos loops vio, si muestreó N, si hizo currículum o si entrenó fijo. No hay paper ni technical report; la card solo declara 6 T + 300 B y las etapas SFT/RL.
- **Coste real de `last_loop` en accuracy**: la card dice “close to per-loop” sin tabla; no hay medición independiente.
- **Benchmarks de Dory**: todos son runs de NVIDIA con su harness (NeMo Gym / Evaluator SDK); no hay replicación de terceros.
- **Detalles de entrenamiento del puente de Dory**: el código de inferencia usa `justnorm(state + input_h)` fijo, pero no se sabe si en entrenamiento hubo una gate aprendida.
- **Si el método training-free funciona en Gemma 4**: el paper prueba 7 familias (Qwen, Llama, Moonlight, DeepSeek-V2-Lite), no Gemma 4.
- **Rendimiento de LoRA con 100–200 M tokens en una conversión de este tipo**: no hay paper que entrene tan poco; la estimación de “paridad o mejora pequeña” es mía.
- **Si el corte 1/5/1 del diseño es mejor que 1/2/4 (E2B) o 1/3/3 (E4B)**: no hay ablación publicada; hay que medirlo en E2B.
- **Impacto de reutilizar el PLE en cada loop**: no estudiado en ningún paper de los revisados; es una decisión abierta del diseño.
