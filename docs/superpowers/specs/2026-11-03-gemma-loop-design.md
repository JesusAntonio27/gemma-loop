# Gemma-Loop: diseño del experimento y del expediente

Fecha: 2026-11-03 · Estado: APROBADO por el autor el 2026-11-03 ("ya puedes preparar el plan completo"). Las llamadas [Mía] de la sección 9 siguen abiertas a revisión.
Idioma de este documento: español. Idioma del paper público: inglés **[Mía]** (cámbialo si prefieres español).

Convenciones:
- **(Autor)** = lo dijo el autor del proyecto.
- **[Mía]** = decisión o suposición mía, corregible.
- **[Sin verificar]** = dato de una fuente secundaria o no confirmado en la documentación oficial.

---

## 1. Intención y por qué

El autor quiere un **expediente verificable** de que puede mejorar un modelo pequeño con arquitecturas que hoy se están experimentando (Autor). Es para su CV: se publicará como un paper en una página de GitHub, con el código en un repo (Autor). Lo que importa es la honestidad y la reproducibilidad, no el tamaño de la mejora (Autor: "quiero ser honesto").

Idea base (Autor): tomar **Gemma 4 E4B** y cambiar su arquitectura a una basada en **loops**: un bloque de capas con pesos compartidos que se ejecuta N veces, como hizo NVIDIA con **Nemotron Labs Dory 1.0 4B**, para mejorar el razonamiento por token. No son loops de workflows: son las capas del propio modelo ejecutadas N veces.

Orden de trabajo (Autor): primero tener el modelo como lo quieren (loops), y después hacer que funcione mejor en **OpenCode**, adaptándolo a su configuración agéntica para mejorar sustancialmente en ese arnés específico.

Inspiración (Autor): un post en X donde alguien usó "Astra" para mejorar Qwen 3.5 0.8B en matemáticas. No encontré el post. El único Astra que encontré es un método de adaptación tipo LoRA (arXiv 2602.19111) **[Sin verificar que sea ese]**. Se puede usar como adaptador opcional (ablación aparte), nunca mezclado con la comparación principal.

Nichos de interés (Autor): programación y diseño creativo.

## 2. Objetivos, no-objetivos y anti-objetivos

**Objetivos**
1. Medir si convertir Gemma 4 a loops mejora la calidad por token generado, con un baseline honesto y reproducible.
2. Medir cuánto mejora el modelo dentro de OpenCode tras adaptarlo, y **atribuir esa mejora por separado** a los loops y a la adaptación.
3. Mostrar evidencia en programación (con pruebas automáticas) y en tareas creativas de diseño (con un juez).
4. Publicar todo de forma que un tercero pueda reproducirlo con una GPU de 12 GB y unos 25 dólares de nube.

**No-objetivos**
- Un modelo listo para producción.
- Un récord o paridad con Dory.
- Preentrenar desde cero.
- Correr SWE-bench completo.

**Anti-objetivos** (lo que sería un fracaso aunque técnicamente funcione) **[Mía, el autor no los enumeró; "honestidad" es suya]**
- Números inflados, elegidos a conveniencia o con criterios fijados después de ver los resultados.
- Atribuir a los loops lo que hizo el entrenamiento de OpenCode, o lo que hicieron los datos.
- Ocultar que los loops cuestan **más cómputo por token**. "Eficiencia por token" no es "eficiencia de cómputo": se reportan ambos.
- Entrenar con prompts o problemas que luego se usan para evaluar.
- Un resultado que nadie pueda reproducir con el hardware y presupuesto declarados.

## 3. Restricciones

- **Hardware del autor (revisado hoy):** RTX 5070 de 12 GB (~2.5 GB ya usados por el escritorio), i5-14600KF, 31.8 GB de RAM, 217 GB libres en C:, CUDA 12.8 instalado, Python 3.11/3.12/3.14, **sin PyTorch**, WSL solo con la distro de Docker (sin Ubuntu).
- **Presupuesto de nube (Autor):** unos 25 USD. Precios vistos en agregadores **[Sin verificar, varían]**: RTX 4090 ~0.35–0.75/h, RTX 5090 ~0.60–1.00/h, A100 80 GB ~1.4–1.5/h, H100 ~2.0–3.3/h.
- **Estimación mía de lo que compran:** del orden de 100–200 M de tokens de entrenamiento LoRA con loops en total. Se calibra con tokens/s medidos en el PC antes de gastar. Tu PC haría ~40 M de tokens al día **[Mía, sin medir]**.
- **Historial de cuelgues del PC:** todo entrenamiento debe guardar checkpoints frecuentes y poder reanudarse.
- **Licencia:** Gemma 4 va bajo Apache 2.0 **[Sin verificar en la model card]**, así que se pueden publicar derivados. No llamarlos "Gemma".
- **Cuota de OpenCode Go (Autor, 2026-11-03):** gastar todo lo posible antes de que termine el mes. Tras verificar los Términos de OpenCode (ver 5.4), la cuota se usa solo para investigación, revisión y para que los agentes de OpenCode escriban el código del proyecto, nunca para datos de entrenamiento ni juicios. **[Mía, no es asesoría legal]**
- **API oficial de DeepSeek (maestro y juez):** pendiente de aprobación del autor. Estimación mía: 30–60 USD para ~1,000–2,000 sesiones de maestro y 3–5 USD por cada 1,000 pares juzgados en los dos órdenes (precios off-peak: 0.15/0.60 USD por M de entrada/salida, 0.003 en acierto de caché). Referencia medida: una sesión de revisión de agente con ~120 K tokens (casi todo razonamiento) costó 0.024 USD. Mi estimación gruesa para ~1,000 sesiones exitosas de OpenCode es de 30–250 USD según longitud y tasa de éxito **[Mía, sin medir]**. Los términos de uso de sus salidas están por verificar.
- **GitHub** limita 100 MB por archivo: los pesos van a Hugging Face y el paper los enlaza **[Mía]**.

## 4. Lo que el autor decidió (tal cual)

- Gemma 4 E4B con arquitectura de loops; prototipo y receta en E2B, luego la misma receta en E4B (Autor: "aplicamos el mismo entrenamiento de uno a otro").
- DeepSeek V4.1 Flash como juez de las tareas creativas (Autor).
- Comparar los benchmarks oficiales contra los propios (Autor).
- Cada benchmark en su arnés propio; OpenCode como arnés de agente y como objetivo de optimización (Autor).
- Primero los loops, después OpenCode (Autor).
- El paper se publica en una página de GitHub, que el autor genera (Autor).

## 5. Diseño experimental propuesto **[Mía, salvo lo marcado]**

### 5.1 Sujetos
- **E2B:** modelo de desarrollo. Todas las variantes, todo en el PC, posibles varias semillas.
- **E4B:** modelo de confirmación. Confirma primero el efecto de los loops (L contra O-H), porque el autor puso los loops antes que OpenCode; la atribución de OpenCode (L+OC contra O-H+OC) se mide en E2B y pasa a E4B solo si sobra presupuesto. El paper declara exactamente qué variante corrió en qué modelo. Si las configuraciones reales de E2B y E4B no encajan en el mismo corte de capas, se permite un corte por modelo y el paper dice si la receta se transfirió tal cual o se reajustó.
- **Semillas [Mía]:** 3 semillas de entrenamiento en E2B para O-H y L si el tiempo lo permite; 1 en E4B. Los intervalos son bootstrap sobre los ítems de evaluación; la dispersión entre semillas de E2B se reporta aparte.

### 5.2 Fase 0: arnés de medición y baselines (sin entrenar)
1. Leer los `config.json` oficiales de E2B y E4B y fijar las cifras reales (hoy hay fuentes contradictorias: ~35 capas con patrón local:global 4:1 en E2B y ~42 capas con 5:1 en E4B **[Sin verificar]**).
2. Reproducir las cifras de Google para el modelo original con el arnés oficial de cada benchmark y los mismos ajustes, y comparar tres columnas: **Google / mi original / mi modificado**. Si "mi original" no coincide con "Google", se reporta la diferencia y se explica; ese es el chequeo de que el arnés propio es válido.
3. Cifras oficiales de E4B (modo con razonamiento) **[Sin verificar, copiadas de espejos de la model card]**: MMLU Pro 69.4, GPQA Diamond 58.6, LiveCodeBench v6 52.0, AIME 2026 42.5, Codeforces ELO 940. HumanEval no se publica.
4. Un solo servidor de inferencia para **todas** las variantes (original y modificadas): transformers con código propio para el modelo con loops, más una capa pequeña compatible con OpenAI que traduzca llamadas a herramientas. Ollama y llama.cpp estándar no soportan la arquitectura (Dory necesitó una rama propia **[Sin verificar]**).
5. Fijar versiones: OpenCode (1.19.3 hoy), versión del juez, versiones de datasets, semillas y entorno.
6. **Qué se mide (confirmatorio) [Mía]:** LiveCodeBench v6 (principal), GPQA Diamond y AIME 2026 (con varias muestras por problema, declarando el ruido) y un subconjunto fijo de MMLU Pro. Codeforces ELO no se reproduce (necesita envíos en vivo): aparece solo como cifra de Google. **Visión (añadido el 2026-11-03):** MMMU Pro (E4B 52.6 / E2B 44.2) y MATH-Vision (E4B 59.5 / E2B 52.4) entran a la tabla de tres columnas como control de regresión visual. **Hipótesis del autor (exploratoria):** el entrenamiento creativo puede sacrificar algo de razonamiento visual pero ganar razonamiento espacial; se mide con un benchmark de razonamiento espacial elegido antes de correr (ver `references/gemma-loop/08`), se reporta como exploratorio y no como criterio de éxito. **Validez del arnés:** "mi original" cuenta como válido en un benchmark si cae a menos de 3 puntos de la cifra de Google o dentro de su intervalo de bootstrap; si no, se marca "no validado" y se explica.
7. **Qué es "original" [Mía]:** el checkpoint instruction-tuned con modo de razonamiento activado (como las cifras de Google), con los mismos ajustes de decodificación y la misma precisión en todas las variantes. **Visión incluida (Autor, 2026-11-03: "sí es importante")**: se carga la torre de visión y queda congelada; la de audio no se carga. El LoRA va solo en la pila de lenguaje. Si bf16 no cabe en 12 GB, todas las variantes se evalúan con la misma cuantización y el paper lo dice.
8. **Puertas antes de gastar dinero:** (a) probar con una captura real que el juez ve imágenes; (b) comprobar que los términos de DeepSeek permiten entrenar con sus salidas; (c) comprobar en la model card que Apache 2.0 permite publicar los pesos fusionados; (d) comprobar que el maestro corre en OpenCode con llamadas a herramientas (el autor ya lo usa así); (e) leer los `config.json` oficiales. Si (a), (b) o (c) fallan, se rediseña la parte afectada antes de seguir.

### 5.3 Fase 1: loops (contribución B)
- **Estructura (revisada el 2026-11-03 con los `config.json` oficiales, ver `references/gemma-loop/01` y `02`):** prelude, bloque en loop y coda. Los números se confirmaron: 35 = 7×5 (E2B) y 42 = 7×6 (E4B). Pero Gemma 4 comparte la caché KV en sus últimas capas (E2B: 15–34; E4B: 24–41), y el corte 1/5/1 cruzaba esa frontera. **Regla del corte [Mía]: prelude = todas las capas que producen KV; coda = el último periodo; loop = el resto.** E2B: prelude 0–14, loop 15–29, coda 30–34. E4B: prelude 0–23, loop 24–35, coda 36–41. Ventajas: N = 1 es exactamente el original, el loop no escribe caché (las capas consumidoras leen el KV del productor), así que la memoria KV no crece con N, y el ancho del MLP es uniforme dentro del bloque. Limitación que el paper declara: en los loops, cada token refina su estado contra una memoria K/V fija (la del prelude); no hay intercambio nuevo de información entre tokens dentro del loop. Ablación opcional en E2B: bloque en la región productora (E2B 5–14) con caché por pasada.
- **Puente entre loops [Mía]:** en el loop 1 el bloque recibe la salida del prelude tal cual (identidad). Desde el loop 2 recibe `w·h_prelude + (1−w)·h_anterior`, con `w = sigmoid(a)` aprendido e inicializado en 0.99 (receta de "Closing the Loop", arXiv 2610.00673).
- **Qué se reutiliza:** los pesos del bloque se comparten entre loops; N va de 1 a 4 en evaluación. **N primario = 2 [Mía], fijado antes de correr:** es el caso más barato y donde Dory reporta el mayor salto **[Sin verificar]**. N = 3 y 4 son exploratorios, se etiquetan así y llevan corrección por comparaciones múltiples si se reportan pruebas.
- **Decisiones técnicas abiertas** (las resuelve el constructor con ablaciones pequeñas en E2B; defecto en negrita):
  - Per-layer embeddings (PLE) en cada loop: **compartir** la misma tabla, o añadir un vector aprendido por loop.
  - Caché KV: con el corte revisado no hace falta caché extra (el bloque está en la región que solo lee KV compartido). Solo la ablación de la región productora necesita caché por pasada.
- **Entrenamiento de adaptación:** LoRA/QLoRA, con N sorteado entre 1 y 4 por lote para que un solo checkpoint sirva a varios N. Datos públicos de licencia permisiva (código y razonamiento); elección concreta a cargo del constructor, excluyendo cualquier contaminación con los benchmarks.
- **Variantes.** Regla principal **[Mía]**: misma cantidad de **tokens de datos** e iguales hiperparámetros. L gasta más cómputo por token de datos, así que se reporta también el cómputo y, solo en E2B, una variante **O-H extendida** con los FLOPs igualados. Sin ella el paper puede afirmar "mejor calidad con más cómputo", no "mejor uso del cómputo".
  - **O**: original sin tocar.
  - **O-H**: original con el mismo entrenamiento de adaptación, **sin loops**. Es el control más importante: separa el efecto de los datos del efecto de la arquitectura.
  - **R**: capas repetidas en inferencia sin entrenar (N = 2, 3, 4). Control gratis. Se espera que degrade (arXiv 2605.23872 mide hasta −17.7 pp).
  - **R-damped**: igual que R pero con el paso amortiguado sin entrenamiento de "Training-Free Looped Transformers" (`x ← (1−1/N)·x + (1/N)·bloque(x)`). Control gratis adicional **[Mía]**.
  - **L**: con loops y entrenado, evaluado en N = 1, 2, 3, 4.

### 5.4 Fase 2: adaptación a OpenCode (contribución A)
- **Restricción legal (verificada el 2026-11-03):** los Términos de OpenCode (servicios hospedados, incluido OpenCode Go; vigentes desde el 15-ago-2026) prohíben "automatically or programmatically extracts data or Output" y "uses Output to develop artificial intelligence models that compete with the Services or any Third Party Models". Por eso **ningún dato de entrenamiento ni juicio sale de OpenCode Go**. El maestro y el juez usan la **API oficial de DeepSeek** (modelo `deepseek-flash`, clave propia del autor), cuyos términos (§4.2(3), 27-mar-2026) permiten "training other models (such as model distillation)". OpenCode se usa como software libre (MIT), apuntando a esa API. Ese gasto de API está pendiente de aprobación del autor.
- **Datos:** sesiones de DeepSeek V4.1 Flash trabajando dentro de OpenCode sobre un conjunto de tareas (código y diseño front-end), conservando solo las exitosas (pasan pruebas o el juez las aprueba). Se entrena solo sobre los turnos del asistente. Las tareas del maestro se descontaminan contra todos los benchmarks y prompts de evaluación. El maestro debería ser de una familia distinta del juez creativo **[Mía]**; si no es posible, se usa un segundo juez (ver 5.5) y se declara la limitación.
- **Configuración objetivo [Mía]:** OpenCode de serie (agente por defecto), versión fijada y configuración mínima documentada. No los agentes personalizados `oc-*` del autor, para que un tercero pueda reproducirlo.
- **Variantes:**
  - **L+OC**: L con la adaptación a OpenCode.
  - **O-H+OC**: O-H con la misma adaptación. Es la comparación principal (mismos tokens de entrenamiento).
  - **O+OC**: original adaptado directamente. Es el "baseline práctico" y puede correr solo en E2B si el presupuesto lo exige.
- **Evaluación en OpenCode:** conjunto de tareas **separado** de las de entrenamiento, con pruebas automáticas. Métricas: tasa de llamadas a herramientas bien formadas, tareas resueltas, pasos y tokens por tarea resuelta. Se incluye el resultado del maestro como línea de referencia.
- **Qué benchmark existente corre con OpenCode como agente:** no lo sé. Si no hay uno adecuado, el conjunto de tareas es propio y el paper lo declara como autoconstruido, no estándar.
- **Chequeo de generalización:** repetir las evaluaciones oficiales de la Fase 0 tras la adaptación. Se espera cierta regresión y se reporta.

### 5.5 Eje creativo (diseño)
- **Tareas:** front-end (páginas, componentes) y SVG **[Mía; el autor aún no precisó qué quiere ver]**. En la Fase 1 en un solo intento; en la Fase 2 dentro de OpenCode.
- **Juez:** DeepSeek V4.1 Flash, que según agregadores tiene visión nativa **[Sin verificar]**. Se renderiza lo generado con un navegador sin interfaz y se juzgan capturas.
- **Salvaguardas:**
  - Comparaciones por pares, cada una en los dos órdenes y a ciegas.
  - Rúbrica fija escrita antes de correr.
  - Prompts de evaluación nunca usados para entrenar.
  - El autor califica a mano, a ciegas, un piloto de 40 pares y luego 120 pares para el análisis principal **[Mía: 40 pares dan ±0.25 de kappa, 120 dan ±0.13]**, para reportar el acuerdo juez-humano.
  - Porcentaje de victorias con intervalo de confianza.
  - Versión del juez fijada y juicios crudos guardados.
  - Un segundo juez de otra familia sobre una submuestra, para cruzar resultados y detectar sesgo de estilo, sobre todo si DeepSeek también fue el maestro.

### 5.6 Criterios de éxito (se fijan antes de la primera corrida final) **[Mía: valores por defecto, el autor los confirma o cambia]**
- **S1, sanidad:** L con N = 1 queda dentro del ruido de O en todas las evaluaciones.
- **S2, los loops ayudan:** el intervalo de confianza del 95% (bootstrap pareado) de la diferencia L (N = 2) − O-H en LiveCodeBench v6 excluye el cero a favor de L, **o** iguala la precisión con menos tokens generados. Siempre se reportan también los FLOPs por problema.
- **S3, creativo:** el límite inferior del intervalo del 95% del porcentaje de victorias de L sobre O-H es mayor que 50%.
- **S4, OpenCode:** L+OC supera a O-H+OC en tareas resueltas, con la misma regla de intervalo; se reporta aparte la mejora de cualquier variante +OC sobre la no adaptada.
- **S5, regresión:** se reporta, sin umbral de "aprobado"; el paper dice dónde empeoró, incluida la visión (MMMU Pro, MATH-Vision). El autor acepta de antemano una posible pérdida de razonamiento visual.
- Si un criterio no se cumple, el paper dice "no hay evidencia de mejora" para esa afirmación. Con conjuntos de evaluación pequeños (AIME 2026 tiene 30 problemas) hay que calcular y declarar el efecto mínimo detectable.

## 6. Entregables
- **Repo:** código, configuraciones, scripts de evaluación, generaciones crudas, juicios del juez, semillas y entorno bloqueado. Fuera de OneDrive (para que no sincronice checkpoints) **[Mía]**.
- **Hugging Face:** adaptadores y pesos con model card honesta (limitaciones incluidas).
- **Página de GitHub:** el paper completo, que genera el autor: resumen, método, tabla de tres columnas, resultados negativos y limitaciones, y galería lado a lado de las tareas creativas.

## 7. Riesgos
- Con ≤ 200 M de tokens, los loops quizá solo recuperen la calidad original o mejoren algo estrecho. Resultado parcial posible y aceptable si se reporta.
- El contexto largo de OpenCode (prompt de sistema + herramientas + historial) aprieta los 12 GB durante el entrenamiento; habrá que truncar o entrenar por turno.
- Las herramientas de Windows para cuantización y kernels son frágiles; probablemente haga falta Ubuntu en WSL2.
- Los presupuestos de nube y de la API de DeepSeek (maestro y juez) aún no están estimados.
- El juez puede ser ruidoso o sesgado; de ahí las salvaguardas.
- Las cifras de arquitectura y los benchmarks oficiales vienen de fuentes secundarias.

## 8. Lo que se deja al constructor
Datasets de adaptación, hiperparámetros y rango de LoRA, capa de servidor, resolución de las decisiones técnicas abiertas (PLE, KV), construcción del conjunto de tareas de OpenCode, reparto exacto del presupuesto entre ejecuciones locales y de nube, y el diseño del cuaderno de resultados.

## 9. Llamadas mías para que las revises
1. Paper en inglés.
2. Documento de diseño dentro de la carpeta de trabajo de la sesión (para poder abrirlo en la app); al aprobarse se mueve al repo, fuera de OneDrive.
3. E2B como sujeto principal y E4B solo como confirmación parcial.
4. Corte revisado: prelude = capas productoras de KV, coda = último periodo, loop = el resto (E2B 0–14 / 15–29 / 30–34; E4B 0–23 / 24–35 / 36–41).
5. PLE compartido entre loops; puente escalar `w` inicializado en 0.99; R-damped como control extra.
6. Tareas creativas = front-end y SVG.
7. Valores por defecto de los criterios S1–S5.
8. Control O-H como comparación principal, y O+OC como baseline práctico opcional.
9. N primario = 2; E4B confirma primero los loops.
10. Conjunto confirmatorio de benchmarks (sin Codeforces ELO) y tolerancia de 3 puntos para validar el arnés.
11. Igualar tokens de datos, con O-H extendida (FLOPs iguales) solo en E2B.
12. Semillas: 3 en E2B, 1 en E4B.
13. OpenCode de serie como configuración objetivo.
14. Maestro y juez de familias distintas, con un segundo juez sobre una submuestra.
15. "Original" = instruction-tuned, razonamiento activado, solo pila de lenguaje.
