# Auditoría de URLs — referencias Gemma Loop

## Resumen

Auditadas **209 URLs distintas** de los siete documentos (`01`–`07`). Quedaron fuera, por no ser auditables, `http://localhost:8000`, `http://localhost:8000/v1`, `http://localhost:8000/v1/chat/completions`, `http://127.0.0.1:4096`, `http://127.0.0.1:4000`, `http://127.0.0.1:8000/v1@4000` y la plantilla `https://pypi.org/pypi/<paquete>/json`.

**Estados:** OK 201, HTTP 401 4, HTTP 404 3, HTTP 400 1.

**Veredictos:** SUPPORTS 202, PARTIAL 4, COULD-NOT-CHECK 3.

URLs que no cumplen (estado OK + veredicto SUPPORTS):

- `https://ai.google.dev/gemma/prohibited_use_policy` — OK, PARTIAL: Existe y el cuerpo ya se extrae; la nota «no se pudo extraer» está desactualizada. Sigue sin confirmarse su vinculación a los pesos.
- `https://console.vast.ai/api/v0/bundles/?q=` — HTTP 400, COULD-NOT-CHECK: URL incompleta (q vacío) → HTTP 400; la variante con q={limit:5} responde 200.
- `https://api.runpod.io/v2/network-volumes` — HTTP 401, COULD-NOT-CHECK: requiere API key; la doc confirma el endpoint POST.
- `https://console.vast.ai/api/v0/bundles/?q=%7B%22limit%22%3A5%7D` — OK, PARTIAL: RTX 5090 0.2689/h, compute_cap 1200, driver 580.126.09 confirmados; A100 40 GB 0.4030 y H100 0.60 no aparecen en esta respuesta (solo 5 ofertas).
- `https://opencode.ai/inference/go/openai/v1/chat/completions` — HTTP 404, SUPPORTS: GET sin auth devuelve 404; la ruta aparece literalmente en la doc de Go.
- `https://opencode.ai/inference/go/openai/v1` — HTTP 404, SUPPORTS: Base URL confirmada en la doc de Go; GET sin auth 404.
- `https://opencode.ai/inference/go/openai/v1@4000` — HTTP 404, COULD-NOT-CHECK: Sintaxis de mitmproxy reverse (@puerto), no es una URL navegable.
- `https://arena.ai/blog/llm-judge-self-preference` — OK, PARTIAL: 58%/34% y 71%/68% confirmados en prosa; «−7.1 pp» solo aparece en los datos del gráfico.
- `https://ui-bench.dev/ux-judge/methodology` — OK, PARTIAL: 10 criterios, pesos, umbrales 82/65 y viewports confirmados; «80 diseños / 6 modelos / 10 variaciones» está en /ux-judge, no aquí.
- `https://api.deepseek.com` — HTTP 401, SUPPORTS: Base URL OpenAI; HTTP 401 sin API key (host alcanzable).
- `https://api.deepseek.com/anthropic` — HTTP 401, SUPPORTS: Base URL Anthropic; HTTP 401 sin API key.
- `https://api.deepseek.com/beta` — HTTP 401, SUPPORTS: Base URL beta para strict; HTTP 401 sin API key.

Método: HTTP GET con user-agent de navegador el 2026-11-03; para los `.jsonl` de LiveCodeBench se descargó por streaming (superan el límite de 5 MB) y se contaron filas y fechas. Los binarios `.safetensors` se validaron por cabecera/range request y por el tamaño del tree API. `pypi.org/pypi/transformers/json` excede 5 MB y se truncó; la versión se confirmó en la página del proyecto.

## Tabla

| archivo | URL | estado | título | veredicto | nota |
|---|---|---|---|---|---|
| 01, 02, 03, 07 | https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json | OK | raw · config.json (E2B-it) | SUPPORTS | 35 capas, patrón 4:1, PLE 256, KV shared 20, vocab 262 144. |
| 01, 02, 03, 07 | https://huggingface.co/google/gemma-4-E4B-it/raw/main/config.json | OK | raw · config.json (E4B-it) | SUPPORTS | 42 capas, patrón 5:1, KV shared 18, head_dim 256/global 512. |
| 01 | https://huggingface.co/google/gemma-4-E2B/raw/main/config.json | OK | raw · config.json (E2B base) | SUPPORTS | text_config idéntico al instruct; solo cambia eos. |
| 01 | https://huggingface.co/google/gemma-4-E4B/raw/main/config.json | OK | raw · config.json (E4B base) | SUPPORTS | text_config idéntico al instruct; solo cambia eos. |
| 01 | https://huggingface.co/google/gemma-4-E2B-it/resolve/main/model.safetensors | OK | binario · model.safetensors (E2B-it) | SUPPORTS | Binario de 10 246 621 918 B; cabecera y shapes verificados por range request. |
| 01 | https://huggingface.co/google/gemma-4-E4B-it/resolve/main/model.safetensors | OK | binario · model.safetensors (E4B-it) | SUPPORTS | Binario de 15 992 595 884 B; cabecera y shapes por range request. |
| 01 | https://ai.google.dev/gemma/docs/gemma_4_license | OK | Apache License 2.0 · Gemma · Google AI for Developers | SUPPORTS | Apache 2.0 íntegro y sin cláusulas extra; «Last updated 2026-04-01» presente. |
| 01 | https://ai.google.dev/gemma/terms | OK | Gemma Terms of Use · Google AI for Developers | SUPPORTS | Gemma Terms of Use; remite a la licencia Apache 2.0 de Gemma 4. |
| 01, 04 | https://arxiv.org/abs/2607.02770 | OK | [2607.02770] Gemma 4 Technical Report | SUPPORTS | Gemma 4 Technical Report, Google DeepMind, 2026. |
| 01, 02 | https://huggingface.co/api/models/google/gemma-4-E4B-it | OK | JSON · HF model API (E4B-it) | SUPPORTS | gated:false, license apache-2.0, arXiv 2607.02770. |
| 01 | https://huggingface.co/api/models/google/gemma-4-E2B-it | OK | JSON · HF model API (E2B-it) | SUPPORTS | gated:false, license apache-2.0, arXiv 2607.02770. |
| 01 | https://huggingface.co/api/models/google/gemma-4-E4B-it/tree/main | OK | JSON · HF tree (E4B-it) | SUPPORTS | model.safetensors = 15 992 595 884 B. |
| 01 | https://huggingface.co/api/models/google/gemma-4-E2B-it/tree/main | OK | JSON · HF tree (E2B-it) | SUPPORTS | model.safetensors = 10 246 621 918 B. |
| 01 | https://huggingface.co/google/gemma-4-E4B-it/raw/main/README.md | OK | raw · README.md (model card E4B-it) | SUPPORTS | Model card: benchmarks, PLE, «E» = effective, Apache 2.0. |
| 01, 04, 07 | https://huggingface.co/google/gemma-4-E4B-it/raw/main/generation_config.json | OK | raw · generation_config.json | SUPPORTS | temperature 1.0, top_k 64, top_p 0.95, eos [1,106,50]. |
| 01, 04, 07 | https://huggingface.co/google/gemma-4-E4B-it/raw/main/chat_template.jinja | OK | raw · chat_template.jinja | SUPPORTS | Canónico (2026-07-09); enable_thinking, preserve_thinking, strip_thinking. |
| 01, 04, 07 | https://ai.google.dev/gemma/docs/core/model_card_4 | OK | Gemma 4 model card · Google AI for Developers | SUPPORTS | Model card oficial: benchmarks, best practices, thinking, datos. |
| 01 | https://arxiv.org/html/2607.02770 | OK | Gemma 4 Technical Report | SUPPORTS | §2 arquitectura, ratios 4:1/5:1, KV 20/35 y 18/42, Table 5. |
| 01, 03, 07 | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/modeling_gemma4.py | OK | raw · modeling_gemma4.py (main) | SUPPORTS | Clases Gemma4*, PLE y KV sharing; main en cc1451df. |
| 01, 03 | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/configuration_gemma4.py | OK | raw · configuration_gemma4.py | SUPPORTS | kv_sharing_modes, per_layer_config y defaults. |
| 01 | https://raw.githubusercontent.com/huggingface/transformers/v5.22.1/src/transformers/models/gemma4/modeling_gemma4.py | OK | raw · modeling_gemma4.py (v5.22.1) | SUPPORTS | KV sharing inline, sin kv_sharing_modes. |
| 01 | https://raw.githubusercontent.com/huggingface/transformers/v5.5.0/src/transformers/models/gemma4/modeling_gemma4.py | OK | raw · modeling_gemma4.py (v5.5.0) | SUPPORTS | Primera versión con Gemma 4 (HTTP 200). |
| 01 | https://api.github.com/repos/huggingface/transformers/commits?path=src/transformers/models/gemma4/modeling_gemma4.py | OK | JSON · commits API (gemma4) | SUPPORTS | cc1451df, 2026-11-02, «[Improvement] Formalize KV sharing (#47290)». |
| 01, 03 | https://huggingface.co/docs/transformers/main/en/model_doc/gemma4 | OK | Gemma4 · Hugging Face | SUPPORTS | Doc oficial: «contributed 2026-04-02», estable v5.22.1, PLE. |
| 01, 03 | https://pypi.org/pypi/transformers/json | OK | JSON · PyPI transformers | SUPPORTS | 5.5.0 (02-04-2026) y 5.22.1 (30-10-2026); respuesta >5 MB truncada, versiones confirmadas en /project/transformers/. |
| 01 | https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/cache_utils.py | OK | raw · cache_utils.py | SUPPORTS | DynamicLayer.update concatena; base del problema con loops. |
| 01 | https://ai.google.dev/gemma/prohibited_use_policy | OK | Gemma Prohibited Use Policy · Google AI for Developers | PARTIAL | Existe y el cuerpo ya se extrae; la nota «no se pudo extraer» está desactualizada. Sigue sin confirmarse su vinculación a los pesos. |
| 02 | https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | OK | nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 · Hugging Face | SUPPORTS | 8/20/8 pares, loops 1/2/4, KV last_loop, OpenMDW-1.1, release 2026-10-16. |
| 02 | https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/raw/main/config.json | OK | raw · config.json (Dory) | SUPPORTS | 72 sublayers (16/40/16), n_recurrent_loops 2, per_loop. |
| 02 | https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/dory.py | OK | vllm/vllm/model_executor/models/dory.py at main · vllm-project/vllm… | SUPPORTS | Forward del loop: justnorm(state + input_h), caché por loop. |
| 02 | https://openmdw.ai/license/1-1/ | OK | OpenMDW-1.1 · OpenMDW | SUPPORTS | OpenMDW 1.1: uso comercial y no comercial; no es Apache 2.0. |
| 02 | https://github.com/vllm-project/vllm/pull/60149 | OK | [Model] Add Dory support by shan18 · Pull Request #60149 · vllm-pro… | SUPPORTS | «[Model] Add Dory support» (shan18), merged 2026-10-25. |
| 02 | https://github.com/ggml-org/llama.cpp/pull/30306 | OK | dory support for llama.cpp by brianchmielNV · Pull Request #30306 ·… | SUPPORTS | «dory support for llama.cpp», Draft y abierto (creado 2026-10-10). |
| 02 | https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py | OK | transformers/src/transformers/models/gemma4/modeling_gemma4.py at m… | SUPPORTS | Mismas clases Gemma4*, PLE y KV sharing (vista GitHub). |
| 02 | https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/configuration_gemma4.py | OK | transformers/src/transformers/models/gemma4/configuration_gemma4.py… | SUPPORTS | Config con KV sharing por capas (vista GitHub). |
| 02 | https://arxiv.org/abs/2410.20672 | OK | [2410.20672] Relaxed Recursive Transformers: Effective Parameter Sh… | SUPPORTS | Relaxed Recursive Transformers: Stepwise/Average/Lower, LoRA por loop. |
| 02 | https://arxiv.org/abs/2507.10524 | OK | [2507.10524] Mixture-of-Recursions: Learning Dynamic Recursive Dept… | SUPPORTS | Mixture-of-Recursions: routers por token, KV sharing, Pareto isoFLOPs. |
| 02 | https://arxiv.org/abs/2502.05171 | OK | [2502.05171] Scaling up Test-Time Compute with Latent Reasoning: A … | SUPPORTS | Huginn: recurrencia de profundidad, Poisson log-normal, TBPTT k=8. |
| 02 | https://arxiv.org/abs/2510.25741 | OK | [2510.25741] Scaling Latent Reasoning via Looped Language Models | SUPPORTS | Ouro/LoopLM: 24 capas, T=4, gate por entropía, 7.7 T tokens. |
| 02 | https://arxiv.org/abs/2511.07384 | OK | [2511.07384] Teaching Pretrained Language Models to Think Deeper wi… | SUPPORTS | Retrofitted Recurrence: prelude/bloque/coda, currículo 1−sqrt, Muon. |
| 02 | https://github.com/mcleish7/retrofitting-recurrence | OK | GitHub - mcleish7/retrofitting-recurrence: Teaching Pretrained Lang… | SUPPORTS | convert_llama.py, verificación de identidad N=1, scripts. |
| 02 | https://arxiv.org/abs/2610.00673 | OK | [2610.00673] Closing the Loop: Practical Training Recipes for Loope… | SUPPORTS | Closing the Loop: w=0.99, smoothed exit loss, LoRA por paso inútil. |
| 02 | https://arxiv.org/abs/2608.11233 | OK | [2608.11233] Retrofitting Recurrent Depth into a Pretrained Languag… | SUPPORTS | Split bridge, identidad T=1, 6 M vs 180 M, extrapolación 1.5×. |
| 02 | https://arxiv.org/abs/2605.23872 | OK | [2605.23872] Training-Free Looped Transformers | SUPPORTS | Training-Free Looped Transformers: Euler/RK amortiguado, ventana del medio. |
| 02 | https://arxiv.org/abs/1807.03819 | OK | [1807.03819] Universal Transformers | SUPPORTS | Universal Transformers: transición compartida, ACT, coordinate embeddings. |
| 02 | https://arxiv.org/abs/2608.18171 | OK | [2608.18171] Looped Language Models Improve Compositional Tool Calling | SUPPORTS | Loops y tool calling composicional; profundidad adaptativa. |
| 02 | https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/dory.py | OK | vllm/vllm/transformers_utils/configs/dory.py at main · vllm-project… | SUPPORTS | DoryConfig, validaciones y semántica de last_loop. |
| 03 | https://download.pytorch.org/whl/cu128/torch/ | OK | Links for torch | SUPPORTS | Última wheel cu128: torch 2.11.0. |
| 03 | https://download.pytorch.org/whl/cu130/torch/ | OK | Links for torch | SUPPORTS | Rango 2.9.0 → 2.14.1. |
| 03 | https://download.pytorch.org/whl/cu132/torch/ | OK | Links for torch | SUPPORTS | Rango 2.12.0 → 2.15.0 (2.15.0 sin cp310). |
| 03 | https://download.pytorch.org/whl/cu128/torchvision/ | OK | Links for torchvision | SUPPORTS | Máximo 0.26.x en cu128. |
| 03 | https://download.pytorch.org/whl/cu128/torchaudio/ | OK | Links for torchaudio | SUPPORTS | Máximo 2.11.0 en cu128; pareja exacta con torch 2.11.0. |
| 03 | https://github.com/pytorch/pytorch/releases/tag/v2.7.0 | OK | Release PyTorch 2.7.0 Release · pytorch/pytorch · GitHub | SUPPORTS | Notas 2.7: soporte Blackwell y fix sm100/sm120. |
| 03 | https://github.com/pytorch/pytorch/releases | OK | Releases · pytorch/pytorch · GitHub | SUPPORTS | 2.15.0 (28-oct-2026), CUDA 13.2 por defecto. |
| 03 | https://huggingface.co/docs/bitsandbytes/main/en/installation | OK | Installation Guide · Hugging Face | SUPPORTS | Tabla por OS/CUDA: Windows x86-64 CUDA 12.8, sm70–sm120. |
| 03 | https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/README.md | OK | raw · README.md (bitsandbytes) | SUPPORTS | Features y requisitos de bitsandbytes. |
| 03 | https://pypi.org/pypi/bitsandbytes/json | OK | JSON · PyPI bitsandbytes | SUPPORTS | Última 0.50.2; un wheel por OS. |
| 03 | https://unsloth.ai/docs/models/gemma-4 | OK | Gemma 4 - How to Run Locally · Unsloth Documentation | SUPPORTS | E2B/E4B/12B/26B-A4B/31B; texto, visión y audio; NVFP4 y vLLM. |
| 03, 07 | https://unsloth.ai/docs/models/gemma-4/train | OK | Gemma 4 Fine-tuning Guide · Unsloth Documentation | SUPPORTS | QLoRA: E2B 8 GB, E4B 10 GB; bugs de Gemma 4. |
| 03 | https://raw.githubusercontent.com/unslothai/unsloth/main/README.md | OK | raw · README.md (unsloth) | SUPPORTS | Windows/Linux/WSL e instalación con torch-backend. |
| 03 | https://unsloth.ai/docs/blog/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth | OK | Fine-tuning LLMs with Blackwell, RTX 50 series & Unsloth · Unsloth … | SUPPORTS | Blackwell: TORCH_CUDA_ARCH_LIST=12.0 y xformers. |
| 03 | https://huggingface.co/unsloth/gemma-4-E4B-it | OK | unsloth/gemma-4-E4B-it · Hugging Face | SUPPORTS | E2B 2.3 B efectivos/5.1 B; E4B 4.5 B/8 B. |
| 03 | https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/README.md | OK | raw · README.md (flash-attention) | SUPPORTS | FA2/FA3/FA4 por arquitectura; FA3 solo Hopper. |
| 03 | https://huggingface.co/docs/transformers/main/en/attention_interface | OK | Attention backends · Hugging Face | SUPPORTS | Backends sdpa, flash_attention_2, flex_attention y kernels del Hub. |
| 03 | https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html | OK | GPU - vLLM | SUPPORTS | Instalación GPU, CUDA 12.9 por defecto, no Windows nativo. |
| 03 | https://docs.vllm.ai/en/latest/models/supported_models/ | OK | Supported Models - vLLM | SUPPORTS | Gemma 4 y Dory nativos; backend Transformers. |
| 03 | https://docs.vllm.ai/en/latest/contributing/model/registration.html | OK | Registering a Model - vLLM | SUPPORTS | ModelRegistry.register_model. |
| 03 | https://docs.vllm.ai/en/latest/design/plugin_system/ | OK | Plugin System - vLLM | SUPPORTS | Entry point vllm.general_plugins. |
| 03 | https://huggingface.co/docs/transformers/main/en/serving | OK | Serve CLI · Hugging Face | SUPPORTS | transformers serve, endpoints y tool calling. |
| 03 | https://huggingface.co/docs/transformers/main/en/main_classes/trainer | OK | Trainer · Hugging Face | SUPPORTS | resume_from_checkpoint, save_only_model, save_total_limit. |
| 03 | https://www.runpod.io/pricing | OK | GPU Cloud Pricing · Per-Second H100, A100, RTX · Runpod | SUPPORTS | Precios on-demand y storage; actualizado 28-oct-2026. |
| 03 | https://docs.runpod.io/pods/storage/create-network-volumes | OK | Network volumes - Runpod Documentation | SUPPORTS | Network volumes, montaje en /workspace, no añadible después. |
| 03 | https://console.vast.ai/api/v0/bundles/?q= | HTTP 400 | JSON · Vast.ai bundles (URL incompleta) | COULD-NOT-CHECK | URL incompleta (q vacío) → HTTP 400; la variante con q={limit:5} responde 200. |
| 03 | https://download.pytorch.org/whl/cu128 | OK | index · wheels cu128 | SUPPORTS | Índice de wheels cu128 usado en --index-url. |
| 03 | https://api.runpod.io/v2/network-volumes | HTTP 401 | network-volumes | COULD-NOT-CHECK | HTTP 401: requiere API key; la doc confirma el endpoint POST. |
| 03, 07 | https://github.com/huggingface/transformers/issues/45242 | OK | [Gemma 4] `use_cache=False` corrupts attention computation, produci… | SUPPORTS | use_cache=False corrompe la atención; fix en PR #45312 (2026-04-09). |
| 03 | https://docs.nvidia.com/cuda/wsl-user-guide/index.html | OK | CUDA on WSL User Guide — CUDA on WSL 13.4 documentation | SUPPORTS | CUDA on WSL 13.4; no instalar driver Linux dentro de WSL. |
| 03 | https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl | OK | Enable NVIDIA CUDA on WSL 2 · Microsoft Learn | SUPPORTS | Requisitos de WSL 2 y kernel para CUDA. |
| 03 | https://docs.vast.ai/cli/get-started | OK | CLI Hello World - Vast.ai Documentation: Affordable GPU Cloud Marke… | SUPPORTS | Hello World: search offers, create instance, copy, destroy (redirige a /cli/hello-world). |
| 03 | https://docs.vast.ai/cli/reference/create-volume | OK | vastai create volume - Vast.ai Documentation: Affordable GPU Cloud … | SUPPORTS | Referencia de vastai create volume. |
| 03 | https://vast.ai/install.sh | OK | script · install.sh (CLI Vast.ai) | SUPPORTS | Instalador del CLI Vast.ai (bash, sin Python, sin sudo). |
| 03 | https://console.vast.ai/api/v0/bundles/?q=%7B%22limit%22%3A5%7D | OK | JSON · Vast.ai bundles (limit=5) | PARTIAL | RTX 5090 0.2689/h, compute_cap 1200, driver 580.126.09 confirmados; A100 40 GB 0.4030 y H100 0.60 no aparecen en esta respuesta (solo 5 ofertas). |
| 04 | https://huggingface.co/google/gemma-4-E4B-it | OK | google/gemma-4-E4B-it · Hugging Face | SUPPORTS | Model card: tabla (MMLU-Pro 69.4, AIME 42.5, LCB 52.0), sampling, Apache 2.0. |
| 04 | https://arxiv.org/html/2607.02770v2 | OK | Gemma 4 Technical Report | SUPPORTS | v2 del technical report; Tabla 5 y modo thinking. |
| 04 | https://ai.google.dev/gemma/docs/capabilities/thinking | OK | Thinking mode in Gemma · Google AI for Developers | SUPPORTS | Activación de thinking, tokens y parse_response. |
| 04 | https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/main/README.md | OK | raw · README.md (LiveCodeBench) | SUPPORTS | release_v6 = 1055 problemas (may-2023→abr-2025); n=10. |
| 04 | https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/main/lcb_runner/runner/parser.py | OK | raw · parser.py (LCB) | SUPPORTS | n=10, temp 0.2, top_p 0.95, max_tokens 2000. |
| 04 | https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/main/lcb_runner/evaluation/pass_k_utils.py | OK | raw · pass_k_utils.py | SUPPORTS | Estimador pass@k. |
| 04 | https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl | OK | JSONL · test6.jsonl | SUPPORTS | 175 filas, 2025-01-04 → 2025-04-06 (contado por streaming; >5 MB para webfetch). |
| 04 | https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test5.jsonl | OK | JSONL · test5.jsonl | SUPPORTS | 167 filas; 7 con fecha ≥2025-01-01 → 182 junto con test6. |
| 04 | https://api.github.com/repos/LiveCodeBench/LiveCodeBench/contents/lcb_runner/runner | OK | JSON · GitHub contents (LCB runner) | SUPPORTS | Inventario real del runner (parser, custom_evaluator, etc.). |
| 04 | https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/main/lcb_runner/runner/custom_evaluator.py | OK | raw · custom_evaluator.py | SUPPORTS | Evaluación de salidas custom. |
| 04 | https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/raw/main/README.md | OK | raw · README.md (MMLU-Pro) | SUPPORTS | 12 032 test / 70 validation, hasta 10 opciones, MIT. |
| 04 | https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/mmlu_pro/_default_template_yaml | OK | raw · _default_template_yaml (mmlu_pro) | SUPPORTS | 5-shot y regex de extracción. |
| 04 | https://huggingface.co/datasets/Idavidrein/gpqa | OK | Idavidrein/gpqa · Datasets at Hugging Face | SUPPORTS | Gated, 448 preguntas, CC BY 4.0. |
| 04 | https://ar5iv.labs.arxiv.org/html/2311.12022 | OK | [2311.12022] GPQA: A Graduate-Level Google-Proof Q&A Benchmark | SUPPORTS | Paper GPQA: 448 preguntas, Diamond = 198. |
| 04 | https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/gpqa/README.md | OK | raw · README.md (gpqa) | SUPPORTS | Variantes GPQA (main, cot, zeroshot). |
| 04 | https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/gpqa/cot_zeroshot/_gpqa_cot_zeroshot_yaml | OK | raw · _gpqa_cot_zeroshot_yaml | SUPPORTS | CoT y filtros de GPQA. |
| 04 | https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/docs/interface.md | OK | raw · interface.md (lm-eval) | SUPPORTS | enable_thinking y think_end_token en el CLI. |
| 04 | https://huggingface.co/datasets/MathArena/aime_2026/raw/main/README.md | OK | raw · README.md (aime_2026) | SUPPORTS | 30 problemas, CC BY-NC-SA-4.0. |
| 04 | https://huggingface.co/datasets/MathArena/aime_2026/raw/main/eval.yaml | OK | raw · eval.yaml (aime_2026) | SUPPORTS | Configuración de evaluación de AIME 2026. |
| 04 | https://raw.githubusercontent.com/eth-sri/matharena/main/README.md | OK | raw · README.md (MathArena) | SUPPORTS | --n 4 y competiciones soportadas. |
| 04 | https://raw.githubusercontent.com/eth-sri/matharena/main/src/matharena/grader.py | OK | raw · grader.py (MathArena) | SUPPORTS | Extracción de \boxed{} y grading. |
| 04 | https://huggingface.co/datasets/math-ai/aime26/raw/main/README.md | OK | raw · README.md (aime26) | SUPPORTS | 30 ejemplos, Apache 2.0, split test. |
| 04 | https://github.com/LiveCodeBench/LiveCodeBench.git | OK | GitHub - LiveCodeBench/LiveCodeBench: Official repository for the p… | SUPPORTS | Repo oficial (redirige a github.com/LiveCodeBench/LiveCodeBench). |
| 04 | https://github.com/eth-sri/matharena.git | OK | GitHub - eth-sri/matharena: Evaluation of LLMs on latest math compe… | SUPPORTS | Repo oficial de MathArena (redirige a la vista web). |
| 04 | https://arxiv.org/html/2411.00640v1 | OK | Adding Error Bars to Evals: A Statistical Approach to Language Mode… | SUPPORTS | §4.2: SE pareado, potencia y MDE. |
| 04 | https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html | OK | statsmodels.stats.proportion.proportion_confint — statsmodels 0.15.0 | SUPPORTS | proportion_confint con método Wilson. |
| 04 | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html | OK | bootstrap — SciPy v1.18.0 Manual | SUPPORTS | Bootstrap pareado y BCa. |
| 04 | https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html | OK | statsmodels.stats.multitest.multipletests — statsmodels 0.15.0 | SUPPORTS | multipletests (Holm). |
| 04 | https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4 | OK | Function calling with Gemma 4 · Google AI for Developers | SUPPORTS | Formato nativo de tool calls de Gemma 4. |
| 04 | https://arxiv.org/abs/2411.00640 | OK | [2411.00640] Adding Error Bars to Evals: A Statistical Approach to … | SUPPORTS | Miller 2024, «Adding Error Bars to Evals». |
| 04 | https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/README.md | OK | raw · README.md (BFCL) | SUPPORTS | CLI y endpoint de BFCL. |
| 04 | https://docs.vllm.ai/en/latest/models/supported_models.html | OK | Supported Models - vLLM | SUPPORTS | Gemma4ForCausalLM y Gemma4ForConditionalGeneration (redirige a la ruta sin .html). |
| 04 | https://docs.vllm.ai/en/latest/features/lora.html | OK | LoRA Adapters - vLLM | SUPPORTS | --enable-lora y --max-lora-rank. |
| 05 | https://api.github.com/repos/sst/opencode | OK | JSON · GitHub repo (sst/opencode) | SUPPORTS | Redirige a repositories/975734319; full_name anomalyco/opencode, licencia MIT. |
| 05 | https://opencode.ai/docs/cli/#paths | OK | CLI · OpenCode | SUPPORTS | opencode debug paths: rutas de datos/config. |
| 05 | https://opencode.ai/docs/config/#precedence-order | OK | Config · OpenCode | SUPPORTS | Orden de precedencia de configuración. |
| 05 | https://opencode.ai/docs/providers/#custom-provider | OK | Providers · OpenCode | SUPPORTS | @ai-sdk/openai-compatible, baseURL y límites. |
| 05 | https://opencode.ai/config.json | OK | JSON · config schema (OpenCode) | SUPPORTS | Esquema v2: permissions/agents/plugins/providers/references; tool_call, limit, reasoning. |
| 05 | https://opencode.ai/docs/config/#models | OK | Config · OpenCode | SUPPORTS | Modelos por agente y modelo de título. |
| 05 | https://opencode.ai/docs/agents/ | OK | Agents · OpenCode | SUPPORTS | Definición de agentes. |
| 05 | https://opencode.ai/docs/config/#default-agent | OK | Config · OpenCode | SUPPORTS | default_agent. |
| 05 | https://opencode.ai/docs/permissions/ | OK | Permissions · OpenCode | SUPPORTS | allow/ask/deny, --auto, doom_loop. |
| 05 | https://opencode.ai/docs/cli/#run-1 | OK | CLI · OpenCode | SUPPORTS | --thinking, --session, --attach. |
| 05 | https://opencode.ai/docs/server/ | OK | Server · OpenCode | SUPPORTS | API HTTP, OpenAPI, SSE y /experimental/tool. |
| 05 | https://opencode.ai/docs/sdk/ | OK | SDK · OpenCode | SUPPORTS | SDK de OpenCode. |
| 05 | https://opencode.ai/docs/cli/#export | OK | CLI · OpenCode | SUPPORTS | opencode export <sessionID>. |
| 05 | https://opencode.ai/docs/cli/#environment-variables | OK | CLI · OpenCode | SUPPORTS | OPENCODE_DISABLE_*, OPENCODE_DB. |
| 05 | https://www.tbench.ai/docs | OK | Run | SUPPORTS | Terminal-Bench 4.0 sobre Harbor (redirige a /run). |
| 05 | https://docs.harborframework.com/agents/pre-integrated-agents.md | OK | md · pre-integrated-agents | SUPPORTS | Agentes preintegrados, opencode, ATIF y opciones. |
| 05 | https://opencode.ai/v2/docs/console/go/ | OK | Go · OpenCode | SUPPORTS | Go $10, Plus $40, Teams $20/miembro; 43 330/108 330/216 670 para V4.1 Flash. |
| 05 | https://opencode.ai/inference/go/openai/v1/chat/completions | HTTP 404 | endpoint · chat/completions (OpenCode Go) | SUPPORTS | GET sin auth devuelve 404; la ruta aparece literalmente en la doc de Go. |
| 05 | https://opencode.ai/inference/go/openai/v1 | HTTP 404 | endpoint · base /v1 (OpenCode Go) | SUPPORTS | Base URL confirmada en la doc de Go; GET sin auth 404. |
| 05 | https://opencode.ai/legal/terms-of-service | OK | OpenCode · Terms of Service | SUPPORTS | Vigente 15-ago-2026; prohíbe extracción automática de Output y entrenar modelos competidores. |
| 05 | https://opencode.ai/inference/go/openai/v1@4000 | HTTP 404 | arg · reverse proxy @4000 | COULD-NOT-CHECK | Sintaxis de mitmproxy reverse (@puerto), no es una URL navegable. |
| 05 | https://github.com/anomalyco/opencode | OK | GitHub - anomalyco/opencode: The open source coding agent. · GitHub | SUPPORTS | Repo real (antes sst/opencode), MIT. |
| 05 | https://opencode.ai/docs/config/ | OK | Config · OpenCode | SUPPORTS | Formato, ubicaciones, precedencia, compaction. |
| 05 | https://opencode.ai/docs/providers/ | OK | Providers · OpenCode | SUPPORTS | Proveedores y proveedor custom. |
| 05 | https://opencode.ai/docs/cli/ | OK | CLI · OpenCode | SUPPORTS | run/serve/export/import/session/db/debug. |
| 05 | https://raw.githubusercontent.com/harbor-framework/harbor/main/src/harbor/agents/installed/opencode.py | OK | raw · opencode.py (Harbor) | SUPPORTS | Integración real: instalación, config, comando y parser. |
| 05 | https://raw.githubusercontent.com/anomalyco/opencode/v1.19.3/packages/opencode/src/session/system.ts | OK | raw · system.ts (v1.19.3) | SUPPORTS | Selección de prompt por proveedor/modelo (PROMPT_*). |
| 05 | https://raw.githubusercontent.com/anomalyco/opencode/v1.19.3/packages/opencode/src/session/llm/request.ts | OK | raw · request.ts (v1.19.3) | SUPPORTS | Ensamblado del system, orden de tools, x-opencode-session-id. |
| 05 | https://raw.githubusercontent.com/anomalyco/opencode/v1.19.3/packages/opencode/src/session/prompt/default.txt | OK | raw · default.txt (v1.19.3) | SUPPORTS | Prompt base que ve el modelo. |
| 05 | https://raw.githubusercontent.com/anomalyco/opencode/v1.19.3/packages/opencode/src/tool/registry.ts | OK | raw · registry.ts (v1.19.3) | SUPPORTS | ids, apply_patch solo para gpt-* no-oss, websearch. |
| 05 | https://raw.githubusercontent.com/anomalyco/opencode/v1.19.3/packages/opencode/src/cli/cmd/run.ts | OK | raw · run.ts (v1.19.3) | SUPPORTS | Eventos JSON, --thinking, sin evento user. |
| 05 | https://docs.mitmproxy.org/stable/concepts/modes/ | OK | Proxy Modes | SUPPORTS | Modo reverse y captura local. |
| 06 | https://api-docs.deepseek.com/quick_start/pricing | OK | Models & Pricing · DeepSeek API Docs | SUPPORTS | deepseek-flash / V4.1-Flash-1017, 1M ctx, 384K out, precios off-peak y peak. |
| 06 | https://api-docs.deepseek.com/guides/vision | OK | Vision · DeepSeek API Docs | SUPPORTS | data URL, 48 MiB, 600 imágenes, detail low = 512×512. |
| 06 | https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html | OK | DeepSeek Terms of Use | SUPPORTS | Actualizado 27-mar-2026; §4.2(3) «training other models (such as model distillation)». |
| 06 | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/LICENSE | OK | raw · LICENSE (MIT) | SUPPORTS | MIT, «Copyright (c) 2023 DeepSeek». |
| 06 | https://arxiv.org/html/2306.05685v4 | OK | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | SUPPORTS | 85% vs 81% de acuerdo; 65.0% de consistencia al invertir el orden. |
| 06 | https://arena.ai/blog/llm-judge-self-preference | OK | Do LLM Judges Favor Themselves? - Arena.ai | PARTIAL | 58%/34% y 71%/68% confirmados en prosa; «−7.1 pp» solo aparece en los datos del gráfico. |
| 06 | https://arxiv.org/html/2506.06251v3 | OK | DesignBench: A Comprehensive Benchmark for MLLM-based Front-end Cod… | SUPPORTS | kappa 0.8648/0.8428; accuracy 95.54%/91.89%. |
| 06 | https://ui-bench.dev/ux-judge/methodology | OK | UI Bench · The Frontend Design Benchmark for LLMs | PARTIAL | 10 criterios, pesos, umbrales 82/65 y viewports confirmados; «80 diseños / 6 modelos / 10 variaciones» está en /ux-judge, no aquí. |
| 06 | https://playwright.dev/python/docs/screenshots | OK | Screenshots · Playwright Python | SUPPORTS | full_page, animations, caret, scale, screenshots de elemento. |
| 06 | https://playwright.dev/python/docs/network | OK | Network · Playwright Python | SUPPORTS | route, abort, offline. |
| 06 | https://api-docs.deepseek.com/api/create-chat-completion | OK | Chat Completions API · DeepSeek API Docs | SUPPORTS | Esquema, defaults y ausencia de seed. |
| 06 | https://api.deepseek.com | HTTP 401 | api.deepseek.com | SUPPORTS | Base URL OpenAI; HTTP 401 sin API key (host alcanzable). |
| 06 | https://api.deepseek.com/anthropic | HTTP 401 | anthropic | SUPPORTS | Base URL Anthropic; HTTP 401 sin API key. |
| 06 | https://api-docs.deepseek.com/guides/thinking_mode | OK | Thinking Mode · DeepSeek API Docs | SUPPORTS | Thinking on por defecto, effort high, reasoning_content. |
| 06 | https://api-docs.deepseek.com/guides/tool_calls | OK | Tool Calls · DeepSeek API Docs | SUPPORTS | Strict mode beta y tipos admitidos. |
| 06 | https://api-docs.deepseek.com/guides/json_mode | OK | JSON Output · DeepSeek API Docs | SUPPORTS | JSON Output y content vacío. |
| 06 | https://api.deepseek.com/beta | HTTP 401 | beta | SUPPORTS | Base URL beta para strict; HTTP 401 sin API key. |
| 06 | https://api-docs.deepseek.com/guides/kv_cache | OK | Context Caching · DeepSeek API Docs | SUPPORTS | Context caching y cache units. |
| 06 | https://arxiv.org/html/2403.03163v3 | OK | Design2Code: Benchmarking Multimodal Code Generation for Automated … | SUPPORTS | Design2Code: 484 páginas y métricas automáticas. |
| 06 | https://lmarena.ai/blog/webdev-arena/ | OK | WebDev Arena: A Live LLM Leaderboard for Web App Development - Aren… | SUPPORTS | WebDev Arena: ties, «both bad», structured output (redirige a arena.ai). |
| 06 | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html | OK | cohen_kappa_score — scikit-learn 1.9.1 documentation | SUPPORTS | Fórmula y API de cohen_kappa_score. |
| 06 | https://playwright.dev/python/docs/intro | OK | Installation · Playwright Python | SUPPORTS | Instalación, Python ≥3.8, Windows/WSL. |
| 06 | https://playwright.dev/python/docs/emulation | OK | Emulation · Playwright Python | SUPPORTS | Viewport, DPR y reduced_motion. |
| 06 | https://playwright.dev/python/docs/api/class-page | OK | Page · Playwright Python | SUPPORTS | page.screenshot: animations, caret, scale. |
| 06 | https://playwright.dev/python/docs/navigations | OK | Navigations · Playwright Python | SUPPORTS | goto espera load; hidratación y centinelas. |
| 06 | https://playwright.dev/python/docs/clock | OK | Clock · Playwright Python | SUPPORTS | set_fixed_time, install, pause_at, fast_forward. |
| 06 | https://api-docs.deepseek.com/quick_start/rate_limit | OK | Rate Limit & Isolation · DeepSeek API Docs | SUPPORTS | 2 500 concurrentes y user_id. |
| 06 | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash | OK | deepseek-ai/DeepSeek-V4.1-Flash · Hugging Face | SUPPORTS | Model card: MIT, fp8. |
| 06 | https://arxiv.org/abs/2306.05685 | OK | [2306.05685] Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | SUPPORTS | MT-Bench, sesgos y acuerdo. |
| 06, 07 | https://arxiv.org/abs/2403.03163 | OK | [2403.03163] Design2Code: Benchmarking Multimodal Code Generation f… | SUPPORTS | Design2Code: métricas y evaluación humana. |
| 06 | https://arxiv.org/abs/2506.06251 | OK | [2506.06251] DesignBench: A Comprehensive Benchmark for MLLM-based … | SUPPORTS | DesignBench: rúbrica 0–10 y kappa. |
| 06 | https://ui-bench.dev/ux-judge | OK | UI Bench · The Frontend Design Benchmark for LLMs | SUPPORTS | 80 landing pages de 6 modelos × 10 variaciones; enlaza a methodology. |
| 07 | https://ai.google.dev/gemma/docs/capabilities/function-calling | OK | Function calling with Gemma 4 · Google AI for Developers | SUPPORTS | Redirige a function-calling-gemma4: tokens, flujo completo y transformers>=5.10.1. |
| 07 | https://huggingface.co/google/gemma-4-E4B-it/raw/main/tokenizer_config.json | OK | raw · tokenizer_config.json | SUPPORTS | sot/eot/std/etd/stc/etc/str/etr/soc/eoc/escape/think exactos; response_schema y response_template. |
| 07 | https://huggingface.co/docs/trl/sft_trainer | OK | SFT Trainer · Hugging Face | SUPPORTS | assistant_only_loss, packing bfd, max_length 1024, truncation keep_start. |
| 07 | https://api.github.com/repos/huggingface/trl/contents/trl/chat_templates | OK | JSON · TRL chat_templates | SUPPORTS | gemma4.jinja + gemma4_v2..v10.jinja; no existe gemma4_training.jinja. |
| 07 | https://raw.githubusercontent.com/huggingface/trl/main/trl/chat_template_utils.py | OK | raw · chat_template_utils.py | SUPPORTS | Sin rama Gemma4 en get_training_chat_template → ValueError final (línea ~1470). |
| 07 | https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/sft_trainer.py | OK | raw · sft_trainer.py | SUPPORTS | Chequeo de {% generation %} y swap de template. |
| 07 | https://huggingface.co/docs/trl/dataset_formats | OK | Dataset formats and types · Hugging Face | SUPPORTS | messages + tools, on_mixed_types, datasets>=4.7. |
| 07 | https://huggingface.co/datasets/nvidia/OpenCodeReasoning | OK | nvidia/OpenCodeReasoning · Datasets at Hugging Face | SUPPORTS | CC BY 4.0; 735 255 = 567 850 (split_0) + 167 405 (split_1); sin campo de fecha. |
| 07 | https://huggingface.co/datasets/nvidia/OpenCodeInstruct/raw/main/README.md | OK | raw · README.md (OpenCodeInstruct) | SUPPORTS | CC BY 4.0, 5 000 000 filas; 6,86 GB de descarga (6,4 GiB) y 19,07 GB de dataset. |
| 07 | https://huggingface.co/datasets/nvidia/OpenMathReasoning/raw/main/README.md | OK | raw · README.md (OpenMathReasoning) | SUPPORTS | 306K problemas, 3,2M CoT, 1,7M TIR, 566K GenSelect, 193K sin solución; 49,6 GB de descarga. |
| 07 | https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M/raw/main/README.md | OK | raw · README.md (OpenThoughts3-1.2M) | SUPPORTS | Apache 2.0, 1,2M filas (850k math / 250k code / 100k science), trazas QwQ-32B. |
| 07 | https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/raw/main/README.md | OK | raw · README.md (OpenR1-Math-220k) | SUPPORTS | Apache 2.0; default 93 733 y all 225 129. |
| 07 | https://huggingface.co/datasets/AI-MO/NuminaMath-1.5/raw/main/README.md | OK | raw · README.md (NuminaMath-1.5) | SUPPORTS | Apache 2.0, 896 215 problemas; amc_aime = 5 872 (redirige a project-numina). |
| 07 | https://huggingface.co/datasets/HuggingFaceM4/WebSight/raw/main/README.md | OK | raw · README.md (WebSight) | SUPPORTS | CC BY 4.0; v0.2 = 1 922 671 y v0.1 = 822 987. |
| 07 | https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k/raw/main/README.md | OK | raw · README.md (self-oss-instruct-sc2) | SUPPORTS | ODC-By, 50 661 filas, response validada por ejecución. |
| 07 | https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/raw/main/README.md | OK | raw · README.md (Dory) | SUPPORTS | 8/20/8, loops 1/2/4, 256K, OpenMDW-1.1, SFT + GRPO, filtros agresivos. |
| 07 | https://livecodebench.github.io/ | OK | LiveCodeBench: Holistic and Contamination Free Evaluation of Large … | SUPPORTS | Evaluación por fecha de publicación; caída post-cutoff. |
| 07 | https://arxiv.org/abs/2504.01943 | OK | [2504.01943] OpenCodeReasoning: Advancing Data Distillation for Com… | SUPPORTS | 28 904 preguntas únicas, coseno ≥0.7, 3 épocas, lr 5e-5. |
| 07 | https://arxiv.org/abs/2506.04178 | OK | [2506.04178] OpenThoughts: Data Recipes for Reasoning Models | SUPPORTS | Indel 75% y N=13 con tokenizer Qwen2-7B-Instruct. |
| 07 | https://raw.githubusercontent.com/lm-sys/llm-decontaminator/main/README.md | OK | raw · README.md (llm-decontaminator) | SUPPORTS | 15.9% HumanEval y 12.8% CodeAlpaca en los subconjuntos medidos. |
| 07 | https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/index.html | OK | Deduplication · NeMo Curator | SUPPORTS | MD5, MinHash+LSH (char_ngrams=24, num_bands=20, minhashes_per_band=13) y semántica. |
| 07 | https://arxiv.org/abs/2412.21139 | OK | [2412.21139] Training Software Engineering Agents and Verifiers wit… | SUPPORTS | 2 438 instancias, 491 trayectorias, ~19k tokens, ctx 32768, +19%. |
| 07 | https://arxiv.org/abs/2504.21798 | OK | [2504.21798] SWE-smith: Scaling Data for Software Engineering Agents | SUPPORTS | 50k instancias, 128 repos, 1 003 + 349 trayectorias, truncado a 5 tool outputs. |
| 07 | https://arxiv.org/abs/2310.12823 | OK | [2310.12823] AgentTuning: Enabling Generalized Agent Abilities for … | SUPPORTS | 1 866 interacciones, η=0.2, lr 5e-5, batch 64. |
| 07 | https://arxiv.org/abs/2407.03502 | OK | [2407.03502] AgentInstruct: Toward Generative Teaching with Agentic… | SUPPORTS | 25M pares sintéticos; Mistral-7B → Orca-3. |
| 07 | https://arxiv.org/abs/2305.14314 | OK | [2305.14314] QLoRA: Efficient Finetuning of Quantized LLMs | SUPPORTS | NF4, double quantization, paged optimizers, r=64/α=16. |
| 07 | https://arxiv.org/abs/2405.09673 | OK | [2405.09673] LoRA Learns Less and Forgets Less | SUPPORTS | LoRA aprende menos y olvida menos que full FT. |
| 07 | https://huggingface.co/docs/trl/chat_templates | OK | Chat Templates · Hugging Face | SUPPORTS | Gemma4 está entre las familias reconocidas, pero no hay training template para ella. |
| 07 | https://arxiv.org/abs/2403.07974 | OK | [2403.07974] LiveCodeBench: Holistic and Contamination Free Evaluat… | SUPPORTS | LiveCodeBench: evaluación por fecha de publicación. |
| 07 | https://arxiv.org/abs/2406.01574 | OK | [2406.01574] MMLU-Pro: A More Robust and Challenging Multi-Task Lan… | SUPPORTS | MMLU-Pro: 10 opciones y sensibilidad al prompt. |
