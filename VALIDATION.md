# TRELLIS2 — Guia de Execução (Validação)

> **Objetivo:** Validar o pipeline end-to-end do **modelo de difusão de shape** (Shape Flow).  
> O VAE será carregado dos pesos pré-treinados do `TRELLIS.2-4B`; o que treinamos é o modelo de difusão.  
> **Hardware:** RTX 3090 (24 GB VRAM) via WSL2  
> **Ambiente conda:** `trellis2`

---

## Visão geral do pipeline

```
[mesh_dumps]  ──► dual_grid_512 ──► encode_shape_latents
[raw/]        ──► renders_cond
                                              │
                                              ▼
                                   train Shape Flow model
                                   (mini-model, 200 steps)
```

O pipeline de difusão tem 3 etapas:
1. **SS Flow** — gera a estrutura esparsa (sparse structure)
2. **Shape Flow** — gera os latentes de shape condicionados numa imagem  ← **este aqui**
3. **Texture Flow** — gera os latentes de textura (PBR)

Para validação usamos apenas o **Shape Flow**, que é o mais direto e não depende de textura.

---

## Estado atual

| Etapa | Status |
|---|---|
| Ambiente conda `trellis2` + todos os pacotes | ✅ Concluído |
| `datasets/ObjaverseXL_sketchfab` — 21 objetos baixados | ✅ Concluído |
| Mesh dumps (Blender) | ✅ 21/21 |
| PBR dumps (Blender) | ✅ 19/21 (2 falharam — normal) |
| O-Voxels `dual_grid_256` (para SC-VAE) | ✅ 21/21 |
| O-Voxels `dual_grid_512` (para Shape Flow) | ⏳ Passo 04 |
| Shape latents (`shape_enc_next_dc_f16c32_fp16_512`) | ⏳ Passo 04 |
| Renders de condição (`renders_cond/`) | ⏳ Passo 05 |
| Treino Shape Flow (200 steps) | ⏳ Passo 06 |

---

## Scripts disponíveis

Todos os scripts devem ser executados da **raiz do repositório**.

```
scripts/
  01_build_dataset.sh          # ✅ Baixa e processa o dataset (já feito)
  02_tryrun.sh                 # ✅ Dry-run do SC-VAE (já executado)
  03_train_validation.sh       # (SC-VAE — não é mais o objetivo)
  04_encode_shape_latents.sh   # Gera dual_grid_512 + encoda latentes de shape
  05_render_cond.sh            # Renderiza imagens multi-view via Blender
  06_train_shape_flow.sh       # Treino de validação do Shape Flow (200 steps)
```

---

## Próximos passos (executar em sequência)

### Passo 4 — Encoding de latentes de shape

```bash
bash scripts/04_encode_shape_latents.sh
```

O que faz:
- Gera `dual_grid_512/` a partir dos `mesh_dumps` existentes (CPU, ~5 min)
- Baixa o encoder pré-treinado `microsoft/TRELLIS.2-4B` do HuggingFace (~2 GB) — **requer login**
- Codifica cada objeto em latentes de shape (`shape_latents/shape_enc_next_dc_f16c32_fp16_512/`)
- Atualiza `metadata.csv`

Verificar resultado:
```bash
ls datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_512/*.npz | wc -l
# Esperado: ~21 arquivos
```

---

### Passo 5 — Render de condições (multi-view)

```bash
bash scripts/05_render_cond.sh
```

O que faz:
- Usa os arquivos `.blend` originais em `raw/` e Blender 3.0.1 (já instalado em `/tmp`)
- Renderiza 16 vistas por objeto → `renders_cond/<sha256>/`
- Atualiza `metadata.csv`

Verificar resultado:
```bash
ls datasets/ObjaverseXL_sketchfab/renders_cond/ | wc -l
# Esperado: ~21 pastas (uma por objeto)
```

---

### Passo 6 — Treino do Shape Flow

```bash
bash scripts/06_train_shape_flow.sh
```

O que faz:
- Carrega o modelo mini de difusão (256ch, 4 blocks — ~10M params, apenas para validação de pipeline)
- Baixa o decoder TRELLIS.2-4B (~3-4 GB) para sampling durante validação — **requer login**
- Baixa DINOv3 ViT-L/16 (~1.2 GB) para encoding de imagem de condição
- Treina por 200 steps
- Salva checkpoints a cada 100 steps em `results/slat_flow_shape_validation/`

---

## O que esperar durante o treino

```
results/slat_flow_shape_validation/
  ckpts/
    misc_step100.pt
    ema_0.9999_step100.pt
    misc_step200.pt          ← checkpoint final
    ema_0.9999_step200.pt
  samples/
    step100/                 ← amostras de validação visual
    step200/
  tb_logs/                   ← TensorBoard (opcional)
```

TensorBoard em tempo real:
```bash
conda activate trellis2
tensorboard --logdir results/slat_flow_shape_validation/tb_logs
```

---

## Solução de problemas comuns

### OOM (Out of Memory) na RTX 3090

O config já usa `batch_size_per_gpu: 2` e `batch_split: 1`. Se ainda ocorrer OOM, edite
`configs/gen/slat_flow_img2shape_validation.json` e reduza `max_tokens`:

```json
"max_tokens": 4096
```

### `shape_latent_encoded == True` não encontrado

O passo 4 ainda não foi executado, ou a metadata não foi atualizada. Re-execute:
```bash
bash scripts/04_encode_shape_latents.sh
```

### `cond_rendered == True` não encontrado

O passo 5 ainda não foi executado. Re-execute:
```bash
bash scripts/05_render_cond.sh
```

### HuggingFace authentication error

O encoder TRELLIS.2-4B e o decoder requerem login:
```bash
conda activate trellis2
huggingface-cli login
```

### Executar sempre da raiz do repositório

```bash
# CORRETO
bash scripts/06_train_shape_flow.sh

# ERRADO
cd scripts && bash 06_train_shape_flow.sh
```

---

## Referência rápida dos configs

### `configs/gen/slat_flow_img2shape_validation.json` vs original

| Parâmetro | Original (1.3B) | Validação (mini) |
|---|---|---|
| `models.denoiser.model_channels` | 1536 | **256** |
| `models.denoiser.num_blocks` | 30 | **4** |
| `models.denoiser.num_heads` | 12 | **4** |
| `trainer.args.max_steps` | 1 000 000 | **200** |
| `trainer.args.batch_size_per_gpu` | 8 | **2** |
| `trainer.args.batch_split` | 2 | **1** |
| `trainer.args.i_log` | 500 | **20** |
| `trainer.args.i_save` | 10 000 | **100** |
| `trainer.args.i_sample` | 10 000 | **100** |
| `dataset.args.min_aesthetic_score` | 4.5 | **0.0** |

> O modelo mini (~10M params) é apenas para validar o pipeline. O modelo de produção tem 1.3B params.

---

## Estrutura do dataset gerado

```
datasets/ObjaverseXL_sketchfab/
  metadata.csv                                  ← índice principal
  raw/                                          ← arquivos originais do Sketchfab
  mesh_dumps/                                   ← geometria processada (.pickle, Blender)
  pbr_dumps/                                    ← texturas PBR processadas (.pickle, Blender)
  dual_grid_256/                                ← O-Voxels 256³ (.vxz) — SC-VAE
  dual_grid_512/                                ← O-Voxels 512³ (.vxz) — Shape Flow
  shape_latents/
    shape_enc_next_dc_f16c32_fp16_512/          ← latentes de shape (.npz) — Shape Flow
  renders_cond/
    <sha256>/                                   ← imagens multi-view + transforms.json
  asset_stats/                                  ← estatísticas de cada asset
```
