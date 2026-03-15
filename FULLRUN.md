# TRELLIS2 — Treino Completo (10k objetos)

> **Objetivo:** Processar ~10 000 objetos do ObjaverseXL e treinar o Shape Flow model completo (1.3B params) para ver se o modelo consegue gerar shapes 3D plausíveis a partir de imagens.  
> **Hardware:** RTX 3090 (24 GB VRAM) via WSL2  
> **Ambiente conda:** `trellis2`

---

## Visão geral

```
[ObjaverseXL]
      │
      ▼ download ~10k objetos
      │
      ├─ dump_mesh   (Blender)  ─┐
      ├─ dump_pbr    (Blender)   ├─► dual_grid_512 ─► encode_shape_latents (GPU)
      └─ render_cond (Blender) ──┘                              │
                                                                ▼
                                                    Train Shape Flow (1.3B)
                                                    batch=2, bf16, 100k steps
```

---

## Scripts disponíveis

| Script | O que faz | Tempo estimado |
|---|---|---|
| `scripts/10_build_dataset_10k.sh` | Pipeline completo de dataset (todos os estágios) | 30–50 h total |
| `scripts/11_train_10k.sh` | Treino Shape Flow completo — 100k steps | 30–60 h |

Todos devem ser executados da **raiz do repositório**.

---

## Passo a passo

### 1. Preparar o dataset

```bash
bash scripts/10_build_dataset_10k.sh
```

O script executa automaticamente os 8 estágios na ordem correta e é **idempotente** — se interrompido, continua de onde parou ao ser re-executado.

**Estágios internos:**

| # | Estágio | Ferramenta | Tempo est. |
|---|---|---|---|
| 1 | Inicializar metadata | Python | segundos |
| 2 | Download ~9 900 objetos | objaverse.xl | 2–6 h |
| 3 | Mesh dump | Blender (8 workers) | 8–12 h |
| 4 | PBR dump | Blender (8 workers) | 8–12 h |
| 5 | O-Voxels 512 + asset stats | CPU puro | 2–4 h |
| 6 | Encode shape latents | GPU (RTX 3090) | 15–30 min |
| 7 | Render condições (16 views) | Blender (8 workers) | 15–25 h |
| 8 | Metadata final | Python | segundos |

> **Gargalo:** os estágios de Blender (3, 4, 7) são os mais lentos. Veja a seção [Paralelização](#paralelização) abaixo para acelerá-los.

---

### 2. Verificar o dataset antes de treinar

```bash
python -c "
import pandas as pd
df = pd.read_csv('datasets/ObjaverseXL_sketchfab/metadata.csv')
ready = ((df['shape_latent_encoded'] == True) & (df['cond_rendered'] == True)).sum()
print(f'Objetos prontos para treino: {ready}')
"
```

Recomendamos iniciar o treino quando houver pelo menos **5 000 objetos prontos** — não é necessário esperar o dataset inteiro. Veja a seção [Treino com dataset parcial](#treino-com-dataset-parcial).

---

### 3. Treinar o Shape Flow

```bash
bash scripts/11_train_10k.sh
```

Parâmetros principais do config (`configs/gen/slat_flow_img2shape_10k.json`):

| Parâmetro | Valor | Observação |
|---|---|---|
| Modelo | `ElasticSLatFlowModel` | 1.3B params — arquitetura completa |
| `max_steps` | 100 000 | ~30–60 h no RTX 3090 |
| `batch_size_per_gpu` | 2 | Ajustado para 24 GB VRAM |
| `batch_split` | 2 | Acumulação de gradiente (efetivo: batch 4) |
| `max_tokens` | 4 096 | Reduzido de 8192 para poupar VRAM |
| `mix_precision_dtype` | bfloat16 | Modo AMP |
| `i_save` / `i_sample` | 5 000 | Checkpoint + amostras a cada 5k steps |
| `min_aesthetic_score` | 0.0 | Sem filtro — usa todos os objetos baixados |

---

### 4. Monitorar o treino

**TensorBoard (em tempo real):**
```bash
conda activate trellis2
tensorboard --logdir results/slat_flow_shape_10k/tb_logs
```
Acesse `http://localhost:6006` no browser.

**Amostras visuais geradas durante o treino:**
```
results/slat_flow_shape_10k/
  samples/
    step0005000/   ← imagens de validação geradas a cada 5k steps
    step0010000/
    ...
  ckpts/           ← checkpoints (modelo + EMA + optimizer)
  tb_logs/         ← TensorBoard logs
  log.txt          ← histórico de loss em texto
```

**Verificar loss atual:**
```bash
tail -20 results/slat_flow_shape_10k/log.txt
```

---

### 5. Retomar após interrupção

Se o treino for interrompido, basta re-executar o mesmo script — ele **detecta automaticamente** o último checkpoint:
```bash
bash scripts/11_train_10k.sh
```

Para definir um número menor de steps (ex: testar com 20k):
```bash
MAX_STEPS=20000 bash scripts/11_train_10k.sh
```

---

## Paralelização

Os estágios de Blender (mesh, PBR, render) podem ser acelerados rodando múltiplas instâncias em paralelo usando `--rank` e `--world_size`.

**Exemplo: dividir o render_cond entre 4 terminais simultâneos:**

```bash
# Terminal 1
python data_toolkit/render_cond.py ObjaverseXL \
    --root datasets/ObjaverseXL_sketchfab --source sketchfab \
    --num_cond_views 16 --max_workers 4 --rank 0 --world_size 4

# Terminal 2
python data_toolkit/render_cond.py ObjaverseXL \
    --root datasets/ObjaverseXL_sketchfab --source sketchfab \
    --num_cond_views 16 --max_workers 4 --rank 1 --world_size 4

# Terminal 3
python data_toolkit/render_cond.py ObjaverseXL \
    --root datasets/ObjaverseXL_sketchfab --source sketchfab \
    --num_cond_views 16 --max_workers 4 --rank 2 --world_size 4

# Terminal 4
python data_toolkit/render_cond.py ObjaverseXL \
    --root datasets/ObjaverseXL_sketchfab --source sketchfab \
    --num_cond_views 16 --max_workers 4 --rank 3 --world_size 4
```

Isso divide os ~10k objetos em 4 partes processadas simultaneamente, reduzindo o tempo de ~20 h para ~5 h.

O mesmo padrão funciona para `dump_mesh.py` e `dump_pbr.py`.

Após terminar, rode o `build_metadata.py` para consolidar todos os `new_records`:
```bash
conda activate trellis2
python data_toolkit/build_metadata.py ObjaverseXL \
    --source sketchfab --root datasets/ObjaverseXL_sketchfab
```

---

## Treino com dataset parcial

Não é necessário esperar o dataset inteiro para começar o treino. Com ~5 000 objetos o modelo já começa a aprender.

Para iniciar o treino enquanto o dataset ainda está sendo construído:

1. Aguarde o estágio 7 (render_cond) atingir pelo menos 5 000 objetos
2. Inicie o treino em um segundo terminal:
   ```bash
   bash scripts/11_train_10k.sh
   ```
3. O treino usará os objetos disponíveis (cycling automático do dataloader)
4. Conforme mais objetos ficam prontos e o `build_metadata.py` é rodado, eles são incorporados **na próxima vez que o treino for retomado** a partir de um checkpoint

---

## O que esperar do treino

| Steps | Comportamento esperado |
|---|---|
| 0–5 000 | Loss alta, amostras aleatórias/ruidosas |
| 5 000–20 000 | Estruturas começa a emergir (shapes blobby/imprecisos) |
| 20 000–50 000 | Shapes reconhecíveis para categorias comuns (móveis, veículos) |
| 50 000–100 000 | Qualidade melhorando, mais fidelidade aos objetos de entrada |

> **Nota:** Com apenas 10k objetos, o modelo de 1.3B params pode apresentar algum nível de overfitting. Isso é esperado — o objetivo é validar o pipeline de ponta a ponta.

---

## OOM (Out of Memory) e ajustes

Se ocorrer erro de memória, edite `configs/gen/slat_flow_img2shape_10k.json`:

```json
"batch_size_per_gpu": 1,   // ← reduzir de 2 para 1
"max_tokens": 2048         // ← reduzir de 4096 para 2048
```

Outra opção: ativar o controlador de memória elástica (disponível para o `ElasticSLatFlowModel`) adicionando dentro de `trainer.args`:

```json
"elastic": {
    "name": "LinearMemoryController",
    "args": {
        "target_ratio": 0.75,
        "max_mem_ratio_start": 0.5
    }
}
```

---

## Diferenças em relação ao config de validação

| Parâmetro | Validação (mini) | 10k (completo) |
|---|---|---|
| `model_channels` | 256 | **1536** |
| `num_blocks` | 4 | **30** |
| `num_heads` | 4 | **12** |
| `max_steps` | 200 | **100 000** |
| `batch_size_per_gpu` | 2 | **2** |
| `max_tokens` | 8 192 | **4 096** |
| `i_save` / `i_sample` | 100 | **5 000** |
| Parâmetros totais | ~6M | **~1.3B** |

---

## Estrutura de arquivos

```
datasets/ObjaverseXL_sketchfab/
  metadata.csv
  raw/                                    ← arquivos originais
  mesh_dumps/                             ← geometria processada
  pbr_dumps/                              ← texturas PBR
  dual_grid_512/                          ← O-Voxels 512³
  shape_latents/
    shape_enc_next_dc_f16c32_fp16_512/    ← latentes de shape (~10k .npz)
  renders_cond/
    <sha256>/                             ← 16 renders por objeto
  asset_stats/

results/slat_flow_shape_10k/
  ckpts/                                  ← checkpoints a cada 5k steps
  samples/                                ← amostras visuais a cada 5k steps
  tb_logs/                                ← TensorBoard
  log.txt                                 ← histórico de loss
  config.json                             ← cópia do config usado
  command.txt                             ← comando exato usado
```
