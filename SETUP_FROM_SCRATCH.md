# TRELLIS2 — Setup Completo do Zero

Guia para replicar o ambiente de validação em uma nova máquina.

> **Hardware testado:** RTX 3090 (24 GB VRAM), WSL2 / Ubuntu 24.04  
> **Requisito mínimo de VRAM:** 24 GB  
> **CUDA Toolkit:** 12.x (testado com 12.6)

---

## Índice

1. [Pré-requisitos do sistema](#1-pré-requisitos-do-sistema)
2. [Clonar o repositório](#2-clonar-o-repositório)
3. [Instalar o ambiente Python](#3-instalar-o-ambiente-python)
4. [Configurar o HuggingFace](#4-configurar-o-huggingface)
5. [Preparar o dataset mínimo](#5-preparar-o-dataset-mínimo)
6. [Verificar o pipeline (dry-run)](#6-verificar-o-pipeline-dry-run)
7. [Rodar o treino de validação](#7-rodar-o-treino-de-validação)
8. [Referência: modificações feitas no repo](#8-referência-modificações-feitas-no-repo)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Pré-requisitos do sistema

### CUDA Toolkit

O CUDA Toolkit (não só o driver) precisa estar instalado para compilar os pacotes customizados.  
Recomendado: **12.4** ou superior. Para verificar:

```bash
nvcc --version
# Deve mostrar: Cuda compilation tools, release 12.x
```

Se `nvcc` não estiver disponível, instale o toolkit:

```bash
# Exemplo para CUDA 12.4 no Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-4
export CUDA_HOME=/usr/local/cuda-12.4
```

> **WSL2:** O driver NVIDIA fica no Windows. Apenas o CUDA Toolkit precisa ser instalado no WSL.

### Conda / Miniconda

```bash
# Instalar Miniconda se não tiver
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Reinicie o terminal após instalar
```

### Dependências do sistema (para Blender)

O `dump_mesh.py` e `dump_pbr.py` usam Blender headless. As bibliotecas necessárias são instaladas automaticamente pelos scripts, mas caso precise instalar manualmente:

```bash
sudo apt install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 libgl1
```

---

## 2. Clonar o repositório

```bash
git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
cd TRELLIS.2
```

> **Atenção:** o `--recursive` é necessário para inicializar o submódulo `o-voxel/third_party/eigen`.

Depois de clonar, adicione o módulo `datasets` ao data_toolkit (não está no repo original):

```bash
# O diretório data_toolkit/datasets/ deve existir com os scripts de dataset.
# Verifique:
ls data_toolkit/datasets/ObjaverseXL.py
```

Se o diretório não existir, obtenha-o a partir da branch/fork que contém as modificações deste guia, ou crie manualmente conforme a [Seção 8](#8-referência-modificações-feitas-no-repo).

---

## 3. Instalar o ambiente Python

### 3.1 Identificar a versão do CUDA instalada

```bash
export CUDA_HOME=/usr/local/cuda   # ajuste se tiver múltiplas versões
nvcc --version
```

### 3.2 Criar o ambiente e instalar dependências principais

Este passo compila vários pacotes CUDA — pode demorar **20–40 minutos**.

```bash
# Criar ambiente conda com Python 3.10 + PyTorch 2.6 (CUDA 12.4 wheels)
export CUDA_HOME=/usr/local/cuda
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
```

> `setup.sh` instala: PyTorch 2.6+cu124, flash-attn 2.7.3, nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, o-voxel.

Se preferir instalar passo a passo (útil para depurar falhas):

```bash
export CUDA_HOME=/usr/local/cuda
. ./setup.sh --new-env --basic       # ambiente + pacotes básicos
. ./setup.sh --flash-attn            # flash-attention (~10 min)
. ./setup.sh --nvdiffrast            # nvdiffrast (~2 min)
. ./setup.sh --nvdiffrec             # nvdiffrec renderer (~1 min)
. ./setup.sh --cumesh                # CuMesh CUDA (~5 min)
. ./setup.sh --flexgemm              # FlexGEMM Triton (~1 min)
. ./setup.sh --o-voxel               # O-Voxel (~3 min)
```

### 3.3 Instalar dependências do data_toolkit

```bash
conda activate trellis2
. ./data_toolkit/setup.sh
```

Isso instala: `open3d`, `objaverse`, `open_clip_torch`, `huggingface_hub`, entre outros.

### 3.4 Verificar a instalação

```bash
conda activate trellis2
python -c "
import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
import flash_attn; print('flash_attn:', flash_attn.__version__)
import nvdiffrast; print('nvdiffrast: ok')
import nvdiffrec_render; print('nvdiffrec_render: ok')
import cumesh; print('cumesh: ok')
import flexgemm; print('flexgemm: ok')
import o_voxel; print('o_voxel: ok')
print('Todos os pacotes OK')
"
```

Saída esperada:
```
torch: 2.6.0+cu124 | cuda: True
flash_attn: 2.7.3
nvdiffrast: ok
nvdiffrec_render: ok
cumesh: ok
flexgemm: ok
o_voxel: ok
Todos os pacotes OK
```

---

## 4. Configurar o HuggingFace

O `build_metadata.py` baixa a lista filtrada de assets do dataset **TRELLIS-500K** via HuggingFace. Este é um dataset público, mas o HuggingFace exige autenticação para acessar via `hf://`:

```bash
conda activate trellis2
huggingface-cli login
# Cole o token quando solicitado (disponível em https://huggingface.co/settings/tokens)
```

> O token pode ser de **read-only**. Não é necessário permissão de escrita.

---

## 5. Preparar o dataset mínimo

O script `01_build_dataset.sh` executa todo o pipeline de dados:

1. Baixa o índice de 168 307 assets do TRELLIS-500K (sketchfab)
2. Seleciona e baixa **~21 objetos** (`world_size=8000`)
3. Processa meshes via **Blender** (baixado automaticamente em `/tmp/`)
4. Processa texturas PBR via **Blender**
5. Converte para **O-Voxels** em resolução 256
6. Calcula **asset stats**

```bash
# Execute da raiz do repositório
conda activate trellis2
bash scripts/01_build_dataset.sh
```

**Tempo estimado:** 15–30 minutos (inclui download do Blender ~390 MB na primeira execução).

**O script é idempotente** — se interrompido, pode ser re-executado e pula etapas já concluídas.

### O que esperar no terminal

```
[1/6] Building initial metadata...
  → Baixa ObjaverseXL_sketchfab.csv do HuggingFace (~3 MB)

[2/6] Downloading objects (world_size=8000, rank=0)...
  → Downloading 21 new objects across 12 processes
  100%|██████████| 21/21 [00:04<00:00]

[4/6] Dumping meshes (Blender)...
  → Installs Blender 4.5.1 if needed (~390 MB, only once)
  → Dumping mesh: 100%|██████████| 21/21

[4b/6] Dumping PBR textures (Blender)...
  → Dumping PBR: ~19/21 (alguns modelos não têm PBR completo — normal)

[6/6] Converting to O-Voxels (resolution=256)...
  → Dual griding: 100%|██████████| 21/21

Dataset preparation COMPLETE
```

### Verificar o resultado

```bash
python data_toolkit/build_metadata.py ObjaverseXL \
    --source sketchfab \
    --root datasets/ObjaverseXL_sketchfab

# Saída esperada em statistics.txt:
#   - Number of assets downloaded: 21
#   - Number of assets with mesh dumped: 21
#   - Number of assets with dual grid:
#     - 256: 21
```

---

## 6. Verificar o pipeline (dry-run)

O dry-run confirma que o trainer carrega o dataset e os modelos sem erros, sem executar nenhum step de treino:

```bash
bash scripts/02_tryrun.sh
```

Saída esperada no final:
```
Trainer initialized.
ShapeVaeTrainer
  - Dataset: FlexiDualGridDataset
    - Total instances: 20
  - Number of steps: 200
  - Batch size per GPU: 4
  ...
Dry-run completed successfully — pipeline loads without errors.
```

> `Total instances: 20` (não 21) é normal — 1 objeto foi filtrado por exceder 1 M de faces.

---

## 7. Rodar o treino de validação

```bash
bash scripts/03_train_validation.sh
```

Ou pelo script master (encadeia tudo):

```bash
# Dataset já feito → pula etapa 1
bash run_validation.sh --skip-dataset

# Dataset + dry-run já feitos → só treino
bash run_validation.sh --skip-dataset --skip-tryrun
```

**Tempo estimado:** 10–30 minutos para 200 steps na RTX 3090.

### Saídas esperadas

```
results/shape_vae_validation/
  ckpts/
    misc_step100.pt          ← state dict completo
    ema_0.9999_step100.pt    ← EMA dos pesos
    misc_step200.pt
    ema_0.9999_step200.pt
  samples/
    step100/                 ← renders de validação (depth, normal, mask)
    step200/
  tb_logs/                   ← eventos do TensorBoard
  encoder_model_summary.txt
  decoder_model_summary.txt
```

Para acompanhar o TensorBoard em paralelo:
```bash
conda activate trellis2
tensorboard --logdir results/shape_vae_validation/tb_logs --port 6006
# Acesse http://localhost:6006
```

---

## 8. Referência: modificações feitas no repo

As seguintes modificações foram feitas em relação ao repositório original do TRELLIS2.

### 8.1 `data_toolkit/datasets/` — módulo ausente no repo original

O repo original não inclui o módulo `datasets` que os scripts de data_toolkit importam via `importlib.import_module('datasets.ObjaverseXL')`. Este diretório precisa existir em `data_toolkit/datasets/`.

Arquivos necessários:
- `data_toolkit/datasets/__init__.py` — vazio
- `data_toolkit/datasets/ObjaverseXL.py` — wrapper do `objaverse.xl`

O `ObjaverseXL.py` expõe as funções:

| Função | Descrição |
|---|---|
| `add_args(parser)` | Adiciona `--source sketchfab\|github` ao argparse |
| `get_metadata(source, **kwargs)` | Lê o CSV do TRELLIS-500K via HuggingFace |
| `download(metadata, *, download_root, ...)` | Baixa objetos via `objaverse.xl.download_objects` |
| `foreach_instance(metadata, output_dir, func, ..., no_file=False)` | Itera sobre instâncias chamando `func(file_path, metadatum_dict)` |

> **Nota sobre `no_file=True`:** usado por `dual_grid.py` e `voxelize_pbr.py`, que lêem diretamente dos `mesh_dumps/` — não precisam do arquivo original. Neste modo, `func(None, metadatum)` é chamado.

### 8.2 `configs/scvae/shape_vae_next_dc_f16c32_fp16_validation.json`

Config do SC-VAE Shape adaptado para validação rápida:

| Parâmetro | Original | Validação |
|---|---|---|
| `trainer.args.max_steps` | 1 000 000 | **200** |
| `trainer.args.batch_size_per_gpu` | 8 | **4** |
| `trainer.args.i_log` | 500 | **20** |
| `trainer.args.i_save` | 10 000 | **100** |
| `trainer.args.i_sample` | 10 000 | **100** |
| `dataset.args.min_aesthetic_score` | 4.5 | **0.0** |

`min_aesthetic_score: 0.0` é necessário para aceitar os ~21 objetos baixados (sem filtro de qualidade estética).

### 8.3 Scripts criados

| Arquivo | Função |
|---|---|
| `scripts/01_build_dataset.sh` | Pipeline completo de preparação de dados |
| `scripts/02_tryrun.sh` | Dry-run do trainer |
| `scripts/03_train_validation.sh` | Treino de 200 steps |
| `run_validation.sh` | Master script com flags `--skip-dataset` e `--skip-tryrun` |

---

## 9. Troubleshooting

### `ModuleNotFoundError: No module named 'datasets'`

Os scripts de data_toolkit importam `datasets.ObjaverseXL` — o módulo precisa estar em `data_toolkit/datasets/`. Confirme:

```bash
ls data_toolkit/datasets/ObjaverseXL.py   # deve existir
ls data_toolkit/datasets/__init__.py       # deve existir (pode ser vazio)
```

### `ModuleNotFoundError: No module named 'flash_attn'`

O flash-attn não foi instalado ou precisa ser compilado. Execute:

```bash
conda activate trellis2
pip install flash-attn==2.7.3 --no-build-isolation
```

> A compilação demora ~10 minutos. Requer `CUDA_HOME` apontando para o toolkit instalado.

### `CUDA out of memory`

Reduza o batch size no config de validação:

```bash
# Edite configs/scvae/shape_vae_next_dc_f16c32_fp16_validation.json
"batch_size_per_gpu": 2,
"batch_split": 1
```

Adicione também antes de rodar:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Blender não abre / falha silenciosa no `dump_mesh`

Verifique se o Blender está instalado em `/tmp/blender-4.5.1-linux-x64/`:

```bash
/tmp/blender-4.5.1-linux-x64/blender --version
```

Se não estiver, o script baixa automaticamente. Em caso de falha de download, baixe manualmente:

```bash
wget https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz -P /tmp
tar -xf /tmp/blender-4.5.1-linux-x64.tar.xz -C /tmp
```

### `HuggingFace authentication error` ao rodar `build_metadata.py`

```bash
conda activate trellis2
huggingface-cli login
```

### `error reading part_0.csv: No columns to parse`

Aviso benigno — ocorre quando uma etapa foi re-executada e não encontrou novos objetos para processar (CSV de records ficou vazio). Pode ser ignorado.

### `nvcc` não encontrado ao compilar extensões

```bash
which nvcc        # deve retornar /usr/local/cuda/bin/nvcc ou similar
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Adicione essas linhas ao `~/.bashrc` para persistir.
