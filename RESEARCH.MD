---
name: Micro-Budget 3D Generation Paper
overview: Plan a research paper demonstrating that competitive image-to-3D shape generation (TRELLIS2's Shape Flow) can be trained on a single consumer GPU (RTX 3090) through systematic optimizations to the training pipeline, inspired by MicroDiT and Sana's micro-budget approaches for 2D diffusion.
todos:
  - id: ablation-framework
    content: Create an experiment tracking framework (configs, logging, comparison scripts) for systematic ablations
    status: pending
  - id: token-masking
    content: Implement sparse token masking (naive + deferred) in the flow matching trainer
    status: pending
  - id: cond-caching
    content: Implement DINOv3 feature caching (offline pre-computation) to eliminate conditioning overhead
    status: pending
  - id: layer-reuse
    content: Implement weight-tied layer reuse with iteration embeddings in SLatFlowModel
    status: pending
  - id: arch-sweep
    content: Set up IsoFLOPs model size sweep configs (4/8/12/16/24 blocks, layer reuse variants)
    status: pending
  - id: training-recipe
    content: Ablate LR, weight decay, time schedule, batch size with gradient checkpointing
    status: pending
  - id: dataset-scaling
    content: Build 5k and 10k datasets; run dataset size scaling study
    status: pending
  - id: flow-dpm-solver
    content: Implement Flow-DPM-Solver for faster sampling during validation
    status: pending
  - id: evaluation
    content: "Build evaluation pipeline: Chamfer distance, IoU, rendered FID against GT meshes"
    status: pending
  - id: full-training
    content: Run full-scale training with best config; compare against TRELLIS2-4B pretrained
    status: pending
  - id: paper-writing
    content: "Write the paper: introduction, method, experiments, analysis"
    status: pending
isProject: false
---

# Micro-Budget 3D Generation: Training TRELLIS2 Shape Flow on a Single Consumer GPU

## Thesis

While state-of-the-art 3D generation models like TRELLIS2 were trained on large GPU clusters, we demonstrate that competitive image-to-3D shape generation can be achieved on a **single RTX 3090 (24 GB)** through a systematic combination of architectural, training, and data optimizations. This is the first "micro-budget" study for 3D diffusion transformers.

## Novelty Angle

MicroDiT (arxiv 2407.15811) and Sana (arxiv 2410.10629) proved micro-budget training is viable for **2D** diffusion transformers. 3D generation introduces unique challenges:

- **Sparse tensors** with variable token counts (vs. fixed 2D patch grids)
- **Heavier data pipeline** (Blender rendering, voxelization, shape encoding)
- **Conditioning on multi-view images** via a frozen DINOv3 backbone
- **3D-specific evaluation** (Chamfer distance, IoU, visual quality)

No prior work has systematically optimized 3D flow-matching training for micro-budgets.

---

## Current Baseline (what we already have)

- **Model:** ElasticSLatFlowModel, 1.3B params (30 blocks), or 187M (4 blocks)
- **Training:** 1k objects, ~30k steps, ~0.35 s/step (4-block), bfloat16 AMP, AdamW8bit
- **Hardware:** Single RTX 3090, 24 GB VRAM
- **Dataset:** 1,001 ObjaverseXL objects (shape latents + 8-view renders)

---

## Optimization Axes (ordered by expected impact)

### Axis 1: Sparse Token Masking (inspired by MicroDiT)

**Idea:** Randomly mask 50-75% of sparse 3D tokens during training, reducing the effective sequence length for the transformer.

**Key difference from MicroDiT:** Our tokens are already sparse (only non-empty voxels). We mask on top of sparsity. Two strategies:

- **Naive masking:** Drop random tokens before the transformer
- **Deferred masking:** Add a lightweight 2-3 block "token mixer" before masking (MicroDiT's key contribution), adapted for sparse tensors

**Implementation:** Modify `SparseFlowMatchingTrainer.training_losses()` in `[trellis2/trainers/flow_matching/sparse_flow_matching.py](trellis2/trainers/flow_matching/sparse_flow_matching.py)` to mask `x_t.feats`/`x_t.coords` before the denoiser forward pass. Loss is computed only on unmasked tokens.

**Expected gain:** 2-4x fewer FLOPs per step at 75% masking.

### Axis 2: Conditioning Model Efficiency

**Current:** DINOv3 ViT-L/16 (~304M params) runs a full forward pass per training step at 512x512.

**Options:**

- **Cache DINOv3 features offline:** Pre-compute and store per-view features (saves ~40% step time)
- **Replace with smaller encoder:** DINOv2 ViT-S/14 (~~22M) or ViT-B/14 (~~86M) as conditioning backbone
- **Lower resolution:** 224x224 instead of 512x512 (4x fewer patches)

**Implementation:** Modify `DinoV3FeatureExtractor` in `[trellis2/trainers/flow_matching/mixins/image_conditioned.py](trellis2/trainers/flow_matching/mixins/image_conditioned.py)` or add feature caching in the dataset class.

### Axis 3: Weight-Tied Layer Reuse (depth without parameters)

**Idea:** Repeat middle transformer blocks multiple times during the forward pass, sharing weights across repetitions. This gives deeper effective computation with the same parameter count and optimizer memory.

**Scheme (example with 4 unique blocks):**

```
Block 0 (1x) -> Block 1 (3x) -> Block 2 (3x) -> Block 3 (1x)  = 8 effective layers
```

Parameters: 4 blocks (~187M). Effective depth: 8 passes. Optimizer states: same as 4 blocks.

**Key design: Iteration Embeddings.** Add a small learned embedding per repetition index, injected via the existing AdaLN modulation mechanism. This lets the same weights behave differently at each repetition (~9K extra params per iteration) without breaking the weight sharing.

```python
# Instead of:
for block in self.blocks:
    h = block(h, t_emb, cond)

# With reuse + iteration embeddings:
for block, repeats in zip(self.blocks, self.repeat_schedule):
    for i in range(repeats):
        h = block(h, t_emb + self.iter_emb[block_idx][i], cond)
```

**Trade-offs:**

- VRAM for weights/optimizer: unchanged (same 4 blocks)
- VRAM for activations: scales with effective depth (mitigated by gradient checkpointing, already supported via `use_checkpoint` in each block)
- FLOPs per step: proportional to effective depth
- Quality: ~70-85% of a true N-block model with unique weights, but significantly better than the base depth

**Precedent:** ALBERT (Lan et al., 2019), Universal Transformer (Dehghani et al., 2018). No prior work has applied this to sparse 3D diffusion transformers.

**Ablations:**

- Repeat factor: 1x (baseline), 2x, 3x on middle blocks
- Which blocks to repeat (middle only vs. all)
- With vs. without iteration embeddings
- Combined with gradient checkpointing for activation memory control

**Implementation:** Modify `SLatFlowModel.forward()` in [`trellis2/models/structured_latent_flow.py`](trellis2/models/structured_latent_flow.py). Add `repeat_schedule` and `iter_embeddings` as config-driven parameters. The `repeat_schedule` is a list (e.g., `[1, 3, 3, 1]`) matching the number of unique blocks.

### Axis 4: Architecture Scaling Study

**Current:** 1.3B (30 blocks) vs. 187M (4 blocks) — big gap. Find the sweet spot.

**Study:** Train models under IsoFLOPs budget, comparing:

- Unique blocks: 4, 8, 12, 16, 24
- Layer reuse variants: 4 blocks x [1,2,2,1], 4 blocks x [1,3,3,1], 8 blocks x [1,2,2,2,2,2,2,1]
- Convergence speed, final quality, memory footprint

**Additional architectural changes:**

- **MoE layers** (expert-choice routing, from MicroDiT) in alternate blocks
- **Layer-wise scaling** (wider deep layers, narrower shallow ones)
- **Linear attention** (from Sana) adapted for sparse tensors

### Axis 4: Training Recipe Optimization

- **Learning rate:** Ablate higher LR (MicroDiT found 2x higher LR helps)
- **Weight decay:** Test 0.1 (MicroDiT's finding) vs current 0.01
- **Time schedule:** Compare `uniform` vs `logitNormal` vs shifted schedule
- **EMA schedule:** Delayed EMA (start later, save early compute)
- **Batch size:** Enable elastic gradient checkpointing to push batch_size_per_gpu from 2 to 4-8
- **Gradient accumulation:** Larger effective batch for smoother gradients

### Axis 5: Dataset Efficiency

- **Scale study:** 1k vs 5k vs 10k vs 50k objects — where does quality plateau?
- **Quality filtering:** Aesthetic score thresholds, complexity filtering
- **View count:** 4 vs 8 vs 16 conditioning views — what's sufficient?
- **Data pipeline cost:** Quantify Blender rendering time, propose parallelization strategies
- **Curriculum:** Start with simple objects, gradually increase complexity

### Axis 6: Inference/Sampling Efficiency

- **Flow-DPM-Solver** (from Sana): Adapt DPM-Solver++ for rectified flow, reducing sampling from 12 to 5-8 steps
- **CFG strength ablation:** Lower guidance may work with better training

---

## Proposed Experiments

### Phase 1: Ablations (1k dataset, 10-30k steps each)


| Experiment | Variable                                         | Metric                         |
| ---------- | ------------------------------------------------ | ------------------------------ |
| E1.1       | Token masking ratio (0%, 25%, 50%, 75%)          | Loss curve, Chamfer, visual    |
| E1.2       | Deferred vs naive masking                        | Same                           |
| E1.3       | DINOv3 caching vs online                         | Step time, same final quality? |
| E1.4       | Cond encoder size (ViT-S vs ViT-B vs ViT-L)     | Quality vs speed tradeoff      |
| E1.5       | Cond image resolution (224 vs 384 vs 512)        | Same                           |
| E1.6       | Layer reuse: repeat factor 1x/2x/3x on middle   | Quality vs FLOPs vs VRAM       |
| E1.7       | Layer reuse: with vs without iteration embedding | Quality delta                  |
| E1.8       | Model size sweep (4-24 blocks + reuse, IsoFLOPs) | Quality per FLOP               |
| E1.9       | LR / weight decay / time schedule                | Convergence speed              |
| E1.10      | Batch size with gradient checkpointing           | Throughput vs quality          |


### Phase 2: Scaled Training (10k+ dataset, 50-100k steps)

- Best config from Phase 1
- Train on 10k objects with optimized pipeline
- Compare against TRELLIS2 pretrained model (microsoft/TRELLIS.2-4B)
- Report total training cost ($, GPU-hours, wall-clock time)

### Phase 3 (stretch): Texture Flow

- Apply same optimizations to the texture flow model
- Full image-to-textured-3D pipeline on a single GPU

---

## Evaluation

- **3D Quality:** Chamfer Distance, IoU, F-Score against ground-truth meshes
- **Visual Quality:** FID on rendered views, user study
- **Cost Metrics:** GPU-hours, wall-clock time, estimated $ cost (at $0.80/h for 3090 spot pricing)
- **Comparisons:** TRELLIS2-4B pretrained, Point-E, Shap-E, other single-image-to-3D methods

---

## Paper Structure Draft

1. **Introduction:** Micro-budget training for 3D generation (motivation, contributions)
2. **Related Work:** TRELLIS2, MicroDiT, Sana, efficient training, image-to-3D methods
3. **Method:**
  - 3.1 Sparse token masking for 3D flow matching
  - 3.2 Weight-tied layer reuse with iteration embeddings
  - 3.3 Conditioning efficiency (caching, smaller encoder)
  - 3.4 Architecture optimizations (scaling study, MoE, linear attention)
  - 3.5 Training recipe
4. **Experiments:**
  - 4.1 Ablation studies (Phase 1)
  - 4.2 Scaled training and comparison (Phase 2)
  - 4.3 Cost analysis
5. **Conclusion and Limitations**

---

## Key Files to Modify

- **Token masking:** `[trellis2/trainers/flow_matching/sparse_flow_matching.py](trellis2/trainers/flow_matching/sparse_flow_matching.py)` (training_losses)
- **Conditioning:** `[trellis2/trainers/flow_matching/mixins/image_conditioned.py](trellis2/trainers/flow_matching/mixins/image_conditioned.py)`
- **Architecture:** `[trellis2/modules/sparse/transformer/modulated.py](trellis2/modules/sparse/transformer/modulated.py)`
- **Model def:** `[trellis2/models/structured_latent_flow.py](trellis2/models/structured_latent_flow.py)`
- **Trainer base:** `[trellis2/trainers/basic.py](trellis2/trainers/basic.py)`
- **Configs:** `[configs/gen/](configs/gen/)`
- **Sampler:** `[trellis2/pipelines/samplers/flow_euler.py](trellis2/pipelines/samplers/flow_euler.py)`

---

## Implementation Plan: Layer Reuse + Patch Mixer

### Overview

Two orthogonal features that compose well:

```
[Input] → [Input Layer] → [PE] → [Patch Mixer (all tokens)] → [MASK] → [Main Blocks w/ Reuse (masked tokens)] → [MERGE] → [Output Layer]
```

- **Layer Reuse**: repeat middle main blocks N times with iteration embeddings (deeper computation, same parameters)
- **Patch Mixer**: lightweight pre-transformer that sees ALL tokens before masking (MicroDiT-style deferred masking)
- During **inference**: no masking, mixer + main blocks run on all tokens (seamless)

---

### Feature 1: Weight-Tied Layer Reuse

#### 1.1 Data Flow

```
t_emb_base = TimestepEmbedder(t)                    # [B, model_channels]

# Current (no reuse):
t_emb = adaLN_modulation(t_emb_base)                # [B, 6 * model_channels]  (share_mod=True)
for block in blocks:
    h = block(h, t_emb, cond)

# With reuse (repeat_schedule = [1, 3, 3, 1]):
iter_idx = 0
for block, repeats in zip(blocks, repeat_schedule):
    for rep in range(repeats):
        t_emb_i = t_emb_base + iter_embeddings[iter_idx]   # [B, model_channels]
        if share_mod:
            t_emb_i = adaLN_modulation(t_emb_i)            # [B, 6 * model_channels]
        h = block(h, t_emb_i, cond)
        iter_idx += 1
```

The `adaLN_modulation` is applied per-iteration so each repetition gets a unique modulation signal. Cost is negligible (one small Linear per iteration).

#### 1.2 File Changes

**`trellis2/models/structured_latent_flow.py` — `SLatFlowModel`**

`__init__()` additions:
```python
def __init__(self, ..., repeat_schedule: Optional[List[int]] = None):
    # ...existing init...
    self.repeat_schedule = repeat_schedule or [1] * num_blocks
    assert len(self.repeat_schedule) == num_blocks
    self.effective_depth = sum(self.repeat_schedule)
    self._has_layer_reuse = self.effective_depth > self.num_blocks

    if self._has_layer_reuse:
        # One learned embedding per (block, iteration) pair; zero-init for identity start
        self.iter_embeddings = nn.Parameter(torch.zeros(self.effective_depth, model_channels))
```

`initialize_weights()` change:
```python
# In 'scaled' mode, use effective_depth instead of num_blocks:
std=1.0 / np.sqrt(5 * self.effective_depth * self.model_channels)
```
This accounts for the deeper effective residual stream.

`forward()` restructured:
```python
h = self.input_layer(x)
h = manual_cast(h, self.dtype)
t_emb_base = self.t_embedder(t)
t_emb_base = manual_cast(t_emb_base, self.dtype)
cond = manual_cast(cond, self.dtype)

if self.pe_mode == "ape":
    h = h + manual_cast(self.pos_embedder(h.coords[:, 1:]), self.dtype)

if self._has_layer_reuse:
    iter_idx = 0
    for block_idx, (block, repeats) in enumerate(zip(self.blocks, self.repeat_schedule)):
        for rep in range(repeats):
            t_emb_i = t_emb_base + self.iter_embeddings[iter_idx]
            if self.share_mod:
                t_emb_i = self.adaLN_modulation(t_emb_i)
            h = block(h, t_emb_i, cond)
            iter_idx += 1
else:
    # Original path (backward compatible)
    if self.share_mod:
        t_emb = self.adaLN_modulation(t_emb_base)
    else:
        t_emb = t_emb_base
    for block in self.blocks:
        h = block(h, t_emb, cond)
```

**`trellis2/models/sparse_elastic_mixin.py` — `SparseTransformerElasticMixin`**

`with_mem_ratio()` — checkpoint allocation uses effective depth:
```python
@contextmanager
def with_mem_ratio(self, mem_ratio=1.0):
    if mem_ratio == 1.0:
        yield 1.0
        return
    num_blocks = len(self.blocks)
    effective_depth = getattr(self, 'effective_depth', num_blocks)
    num_checkpoint_blocks = min(math.ceil((1 - mem_ratio) * num_blocks) + 1, num_blocks)
    exact_mem_ratio = 1 - (num_checkpoint_blocks - 1) / num_blocks
    for i in range(num_blocks):
        self.blocks[i].use_checkpoint = i < num_checkpoint_blocks
    yield exact_mem_ratio
    for i in range(num_blocks):
        self.blocks[i].use_checkpoint = False
```
Since checkpoint is per-block and all repetitions share the flag, this naturally applies to all iterations.

#### 1.3 Config

```json
{
    "models": {
        "denoiser": {
            "args": {
                "num_blocks": 4,
                "repeat_schedule": [1, 3, 3, 1]
            }
        }
    }
}
```

Ablation configs: `[1,1,1,1]` (baseline), `[1,2,2,1]` (6 effective), `[1,3,3,1]` (8 effective), `[1,4,4,1]` (10 effective).

#### 1.4 Parameter Budget

| Config | Unique Blocks | Effective Depth | Params | Iter Embeddings |
|--------|-------------|-----------------|--------|-----------------|
| [1,1,1,1] | 4 | 4 | 187M | 0 |
| [1,2,2,1] | 4 | 6 | 187M | +56K (6 × 1536 × 4B) |
| [1,3,3,1] | 4 | 8 | 187M | +74K (8 × 1536 × 4B) |
| [1,4,4,1] | 4 | 10 | 187M | +92K (10 × 1536 × 4B) |

Overhead: < 0.05% extra parameters.

#### 1.5 Compatibility

- **Existing checkpoints**: `repeat_schedule=None` → `[1]*N`, `iter_embeddings` absent → `strict=False` in `load_state_dict` (already used)
- **Inference**: transparent, same `forward()` signature
- **Sampler**: calls `model(noise, t, cond)` — no change needed
- **ElasticSLatFlowModel**: mixin calls `super().forward()` → automatically uses reuse logic

---

### Feature 2: Patch Mixer (Deferred Masking)

#### 2.1 Architecture

The patch mixer is a lightweight set of `ModulatedSparseTransformerCrossBlock` blocks placed BEFORE the main transformer. It processes ALL tokens at full resolution, providing contextualized representations before token masking reduces the sequence length for the heavier main blocks.

```
         ALL TOKENS                    MASKED TOKENS              ALL TOKENS
[Input Layer + PE] → [Mixer × M] → [random mask] → [Main Blocks × N] → [scatter back] → [Output Layer]
                     (2-3 blocks)    (keep 25-50%)   (4+ blocks w/ reuse)  (mixer fills gaps)
```

During **training with masking**:
1. All T tokens go through M mixer blocks (cheap, M=2-3)
2. Random subset of tokens is selected (e.g., 50%)
3. Selected tokens go through N main blocks (expensive, but ~2x faster due to masking)
4. Predictions scatter back; mixer output fills masked positions
5. Loss computed on ALL T tokens

During **inference** (no masking):
1. All tokens go through mixer + main blocks sequentially
2. Mixer acts as additional depth (beneficial, not wasted)

#### 2.2 Sparse Masking Utility

New function in `trellis2/modules/sparse/basic.py`:

```python
def sparse_random_mask(
    x: SparseTensor,
    mask_ratio: float,
    generator: Optional[torch.Generator] = None
) -> Tuple[SparseTensor, torch.LongTensor]:
    """
    Randomly keep (1 - mask_ratio) fraction of tokens per batch element.

    Returns:
        masked: SparseTensor with subset of tokens
        keep_indices: LongTensor of kept global indices (sorted)
    """
    keep_indices = []
    for s in x.layout:
        n = s.stop - s.start
        n_keep = max(1, round(n * (1 - mask_ratio)))
        perm = torch.randperm(n, device=x.device, generator=generator)[:n_keep]
        keep_indices.append(perm + s.start)
    keep_indices = torch.cat(keep_indices).sort()[0]
    return SparseTensor(x.feats[keep_indices], x.coords[keep_indices]), keep_indices


def sparse_scatter_back(
    full: SparseTensor,
    masked: SparseTensor,
    keep_indices: torch.LongTensor
) -> SparseTensor:
    """
    Scatter masked predictions back into full-resolution tensor.
    Positions NOT in keep_indices retain their original values (from mixer).
    """
    new_feats = full.feats.clone()
    new_feats[keep_indices] = masked.feats
    return full.replace(new_feats)
```

Key properties:
- Maintains sorted batch indices → valid SparseTensor layout
- At least 1 token kept per batch element → no empty batches
- Differentiable: indexing and scatter are autograd-compatible

#### 2.3 File Changes

**`trellis2/models/structured_latent_flow.py` — `SLatFlowModel`**

`__init__()` additions:
```python
def __init__(self, ..., patch_mixer_blocks: int = 0):
    # ...existing init...
    self.patch_mixer_blocks = patch_mixer_blocks
    if patch_mixer_blocks > 0:
        self.mixer_blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels, cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                rope_freq=rope_freq,
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(patch_mixer_blocks)
        ])
```

`forward()` additions:
```python
def forward(self, x, t, cond, ..., mask_indices=None):
    h = self.input_layer(x)
    # ...t_emb, PE, etc...

    # Phase 1: Mixer (all tokens)
    if self.patch_mixer_blocks > 0:
        for block in self.mixer_blocks:
            h = block(h, t_emb_mixer, cond)

    # Phase 2: Mask
    if mask_indices is not None:
        h_full = h  # save reference for scatter-back
        h = SparseTensor(h.feats[mask_indices], h.coords[mask_indices])

    # Phase 3: Main blocks (masked tokens only during training)
    # ...(existing block loop, with or without reuse)...

    # Phase 4: Scatter back
    if mask_indices is not None:
        h = sparse_scatter_back(h_full, h, mask_indices)

    h = manual_cast(h, x.dtype)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    h = self.out_layer(h)
    return h
```

**`trellis2/trainers/flow_matching/sparse_flow_matching.py` — `SparseFlowMatchingTrainer`**

`__init__()`: accept `mask_ratio: float = 0.0` from config

`training_losses()`:
```python
def training_losses(self, x_0, cond=None, **kwargs):
    noise = x_0.replace(torch.randn_like(x_0.feats))
    t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
    x_t = self.diffuse(x_0, t, noise=noise)
    cond = self.get_cond(cond, **kwargs)

    # Generate mask if enabled
    mask_indices = None
    if self.mask_ratio > 0 and self.training:
        _, mask_indices = sparse_random_mask(x_t, self.mask_ratio)

    pred = self.training_models['denoiser'](x_t, t * 1000, cond, mask_indices=mask_indices, **kwargs)

    target = self.get_v(x_0, noise, t)
    terms = edict()
    terms["mse"] = F.mse_loss(pred.feats, target.feats)  # loss on ALL tokens
    terms["loss"] = terms["mse"]
    # ...bin logging unchanged...
    return terms, {}
```

**`trellis2/modules/sparse/basic.py`**

Add `sparse_random_mask()` and `sparse_scatter_back()` utility functions.

#### 2.4 Config

```json
{
    "models": {
        "denoiser": {
            "args": {
                "patch_mixer_blocks": 2,
                "num_blocks": 4,
                "repeat_schedule": [1, 3, 3, 1]
            }
        }
    },
    "trainer": {
        "args": {
            "mask_ratio": 0.5
        }
    }
}
```

Ablation configs:
- `mask_ratio=0.0, patch_mixer_blocks=0` → baseline (no masking)
- `mask_ratio=0.5, patch_mixer_blocks=0` → naive masking (no mixer, loss only on kept tokens)
- `mask_ratio=0.5, patch_mixer_blocks=2` → deferred masking (mixer + mask, loss on all tokens)
- `mask_ratio=0.75, patch_mixer_blocks=2` → aggressive deferred masking

#### 2.5 FLOPs Analysis

For 4-block model with repeat_schedule=[1,3,3,1] (8 effective depth), 2 mixer blocks, 50% mask ratio:

| Component | Tokens | Effective Blocks | Relative FLOPs |
|-----------|--------|-----------------|----------------|
| Mixer | T (all) | 2 | 2T |
| Main blocks (no mask) | T | 8 | 8T |
| Main blocks (50% mask) | T/2 | 8 | 4T |
| **Total (no mask)** | | | **10T** |
| **Total (50% mask)** | | | **6T** (40% saving) |
| **Total (75% mask)** | | | **4T** (60% saving) |

The mixer's cost (2T) is paid once; the savings from masking (4T→2T at 50%) compound with the main block depth.

---

### Interaction Matrix

| | No Reuse | Reuse [1,3,3,1] |
|---|---|---|
| **No Mask** | baseline (4 passes, 4T FLOPs) | +depth, same params (8 passes, 8T FLOPs) |
| **Naive Mask 50%** | faster training (4 passes, 2T FLOPs) | best of both (8 passes, 4T FLOPs) |
| **Deferred Mask 50%** | mixer + main (2+4 passes, 4T FLOPs) | full combo (2+8 passes, 6T FLOPs) |

The "sweet spot" is likely **Reuse [1,3,3,1] + Naive Mask 50%**: 8 effective depth at the FLOPs cost of a 4-block model on half the tokens.

---

### Implementation Order

1. **Layer Reuse** (self-contained, no new dependencies)
   - Modify `SLatFlowModel.__init__` + `forward` + `initialize_weights`
   - Update `SparseTransformerElasticMixin.with_mem_ratio`
   - Create ablation config
   - Test: run 100 steps with [1,3,3,1] and verify loss decreases

2. **Sparse Masking Utilities**
   - Add `sparse_random_mask` + `sparse_scatter_back` to `basic.py`
   - Unit test: mask and scatter on synthetic SparseTensor

3. **Patch Mixer + Deferred Masking**
   - Add mixer blocks to `SLatFlowModel`
   - Add `mask_indices` support to `forward()`
   - Add `mask_ratio` to trainer
   - Test: run 100 steps with mask_ratio=0.5, mixer=2, verify loss

4. **Ablation Configs** — one JSON per experiment variant

