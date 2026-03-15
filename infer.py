#!/usr/bin/env python3
"""
Real-world inference: image(s) → 3D mesh, using a trained SLat flow checkpoint.

The script loads:
  1. Pretrained sparse-structure flow + decoder  (microsoft/TRELLIS.2-4B or local path)
  2. Our trained ElasticSLatFlowModel checkpoint  (from a result dir)
  3. Pretrained shape-SLat decoder               (referenced in the training config)

Then it runs the full image → voxel structure → shape SLat → mesh pipeline and saves:
  • <output>/<stem>/mesh.glb        – untextured GLB mesh
  • <output>/<stem>/preview.png     – 4-view normal-map render (same style as training snapshots)
  • <output>/<stem>/input.png       – preprocessed input image

Usage
-----
  python infer.py \\
    --images assets/chair.png assets/car.jpg \\
    --result_dir results/E_combined_reuse_mixer_mask50_1k \\
    [--ckpt_step 30000]  [--use_ema] \\
    [--pretrained microsoft/TRELLIS.2-4B] \\
    [--output_dir outputs/infer] \\
    [--steps 25] [--guidance 3.0] [--seed 42] \\
    [--no_rembg]
"""

import os
import sys
import json
import argparse
import glob as _glob

import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Make sure the repo root is on the Python path.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_step(ckpt_dir: str, use_ema: bool, ema_rate: float = 0.9999) -> int:
    """Return the highest step number that has a saved checkpoint."""
    prefix = f"denoiser_ema{ema_rate}" if use_ema else "denoiser"
    pattern = os.path.join(ckpt_dir, f"{prefix}_step*.pt")
    files = _glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint matching '{pattern}' found.")
    steps = []
    for f in files:
        stem = os.path.basename(f).replace(".pt", "")
        step_str = stem.split("step")[-1]
        steps.append(int(step_str))
    return max(steps)


def _ckpt_path(ckpt_dir: str, step: int, use_ema: bool, ema_rate: float = 0.9999) -> str:
    prefix = f"denoiser_ema{ema_rate}" if use_ema else "denoiser"
    return os.path.join(ckpt_dir, f"{prefix}_step{step:07d}.pt")


def _build_pipeline(pretrained_path: str, our_model, normalization: dict):
    """
    Monta a Trellis2ImageTo3DPipeline sem usar from_pretrained da subclasse
    (que crasharia ao tentar carregar o rembg com meta tensors).

    Carrega do pretrained APENAS:
      • sparse_structure_flow_model  (determina quais voxels existem)
      • sparse_structure_decoder     (binariza estrutura esparsa)
      • shape_slat_decoder           (SLat → mesh)

    Injeta NOSSO modelo treinado no slot shape_slat_flow_model_512.
    """
    import json
    from trellis2 import models as trellis_models
    from trellis2.pipelines import samplers
    from trellis2.pipelines import rembg as _rembg
    from trellis2.modules import image_feature_extractor
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    # Resolve pipeline.json (local ou HuggingFace)
    is_local = os.path.exists(os.path.join(pretrained_path, "pipeline.json"))
    if is_local:
        config_file = os.path.join(pretrained_path, "pipeline.json")
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(pretrained_path, "pipeline.json")

    with open(config_file) as f:
        pargs = json.load(f)["args"]

    # Carrega só os modelos pretrained que precisamos
    needed = {
        "sparse_structure_flow_model": pargs["models"]["sparse_structure_flow_model"],
        "sparse_structure_decoder":    pargs["models"]["sparse_structure_decoder"],
        "shape_slat_decoder":          pargs["models"]["shape_slat_decoder"],
    }
    loaded = {}
    for key, rel_path in needed.items():
        print(f"  carregando {key} …")
        try:
            loaded[key] = trellis_models.from_pretrained(f"{pretrained_path}/{rel_path}")
        except Exception:
            loaded[key] = trellis_models.from_pretrained(rel_path)

    # Injeta nosso modelo treinado (já com dtypes corretos do _load_model_from_config_and_ckpt:
    # t_embedder/adaLN_modulation = float32, blocks/input_layer/out_layer = bfloat16)
    # NÃO chamar .bfloat16() aqui — isso desfaz a restauração float32 dos módulos de condicionamento
    loaded["shape_slat_flow_model_512"] = our_model

    # Cria pipeline manualmente (evita from_pretrained que crasharia no rembg)
    pipeline = Trellis2ImageTo3DPipeline(loaded)
    pipeline._pretrained_args = pargs

    # Samplers
    ss = pargs["sparse_structure_sampler"]
    pipeline.sparse_structure_sampler        = getattr(samplers, ss["name"])(**ss["args"])
    pipeline.sparse_structure_sampler_params = ss["params"]

    sl = pargs["shape_slat_sampler"]
    pipeline.shape_slat_sampler        = getattr(samplers, sl["name"])(**sl["args"])
    pipeline.shape_slat_sampler_params = sl["params"]

    # Normalização do nosso treino (sobrescreve a do pretrained)
    pipeline.shape_slat_normalization = normalization
    pipeline.tex_slat_normalization   = pargs.get("tex_slat_normalization", {})

    # Image conditioning (DINOv3) — mesmo usado no treino
    ic = pargs["image_cond_model"]
    pipeline.image_cond_model = getattr(image_feature_extractor, ic["name"])(**ic["args"])

    # rembg — tenta carregar mas não falha se não conseguir
    try:
        rb = pargs["rembg_model"]
        pipeline.rembg_model = getattr(_rembg, rb["name"])(**rb["args"])
        print("  rembg (background removal) carregado.")
    except Exception as e:
        print(f"  [warn] rembg não carregou ({type(e).__name__}). Use --no_rembg para imagens sem fundo.")
        pipeline.rembg_model = None

    pipeline.low_vram            = pargs.get("low_vram", True)
    pipeline.default_pipeline_type = pargs.get("default_pipeline_type", "1024_cascade")
    pipeline.pbr_attr_layout     = {
        "base_color": slice(0, 3),
        "metallic":   slice(3, 4),
        "roughness":  slice(4, 5),
        "alpha":      slice(5, 6),
    }
    pipeline._device = "cpu"
    return pipeline


def _load_model_from_config_and_ckpt(config: dict, ckpt_path: str) -> torch.nn.Module:
    """
    Instancia ElasticSLatFlowModel e carrega o checkpoint.

    O checkpoint pode ter sido salvo com o modelo inteiro em bfloat16 (quando o
    trainer chamou model.bfloat16() antes de salvar). Porém, por design do forward():
      • t_embedder, adaLN_modulation, mixer_t_proj, iter_embeddings → float32
      • blocks, mixer_blocks, mixer_proj_in/out               → bfloat16 (via dtype)

    Após carregar o state_dict, chamamos convert_to(bfloat16) para setar self.dtype
    e então restauramos explicitamente as camadas que DEVEM ficar em float32.
    """
    from trellis2.models.structured_latent_flow import ElasticSLatFlowModel
    model_args = config["models"]["denoiser"]["args"]
    model = ElasticSLatFlowModel(**model_args)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # O checkpoint foi salvo com AMP (Automatic Mixed Precision), que mantém os
    # parâmetros em float32. Precisamos converter EXPLICITAMENTE para bfloat16
    # para que FlashAttention funcione (só aceita fp16/bf16).
    #
    # convert_to() converte blocks, mixer_blocks e proj layers, mas NÃO cobre
    # input_layer e out_layer — esses precisam ser convertidos manualmente.
    from functools import partial
    from trellis2.modules.utils import convert_module_to
    target_dtype = torch.bfloat16

    model.convert_to(target_dtype)  # seta self.dtype + converte blocks/mixer
    model.input_layer.apply(partial(convert_module_to, dtype=target_dtype))
    model.out_layer.apply(partial(convert_module_to, dtype=target_dtype))

    # t_embedder, adaLN_modulation, mixer_t_proj e iter_embeddings DEVEM ficar
    # em float32: recebem timesteps em float32 e o forward() os usa ANTES do
    # manual_cast para bfloat16.
    model.t_embedder.float()
    if hasattr(model, "adaLN_modulation"):
        model.adaLN_modulation.float()
    if hasattr(model, "mixer_t_proj") and model.mixer_t_proj is not None:
        model.mixer_t_proj.float()
    if hasattr(model, "iter_embeddings") and model.iter_embeddings is not None:
        model.iter_embeddings.data = model.iter_embeddings.data.float()

    model.eval()
    return model


def _render_preview(mesh, output_path: str, resolution: int = 512):
    """Render a 4-view normal-map grid of the mesh and save as PNG."""
    from trellis2.utils.render_utils import render_snapshot
    frames = render_snapshot(mesh, resolution=resolution, nviews=4)
    imgs = frames.get("normal", frames.get("color", []))
    if not imgs:
        return
    # Arrange as a 2×2 grid
    row1 = np.concatenate(imgs[:2], axis=1)
    row2 = np.concatenate(imgs[2:4], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    Image.fromarray(grid).save(output_path)


def _sample_shape_slat(pipeline, cond: dict, our_model, coords, normalization: dict,
                       steps: int, guidance: float) -> "SparseTensor":
    """
    Versão de sample_shape_slat que cria o noise já no dtype do modelo (bfloat16)
    e garante que cond está no dtype correto.

    Roda SEM torch.autocast para que manual_cast() funcione corretamente:
    - SparseTensor.type(dtype) existe e converte os feats internos
    - LayerNorm32 usa manual_cast para cast float32→bfloat16 no output
    """
    from trellis2.modules.sparse import SparseTensor as _SparseTensor

    # dtype dos blocos (bfloat16 se o modelo foi treinado em bfloat16)
    model_dtype = our_model.dtype   # setado por convert_to() no _load_model_from_config_and_ckpt

    noise = _SparseTensor(
        feats=torch.randn(coords.shape[0], our_model.in_channels,
                          dtype=model_dtype, device=pipeline.device),
        coords=coords,
    )

    # cond e neg_cond precisam estar no mesmo dtype dos blocos para o cross-attention
    cond_typed = {
        k: v.to(model_dtype) if isinstance(v, torch.Tensor) else v
        for k, v in cond.items()
    }

    sampler_params = {
        **pipeline.shape_slat_sampler_params,
        "steps": steps,
        "guidance_strength": guidance,
    }
    if pipeline.low_vram:
        our_model.to(pipeline.device)
    # Sem autocast: manual_cast() usa SparseTensor.type(dtype) que funciona
    # corretamente convertendo os feats internos. LayerNorm32 também funciona
    # corretamente: manual_cast cast para float32, computa, volta para bfloat16.
    slat = pipeline.shape_slat_sampler.sample(
        our_model,
        noise,
        **cond_typed,
        **sampler_params,
        verbose=True,
        tqdm_desc="Sampling shape SLat",
    ).samples
    if pipeline.low_vram:
        our_model.cpu()

    std  = torch.tensor(normalization["std"])[None].to(slat.device)
    mean = torch.tensor(normalization["mean"])[None].to(slat.device)
    return slat * std + mean


def _save_glb(mesh, output_path: str):
    """Export a Mesh object to GLB using trimesh."""
    import trimesh
    v = mesh.vertices.cpu().float().numpy()
    f = mesh.faces.cpu().numpy()
    # Convert from TRELLIS coordinate system to standard GLB (Y-up, Z-forward)
    # TRELLIS uses Z-up convention; swap Y/Z for GLB compatibility.
    v_out = v.copy()
    v_out[:, 1], v_out[:, 2] = v[:, 2].copy(), -v[:, 1].copy()
    tm = trimesh.Trimesh(vertices=v_out, faces=f, process=False)
    tm.export(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run real-world image-to-3D inference with a trained SLat flow model."
    )
    p.add_argument(
        "--images", nargs="+", required=True,
        help="Input image paths (PNG/JPG, any number). Glob patterns are accepted."
    )
    p.add_argument(
        "--result_dir", required=True,
        help="Path to a training result directory (must contain config.json and ckpts/)."
    )
    p.add_argument(
        "--ckpt_step", type=int, default=None,
        help="Step number of the checkpoint to load (default: latest)."
    )
    p.add_argument(
        "--use_ema", action="store_true", default=True,
        help="Load the EMA checkpoint (default: True)."
    )
    p.add_argument(
        "--no_ema", dest="use_ema", action="store_false",
        help="Load the raw (non-EMA) checkpoint instead."
    )
    p.add_argument(
        "--pretrained", default="microsoft/TRELLIS.2-4B",
        help="Pretrained pipeline path (local dir or HuggingFace repo id)."
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Where to save results (default: <result_dir>/infer/)."
    )
    p.add_argument(
        "--steps", type=int, default=25,
        help="Number of Euler sampler steps (default: 25)."
    )
    p.add_argument(
        "--guidance", type=float, default=3.0,
        help="Classifier-free guidance strength (default: 3.0)."
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )
    p.add_argument(
        "--no_rembg", action="store_true",
        help="Skip background removal (use if image already has transparent background)."
    )
    p.add_argument(
        "--render_resolution", type=int, default=512,
        help="Resolution for the preview render (default: 512)."
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(" TRELLIS2 — Inferência Real (imagem → mesh 3D)")
    print("=" * 60)
    print(f"  Modelo treinado : {args.result_dir}")
    print(f"  EMA             : {args.use_ema}")
    print(f"  Pretrained (HF) : {args.pretrained}")
    print(f"    ↳ sparse structure flow + decoders  (NÃO o shape SLat flow)")
    print(f"  Steps / Guidance: {args.steps} / {args.guidance}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Expand image globs
    # ------------------------------------------------------------------
    image_paths = []
    for pattern in args.images:
        expanded = _glob.glob(pattern)
        if expanded:
            image_paths.extend(expanded)
        else:
            image_paths.append(pattern)  # will fail at open time with a clear error
    image_paths = sorted(set(image_paths))
    print(f"[infer] {len(image_paths)} image(s) to process.")

    # ------------------------------------------------------------------
    # 2. Load training config
    # ------------------------------------------------------------------
    config_path = os.path.join(args.result_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {args.result_dir}")
    with open(config_path) as f:
        config = json.load(f)

    normalization = config["dataset"]["args"]["normalization"]

    # ------------------------------------------------------------------
    # 3. Resolve checkpoint
    # ------------------------------------------------------------------
    ckpt_dir = os.path.join(args.result_dir, "ckpts")
    step = args.ckpt_step or _find_latest_step(ckpt_dir, args.use_ema)
    ckpt = _ckpt_path(ckpt_dir, step, args.use_ema)
    if not os.path.exists(ckpt):
        # Fallback: try without EMA
        fallback = _ckpt_path(ckpt_dir, step, use_ema=False)
        if os.path.exists(fallback):
            print(f"  [warn] EMA ckpt not found, falling back to raw: {os.path.basename(fallback)}")
            ckpt = fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    ema_label = "EMA" if args.use_ema else "raw"
    print(f"[infer] Loading {ema_label} checkpoint: step={step}  ({os.path.basename(ckpt)})")

    # ------------------------------------------------------------------
    # 4. Instancia e carrega NOSSO modelo treinado
    # ------------------------------------------------------------------
    print("[infer] Carregando nosso ElasticSLatFlowModel treinado …")
    our_model = _load_model_from_config_and_ckpt(config, ckpt)
    n_params = sum(p.numel() for p in our_model.parameters()) / 1e6
    print(f"         → {n_params:.1f}M parâmetros | checkpoint: {os.path.basename(ckpt)}")

    # ------------------------------------------------------------------
    # 5. Monta pipeline: pretrained (estrutura + decoders) + nosso modelo
    # ------------------------------------------------------------------
    print(f"[infer] Carregando componentes pretrained de: {args.pretrained}")
    print(f"        sparse_structure_flow, sparse_structure_decoder, shape_slat_decoder")
    print(f"        shape_slat_flow_model_512 → nosso checkpoint (step {step})")
    pipeline = _build_pipeline(args.pretrained, our_model, normalization)
    print(f"[infer] ✓ shape_slat_flow_model_512 = nosso checkpoint (step {step})")

    pipeline.to("cuda")
    print("[infer] Pipeline pronta.")

    # ------------------------------------------------------------------
    # 6. Output directory
    # ------------------------------------------------------------------
    output_dir = args.output_dir or os.path.join(args.result_dir, "infer")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 7. Inference loop
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)

    for img_path in image_paths:
        stem = Path(img_path).stem
        out_subdir = os.path.join(output_dir, stem)
        os.makedirs(out_subdir, exist_ok=True)

        print(f"\n[infer] Processing: {img_path}")
        image = Image.open(img_path).convert("RGBA" if not args.no_rembg else "RGB")

        # Preprocess (background removal + crop)
        if not args.no_rembg:
            image = pipeline.preprocess_image(image)
            image.save(os.path.join(out_subdir, "input.png"))
        else:
            image = image.convert("RGB")
            image.save(os.path.join(out_subdir, "input.png"))

        # --- Step A: image conditioning ---
        print("  [1/3] Extracting image features (DINOv3) …")
        cond = pipeline.get_cond([image], resolution=512)

        # --- Step B: sample sparse structure ---
        print("  [2/3] Sampling sparse voxel structure …")
        coords = pipeline.sample_sparse_structure(
            cond, resolution=32, num_samples=1,
            sampler_params={"steps": args.steps, "guidance_strength": args.guidance},
        )
        print(f"         → {coords.shape[0]} occupied voxels")

        # --- Step C: sample shape SLat com our model ---
        print("  [3/3] Sampling shape SLat (our trained model) …")
        shape_slat = _sample_shape_slat(
            pipeline, cond,
            our_model=pipeline.models["shape_slat_flow_model_512"],
            coords=coords,
            normalization=normalization,
            steps=args.steps,
            guidance=args.guidance,
        )

        # --- Decode ---
        print("  Decoding SLat → mesh …")
        meshes, _ = pipeline.decode_shape_slat(shape_slat, resolution=512)
        mesh = meshes[0]
        mesh.fill_holes()

        # --- Save ---
        glb_path = os.path.join(out_subdir, "mesh.glb")
        _save_glb(mesh, glb_path)
        print(f"  Saved GLB: {glb_path}")

        preview_path = os.path.join(out_subdir, "preview.png")
        _render_preview(mesh, preview_path, resolution=args.render_resolution)
        print(f"  Saved preview: {preview_path}")

        torch.cuda.empty_cache()

    print(f"\n[infer] Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
