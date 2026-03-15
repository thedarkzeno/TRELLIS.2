from contextlib import contextmanager
from typing import *
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to, manual_cast, str_to_dtype
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from ..modules.attention import RotaryPositionEmbedder
from ..utils.elastic_utils import ElasticModuleMixin


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = 'float32',
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = 'vanilla',
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        # --- Layer Reuse ---
        repeat_schedule: Optional[List[int]] = None,
        # --- Patch Mixer ---
        patch_mixer_blocks: int = 0,
        patch_mixer_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = str_to_dtype(dtype)

        # --- Layer Reuse setup ---
        self.repeat_schedule = repeat_schedule or [1] * num_blocks
        assert len(self.repeat_schedule) == num_blocks, (
            f"repeat_schedule length {len(self.repeat_schedule)} must equal num_blocks {num_blocks}"
        )
        self.effective_depth = sum(self.repeat_schedule)
        self._has_layer_reuse = self.effective_depth > self.num_blocks
        if self._has_layer_reuse:
            # One embedding per effective layer step; zero-init → identity at start
            self.iter_embeddings = nn.Parameter(
                torch.zeros(self.effective_depth, model_channels)
            )

        # --- Patch Mixer setup ---
        self.patch_mixer_blocks = patch_mixer_blocks
        self._mixer_dim = patch_mixer_dim or model_channels
        self._mixer_needs_proj = (patch_mixer_blocks > 0) and (self._mixer_dim != model_channels)

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)
        elif pe_mode == "rope":
            pos_embedder = RotaryPositionEmbedder(self.model_channels // self.num_heads, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            rope_phases = pos_embedder(coords)
            self.register_buffer("rope_phases", rope_phases)

        if pe_mode != "rope":
            self.rope_phases = None

        self.input_layer = nn.Linear(in_channels, model_channels)

        # Patch mixer projection layers + blocks (MicroDiT-style deferred masking)
        if patch_mixer_blocks > 0:
            if self._mixer_needs_proj:
                self.mixer_proj_in = nn.Linear(model_channels, self._mixer_dim)
                self.mixer_proj_out = nn.Linear(self._mixer_dim, model_channels)
                # Project timestep embedding to mixer dim for mixer's adaLN
                self.mixer_t_proj = nn.Linear(model_channels, self._mixer_dim)
            _mixer_num_heads = max(1, self._mixer_dim // (model_channels // self.num_heads))
            self.mixer_blocks = nn.ModuleList([
                ModulatedTransformerCrossBlock(
                    self._mixer_dim,
                    cond_channels,
                    num_heads=_mixer_num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attn_mode='full',
                    use_checkpoint=self.use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    rope_freq=rope_freq,
                    share_mod=False,  # own adaLN; works at any dim
                    qk_rms_norm=self.qk_rms_norm,
                    qk_rms_norm_cross=self.qk_rms_norm_cross,
                )
                for _ in range(patch_mixer_blocks)
            ])

        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                rope_freq=rope_freq,
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels)

        self.initialize_weights()
        self.convert_to(self.dtype)

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to(self, dtype: torch.dtype) -> None:
        """
        Convert the torso of the model to the specified dtype.
        """
        self.dtype = dtype
        self.blocks.apply(partial(convert_module_to, dtype=dtype))
        if self.patch_mixer_blocks > 0:
            self.mixer_blocks.apply(partial(convert_module_to, dtype=dtype))
            if self._mixer_needs_proj:
                # proj_in/out operate on the hidden features; convert to dtype
                # mixer_t_proj takes float32 t_emb_base as input → leave it float32
                self.mixer_proj_in.apply(partial(convert_module_to, dtype=dtype))
                self.mixer_proj_out.apply(partial(convert_module_to, dtype=dtype))

    def initialize_weights(self) -> None:
        if self.initialization == 'vanilla':
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            if self.patch_mixer_blocks > 0:
                for block in self.mixer_blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

        elif self.initialization == 'scaled':
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=np.sqrt(2.0 / (5.0 * self.model_channels)))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            # Use effective_depth for scaled init — deeper effective residual stream
            # requires smaller per-layer contribution to maintain the same output variance
            def _scaled_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(
                        module.weight,
                        std=1.0 / np.sqrt(5 * self.effective_depth * self.model_channels)
                    )
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

            for block in self.blocks:
                block.self_attn.to_out.apply(_scaled_init)
                block.cross_attn.to_out.apply(_scaled_init)
                block.mlp.mlp[2].apply(_scaled_init)

            # Mixer output projections also get scaled init (they feed into main residual stream)
            if self.patch_mixer_blocks > 0:
                for block in self.mixer_blocks:
                    block.self_attn.to_out.apply(_scaled_init)
                    block.cross_attn.to_out.apply(_scaled_init)
                    block.mlp.mlp[2].apply(_scaled_init)
                if self._mixer_needs_proj:
                    # proj_out feeds into main stream; scaled init
                    _scaled_init(self.mixer_proj_out)
                    # proj_in and mixer_t_proj: normal init (not in residual path)
                    nn.init.normal_(
                        self.mixer_proj_in.weight,
                        std=np.sqrt(2.0 / (5.0 * self.model_channels))
                    )
                    nn.init.zeros_(self.mixer_proj_in.bias)
                    nn.init.normal_(
                        self.mixer_t_proj.weight,
                        std=np.sqrt(2.0 / (5.0 * self.model_channels))
                    )
                    nn.init.zeros_(self.mixer_t_proj.bias)

            nn.init.normal_(self.input_layer.weight, std=1.0 / np.sqrt(self.in_channels))
            nn.init.zeros_(self.input_layer.bias)

            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            if self.patch_mixer_blocks > 0:
                for block in self.mixer_blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        keep_indices: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        B = x.shape[0]

        # [B, C, R, R, R] → [B, R^3, C]
        h = x.view(B, self.in_channels, -1).permute(0, 2, 1).contiguous()

        h = self.input_layer(h)
        if self.pe_mode == "ape":
            h = h + self.pos_emb[None]

        # t_embedder runs in float32; cast to model dtype after
        t_emb_base = self.t_embedder(t)   # [B, model_channels], float32
        cond = manual_cast(cond, self.dtype)

        # rope phases subset: if masking, attend only over kept positions
        rp_full = self.rope_phases         # [R^3, head_dim] or None
        rp = rp_full[keep_indices] if (keep_indices is not None and rp_full is not None) else rp_full

        # --- Phase 1: Patch Mixer (ALL tokens) ---
        if self.patch_mixer_blocks > 0:
            t_mixer_fp32 = self.mixer_t_proj(t_emb_base) if self._mixer_needs_proj else t_emb_base
            t_mixer = manual_cast(t_mixer_fp32, self.dtype)
            h_mixer = self.mixer_proj_in(h) if self._mixer_needs_proj else h
            h_mixer = manual_cast(h_mixer, self.dtype)
            for block in self.mixer_blocks:
                h_mixer = block(h_mixer, t_mixer, cond, rp_full)

            # --- Phase 2: Mask (keep only subset for main blocks) ---
            if keep_indices is not None:
                h_mixer = h_mixer[:, keep_indices, :]
            h = self.mixer_proj_out(h_mixer) if self._mixer_needs_proj else h_mixer
        elif keep_indices is not None:
            # Naive masking: no mixer, just drop tokens
            h = manual_cast(h, self.dtype)[:, keep_indices, :]
        else:
            h = manual_cast(h, self.dtype)

        # --- Phase 3: Main blocks (with optional layer reuse) ---
        if self._has_layer_reuse:
            iter_idx = 0
            for block, repeats in zip(self.blocks, self.repeat_schedule):
                for _ in range(repeats):
                    # iter_embeddings is float32; add before adaLN, then cast
                    t_emb_i = t_emb_base + self.iter_embeddings[iter_idx]
                    if self.share_mod:
                        t_emb_i = self.adaLN_modulation(t_emb_i)
                    h = block(h, manual_cast(t_emb_i, self.dtype), cond, rp)
                    iter_idx += 1
        else:
            if self.share_mod:
                t_emb = self.adaLN_modulation(t_emb_base)
            else:
                t_emb = t_emb_base
            t_emb = manual_cast(t_emb, self.dtype)
            for block in self.blocks:
                h = block(h, t_emb, cond, rp)

        # --- Phase 4: Output ---
        h = manual_cast(h, x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        if keep_indices is not None:
            # Return [B, n_keep, out_channels]; trainer is responsible for aligning target
            return h

        # Reshape back to [B, out_channels, R, R, R]
        h = h.permute(0, 2, 1).view(B, self.out_channels, *[self.resolution] * 3).contiguous()
        return h


class DenseTransformerElasticMixin(ElasticModuleMixin):
    """
    Elastic memory management mixin for dense-tensor transformer models.
    Enables gradient checkpointing on a subset of blocks proportional to mem_ratio.
    """

    def _get_input_size(self, x: torch.Tensor, *args, **kwargs) -> int:
        return x.numel()

    @contextmanager
    def with_mem_ratio(self, mem_ratio: float = 1.0):
        if mem_ratio == 1.0:
            yield 1.0
            return
        num_blocks = len(self.blocks)
        effective_depth = getattr(self, 'effective_depth', num_blocks)
        num_checkpoint_blocks = min(
            math.ceil((1 - mem_ratio) * effective_depth) + 1, num_blocks
        )
        exact_mem_ratio = 1 - (num_checkpoint_blocks - 1) / effective_depth
        for i in range(num_blocks):
            self.blocks[i].use_checkpoint = i < num_checkpoint_blocks
        yield exact_mem_ratio
        for i in range(num_blocks):
            self.blocks[i].use_checkpoint = False


class ElasticSparseStructureFlowModel(DenseTransformerElasticMixin, SparseStructureFlowModel):
    """
    SparseStructureFlowModel with elastic memory management (adaptive gradient checkpointing).
    Used for training with low VRAM.
    """
    pass
