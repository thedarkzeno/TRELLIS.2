from typing import *
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to, manual_cast, str_to_dtype
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules import sparse as sp
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
from .sparse_structure_flow import TimestepEmbedder
from .sparse_elastic_mixin import SparseTransformerElasticMixin
    

class SLatFlowModel(nn.Module):
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
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels)

        # Patch mixer projection layers + blocks (MicroDiT-style)
        if patch_mixer_blocks > 0:
            if self._mixer_needs_proj:
                self.mixer_proj_in = sp.SparseLinear(model_channels, self._mixer_dim)
                self.mixer_proj_out = sp.SparseLinear(self._mixer_dim, model_channels)
                # Project timestep embedding to mixer dim for mixer's adaLN
                self.mixer_t_proj = nn.Linear(model_channels, self._mixer_dim)
            _mixer_num_heads = max(1, self._mixer_dim // (model_channels // self.num_heads))
            self.mixer_blocks = nn.ModuleList([
                ModulatedSparseTransformerCrossBlock(
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
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
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
            for _ in range(num_blocks)
        ])
            
        self.out_layer = sp.SparseLinear(model_channels, out_channels)

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
                # proj_in/out operate on SparseTensor features (dtype); convert them
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
                    # proj_in: normal init is fine (not in residual path)
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
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, List[torch.Tensor]],
        concat_cond: Optional[sp.SparseTensor] = None,
        keep_indices: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> sp.SparseTensor:
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)

        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)
        # t_embedder runs in float32; adaLN_modulation also float32 — cast to dtype AFTER
        t_emb_base = self.t_embedder(t)     # [B, model_channels], float32
        cond = manual_cast(cond, self.dtype)

        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.dtype)

        # --- Phase 1: Patch Mixer (ALL tokens, reduced dim) ---
        if self.patch_mixer_blocks > 0:
            # Mixer timestep: project to mixer_dim if needed, then cast
            t_mixer_fp32 = self.mixer_t_proj(t_emb_base) if self._mixer_needs_proj else t_emb_base
            t_mixer = manual_cast(t_mixer_fp32, self.dtype)
            h_mixer = self.mixer_proj_in(h) if self._mixer_needs_proj else h
            for block in self.mixer_blocks:
                h_mixer = block(h_mixer, t_mixer, cond)

            # --- Phase 2: Mask (keep_indices provided by trainer during training) ---
            if keep_indices is not None:
                h_mixer = sp.SparseTensor(
                    h_mixer.feats[keep_indices],
                    h_mixer.coords[keep_indices]
                )
            # Project back to model_channels AFTER masking (fewer tokens → cheaper)
            h = self.mixer_proj_out(h_mixer) if self._mixer_needs_proj else h_mixer
        elif keep_indices is not None:
            # No mixer but masking requested (naive masking)
            h = sp.SparseTensor(h.feats[keep_indices], h.coords[keep_indices])

        # --- Phase 3: Main blocks (with optional layer reuse) ---
        # adaLN_modulation and iter_embeddings are in float32; cast result to dtype
        if self._has_layer_reuse:
            iter_idx = 0
            for block, repeats in zip(self.blocks, self.repeat_schedule):
                for _ in range(repeats):
                    # iter_embeddings is float32 param; add before adaLN, then cast
                    t_emb_i = t_emb_base + self.iter_embeddings[iter_idx]
                    if self.share_mod:
                        t_emb_i = self.adaLN_modulation(t_emb_i)
                    h = block(h, manual_cast(t_emb_i, self.dtype), cond)
                    iter_idx += 1
        else:
            if self.share_mod:
                t_emb = self.adaLN_modulation(t_emb_base)
            else:
                t_emb = t_emb_base
            t_emb = manual_cast(t_emb, self.dtype)
            for block in self.blocks:
                h = block(h, t_emb, cond)

        # --- Phase 4: Output (operates on whatever tokens are present) ---
        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h


class ElasticSLatFlowModel(SparseTransformerElasticMixin, SLatFlowModel):
    """
    SLat Flow Model with elastic memory management.
    Used for training with low VRAM.
    """
    pass
