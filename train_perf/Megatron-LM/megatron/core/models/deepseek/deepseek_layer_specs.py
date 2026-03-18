# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import warnings
from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp

from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.deepseek_transformer_layer import DeepseekTransformerLayer, DeepseekTransformerLayerSubmodules
from megatron.core.transformer.multi_latent_attention import MLASelfAttention, MLASelfAttentionSubmodules
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.utils import get_te_version, is_te_min_version
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

    warnings.warn('Apex is not installed. Falling back to Torch LayerNorm')
    LNImpl = WrappedTorchLayerNorm



def get_deepseekv2_layer_with_transformer_engine_spec(
    num_experts: int = None, 
    moe_grouped_gemm: bool = False, 
    qk_layernorm: bool = False, 
    fp8: Optional[str] = None,
    multi_latent_attention: bool = False
) -> ModuleSpec:
    mlp = get_mlp_module_spec(
        use_te=True,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        fp8=fp8,
    )

    mlp_dense = get_mlp_module_spec(
        use_te=True, num_experts=None, moe_grouped_gemm=moe_grouped_gemm, fp8=fp8
    )


    if multi_latent_attention:
        model = ModuleSpec(
            module=DeepseekTransformerLayer,
            submodules=DeepseekTransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=TEColumnParallelLinear,
                        linear_q_down_proj=TEColumnParallelLinear,
                        linear_q_up_proj=TEColumnParallelLinear,
                        linear_kv_down_proj=TEColumnParallelLinear,
                        linear_kv_up_proj=TEColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                        q_layernorm=TENorm if qk_layernorm else IdentityOp,
                        kv_layernorm=TENorm if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
                input_layernorm=TENorm if num_experts else IdentityOp,
                mlp=mlp,
                mlp_dense=mlp_dense,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    else:
        qk_norm = TENorm if is_te_min_version("1.9.0") else FusedLayerNorm

        model = ModuleSpec(
            module=DeepseekTransformerLayer,
            submodules=DeepseekTransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=TELayerNormColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                        q_layernorm=qk_norm if qk_layernorm else IdentityOp,
                        k_layernorm=qk_norm if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
                input_layernorm=TENorm if num_experts else IdentityOp,
                mlp=mlp,
                mlp_dense=mlp_dense,
                mlp_bda=get_bias_dropout_add,
            ),
        )
    return model

def get_deepseekv2_layer_local_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.

    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=LNImpl,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=LNImpl if qk_layernorm else IdentityOp,
                    k_layernorm=LNImpl if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=LNImpl,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


def _get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
):
    warnings.warn(
        """This private function is on a deprecation track. Please switch to `get_mlp_module_spec`
        since it will be removed in a future release."""
    )

    return get_mlp_module_spec(
        use_te=use_te,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        fp8=fp8,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )


def get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-arguments
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "_get_mlp_module_spec" has been deprecated'
            ' and will be removed soon. Please update your code accordingly.'
        )

    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        return get_moe_module_spec(
            use_te=use_te,
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        )
