import torch
from functools import partial

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf_provider_base_ops.llm_ops.moe_gather import MoeGatherOp
from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size
import itertools


def count_expert_num_tokens_with_map(topk_ids, num_local_experts, expert_map):

    flat_ids = topk_ids.view(-1)
    local_ids = expert_map[flat_ids]
    mask = local_ids >= 0
    return torch.bincount(local_ids[mask], minlength=num_local_experts).to(torch.int32)

def get_expert_offsets(expert_counts, alignment):
    padded_counts = (expert_counts + alignment - 1) // alignment * alignment
    offsets = torch.zeros_like(expert_counts)
    offsets[1:] = torch.cumsum(padded_counts[:-1], dim=0)
    return offsets, padded_counts.sum().item()


def get_moe_tokens_info(
    num_tokens, num_experts, topk,
    ep_size=1, ep_rank=0
):
    # split tokens / experts
    num_scatter_tokens = num_tokens * topk
    num_scatter_tokens_per_rank = num_scatter_tokens // ep_size
    num_experts_per_rank = num_experts // ep_size

    experts_start_idx = ep_rank * num_experts_per_rank
    experts_end_idx = experts_start_idx + num_experts_per_rank

    expert_map = torch.full((num_experts,), -1, dtype=torch.int32)

    local_expert_ids = torch.arange(num_experts_per_rank, dtype=torch.int32)

    expert_map[experts_start_idx:experts_end_idx] = local_expert_ids

    experts_idx_for_each_rank = []
    for rank_idx in range(ep_size):
        start_idx = rank_idx * num_experts_per_rank
        end_idx = start_idx + num_experts_per_rank
        experts_idx_for_each_rank.append(list(range(start_idx, end_idx)))
    transpose_experts = [list(row) for row in zip(*experts_idx_for_each_rank)]
    experts_array = [num for row in transpose_experts for num in row]

    all_select_experts = []
    all_select_weights = []

    cur_expert = 0
    for token_idx in range(num_tokens):
        cur_token_selections = []
        for topk_idx in range(topk):
            cur_token_selections.append(experts_array[cur_expert])
            cur_expert += 1
            if cur_expert >= num_experts:
                cur_expert = 0
        all_select_experts.append(cur_token_selections)
        all_select_weights.append([1 / topk for _ in range(topk)])

    all_select_experts_tensor = torch.tensor(all_select_experts, dtype=torch.long)
    expert_counts = count_expert_num_tokens_with_map(all_select_experts_tensor, num_experts_per_rank, expert_map)
    expert_offsets, total_rows = get_expert_offsets(expert_counts, 16)

    cur_rank_tokens = {}
    cur_rank_weights = {}
    dispatch_tokens = 0

    for token_idx in range(num_tokens):
        cur_token_dispatch_experts = []
        cur_token_dispatch_weights = []
        for expert_idx, expert_weight in zip(all_select_experts[token_idx], all_select_weights[token_idx]):
            if expert_idx >= experts_start_idx and expert_idx < experts_end_idx:
                cur_token_dispatch_experts.append(expert_idx)
                cur_token_dispatch_weights.append(expert_weight)

        if cur_token_dispatch_experts:
            cur_rank_tokens[token_idx] = cur_token_dispatch_experts
            cur_rank_weights[token_idx] = cur_token_dispatch_weights
            dispatch_tokens += len(cur_token_dispatch_experts)

    used_src_tokens = len(cur_rank_tokens)

    expert_dispatch_tokens = [[] for _ in range(experts_start_idx, experts_end_idx)]
    expert_dispatch_weights = [[] for _ in range(experts_start_idx, experts_end_idx)]
    expert_dispatch_token_count = [0 for _ in range(experts_start_idx, experts_end_idx)]
    expert_dispatch_token_offset = [0 for _ in range(experts_start_idx, experts_end_idx)]

    for token_idx in cur_rank_tokens:
        for topk_idx, expert_idx in enumerate(cur_rank_tokens[token_idx]):
            expert_dispatch_tokens[expert_idx - experts_start_idx].append(token_idx)
            expert_dispatch_weights[expert_idx - experts_start_idx].append(cur_rank_weights[token_idx][topk_idx])
            expert_dispatch_token_count[expert_idx - experts_start_idx] += 1
    expert_dispatch_token_offset = ([0] + list(itertools.accumulate(expert_dispatch_token_count)))[:num_experts_per_rank]

    expert_dispatch_tokens_flatten = [token for tokens in expert_dispatch_tokens for token in tokens]
    expert_dispatch_weights_flatten = [weight for weights in expert_dispatch_weights for weight in weights]

    return (
        num_scatter_tokens,
        num_scatter_tokens_per_rank,
        num_experts_per_rank,
        experts_start_idx,
        experts_end_idx,
        all_select_experts,
        all_select_weights,
        dispatch_tokens,
        used_src_tokens,
        expert_dispatch_tokens,
        expert_dispatch_weights,
        expert_dispatch_tokens_flatten,
        expert_dispatch_weights_flatten,
        expert_dispatch_token_count,
        expert_dispatch_token_offset,
        expert_map,
        expert_offsets,
        total_rows,
        all_select_experts_tensor
    )


def compute_inv_perm_ep_gather(
    num_tokens: int,
    topk: int,
    experts_start_idx: int,
    expert_dispatch_tokens: list,
    expert_offsets: torch.Tensor,
    all_select_experts: list,
) -> torch.Tensor:
    """
    Build inv_perm [num_tokens, topk] for lightop ep_gather: row index in the padded expert buffer
    `scatter_tokens_lightop` for each (token, k), or -1 if not routed to this EP rank.

    Layout matches expert-major scatter: expert e uses rows
    [expert_offsets[e], expert_offsets[e] + count_e) inside [0, total_rows).
    """
    inv = torch.full((num_tokens, topk), -1, dtype=torch.int32)
    for e, tok_list in enumerate(expert_dispatch_tokens):
        global_expert = experts_start_idx + e
        base = int(expert_offsets[e].item())
        for j, tok in enumerate(tok_list):
            row = base + j
            for k in range(topk):
                if all_select_experts[tok][k] == global_expert:
                    inv[tok, k] = row
                    break
    return inv


def fill_scatter_tokens_lightop_from_dense(
    scatter_tokens_lightop: torch.Tensor,
    scatter_tokens: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_dispatch_token_count: list,
    num_experts_per_rank: int,
) -> None:
    """Place dense [dispatch_tokens, H] expert outputs into padded [total_rows, H] buffer (expert-major)."""
    scatter_tokens_lightop.zero_()
    off = 0
    for e in range(num_experts_per_rank):
        cnt = expert_dispatch_token_count[e]
        if cnt == 0:
            continue
        base = int(expert_offsets[e].item())
        scatter_tokens_lightop[base : base + cnt].copy_(scatter_tokens[off : off + cnt])
        off += cnt


try:
    from lightop import op

    @ProviderRegistry.register_vendor_impl("moe_gather", "lightop")
    class LightopMoeGatherOp(MoeGatherOp):
        """
        MoE gather via lightop `op.ep_gather`, aligned with the same dispatch metadata as base `MoeGatherOp`:
        - `scatter_tokens`: dense expert outputs [dispatch_tokens, H] (expert-major flatten order).
        - `scatter_tokens_lightop`: padded buffer [total_rows, H] filled from `scatter_tokens` before the kernel.
        - `inv_perm_lightop`: per (token, k) source row in `scatter_tokens_lightop` (not random).
        - `selected_experts` / `moe_weights` / `expert_map`: arguments to `ep_gather` (global expert ids + map).
        """

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["lightop"]

        def prepare_args(self):
            # Keep base op semantic args parsing, then add lightop-specific dispatch metadata.
            super().prepare_args()

            (
                self.num_scatter_tokens,
                self.num_scatter_tokens_per_rank,
                self.num_experts_per_rank,
                self.experts_start_idx,
                self.experts_end_idx,
                self.all_select_experts,
                self.all_select_weights,
                self.dispatch_tokens,
                self.used_src_tokens,
                self.expert_dispatch_tokens,
                self.expert_dispatch_weights,
                self.expert_dispatch_tokens_flatten,
                self.expert_dispatch_weights_flatten,
                self.expert_dispatch_token_count,
                self.expert_dispatch_token_offset,
                self.expert_map,
                self.expert_offsets,
                self.total_rows,
                self.all_select_experts_tensor,
            ) = get_moe_tokens_info(
                self.num_tokens,
                self.num_experts,
                self.topk,
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
            )

            self.inv_perm_cpu = compute_inv_perm_ep_gather(
                self.num_tokens,
                self.topk,
                self.experts_start_idx,
                self.expert_dispatch_tokens,
                self.expert_offsets,
                self.all_select_experts,
            )

        def vendor_parser(self):
            if self.dtype not in ["bfloat16"]:
                raise ValueError(
                    f"{type(self).__name__} only supports bfloat16 dtype, got {self.dtype}"
                )

        def vendor_impl(self):
            self.torch_dtype = getattr(torch, self.dtype)
            dev = self.backend.get_torch_device_name()

            inv_perm_stored = self.inv_perm_cpu

            self.input_tensor_info = {
                "scatter_tokens": OpTensorInfo(
                    # Feed `ep_gather` the padded expert-major buffer directly to avoid
                    # an extra scatter_tokens -> scatter_tokens_lightop fill/copy.
                    shape=[self.total_rows, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=dev,
                ),
                "selected_experts": OpTensorInfo(
                    shape=[self.num_tokens, self.topk],
                    dtype=torch.long,
                    device=dev,
                    creator=lambda size, dtype, device: torch.tensor(
                        self.all_select_experts, dtype=dtype, device=device
                    ),
                ),
                "moe_weights": OpTensorInfo(
                    shape=[self.num_tokens, self.topk],
                    dtype=torch.float32,
                    device=dev,
                    creator=lambda size, dtype, device: torch.tensor(
                        self.all_select_weights, dtype=dtype, device=device
                    ),
                ),
                "inv_perm_lightop": OpTensorInfo(
                    shape=[self.num_tokens, self.topk],
                    dtype=torch.int32,
                    device=dev,
                    creator=lambda size, dtype, device: torch.randint(low=0, high=self.total_rows, size=(self.num_tokens, self.topk), dtype=dtype, device=device),
                ),

                "expert_map": OpTensorInfo(
                    shape=[self.num_experts],
                    dtype=torch.int32,
                    device=dev,
                    creator=lambda size, dtype, device: self.expert_map.clone().detach().to(dtype=dtype, device=device),
                ),
            }
            if getattr(self, "res_scale", 0.0) != 0.0:
                self.input_tensor_info["residual_tokens"] = OpTensorInfo(
                    shape=[self.num_res_tokens_per_rank, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=dev,
                )

            self.output_tensor_info = {
                "convergent_tokens": OpTensorInfo(
                    shape=[self.num_tokens, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=dev,
                    creator=torch.zeros
                ),
            }

            # calculator (keep consistent with base MoeGatherOp accounting)
            self.input_tensor_size = sum(
                calc_tensor_size(info) for info in self.input_tensor_info.values()
            )
            self.output_tensor_size = sum(
                calc_tensor_size(info) for info in self.output_tensor_info.values()
            )
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = \
                calc_tensor_size(self.input_tensor_info["scatter_tokens"]) * (self.num_tokens / 2) / self.total_rows  + \
                calc_tensor_size(self.input_tensor_info["selected_experts"]) + \
                calc_tensor_size(self.input_tensor_info["moe_weights"]) + \
                calc_tensor_size(self.input_tensor_info["inv_perm_lightop"]) + \
                calc_tensor_size(self.input_tensor_info["expert_map"])

            if self.res_scale:
                self.read_bytes += calc_tensor_size(self.input_tensor_info["residual_tokens"])
            # index_add dst
            self.write_bytes = calc_tensor_size(self.output_tensor_info["convergent_tokens"])
            self.io_bytes = self.read_bytes + self.write_bytes


            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )
            self._run_func = self.vendor_impl_run

        def vendor_impl_run(self, tensor_mapping):
            scatter_tokens = tensor_mapping["scatter_tokens"]
            selected_experts = tensor_mapping["selected_experts"]
            expert_map = tensor_mapping["expert_map"]
            inv_perm_lightop = tensor_mapping["inv_perm_lightop"]
            moe_weights = tensor_mapping["moe_weights"]
            convergent_tokens = tensor_mapping["convergent_tokens"]

            if getattr(self, "res_scale", 0.0) != 0.0:
                residual_tokens = tensor_mapping["residual_tokens"]
                convergent_tokens[self.res_token_start:self.res_token_end] += residual_tokens * self.res_scale

            convergent_tokens = op.ep_gather(
                scatter_tokens,
                selected_experts,
                moe_weights,
                inv_perm_lightop,
                expert_map,
                convergent_tokens,
            )
            return convergent_tokens

except Exception:
    pass
