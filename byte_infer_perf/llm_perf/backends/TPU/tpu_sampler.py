from typing import Any, Dict, List, Tuple, Union

import torch
import torch_tpu

from llm_perf.core.generation import GenerateResult
from llm_perf.core.inferencer import CoreInferencer
from llm_perf.core.sampler import CoreSampler

from llm_perf.utils.logger import logger


class TpuSampler(CoreSampler):
    def __init__(self) -> None:
        super().__init__()

    def sample(
        self, 
        tasks: List[CoreInferencer.Task], 
        logits: torch.FloatTensor
    ) -> List[int]:
        top_p = [p.request.generate_config.top_p for p in tasks]
        if all(p == 1.0 for p in top_p):
            top_p = None

        top_k = [p.request.generate_config.top_k for p in tasks]
        if all(k == 0 for k in top_k):
            top_k = None

        temperature = [p.request.generate_config.temperature for p in tasks]
        if all(t == 1.0 for t in temperature):
            temperature = None

        (
            sp_input_ids,
            sp_cu_seqlens,
            sp_max_seqlens,
            repetition_penalty,
            mask_eos_token,
        ) = (None, None, 0, None, None)
        eos_token_id = [p.request.generate_config.eos_token_id or -1 for p in tasks]

        next_tokens, softmax_out = self._sample(
            logits.float(),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            input_ids=sp_input_ids,
            cu_seqlens=sp_cu_seqlens,
            max_seqlens=sp_max_seqlens,
            repetition_penalty=repetition_penalty,
            mask_eos_token=mask_eos_token,
            min_tokens_to_keep=1,
            eos_token_id=eos_token_id,
        )

        next_tokens = next_tokens.tolist()

        # The aux_data is softmax_out here
        return next_tokens, softmax_out


    def _sample(
        self,
        logits: torch.FloatTensor,
        temperature: Union[List[float], torch.FloatTensor] = None,
        top_k: Union[List[int], torch.IntTensor] = None,
        top_p: Union[List[float], torch.FloatTensor] = None,
        input_ids: Union[List[int], torch.IntTensor] = None,
        cu_seqlens: Union[List[int], torch.IntTensor] = None,
        repetition_penalty: Union[List[float], torch.FloatTensor] = None,
        mask_eos_token: Union[List[int], torch.IntTensor] = None,
        min_tokens_to_keep: int = 1,
        eos_token_id: int = 0,
        max_seqlens: int = 0,
    ) -> Tuple[List[int], torch.FloatTensor]:
        _is_greedy = False
        _is_random = False
        _is_fastpath = False

        if top_k:
            assert all(
                k == top_k[0] for k in top_k
            ), f"expect the same batch top_k, but got {top_k}"
            if all(k == 1 for k in top_k):
                _is_greedy = True
        elif top_p:
            _is_random = True
            if all(p == top_p[0] for p in top_p):
                _is_fastpath = True
                _top_p = top_p[0]
        else:
            raise RuntimeError(
                f"Unsupported sample strategy, parameter top_k: {top_k} top_p: {top_p}"
            )

        if _is_greedy:
            return torch.argmax(logits, dim=-1), torch.nn.functional.softmax(logits, dim=-1)
        else:
            raise NotImplementedError

    def postprocess(
        self,
        tasks: List[CoreInferencer.Task],
        infer_outputs: Dict[str, torch.FloatTensor],
        next_tokens: List[int],
    ) -> List[GenerateResult]:
        generate_result = []
        for i in range(len(tasks)):
            token_id = next_tokens[i]
            task = tasks[i]

            # take current generated token into account
            generate_tokens_len = len(task.generate_ids) + 1

            if token_id == task.request.generate_config.eos_token_id:
                if generate_tokens_len < task.request.generate_config.min_new_tokens:
                    finish_reason = ""
                    token_id = task.request.generate_config.eos_token_id
                else:
                    finish_reason = "stop"
            elif generate_tokens_len >= task.request.generate_config.max_new_tokens:
                finish_reason = "max_length"
            else:
                finish_reason = ""


            if task.request.generate_config.get_input_logits:
                gen_res = GenerateResult(
                    token_id=token_id,
                    finish_reason=finish_reason, 

                    wait_time=task.wait_time[-1], 
                    model_time=task.model_time[-1], 
                    post_process_time=task.post_process_time[-1], 

                    logits=infer_outputs["logits"][i].float().cpu(), 
                    last_logits=infer_outputs["last_logits"][i].float().cpu(), 
                )
            else:
                gen_res = GenerateResult(
                    token_id=token_id,
                    finish_reason=finish_reason, 

                    wait_time=task.wait_time[-1], 
                    model_time=task.model_time[-1], 
                    post_process_time=task.post_process_time[-1], 

                    logits=None, 
                    last_logits=None, 
                )

            generate_result.append(gen_res)

        return generate_result
