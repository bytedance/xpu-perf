from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from llm_perf.core.generation import GenerateResult
from llm_perf.core.inferencer import CoreInferencer


class CoreSampler(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def sample(
        self, 
        tasks: List[CoreInferencer.Task], 
        logits: torch.FloatTensor
    ) -> List[int]:
        """Sample next tokens

        Args:
            packets: sample batch packets
            logits: model inference outputs, shape is (sum(len(input_ids) of each packet), vocab_size)

        Return:
            next_tokens: next token list of each request
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(
        self,
        packets: List[CoreInferencer.Task],
        infer_outputs: Dict[str, torch.FloatTensor],
        next_tokens: List[int],
    ) -> List[GenerateResult]:
        """Postprocess sample result tokens

        Args:
            packets: sample batch packets
            infer_output: inference outputs, contain 'input_logits' and 'last_logits' `{"input_logits": tensor, "last_logits": tensor}`
                input_logits: model inference output input logits
                last_logits: model inference outputs last logits, shape is (sum(len(input_ids) of each packet), vocab_size)
            next_tokens: sample packets next token list

        Return:
            GenerateResult list of packets
        """
        raise NotImplementedError
