import sys
import json
import pathlib
import asyncio
import importlib
from typing import Any, AsyncIterable, Dict, Iterable

from transformers import AutoTokenizer, PreTrainedTokenizer

from llm_perf.core.generation import GenerateConfig, GenerateRequest, GenerateResult
from llm_perf.core.scheduler import CoreScheduler
from llm_perf.utils.logger import logger


class LLMPerfEndpoint:
    def __init__(self, xpu_cfg) -> None:
        super().__init__()

        self.xpu_cfg = xpu_cfg
        hardware_type = xpu_cfg["hardware_type"]
        model_config = xpu_cfg["model_config"]
    
        # load tokenizer
        try:
            tokenizer_config = model_config["tokenizer"]
            tokenizer_path = tokenizer_config["path"]
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=tokenizer_path, 
                local_files_only=True,
                trust_remote_code=True
            )
            self.support_chn = tokenizer_config.get("support_chn", False)
            self.apply_chat_template = tokenizer_config.get("apply_chat_template", False)
        except Exception as e:
            logger.error(f"load tokenizer error: {e}")
            sys.exit(-1)

        logger.info(f"load tokenizer: {tokenizer_path}")
        logger.info("*"*50)
        logger.info(f"bos_token_id: {self.tokenizer.bos_token_id}")
        logger.info(f"eos_token_id: {self.tokenizer.eos_token_id}")
        logger.info(f"unk_token_id: {self.tokenizer.unk_token_id}")
        logger.info(f"pad_token_id: {self.tokenizer.pad_token_id}")
        logger.info("*"*50)
        
        xpu_cfg["bos_token_id"] = self.tokenizer.bos_token_id
        xpu_cfg["eos_token_id"] = self.tokenizer.eos_token_id
        xpu_cfg["unk_token_id"] = self.tokenizer.unk_token_id
        xpu_cfg["pad_token_id"] = self.tokenizer.pad_token_id

        # import setup according to hardware_type
        setup = importlib.import_module(
            ".setup", package=f"llm_perf.backends.{hardware_type}"
        )
        logger.info(f"import setup: {setup}")

        # setup scheduler
        self.scheduler : CoreScheduler = setup.setup_scheduler(xpu_cfg)
        self.scheduler.start()

        self.warmup(xpu_cfg["max_batch_size"])


    def __del__(self):
        if hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.stop()


    def warmup(self, max_batch_size):
        if self.support_chn:
            prompt = "7年前，我的年龄是我的儿子的6倍，我的儿子今年12岁，我今年多少岁？"
        else:
            prompt = "7 years ago, I was 6 times older than my son. My son is 12 years old now. How old am I now?"

        generate_config = {
            "min_new_tokens": 1,
            "max_new_tokens": 512,
            "top_k": 1,
            "get_input_logits": 0
        }
        logger.info(f"warmup prompt: {prompt}, config: {generate_config}")

        async def _steram_warmup():
            message = ""
            async for result in self.streaming_inference(prompt, generate_config):
                message = result["choice"]["message"]
            result["choice"]["message"] = message
            return result

        async def _multiple_warmup():
            tasks = []
            for _ in range(max_batch_size):
                tasks.append(_steram_warmup())
            res = await asyncio.gather(*tasks)
            return res

        single_result = asyncio.run(_steram_warmup())
        message = single_result["choice"]["message"]
        logger.info(f"single warmup response: {message}\n")

        multiple_result = asyncio.run(_multiple_warmup())
        for i, result in enumerate(multiple_result):
            message = result["choice"]["message"]
            logger.info(f"multiple warmup reponse {i}: {message}\n")

    async def prepare_request(
        self, prompt: str, generate_config: Dict[str, Any]
    ) -> GenerateRequest:
        if not self.apply_chat_template:
            input_ids = self.tokenizer.encode(prompt)      
        else:
            input_ids = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt}
                ], 
                add_generation_prompt=True
            )

        # create generate config
        config = GenerateConfig(
            min_new_tokens=generate_config.get("min_new_tokens", 0),
            max_new_tokens=generate_config.get("max_new_tokens", 0),
            top_k=generate_config.get("top_k", 0),
            top_p=generate_config.get("top_p", 1.0),
            temperature=generate_config.get("temperature", 1.0),
            presence_penalty=generate_config.get("presence_penalty", 1.0),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            get_input_logits=bool(generate_config.get("get_input_logits", 0)),
        )
        req = GenerateRequest(input_ids, config)
        return req

    async def streaming_inference(
        self, 
        prompt: str, 
        generate_config: Dict[str, Any]
    ) -> AsyncIterable[Dict[str, Any]]:
        try:
            # create GenerateRequest object
            req = await self.prepare_request(prompt, generate_config)

            prompt_tokens = len(req.input_ids)
            completion_tokens = 0

            tokens_buffer = []

            async for gen_res in self.scheduler.generate(req):
                result = gen_res["result"]
                if result is not None:
                    completion_tokens += 1

                infer_outputs = {
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "choice": {
                        "message": ""
                    }
                }

                if result is not None:
                    tokens_buffer.append(result.token_id)

                    text = self.tokenizer.decode(tokens_buffer, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                    infer_outputs["choice"].update(
                        {
                            "message": text, 
                            "wait_time": result.wait_time, 
                            "model_time": result.model_time, 
                            "post_process_time": result.post_process_time
                        }
                    )

                if req.generate_config.get_input_logits:
                    infer_outputs["choice"].update(
                        {
                            "perplexity": gen_res["perplexity"], 
                            "logits_dump": gen_res["dump_file"]
                        }
                    )

                logger.debug(f"steam inference result: {infer_outputs}")
                yield infer_outputs
        except Exception as e:
            logger.error(f"stream inference error: {e}")
            raise e


