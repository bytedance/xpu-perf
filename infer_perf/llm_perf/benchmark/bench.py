import multiprocessing as mp
import os
import random
import sys
import time
from typing import Any, AsyncIterable, Callable, Dict, Iterable, List

import backoff
import grpc
import pandas as pd
from tqdm import tqdm

from llm_perf.utils.logger import logger
from llm_perf.utils.reporter import ReportType

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from llm_perf.server import server_pb2, server_pb2_grpc
from llm_perf.server.pb import deserialize_value, serialize_value


@backoff.on_exception(backoff.expo, Exception, factor=0.1, max_value=1, max_tries=3)
def gen_stream_request(
    stub: server_pb2_grpc.InferenceStub,
    index: int,
    prompt: str,
    min_new_tokens: int,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    get_input_logits: int,
) -> Iterable[server_pb2.InferenceResponse]:
    req = server_pb2.InferenceRequest(
        req_id=str(index) + "_" + str(int(time.time())),
        inputs={
            "input_messages": serialize_value(prompt),
            "min_new_tokens": serialize_value(min_new_tokens),
            "max_new_tokens": serialize_value(max_new_tokens),
            "top_p": serialize_value(top_p),
            "top_k": serialize_value(top_k),
            "get_input_logits": serialize_value(get_input_logits),
        },
    )
    for res in stub.StreamingInference(req, wait_for_ready=False):
        yield res


def format_question(line) -> str:
    question = line["question"]
    for choice in ["A", "B", "C", "D"]:
        question += f"\n{choice}. {line[f'{choice}']}"
    question += "\n答案："
    return question


def bench_accuracy(stub, workload: Dict[str, Any], result_queue: mp.Queue):
    dataset: pd.DataFrame = pd.read_csv(workload["dataset"])
    result_queue.put("@start")
    for row_index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        question = format_question(row)
        output_messages = ""
        perplexity_list: List[float] = []
        for res in gen_stream_request(
            stub,
            index=0,
            prompt=question,
            min_new_tokens=workload["min_new_tokens"],
            max_new_tokens=workload["max_new_tokens"],
            top_p=0,
            top_k=1,  # use greedy search for accuracy bench
            get_input_logits=1,
        ):
            res = {k: deserialize_value(v) for k, v in res.outputs.items()}
            if res["choice"]["message"] != "":
                output_messages = res["choice"]["message"]
            perplexity = res["choice"]["perplexity"]
            logits_dump = res["choice"]["logits_dump"]
            if not logits_dump:
                perplexity_list.append(perplexity)

        print("*"*150)
        logger.info("question: ")
        print(question)
        print("*"*150)
        logger.info("answer: ")
        print(output_messages)
        print("*"*150)

        result = {
            "output_message": output_messages,
            "perplexity": perplexity_list, 
            "logits_dump": logits_dump
        }
        result_queue.put(result)


def bench_performance(
    stub,
    index: int,
    workload: Dict[str, Any],
    input_tokens: int,
    result_queue: mp.Queue,
):
    result_queue.put("@start")

    accum_time = 0
    perf_time: int = workload["perf_time"] * int(1e9)

    while accum_time < perf_time:
        # make fake prompt, actual input_ids len may exceed input_tokens
        prompt = "我" * input_tokens
    
        st = time.perf_counter_ns()
        first_token_latency = 0

        min_new_tokens = workload["output_tokens"]
        max_new_tokens = workload["output_tokens"]

        output_messages: str = ""
        wait_time = []
        model_time = []
        post_process_time = []

        for res in gen_stream_request(
            stub,
            index=index,
            prompt=prompt,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            top_p=0,
            top_k=1,
            get_input_logits=0,
        ):
            res = {k: deserialize_value(v) for k, v in res.outputs.items()}
            if res["choice"]["message"] != "":
                output_messages = res["choice"]["message"]
            wait_time.append(res["choice"]["wait_time"])
            model_time.append(res["choice"]["model_time"])
            post_process_time.append(res["choice"]["post_process_time"])
            if first_token_latency == 0:
                first_token_latency = (time.perf_counter_ns() - st) / 1e6

        use_time = time.perf_counter_ns() - st
        accum_time += use_time

        # record context and decode len
        prompt_tokens = res["usage"]["prompt_tokens"]
        completion_tokens = res["usage"]["completion_tokens"]

        # seperate context and decode latency
        if completion_tokens > 1:
            per_token_latency = (use_time - first_token_latency) / (completion_tokens - 1) / 1e6
        else:
            per_token_latency = first_token_latency / 1e6

        context_wait_time = wait_time[0]
        context_model_time = model_time[0]
        context_postprocess_time = post_process_time[0]

        decode_wait_time = sum(wait_time[1:]) / len(wait_time[1:])
        decode_model_time = sum(model_time[1:]) / len(model_time[1:])
        decode_postprocess_time = sum(post_process_time[1:]) / len(post_process_time[1:])

        result = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "output_message": output_messages,
            "first_token_latency": first_token_latency,
            "per_token_latency": per_token_latency,

            "context_wait_time": context_wait_time, 
            "context_model_time": context_model_time, 
            "context_postprocess_time": context_postprocess_time, 

            "decode_wait_time": decode_wait_time, 
            "decode_model_time": decode_model_time, 
            "decode_postprocess_time": decode_postprocess_time, 
        }

        logger.debug(f"bench_{index} prompt response: {result}")
        result_queue.put(result)

def benchmark(
    index: int,
    start_wait: int, 
    workload: Dict[str, Any],
    report_type: ReportType,
    input_tokens: int,
    result_queue: mp.Queue,
    host, port
):
    with grpc.insecure_channel(f"{host}:{port}") as channel:
        stub = server_pb2_grpc.InferenceStub(channel)
        logger.debug(f"{report_type.name} bench_{index} start")
        
        # wait for start_wait seconds
        time.sleep(1 * start_wait)
        
        try:
            if report_type == ReportType.ACCURACY:
                bench_accuracy(stub, workload, result_queue)
            elif report_type == ReportType.PERFORMANCE:
                bench_performance(stub, index, workload, input_tokens, result_queue)
        except Exception as e:
            logger.error(f"{report_type.name} bench_{index} error: {e}")
            raise e

    logger.debug(f"{report_type.name} bench_{index} finish")
    result_queue.put(None)
