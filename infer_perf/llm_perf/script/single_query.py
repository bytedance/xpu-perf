import os
import sys
import time
import grpc
import random
import pathlib
import argparse
from tqdm import tqdm
import numpy as np
import prettytable as pt
import csv
import concurrent.futures
import subprocess
import threading
from functools import partial
from typing import List


FILE_DIR = pathlib.Path(__file__).parent.absolute()
INFER_ROOT_DIR = FILE_DIR.parents[1]
CUR_DIR = pathlib.Path.cwd().absolute()


sys.path.insert(0, str(INFER_ROOT_DIR))

from llm_perf.server import server_pb2, server_pb2_grpc
from llm_perf.server.pb import deserialize_value, serialize_value



def gen_stream_request(
    stub: server_pb2_grpc.InferenceStub, 
    index: int, 
    prompt: str, 
    min_new_tokens: int, 
    max_new_tokens: int, 
    top_p: float, 
    top_k: int, 
    get_input_logits: int
): 
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


start_index = 0
start_condition = threading.Condition()

finish_num = 0
finish_num_lock = threading.Lock()

finish_index = 0
finish_condition = threading.Condition()


def thread_func(
    local_rank: int, 
    world_size: int,   
    promp_list: List[str], 
    max_new_tokens: int, 
    save_logits: bool, 
    host: str, 
    port: int
):
    global start_index
    global finish_num
    global finish_index

    prompt = promp_list[local_rank]

    result_list = []
    with grpc.insecure_channel(f"{host}:{port}") as channel:
        # wait for start
        start_condition.acquire()
        while local_rank != start_index:
            start_condition.wait()

        stub = server_pb2_grpc.InferenceStub(channel)
        infer_timeline = tqdm(total=max_new_tokens)
        infer_timeline.set_description(f"infer_{local_rank} (max_new_tokens={max_new_tokens})")
        infer_timeline.set_postfix({"max_new_tokens": max_new_tokens})

        for res in gen_stream_request(
            stub, 
            index=local_rank,
            prompt=prompt,
            min_new_tokens=1,
            max_new_tokens=max_new_tokens,
            top_p=0,
            top_k=1, 
            get_input_logits=1 if save_logits else 0,
        ):
            # make sure that each request that in one decode batch is at different stage (different kv len)
            if local_rank == start_index:
                start_index += 1
                start_condition.notify_all()
                start_condition.release()

            res = {k: deserialize_value(v) for k, v in res.outputs.items()}
            result_list.append(res)
            if res["choice"]["message"] != "":
                infer_timeline.update(1)
        
        # update finish_num
        finish_num_lock.acquire()
        finish_num += 1
        finish_num_lock.release()


        finish_condition.acquire()
        if finish_num == world_size:
            finish_condition.notify_all()
        while finish_num != world_size or local_rank != finish_index:
            finish_condition.wait()
        finish_index += 1
        finish_condition.notify_all()
        finish_condition.release()

    return result_list



def test_infer(
    promp_list: List[str],
    batch_size: int, 
    max_new_tokens: int,
    save_logits: bool,
    workspace: pathlib.Path,
    host: str,
    port: int
):
    collect_results = []
    start_time = time.perf_counter_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        # create tasks
        to_do = []
        for i in range(batch_size):
            future = executor.submit(thread_func, i, batch_size, promp_list, max_new_tokens, save_logits, host, port)
            to_do.append(future)
        # wait for all tasks to complete
        for future in concurrent.futures.as_completed(to_do):
            collect_results.append(future.result())
    duration_s = round((time.perf_counter_ns() - start_time) / 1e9, 3)
    print(f"\nquery duration: {duration_s}s\n")

    valid_res_list = []
    logits_list = []
    for result_list in collect_results:
        if save_logits:
            valid_res_list.append(result_list[:-1])
            logits_file = INFER_ROOT_DIR.joinpath(result_list[-1]["choice"]["logits_dump"])
            logits = np.load(logits_file)
            logits_list.append(logits)
            logits_file.unlink()
        else:
            valid_res_list.append(result_list)


    prompt_tokens_list = [valid_res[-1]["usage"]["prompt_tokens"] for valid_res in valid_res_list]
    completion_tokens_list = [valid_res[-1]["usage"]["completion_tokens"] for valid_res in valid_res_list]
    output_messages_list = [valid_res[-1]["choice"]["message"] for valid_res in valid_res_list]

    
    response_match = [output_messages_list[0] == output_messages_i for output_messages_i in output_messages_list]

    logits_diff = []
    tokens_diff = []
    tokens_fork_index = []
    if save_logits:
        logits = logits_list[0]
        for i in range(batch_size):
            cur_logits_diff = []
            cur_token_diff = []
            # logits shape: [1, completion_tokens, vocab_size]
            for j in range(min(completion_tokens_list[0], completion_tokens_list[i])):
                cur_logits_diff.append(round(abs(logits[0, j, :] - logits_list[i][0, j, :]).mean(), 3))
                cur_token_diff.append(abs(np.argmax(logits[0, j, :]) - np.argmax(logits_list[i][0, j, :])))
            logits_diff.append(cur_logits_diff)
            tokens_diff.append(cur_token_diff)
            tokens_fork_index.append(np.flatnonzero(np.array(cur_token_diff) > 0))


    output_txt = workspace.joinpath("output.txt")
    with open(output_txt, "w") as f:
        print_func = partial(print, file=f)
        print_func("-" * 150)
        print_func(f"* prompts (batch_size={batch_size})")
        for i in range(batch_size):
            print_func(f"{i}: {promp_list[i]}")
        print_func("")
        print_func(f"* response (batch_size={batch_size})")
        for i in range(batch_size):
            print_func(f"{i}: {output_messages_list[i]}")
        print_func("")
        print_func(f"* response match: {response_match}")
        if save_logits:
            print_func("")
            print_func(f"* logits mean diff: {[round(sum(diff) / len(diff), 3) for diff in logits_diff]}")
            print_func("")
            print_func(f"* token diverge index: {[None if len(diff) == 0 else diff[0] for diff in tokens_fork_index]}")
        print_func("-" * 150)
    print(output_txt.read_text())


    subprocess.run(f"rm -rf {workspace}/batch_*", shell=True)
    for index in range(batch_size):
        batch_workspace = workspace.joinpath(f"batch_{index}")
        batch_workspace.mkdir()

        prompt_txt = batch_workspace.joinpath("prompt.txt")
        with open(prompt_txt, "w") as f:
            f.write(promp_list[index])
        response_txt = batch_workspace.joinpath("response.txt")
        with open(response_txt, "w") as f:
            f.write(output_messages_list[index])
        output_txt = batch_workspace.joinpath("output.txt")
        latency_csv = batch_workspace.joinpath("latency.csv")
        if save_logits:
            logits_npy = batch_workspace.joinpath(f"logits.npy")
            np.save(logits_npy, logits_list[index])


        prompt_tokens = prompt_tokens_list[index]
        completion_tokens = completion_tokens_list[index]
        output_messages = output_messages_list[index]
        valid_res = valid_res_list[index]

        model_time_list = [res["choice"]["model_time"] for res in valid_res]
        post_process_time_list = [res["choice"]["post_process_time"] for res in valid_res]
        wait_time_list = [res["choice"]["wait_time"] for res in valid_res]

        first_token_latency = model_time_list[0]
        first_token_latency = round(first_token_latency, 3)
        per_token_latency = sum(model_time_list[1:]) / (completion_tokens - 1) if completion_tokens > 1 else 0
        per_token_latency = round(per_token_latency, 3)

        summart_tb = pt.PrettyTable()
        summart_tb.field_names = ["Metric", "Value"]
        summart_tb.add_row(["Min new tokens", 1])
        summart_tb.add_row(["Max new tokens", f"{max_new_tokens}"])
        summart_tb.add_row(["Prompt tokens", f"{prompt_tokens}"])
        summart_tb.add_row(["Completion tokens", f"{completion_tokens}"])
        summart_tb.add_row(["First token latency", f"{first_token_latency} ms"])
        summart_tb.add_row(["Per token latency (Avg)", f"{per_token_latency} ms"])
        if save_logits:
            summart_tb.add_row(["Output logits shape", f"{logits.shape}"])
            if len(tokens_fork_index[index]) > 0:
                summart_tb.add_row(["Token diverge index", f"{tokens_fork_index[index][0]}"])
            else:
                summart_tb.add_row(["Token diverge index", "None"])


        file_tb = pt.PrettyTable()
        file_tb.field_names = ["File", "Path"]
        file_tb.add_row(["Prompt text", str(prompt_txt)])
        file_tb.add_row(["Response text", str(response_txt)])
        file_tb.add_row(["Output text", str(output_txt)])
        file_tb.add_row(["Latency csv", str(latency_csv)])
        if save_logits:
            file_tb.add_row(["Logits npy", str(logits_npy)])


        with open(output_txt, "w") as file:
            print("-" * 150, file=file)
            print(f"* prompt: ({prompt_tokens} tokens)", file=file)
            print(promp_list[index], file=file)
            print("", file=file)
            print(f"* response: ({completion_tokens} tokens)", file=file)
            print(output_messages, file=file)
            print("-" * 150, file=file)

            print(summart_tb, file=file)
            if save_logits:
                print(f"logits_diff: {logits_diff[index]}", file=file)
                print(f"token_diff: {tokens_diff[index]}", file=file)
            print(file_tb, file=file)



        with open(latency_csv, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "mode", "past_kv", "q_len", "token_generated", 
                "wait_time (ms)", "model_time (ms)", "post_process_time (ms)", 
                "message"])
            for i, res in enumerate(valid_res):
                if i == 0:
                    csv_writer.writerow([
                        "prefill", 
                        i, 
                        prompt_tokens,
                        res["usage"]["completion_tokens"], 
                        round(res["choice"]["wait_time"], 3),
                        round(res["choice"]["model_time"], 3),
                        round(res["choice"]["post_process_time"], 3), 
                        res["choice"]["message"]
                    ])
                else:
                    csv_writer.writerow([
                        "decode", 
                        prompt_tokens + i - 1, 
                        1,
                        res["usage"]["completion_tokens"],
                        round(res["choice"]["wait_time"], 3),
                        round(res["choice"]["model_time"], 3),
                        round(res["choice"]["post_process_time"], 3), 
                        res["choice"]["message"]
                    ])






def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, nargs='+', default=["How are you?"])
    parser.add_argument("--prompt_tokens", type=int)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=51000)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    prompt = args.prompt
    prompt_tokens = args.prompt_tokens
    batch_size = args.batch_size
    max_new_tokens = args.max_new_tokens
    save_logits = args.save_logits
    workspace = args.workspace
    host = args.host
    port = args.port

    # create prompt list
    candidate_words = [
        "you", "who", "that", "info", "debug", "error", "correct", 
        "true", "false", "token", "model", "batch", "launch", "server", 
        "setup"
    ]
    prompt_list = []
    if prompt_tokens is not None:
        for _ in range(batch_size):
            prompt_list.append(" ".join([random.choice(candidate_words) for _ in range(prompt_tokens)]))
    else:  
        if len(prompt) == 1:
            assert batch_size >= 1 and batch_size <= 32
            prompt_list = prompt * batch_size
        else:
            batch_size = len(prompt)
            prompt_list = prompt

    if workspace is not None:
        workspace = pathlib.Path(args.workspace).absolute()
    else:
        workspace = INFER_ROOT_DIR.joinpath(
            "llm_perf", "reports", "single_query")

    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
    
    test_infer(prompt_list, batch_size, max_new_tokens, save_logits, workspace, host, port)
    

