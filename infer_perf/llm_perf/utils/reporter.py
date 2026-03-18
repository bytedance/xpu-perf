import json
import os
import shutil
import subprocess
import sys
import threading
import time
from enum import Enum
from queue import Queue
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from llm_perf.utils.logger import logger

# Time To First Token
__TTFT_AVG__ = "First Token Latency(AVG)"
__TTFT_P90__ = "First Token Latency(P90)"
__CONTEXT_WAIT_AVG__ = "Context Wait Time(AVG)"
__CONTEXT_WAIT_P90__ = "Context Wait Time(P90)"
__CONTEXT_MODEL_AVG__ = "Context Model Time(AVG)"
__CONTEXT_MODEL_P90__ = "Context Model Time(P90)"

# Time Per Output Token
__TPOT_AVG__ = "Per Token Latency(AVG)"
__TPOT_P90__ = "Per Token Latency(P90)"
__DECODE_WAIT_AVG__ = "Decode Wait Time(AVG)"
__DECODE_WAIT_P90__ = "Decode Wait Time(P90)"
__DECODE_MODEL_AVG__ = "Decode Model Time(AVG)"
__DECODE_MODEL_P90__ = "Decode Model Time(P90)"


class ReportType(Enum):
    ACCURACY = 0
    PERFORMANCE = 1


def get_cpu_name():
    command = "lscpu | grep 'Model name' | awk -F: '{print $2}'"
    cpu_name = subprocess.check_output(command, shell=True)
    return cpu_name.decode().strip()


def calc_perplexity(input_logits: torch.FloatTensor, labels: torch.LongTensor) -> float:
    # Shift so that tokens < n predict n
    shift_logits = input_logits
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss)
    logger.debug(f"Loss: {loss}, PPL: {perplexity}")
    return perplexity.tolist()


class Reporter:
    def __init__(
        self,
        task: str,
        backend: str,
        tp_size: int,
        batch_size: int,
        input_tokens: int,
        min_new_tokens: int,
        max_new_tokens: int,
        test_perf: bool,
        test_accuracy: bool,
        version: str="",
    ) -> None:
        self._running: bool = False
        self.cond: threading.Condition = threading.Condition()

        self.accuracy_datas: List[Dict[str:Any]] = []
        self.performance_datas: List[Dict[str:Any]] = []
        self._is_performance: bool = False
        self.logits_diff: List[Dict[str:Any]] = []
        self.token_diff: List[Dict[str:Any]] = []
        self.max_token_diff_num = 16
        self.test_perf = test_perf
        self.test_accuracy = test_accuracy

        self.backend = backend
        self.task: str = task
        # these configs will be update
        self.tp_size = tp_size
        self.batch_size = batch_size
        self.input_tokens = input_tokens
        self.version = version

        # result template
        self.result: Dict[str, Any] = {
            "Model": self.task,
            "Backend": self.backend,
            "Host Info": get_cpu_name(),
            "Version": self.version,
            "Execution Date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Min New Tokens": min_new_tokens,
            "Max New Tokens": max_new_tokens,
            "Accuracy": {"PPL": [], "Token Diff": {}, "Logits Diff": {}},
            "Performance": [
                {
                    "TP Size": self.tp_size,
                    "Batch Size": self.batch_size,
                    "Input Tokens": self.input_tokens,
                }
            ],
        }


    def update_meta(self, tp_size: int, batch_size: int, input_tokens: int):
        # update config 
        self.tp_size = tp_size
        self.batch_size = batch_size
        self.input_tokens = input_tokens

        self.start_time = time.perf_counter_ns()
        self.request = 0
        self.performance_datas.clear()
        logger.info(
            f"Update reporter meta: TP={self.tp_size}, BS={self.batch_size}, Inputs={self.input_tokens}"
        )


    def start(self):
        self._worker = threading.Thread(target=self.worker)
        self._running = True
        self._worker.start()

        self.start_time = time.perf_counter_ns()
        self.request = 0

    def stop(self):
        with self.cond:
            self._running = False
            self.cond.notify()
        self._worker.join()

    def submit(self, data: Dict[str, Any], report_type: ReportType):
        with self.cond:
            if report_type == ReportType.ACCURACY:
                self.accuracy_datas.append(data)
            elif report_type == ReportType.PERFORMANCE:
                self._is_performance = True
                self.performance_datas.append(data)
                self.request += 1
                self.last_submit_time = time.perf_counter_ns()
            self.cond.notify()

    def worker(self):
        with self.cond:
            while self._running:
                self.cond.wait()
                if self._running:
                    self.calc()


    def _calc_performance(self):
        # Calc avg/p99/sum of data, return result

        completion_tokens = 0
        time_since_start = (self.last_submit_time - self.start_time) / 1e9

        ttfts = []
        tpots = []

        context_wait_time = []
        context_model_time = []

        decode_wait_time = []
        decode_model_time = []
        
        for data in self.performance_datas:
            completion_tokens += data["completion_tokens"]

            ttfts.append(data["first_token_latency"])
            tpots.append(data["per_token_latency"])

            context_wait_time.append(data["context_wait_time"])
            context_model_time.append(data["context_model_time"])

            decode_wait_time.append(data["decode_wait_time"])
            decode_model_time.append(data["decode_model_time"])



        # context
        cur_ttft_avg = np.mean(ttfts)        
        cur_ttft_p90 = np.percentile(ttfts, 90)
        cur_context_wait_avg = np.mean(context_wait_time)
        cur_context_wait_p90 = np.percentile(context_wait_time, 90)
        cur_context_model_avg = np.mean(context_model_time)
        cur_context_model_p90 = np.percentile(context_model_time, 90)

        # decode
        cur_tpot_avg = np.mean(tpots)
        cur_tpot_p90 = np.percentile(tpots, 90)
        cur_decode_wait_avg = np.mean(decode_wait_time)
        cur_decode_wait_p90 = np.percentile(decode_wait_time, 90)
        cur_decode_model_avg = np.mean(decode_model_time)
        cur_decode_model_p90 = np.percentile(decode_model_time, 90)


        performance = None
        for perf in self.result["Performance"]:
            if (
                perf["TP Size"] == self.tp_size
                and perf["Batch Size"] == self.batch_size
                and perf["Input Tokens"] == self.input_tokens
            ):
                performance = perf

        if performance is None:
            performance = {
                "TP Size": self.tp_size,
                "Batch Size": self.batch_size,
                "Input Tokens": self.input_tokens,
            }
            self.result["Performance"].append(performance)


        performance["client"] = {
            __TTFT_AVG__: cur_ttft_avg, 
            __TTFT_P90__: cur_ttft_p90, 
            __TPOT_AVG__: cur_tpot_avg, 
            __TPOT_P90__: cur_tpot_p90, 
        }
        performance["server"] = {
            __CONTEXT_WAIT_AVG__ : cur_context_wait_avg, 
            __CONTEXT_WAIT_P90__ : cur_context_wait_p90, 
            __CONTEXT_MODEL_AVG__ : cur_context_model_avg, 
            __CONTEXT_MODEL_P90__ : cur_context_model_p90, 
            __DECODE_WAIT_AVG__ : cur_decode_wait_avg, 
            __DECODE_WAIT_P90__ : cur_decode_wait_p90, 
            __DECODE_MODEL_AVG__ : cur_decode_model_avg, 
            __DECODE_MODEL_P90__ : cur_decode_model_p90, 
        }

        logger.debug(
            f"TTFT(AVG)={cur_ttft_avg}, TTFT(P90)={cur_ttft_p90}, TPOT(AVG)={cur_tpot_avg}, TPOT(P90)={cur_tpot_p90}"
        )

        performance["Token Throughput"] = completion_tokens / time_since_start
        performance["Request Number"] = self.request
        performance["QPS"] = self.request / time_since_start

        logger.info(
            f"Request Number={performance['Request Number']}, Token Throughput={performance['Token Throughput']}, QPS={performance['QPS']}"
        )

    def _calc_accuracy(self):
        accuracy = self.result["Accuracy"]
        perplexity_list = []
        dump_files = []
        for i, data in enumerate(self.accuracy_datas):
            perplexity_list.append(data["perplexity"])
            if data["logits_dump"] != "":
                dump_files.append(data["logits_dump"])

        # 1. PPL
        accuracy["PPL"] = [
            sum(prompt_ppl) / len(prompt_ppl) for prompt_ppl in perplexity_list
        ]
        logger.debug(f"PPL={accuracy['PPL']}")


        # Diff Prepare
        diff_index = -1
        # prepare backend's logits numpy data file
        for i in range(len(dump_files)):
            diff_index += 1
            dump_file = dump_files[i]
            # If not exists, may already move to reports dir
            if not os.path.exists(dump_file):
                continue
            logits_dump_path = (
                "llm_perf/reports/" + self.backend + "/" + self.task + "/logits"
            )
            os.makedirs(logits_dump_path, exist_ok=True)
            shutil.move(dump_file, f"{logits_dump_path}/{i}.npy")
            logger.info(f"move {dump_file} to {logits_dump_path}/{i}.npy")


        # 2. Logits Diff: First token diff
        def calc_logits_diff(diff_index: int):
            # 2.1 Get base logits
            base_file = f"llm_perf/reports/base/{self.task}/logits/{diff_index}.npy"
            base_logits = np.load(base_file).astype(np.float32)

            # 2.2 Get Backend logits
            backend_file = (
                f"llm_perf/reports/{self.backend}/{self.task}/logits/{diff_index}.npy"
            )
            backend_logits = np.load(backend_file).astype(np.float32)

            # check shape
            if base_logits.shape != backend_logits.shape:
                logger.warn(
                    f"base and {self.backend} logits shape mismatch! Make sure generate config is the same. \nGPU: {base_logits.shape}, {self.backend}: {backend_logits.shape}"
                )

            # Only care about first token
            base_logits = base_logits[:, 0:1, :]
            backend_logits = backend_logits[:, 0:1, :]

            # 2.3 Calc Diff
            diff = base_logits - backend_logits
            max_difference = np.max(np.abs(diff))
            mse = np.mean(np.square(diff))
            mae = np.mean(np.abs(diff))
            cos_similarity = np.dot(base_logits.flatten(), backend_logits.flatten()) / (
                np.linalg.norm(base_logits.flatten())
                * np.linalg.norm(backend_logits.flatten())
            )
            logger.info(
                f"Logits Diff: Prompt Index={diff_index}, Max Difference={max_difference}, Mean Squared Error={mse}, Mean Absolute Error={mae}, Cosine Similarity={cos_similarity}"
            )

            last_logits_diff: Dict[str, Any] = {}
            last_logits_diff["Max Difference"] = max_difference
            last_logits_diff["Mean Squared Error"] = mse
            last_logits_diff["Mean Absolute Error"] = mae
            last_logits_diff["Cosine Similarity"] = cos_similarity
            last_logits_diff["Diff Data"] = diff.flatten()
            return last_logits_diff

        if diff_index >= 0:
            _diff = calc_logits_diff(diff_index)
            self.logits_diff.append(_diff)

        result_logits_diff = accuracy["Logits Diff"]
        if len(self.logits_diff) != 0:
            result_logits_diff["Max Difference"] = np.max(
                [l["Max Difference"] for l in self.logits_diff]
            ).tolist()
            result_logits_diff["Mean Squared Error"] = np.mean(
                [l["Mean Squared Error"] for l in self.logits_diff]
            ).tolist()
            result_logits_diff["Mean Absolute Error"] = np.mean(
                [l["Mean Absolute Error"] for l in self.logits_diff]
            ).tolist()
            result_logits_diff["Cosine Similarity"] = np.mean(
                [l["Cosine Similarity"] for l in self.logits_diff]
            ).tolist()

        # 3. Token Diff
        def calc_token_diff(diff_index: int):
            # 2.1 Get GPU base logits
            base_file = f"llm_perf/reports/base/{self.task}/logits/{diff_index}.npy"
            base_logits = np.load(base_file).astype(np.float32)
    
            # 2.2 Get Backend logits
            backend_file = (
                f"llm_perf/reports/{self.backend}/{self.task}/logits/{diff_index}.npy"
            )
            backend_logits = np.load(backend_file).astype(np.float32)

            # check shape
            if base_logits.shape != backend_logits.shape:
                logger.warn(
                    f"GPU and {self.backend} logits shape mismatch! Make sure generate config is the same. \nGPU: {base_logits.shape}, {self.backend}: {backend_logits.shape}"
                )
                return -1

            # Only care about max prob token (greedy search)
            base_logits = np.amax(base_logits, axis=2, keepdims=True)
            backend_logits = np.amax(backend_logits, axis=2, keepdims=True)

            # 2.3 Calc Diff
            diff = np.abs(base_logits - backend_logits)
            max_difference = np.max(diff)
            logger.info(
                f"Token Diff: Prompt Index={diff_index}, Max Difference={max_difference}"
            )

            last_token_diff: Dict[str, Any] = {}
            last_token_diff["Logits Index"] = diff_index
            last_token_diff["Max Difference"] = max_difference
            last_token_diff["Diff Data"] = diff.flatten()
            return last_token_diff


        if diff_index >= 0 and len(self.token_diff) < self.max_token_diff_num:
            _diff = calc_token_diff(diff_index)
            if _diff == -1:
                pass
            else:
                self.token_diff.append(_diff)

        result_token_diff = accuracy["Token Diff"]
        if len(self.token_diff) != 0:
            result_token_diff["Max Difference"] = np.max(
                [l["Max Difference"] for l in self.token_diff]
            ).tolist()
            result_token_diff["Prompt Num"] = len(self.token_diff)



    def calc(self):
        if self.test_accuracy and self.accuracy_datas and not self._is_performance:
            self._calc_accuracy()
        elif self.test_perf and self.performance_datas and self._is_performance:
            self._calc_performance()

    def summary(self):
        logger.info(f"summary...{self.result}")

        output_report_path = f"llm_perf/reports/{self.backend}/{self.task}"
        os.makedirs(output_report_path, exist_ok=True)

        if self.test_accuracy:
            # Save accuracy logits diff plt result
            logits_diff_png_path = f"{output_report_path}/logits_diff.png"
            logits_diff = np.concatenate(
                [l["Diff Data"] for l in self.logits_diff], axis=0
            )
            plt.hist(logits_diff, bins=150, alpha=0.75)
            plt.title("Logits Difference")
            plt.xlabel("Difference")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(logits_diff_png_path, dpi=300)
            self.result["Accuracy"]["Logits Diff"]["Png"] = logits_diff_png_path

            plt.clf()
            plt.cla()
            # Save token diff plt result
            token_diff_png_path = f"{output_report_path}/token_diff.png"
            plt.figure(figsize=(10, 6))
            for diff in self.token_diff:
                plt.plot(diff["Diff Data"], label=f"Prompt {diff['Logits Index']}")
            plt.legend()
            plt.title("Token Difference")
            plt.xlabel("Token")
            plt.ylabel("Difference")
            plt.grid(True)
            plt.savefig(token_diff_png_path, dpi=300)
            self.result["Accuracy"]["Token Diff"]["Png"] = logits_diff_png_path

        if not self.test_perf:
            self.result.pop("Min New Tokens", None)
            self.result.pop("Max New Tokens", None)
            self.result.pop("Performance", None)

        # Save Result
        with open(f"{output_report_path}/result.json", "w") as file:
            json.dump(self.result, file, indent=4)

        logger.info(f"Summary result to {output_report_path}/result.json")
