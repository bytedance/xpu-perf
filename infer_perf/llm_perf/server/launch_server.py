import os
import sys
import json
import pathlib
import argparse
import asyncio
from concurrent import futures
from typing import Any, AsyncIterable, Dict, Iterable, List
import grpc
import signal


CWD_DIR = pathlib.Path.cwd().absolute()
FILE_DIR = pathlib.Path(__file__).parent.absolute()
BYTE_MLPERF_ROOT = FILE_DIR.parents[1]
os.chdir(BYTE_MLPERF_ROOT)
sys.path.insert(0, BYTE_MLPERF_ROOT.__str__())



from llm_perf.server import server_pb2, server_pb2_grpc
from llm_perf.server.pb import deserialize_value, serialize_value
from llm_perf.server.endpoint import LLMPerfEndpoint
from llm_perf.utils.logger import logger, setup_logger


class TestServer(server_pb2_grpc.InferenceServicer):
    def __init__(self, generator: LLMPerfEndpoint) -> None:
        super().__init__()
        self.generator = generator

    async def StreamingInference(
        self, 
        request, 
        context
    ):
        logger.debug(f"StreamingInference request id {request.req_id}")

        req = {k: deserialize_value(v) for k, v in request.inputs.items()}
        prompt = req["input_messages"]
        generate_config = {
            "min_new_tokens": req["min_new_tokens"],
            "max_new_tokens": req["max_new_tokens"],
            "top_p": req["top_p"],
            "top_k": req["top_k"],
            "get_input_logits": req["get_input_logits"],
        }

        # Generating
        async for result in self.generator.streaming_inference(
            prompt=prompt, generate_config=generate_config
        ):
            yield server_pb2.InferenceResponse(
                req_id=request.req_id,
                outputs={k: serialize_value(v) for k, v in result.items()},
            )

async def serve(port, generator: LLMPerfEndpoint) -> None:
    server = grpc.aio.server(
        migration_thread_pool=futures.ThreadPoolExecutor(
            max_workers=min(os.cpu_count(), 100), 
            thread_name_prefix="server"
        )
    )
    server_pb2_grpc.add_InferenceServicer_to_server(
        TestServer(generator), server
    )
    server.add_insecure_port(f"[::]:{port}")

    await server.start()
    logger.info(f"GRPC Server start at {port}")

    fifo_name = "./server_fifo"
    with open(fifo_name, "w") as fifo_fd:
        fifo_fd.write("Server Ready")
        fifo_fd.flush()

    await server.wait_for_termination()    



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--hardware_type", type=str, default="GPU")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--port", type=int, default=51000)
    parser.add_argument("--log_level", type=str, default="info")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logger(args.log_level)

    # check model_config
    if pathlib.Path(args.model_config).is_absolute():
        model_config_path = model_config_path
    else:
        model_config_path = CWD_DIR.joinpath(args.model_config)

    if not model_config_path.exists():
        logger.error(f"model_config_path not exist")
        sys.exit(-1)
    with open(model_config_path, 'r') as file:
        model_config = json.load(file)


    # create xpu config
    xpu_cfg = {
        "hardware_type": args.hardware_type,
        "tp_size": args.tp_size,
        "max_batch_size": args.max_batch_size,
        "model_config": model_config
    }
    generator = LLMPerfEndpoint(xpu_cfg)
    asyncio.run(serve(args.port, generator))


if __name__ == "__main__":
    main()
