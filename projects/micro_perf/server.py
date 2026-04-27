import json
import queue
import pathlib
import argparse
import threading
import traceback
from functools import partial

import torch.multiprocessing as mp

FILE_DIR = pathlib.Path(__file__).parent.absolute()

from xpu_perf.micro_perf.core.perf_engine import XpuPerfServer
from xpu_perf.micro_perf.core.common_utils import logger, setup_logger
from xpu_perf.micro_perf.core.common_utils import get_submodules, existing_dir_path, valid_file

from flask import Flask, request, jsonify, Response, stream_with_context


BYTE_MLPERF_ROOT = FILE_DIR
OP_DEFS_DIR = BYTE_MLPERF_ROOT.joinpath("op_defs")

mp.set_start_method('spawn', force=True)



def parse_args():
    setup_logger("INFO")

    BACKEND_NAME_LIST, BACKEND_MOD_LIST = \
        get_submodules("xpu_perf.micro_perf.backends")

    parser = argparse.ArgumentParser()
    
    # backend config
    parser.add_argument("--backend", type=str, required=True, choices=BACKEND_NAME_LIST)
    parser.add_argument("--op_defs", type=existing_dir_path, default=OP_DEFS_DIR)
    parser.add_argument("--vendor_ops", type=existing_dir_path, default=[],action='append')

    parser.add_argument(
        "--env", 
        type=partial(valid_file, required_suffix=".json"), 
        default=None,
        help="Path to env JSON file (default: backends/{backend}/env.json)"
    )

    # numa config
    parser.add_argument(
        "--numa", type=str, default=None, 
        help="Numa config. "
            "Default is None which create **num_numa_nodes** processes to bench, "
             "each of them run on one numa node and schedule some devices. "
             "Values '-1' or '0' or '1' mean creating one process and specifing all numa nodes or node 0 or node 1. "
             "Value '0,1' means creating 2 processes and assign node 0 and node 1 to them respectively. "
    )
    
    # device config
    parser.add_argument(
        "--device", type=str, default=None, 
        help="Device config."
             "Default is None which use all devices on current machine."
             "Value '0,1' means using device 0 and device 1 on current machine."
    )

    # node config
    parser.add_argument(
        "--node_world_size", type=int, default=1, 
        help="Node world size, default is 1"
    )
    parser.add_argument(
        "--node_rank", type=int, default=0,
        help="Node rank, default is 0"
    )

    # server config
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--server_port", type=int, default=49371)
    parser.add_argument("--host_port", type=int, default=49372)
    parser.add_argument("--device_port", type=int, default=49373)
    
    args = parser.parse_args()

    args.script_dir = pathlib.Path(__file__).parent.absolute()
    args.backend_name_list = BACKEND_NAME_LIST
    args.backend_mod_list = BACKEND_MOD_LIST

    return args


def error_response(error):
    logger.exception("Request handling failed: %s", error)
    return jsonify({"status": "error", "message": str(error)}), 500


def create_stream_response(server_instance, body):
    progress_q = queue.Queue()
    holder = {}

    def worker():
        try:
            def progress_callback(evt):
                progress_q.put(evt)

            holder["result"] = server_instance.bench(body, progress_callback=progress_callback)
        except Exception as error:
            logger.exception("Streaming bench failed: %s", error)
            holder["error"] = str(error)
            holder["traceback"] = traceback.format_exc()
        finally:
            progress_q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        while True:
            item = progress_q.get()
            if item is None:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"
        if holder.get("error"):
            yield json.dumps(
                {"event": "error", "message": holder["error"]},
                ensure_ascii=False,
            ) + "\n"
        else:
            yield json.dumps(
                {"event": "done", "data": holder.get("result", {})},
                ensure_ascii=False,
            ) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


def create_app(server_instance):
    app_instance = Flask("XpuPerfServer")

    @app_instance.route("/info", methods=["GET"])
    def info():
        try:
            return jsonify(server_instance.get_info())
        except Exception as error:
            return error_response(error)

    @app_instance.route("/bench", methods=["POST"])
    def bench():
        try:
            input_dict = request.get_json(force=True, silent=True) or {}
            stream = bool(input_dict.get("stream_progress"))
            body = {k: v for k, v in input_dict.items() if k != "stream_progress"}

            if not stream:
                return jsonify(server_instance.bench(body))

            return create_stream_response(server_instance, body)
        except Exception as error:
            return error_response(error)

    return app_instance


def main():
    args = parse_args()
    server_instance = XpuPerfServer(args)
    server_instance.create()
    app_instance = create_app(server_instance)

    try:
        app_instance.run(
            host=args.master_addr,
            port=args.server_port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
    finally:
        server_instance.destroy()




if __name__ == '__main__':
    main()
