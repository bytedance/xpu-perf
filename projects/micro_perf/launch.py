import argparse
import pathlib
from functools import partial

import torch.multiprocessing as mp

FILE_DIR = pathlib.Path(__file__).parent.absolute()

from xpu_perf.micro_perf.core.perf_engine import XpuPerfServer
from xpu_perf.micro_perf.core.common_utils import logger, setup_logger
from xpu_perf.micro_perf.core.common_utils import get_submodules, existing_dir_path, valid_file
from xpu_perf.micro_perf.core.common_utils import parse_tasks, parse_workload, export_reports


BYTE_MLPERF_ROOT = FILE_DIR
OP_DEFS_DIR = BYTE_MLPERF_ROOT.joinpath("op_defs")

mp.set_start_method('spawn', force=True)


def parse_args():
    setup_logger("INFO")

    backend_name_list, backend_mod_list = get_submodules("xpu_perf.micro_perf.backends")

    parser = argparse.ArgumentParser()

    # backend config
    parser.add_argument(
        "--backend",
        type=str,
        default="GPU",
        choices=backend_name_list,
    )
    parser.add_argument("--op_defs", type=existing_dir_path, default=OP_DEFS_DIR)
    parser.add_argument(
        "--vendor_ops",
        type=existing_dir_path,
        default=[],
        action="append",
    )
    parser.add_argument(
        "--env",
        type=partial(valid_file, required_suffix=".json"),
        default=None,
        help="Path to env JSON file (default: backends/{backend}/env.json)",
    )

    # numa config
    parser.add_argument(
        "--numa",
        type=str,
        default=None,
        help="Numa config. "
        "Default is None which create **num_numa_nodes** processes to bench, "
        "each of them run on one numa node and schedule some devices. "
        "Values '-1' or '0' or '1' mean creating one process and specifing all numa nodes or node 0 or node 1. "
        "Value '0,1' means creating 2 processes and assign node 0 and node 1 to them respectively. ",
    )

    # device config
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device config."
        "Default is None which use all devices on current machine."
        "Value '0,1' means using device 0 and device 1 on current machine.",
    )

    # node config
    parser.add_argument(
        "--node_world_size",
        type=int,
        default=1,
        help="Node world size, default is 1",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="Node rank, default is 0",
    )

    # server config
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--server_port", type=int, default=49371)
    parser.add_argument("--host_port", type=int, default=49372)
    parser.add_argument("--device_port", type=int, default=49373)

    # bench arguments
    parser.add_argument(
        "--task_dir",
        type=str,
        default=str(BYTE_MLPERF_ROOT.joinpath("workloads", "basic")),
    )
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--workload", type=str)

    # report output
    parser.add_argument(
        "--report_dir",
        type=str,
        default=str(BYTE_MLPERF_ROOT.joinpath("reports")),
    )

    args = parser.parse_args()
    args.script_dir = FILE_DIR
    args.backend_name_list = backend_name_list
    args.backend_mod_list = backend_mod_list
    return args


def load_test_cases(args):
    if args.workload is not None:
        return parse_workload(args.workload)
    return parse_tasks(args.task_dir, args.task)


def log_test_cases(test_cases):
    print("*" * 100)
    logger.info("test cases:")
    for op_name, op_cases in test_cases.items():
        logger.info(f"{op_name} has {len(op_cases)} test cases")
    print("*" * 100)


def run_bench(args):
    test_cases = load_test_cases(args)
    if not test_cases:
        logger.error("No valid test cases found. Exiting.")
        raise SystemExit(1)

    log_test_cases(test_cases)

    with XpuPerfServer(args) as server_instance:
        info_dict = server_instance.get_info()
        bench_results = server_instance.normal_bench(test_cases)

    export_reports(
        args.report_dir,
        info_dict,
        test_cases,
        bench_results,
    )


def main():
    args = parse_args()
    run_bench(args)


if __name__ == "__main__":
    main()
