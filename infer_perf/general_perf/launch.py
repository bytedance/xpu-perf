import os
import sys
import argparse
import subprocess
import logging
import json

# ${prj_root}/byte_infer_perf
BYTE_MLPERF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BYTE_MLPERF_ROOT)
sys.path.insert(0, BYTE_MLPERF_ROOT)

from general_perf.core.configs.workload_store import load_workload

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("LANUCH")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="",
        help="The task going to be evaluted, refs to workloads/")
    parser.add_argument(
        "--hardware_type",
        default="CPU",
        help="The backend going to be evaluted, refs to backends/")
    parser.add_argument("--compile_only",
                        action='store_true',
                        help="Task will stoped after compilation finished")
    parser.add_argument("--show_task_list",
                        action='store_true',
                        help="Print all task names")
    parser.add_argument("--show_hardware_list",
                        action='store_true',
                        help="Print all hardware bytemlperf supported")
    args = parser.parse_args()
    return args


def main():
    parsed_args = get_args()

    if parsed_args.show_task_list:
        log.info("******************* Supported Task *******************")
        for file in os.listdir('general_perf/workloads'):
            print(file[:-5])

    if parsed_args.show_hardware_list:
        log.info("***************** Supported Hardware Backend *****************")
        for file in os.listdir('general_perf/backends'):
            if not file.endswith('.py') and not file.startswith('_'):
                print(file)

    if parsed_args.task:
        log.info("******************* Pip Package Installing *******************")
        subprocess.call([
            'python3', '-m', 'pip', 'install', 'pip', '--upgrade', '--quiet'])
        subprocess.call([
            'python3', '-m', 'pip', 'install', '-r', 'general_perf/requirements.txt', '--quiet'])

        workload = load_workload(parsed_args.task)
        with open("general_perf/model_zoo/" + workload['model'] + '.json', 'r') as file:
            model_info = json.load(file)

        if not os.path.exists(model_info['model_path']):
            subprocess.call([
                'bash', 'general_perf/prepare_model_and_dataset.sh',
                model_info['model'], model_info['dataset_name'] or "None"])

        # test numeric
        if workload['test_numeric'] and not parsed_args.compile_only and not workload['compile_only']:
            log.info("******************************************* Running CPU Numeric Checker... *******************************************")
            subprocess.call([
                'bash', 'general_perf/backends/CPU/calculate_cpu_diff.sh',
                workload['model'],
                str(workload['batch_sizes'][0])
            ])

        cmd = f'python3 general_perf/core/perf_engine.py --hardware_type {parsed_args.hardware_type} --task {parsed_args.task}'
        if parsed_args.compile_only:
            cmd += '--compile_only'
        exit_code = subprocess.call(cmd, shell=True)
        sys.exit(exit_code)


if __name__ == '__main__':
    main()
