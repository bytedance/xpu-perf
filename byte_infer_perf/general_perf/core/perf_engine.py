# Copyright 2023 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
import logging
import importlib
import json
import subprocess
import time
import traceback

from typing import Any, Dict, Tuple
import virtualenv
from prompt_toolkit.shortcuts import radiolist_dialog, input_dialog, yes_no_dialog
from prompt_toolkit.styles import Style

BYTE_MLPERF_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(BYTE_MLPERF_ROOT)
sys.path.insert(0, BYTE_MLPERF_ROOT)

import argparse
from general_perf.core.configs.workload_store import load_workload
from general_perf.core.configs.dataset_store import load_dataset
from general_perf.core.configs.backend_store import init_compile_backend, init_runtime_backend
from general_perf.tools.build_pdf import build_pdf

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PerfEngine")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="resnet50-tf-fp32",
        help="The task going to be evaluted, refs to workloads/")
    parser.add_argument(
        "--hardware_type",
        default="GPU",
        help="The backend going to be evaluted, refs to backends/")
    parser.add_argument("--compile_only",
                        action='store_true',
                        help="Run compilation only")

    args = parser.parse_args()
    return args


class PerfEngine:
    def __init__(self) -> None:
        super().__init__()
        self.args = get_args()
        self.workload = load_workload(self.args.task)
        self.backend_type = self.args.hardware_type
        self.compile_backend = None
        self.old_os_path = os.environ['PATH']
        self.prev_sys_path = list(sys.path)
        self.real_prefix = sys.prefix
        self.compile_only_mode = False
        self.version = self.get_version()

    def get_version(self):
        version = ""
        try:
            version_file = os.path.join(str(BYTE_MLPERF_ROOT), "../VERSION")
            with open(version_file) as f:
                _version = f.read().splitlines()
            version = '.'.join(v.split('=')[1] for v in _version)
        except Exception as e:
            traceback.print_exc()
            log.warning(f"get bytemlperf version failed, error msg: {e}")
        return version

    def start_engine(self) -> None:
        '''
        Byte MlPerf will create an virtual env for each backend to avoid dependance conflict
        '''
        success, total = 0, len(self.workload)
        if total == 0:
            return
        log.info("******************* Backend Env Initization *******************")
        status = self.activate_venv(self.backend_type)
        if not status:
            log.warning("Activate virtualenv Failed, Please Check...")

        self.compile_backend = init_compile_backend(self.backend_type)
        self.runtime_backend = init_runtime_backend(self.backend_type)

        output_dir = os.path.abspath('general_perf/reports/' +
                                     self.backend_type)
        os.makedirs(output_dir, exist_ok=True)
        
        status = self.single_workload_perf(self.workload)

    def single_workload_perf(
            self, workload: Dict[str, Any]) -> bool:
        log.info("******************************************* Start to test model: {}. *******************************************".format(workload['model']))

        # Check Compile Only Mode
        self.compile_only_mode = False
        if self.args.compile_only or workload['compile_only']:
            self.compile_only_mode = True

        base_report = {
            "Model": workload['model'].upper(),
            "Backend": self.backend_type,
            "Host Info": self.get_cpu_name()
        }

        # Initalize Model Config Info
        model_info = self.get_model_info(workload['model'])
        pre_compile_config = {"workload": workload, 'model_info': model_info}
        interact_info = self.check_interact_info(pre_compile_config)
        pre_compile_config['interact_info'] = interact_info
        if not model_info['dataset_name']:
            model_info['dataset_name'] = 'fake_dataset'


        '''
        Compile Backend could do some optimization like convert model format here
        '''
        log.info("******************************************* Running Backend Compilation... *******************************************")
        log.info("Running Backend Preoptimization...")
        pre_compile_config = self.compile_backend.pre_optimize(pre_compile_config)


        # Initalize dataset
        dataset = load_dataset(model_info)
        dataset.preprocess()
        base_report['Dataset'] = model_info['dataset_name'].upper(
        ) if model_info['dataset_name'] else None

        #Placeholder Only
        segment_info = self.compile_backend.segment(pre_compile_config)

        best_batch_sizes = self.compile_backend.get_best_batch_size()
        if isinstance(best_batch_sizes, list):
            pre_compile_config['workload'][
                'batch_sizes'] = best_batch_sizes

        log.info("Start to compile the model...")
        start = time.time()
        compile_info = self.compile_backend.compile(pre_compile_config,
                                                    dataset)
        end = time.time()

        graph_compile_report = {}
        graph_compile_report["Compile Duration"] = round(end - start, 5)
        graph_compile_report["Compile Precision"] = compile_info[
            'compile_precision']
        graph_compile_report["Subgraph Coverage"] = compile_info['sg_percent']
        if 'optimizations' in compile_info:
            graph_compile_report['Optimizations'] = compile_info['optimizations']
        if 'instance_count' in compile_info:
            base_report['Instance Count'] = compile_info['instance_count']
        if 'device_count' in compile_info:
            base_report['Device Count'] = compile_info['device_count']
        base_report['Graph Compile'] = graph_compile_report

        # Initalize Output Dir and Reports
        output_dir = os.path.abspath('general_perf/reports/' +
                                     self.backend_type + '/' +
                                     workload['model'])
        os.makedirs(output_dir, exist_ok=True)

        # Compile only mode will stop here
        if self.compile_only_mode:
            base_report.pop("Backend")
            return compile_info["compile_status"], base_report

        base_report["Version"] = self.version
        base_report["Execution Date"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # load runtime backend
        """
        Start Here
        """
        batch_sizes = pre_compile_config['workload']['batch_sizes']
        self.runtime_backend.configs = compile_info
        self.runtime_backend.workload = workload
        self.runtime_backend.model_info = model_info

        self.runtime_backend.load(workload['batch_sizes'][0])
        # test accuracy
        accuracy_report = {}
        AccuracyChecker = self.get_accuracy_checker(
            model_info['dataset_name']
            if model_info['dataset_name'] else 'fake_dataset')
        AccuracyChecker.runtime_backend = self.runtime_backend
        AccuracyChecker.dataloader = dataset
        AccuracyChecker.output_dir = output_dir
        AccuracyChecker.configs = compile_info

        if workload['test_accuracy']:
            log.info("******************************************* Running Accuracy Checker... *******************************************")

            dataset.rebatch(self.runtime_backend.get_loaded_batch_size())
            accuracy_results = AccuracyChecker.calculate_acc(
                workload['data_percent'])

            accuracy_report['Data Percent'] = workload['data_percent']
            accuracy_report.update(accuracy_results)

        # test numeric
        if workload['test_numeric']:
            log.info("******************************************* Running Numeric Checker... *******************************************")

            dataset.rebatch(self.runtime_backend.get_loaded_batch_size())
            if not workload['test_accuracy']:
                accuracy_results = AccuracyChecker.calculate_acc(
                    workload['data_percent'])
            diff_results = AccuracyChecker.calculate_diff()
            accuracy_report.update(diff_results)
            accuracy_report['Diff Dist'] = compile_info['model'] + '-to-' + compile_info['compile_precision'].lower() + ".png"

        if accuracy_report:
            base_report['Accuracy'] = accuracy_report

        # function to test qps and latency
        if workload['test_perf']:
            log.info("******************************************* Runing QPS Checker... *******************************************")
            performance_reports = []
            qs_status = self.runtime_backend.is_qs_mode_supported()
            if qs_status:
                qs_config = self.runtime_backend.generate_qs_config()
                performance_reports = self.qs_benchmark(qs_config)
            else:
                for bs in batch_sizes:
                    self.runtime_backend.load(bs)
                    batch_reports = self.runtime_backend.benchmark(dataset)
                    performance_reports.append(batch_reports)
            base_report['Performance'] = performance_reports

        if "Instance Count" not in base_report:
            log.warning("Vendors need to Add # of instances")
        if "Device Count" not in base_report:
            log.warning("Vendors need to Add # of devices")

        # write output to json file
        output_report_path = output_dir + "/result-" + compile_info['compile_precision'].lower() + ".json"
        with open(output_report_path, 'w') as file:
            json.dump(base_report, file, indent=4)

        base_report.pop("Backend")
        log.info("Testing Finish. Report is saved in path: [ {}/{} ]".
                 format(output_dir[output_dir.rfind('general_perf'):],
                 os.path.basename(output_report_path)))
        build_pdf(output_report_path)
        log.info("PDF Version is saved in path: [ {}/{}-TO-{}.pdf ]".format(
            output_dir[output_dir.rfind('general_perf'):],
            base_report['Model'],
            output_report_path.split('/')[-1].split('-')[1].upper()))

        return compile_info["compile_status"]

    #WIP
    def qs_benchmark(self, qs_config: Dict[str, Any]) -> list:
        return []

    def get_accuracy_checker(self, dataset_name: str):
        AccuracyChecker = importlib.import_module('general_perf.datasets.' +
                                                  dataset_name +
                                                  ".test_accuracy")
        AccuracyChecker = getattr(AccuracyChecker, 'AccuracyChecker')
        return AccuracyChecker()

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        with open("general_perf/model_zoo/" + model_name + '.json',
                  'r') as file:
            model_info = json.load(file)
        return model_info

    def get_cpu_name(self):
        try:
            lscpu_output = subprocess.check_output(["lscpu"], text=True)
            for line in lscpu_output.split('\n'):
                if 'Model name' in line:
                    return line.split(':')[1].strip()
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {e}")
            return None

    def check_interact_info(
            self, pre_compile_config: Dict[str, Dict]) -> Dict[str, Any]:
        interact_info = self.compile_backend.get_interact_profile(
            pre_compile_config)

        answer = {}
        if len(interact_info) == 0:
            return answer

        dialog_style = Style.from_dict({
            'dialog': 'bg:#88b8ff',
            'dialog frame.label': 'bg:#ffffff #000000',
            'dialog.body': 'bg:#000000 #a0acde',
            'dialog shadow': 'bg:#004aaa',
        })

        input_style = Style.from_dict({
            'dialog': 'bg:#88b8ff',
            'dialog frame.label': 'bg:#ffffff #000000',
            'dialog.body': 'bg:#000000 #a0acde',
            'dialog shadow': 'bg:#004aaa',
            'text-area.prompt': 'bg:#ffffff',
            'text-area': '#000000',
        })

        option = yes_no_dialog(title=self.backend_type + '编译配置',
                               text='[请选择]：是否进行编译后端配置:',
                               style=dialog_style).run()
        if option:
            sum_question = len(interact_info)
            for i, question in enumerate(interact_info):
                if question['depends']:
                    state = 0
                    for title in question['depends'].split(','):
                        if not answer[title]:
                            state = 1
                    if state:
                        continue
                if question['dialog_type'] == 'Yes/No Dialog':
                    option = yes_no_dialog(
                        title=self.backend_type + '编译配置进度(' + str(i + 1) +
                        '/' + str(sum_question) + ')',
                        text="[Backend " + self.backend_type + "]: " +
                        question['note'],
                        style=dialog_style).run()
                elif question['dialog_type'] == "Input Dialog":
                    option = input_dialog(
                        title=self.backend_type + '编译配置进度(' + str(i + 1) +
                        '/' + str(sum_question) + ')',
                        text="[Backend " + self.backend_type + "]: " +
                        question['note'],
                        style=input_style).run()
                elif question['dialog_type'] == "Radiolist Dialog":
                    choice = [(i, text)
                              for i, text in enumerate(question['options'])]
                    num = radiolist_dialog(
                        title=self.backend_type + '编译配置进度(' + str(i + 1) +
                        '/' + str(sum_question) + ')',
                        text="[Backend " + self.backend_type + "]: " +
                        question['note'],
                        values=choice,
                        style=dialog_style).run()
                    option = question['options'][num] if num is not None else question[
                        'default']
                answer[question['name']] = option

        return answer

    def activate_venv(self, hardware_type: str) -> bool:
        if os.path.exists('general_perf/backends/' + hardware_type +
                          '/requirements.txt'):
            log.info("Activating Virtual Env for " + hardware_type)

            venv_dir = os.path.join("general_perf/backends",
                                    hardware_type + "/venv")
            activate_file = os.path.join(venv_dir, 'bin', 'activate_this.py')
            if not os.path.exists(venv_dir):
                log.info("venv not exist, Creating Virtual Env for " +
                         hardware_type)
                if (hardware_type == "HPU"):
                    virtualenv.create_environment(venv_dir,True)
                else:
                    virtualenv.create_environment(venv_dir)
                exec(open(activate_file).read(), {'__file__': activate_file})
                python_path = os.path.join(venv_dir, 'bin', 'python3')
                subprocess.call([
                    python_path, '-m', 'pip', 'install', '--upgrade', 'pip', '--quiet'
                ])
                subprocess.call([
                    python_path, '-m', 'pip', 'install', '-r', 'general_perf/backends/' +
                    hardware_type + '/requirements.txt', '-q'
                ])
            else:
                exec(open(activate_file).read(), {'__file__': activate_file})
                '''
                just in case install failed in pre-run.
                '''
                python_path = os.path.join(venv_dir, 'bin', 'python3')
                subprocess.call([
                    python_path, '-m', 'pip', 'install', '--upgrade', 'pip', '--quiet'
                ])
                subprocess.call([
                    python_path, '-m', 'pip', 'install', '-r', 'general_perf/backends/' +
                    hardware_type + '/requirements.txt', '-q'
                ])

                if not hasattr(sys, 'real_prefix'):
                    return False
                return True
        return True

    def deactivate_venv(self):
        sys.path[:
                 0] = self.prev_sys_path  #will also revert the added site-packages
        sys.prefix = self.real_prefix
        os.environ['PATH'] = self.old_os_path


if __name__ == "__main__":
    engine = PerfEngine()
    engine.start_engine()
