import argparse
import logging
import os
import importlib
import json
import sys
BYTE_MLPERF_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(BYTE_MLPERF_ROOT)
sys.path.insert(0, BYTE_MLPERF_ROOT)

from general_perf.core.configs.workload_store import load_workload
from general_perf.core.configs.dataset_store import load_dataset
from general_perf.core.configs.backend_store import init_compile_backend, init_runtime_backend

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("CPUBase")


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='resnet50-tf-fp32')
    parser.add_argument("--hardware_type", default="CPU")
    parser.add_argument("--batch_size",
                        type=int,
                        help="Batch sizes we will test in performace mode")
    parser.add_argument(
        "--data_percent",
        type=int,
        help=
        "Data percent we will used in the whole data set when we will test in accuracy mode"
    )
    args = parser.parse_args()
    return args


class PerfEngine(object):
    def __init__(self) -> None:
        super().__init__()
        self.args = get_args()
        self.workload = load_workload(self.args.task)
        self.backend_type = self.args.hardware_type

    def start_engine(self):
        '''
        Byte MlPerf will create an virtual env for each backend to avoid dependance conflict
        '''
        log.info("Runing CPU Base...")

        self.compile_backend = init_compile_backend(self.args.hardware_type)
        self.runtime_backend = init_runtime_backend(self.args.hardware_type)
        if self.workload:
            return self.workload_perf(self.workload)

    def workload_perf(self, workload):
        # set reports dir
        output_dir = os.path.abspath('general_perf/reports/' + self.args.hardware_type +
                                     '/' + workload['model'])
        os.makedirs(output_dir, exist_ok=True)

        model_info = self.get_model_info(workload['model'])

        ds = load_dataset(model_info)
        ds.preprocess()

        compile_info = self.compile_backend.compile({
            "workload": workload,
            'model_info': model_info
        })

        # load runtime backend
        runtime_backend = self.runtime_backend
        runtime_backend.configs = compile_info
        runtime_backend.workload = workload
        runtime_backend.model_info = model_info
        runtime_backend.load(workload['batch_sizes'][0])
        # test accuracy
        if workload['test_accuracy'] or workload['test_numeric']:
            ds.rebatch(self.args.batch_size)
            AccuracyChecker = self.get_accuracy_checker(
                model_info['dataset_name']
                if model_info['dataset_name'] else 'fake_dataset')
            AccuracyChecker.runtime_backend = runtime_backend
            AccuracyChecker.dataloader = ds
            AccuracyChecker.output_dir = output_dir
            AccuracyChecker.configs = compile_info
            AccuracyChecker.calculate_acc(workload['data_percent'])

        return

    def get_accuracy_checker(self, dataset_name: str):
        AccuracyChecker = importlib.import_module('general_perf.datasets.' +
                                                  dataset_name +
                                                  ".test_accuracy")
        AccuracyChecker = getattr(AccuracyChecker, 'AccuracyChecker')
        return AccuracyChecker()

    def get_model_info(self, model_name: str):
        with open("general_perf/model_zoo/" + model_name + '.json', 'r') as f:
            model_info = json.load(f)
        return model_info


if __name__ == "__main__":
    engine = PerfEngine()
    engine.start_engine()
