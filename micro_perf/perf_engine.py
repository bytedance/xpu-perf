import sys
import pathlib
from typing import Set

FILE_DIR = pathlib.Path(__file__).parent.absolute()
BYTE_MLPERF_ROOT = FILE_DIR
BACKENDS_DIR = BYTE_MLPERF_ROOT.joinpath("backends")
sys.path.insert(0, str(BYTE_MLPERF_ROOT))


from core.ops import OP_ENGINE_MAPPING
from core.engine import ComputeEngine, XCCLEngine
from core.utils import logger


ENGINE_TYPE_MAPPING = {
    "ComputeEngine": ComputeEngine,
    "XCCLEngine": XCCLEngine
}


class XpuPerfServer:
    def __init__(self, args_dict, required_engines: Set[str] = set()):
        self.args_dict = args_dict
        # create internal engines
        self.backend_instance = args_dict["backend_instance"]
        self.node_world_size = args_dict.get("node_world_size", 1)
        self.node_rank = args_dict.get("node_rank", 0)
        self.device_ids = args_dict["device_ids"]
            
        self.started_engines = {}
        self.required_engines = required_engines if required_engines else list(ENGINE_TYPE_MAPPING.keys())

        for engine_name in self.required_engines:
            if engine_name == "XCCLEngine" and len(self.device_ids) * self.node_world_size <= 1:
                continue
            self.started_engines[engine_name] = ENGINE_TYPE_MAPPING[engine_name](args_dict)
    
    def __del__(self):
        self.destroy()


    def __enter__(self):
        self.create()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.destroy()


    def create(self):
        self.destroy()        
        for engine_name, engine_instance in self.started_engines.items():
            engine_instance.start()

    def destroy(self):
        for engine_name, engine_instance in self.started_engines.items():
            engine_instance.stop()
        self.backend_instance.clean_extra_files()


    def get_info(self):
        info_dict = {
            "backend_type": self.backend_instance.backend_type, 
            "common": self.backend_instance.common_info, 
            "provider": self.backend_instance.provider_info, 
            "backend": self.backend_instance.backend_info, 
            "runtime": {
                "device_mapping": self.args_dict["device_mapping"], 
                "device_ids": self.args_dict["device_ids"], 
                "numa_num": self.args_dict["numa_num"], 
                "numa_order": self.args_dict["numa_order"], 
                "node_world_size": self.args_dict["node_world_size"], 
                "node_rank": self.args_dict["node_rank"], 
                "all_numa_num": self.args_dict["all_numa_num"], 
            }
        }
        return info_dict


    def bench(self, input_dict):
        result_dict = {}

        """
        通信协议:
        {
            "type": "normal", 
            "data": {
                "flash_attention": [], 
                "gemm": []
            }
        }
        """
        try:
            input_type = input_dict["type"]
            data_dict = input_dict["data"]
            if input_type == "normal":
                result_dict = self.normal_bench(data_dict)
        except Exception as e:
            print(e)

        return result_dict


    def normal_bench(self, test_cases):
        """
        {
            "flash_attention": [
                common_case_0
                common_case_1
            ], 
            "gemm": [
                common_case_0
                common_case_1
            ]
        }
        - key是op_name, value是列表, 包含多个待测试的公共配置参数
        - 每一个op在特定backend上可能有多个provider实现
        - 每一个op在特定provider实现上可能有多个实现方式, 比如高精、低精, 不同数据类型等,
        而这个参数是provider/backend特定的, 无法表示在workload上。
        """


        # 将输入测试用例分发到不同的engine上
        engine_tasks = {}
        # 对等地创建results
        all_results = {}

        for op_name, cases in test_cases.items():
            # 如果op_name不被当前框架支持, 创建空结果
            if op_name not in OP_ENGINE_MAPPING:
                logger.error(f"op_name: {op_name} not in OP_ENGINE_MAPPING")
                all_results[op_name] = [{} for _ in cases]
                continue
            engine_name = OP_ENGINE_MAPPING[op_name]
            engine_tasks[engine_name] = engine_tasks.get(engine_name, {})
            engine_tasks[engine_name][op_name] = cases
            

        for engine_name, engine_test_cases in engine_tasks.items():
            # 如果engine_name没有在当前测试启动, 创建空结果
            if engine_name not in self.started_engines:
                logger.error(f"engine_name: {engine_name} not in self.started_engines")
                for op_name in engine_test_cases:
                    all_results[op_name] = [{} for _ in engine_test_cases[op_name]]
                continue

            num_ops = len(engine_test_cases)
            num_cases = sum([len(cases) for cases in engine_test_cases.values()])

            # 直接dispatch到对应的engine上
            logger.info(f"dispatch {num_ops} ops, {num_cases} cases to {engine_name}")
            cur_results = self.started_engines[engine_name].dispatch(engine_test_cases)
            all_results.update(cur_results)
            
        return all_results
        
