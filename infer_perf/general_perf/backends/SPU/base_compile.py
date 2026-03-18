from torch.utils.data import DataLoader as DataLoaderX
from dataset.dataset import ImageNetDataset,MZJBertDataset,DummyDataset
from nn_compiler.common.constants import OpType
from common_compile import SparsertBaseBuilder
import onnx

def get_onnx_input_info(onnx_model_path):
    # Load ONNX model
    model = onnx.load(onnx_model_path)

    # Initialize an empty dictionary to store input names and shapes
    input_info = {}

    # Iterate through the inputs of the model
    for input in model.graph.input:
        input_name = input.name
        input_shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        input_info[input_name] = input_shape

    return input_info

def get_model_input_info(onnx_input_info,batch_size):
    config_input_dict = {}
    input_shape_dict = {}
    for input_name,input_shape in onnx_input_info.items():
        config_input_dict[input_name] = input_name
        input_shape[0] = batch_size
        input_shape_dict[input_name] = input_shape
    return config_input_dict,input_shape_dict

class Resnet50Builder(SparsertBaseBuilder):
    def __init__(self, onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs):
        super(Resnet50Builder, self).__init__(
            onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs)

    def set_dataset_config(self):
        # calibration dataset config
        dataset = ImageNetDataset(self.dataset_dir, transform_file=self.dataset_cfg)
        self.config.dataloader = DataLoaderX(dataset, batch_size=self.batch_size)
        self.config.calib_batch = 1

        # model inputs info
        onnx_input_info = get_onnx_input_info(self.onnx_path)
        self.config.input_dict,self.input_shape_dict = get_model_input_info(onnx_input_info,self.batch_size)

        
        # you can also set other configs here
        self.config.do_kl = True
        self.config.opt_level = 8
        self.config.total_cores = 1


class BertBaseBuilder(SparsertBaseBuilder):
    def __init__(self, onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs):
        super(BertBaseBuilder, self).__init__(
            onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs)

    def set_dataset_config(self):
        # model inputs info
        onnx_input_info = get_onnx_input_info(self.onnx_path)
        self.config.input_dict,self.input_shape_dict = get_model_input_info(onnx_input_info,self.batch_size)

        # calibration dataset config
        dataset = MZJBertDataset(data_path=self.dataset_dir, input_info=self.config.input_dict)
        self.config.dataloader = DataLoaderX(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.config.calib_batch = 1

        # you can also set other configs here
        self.config.do_kl = False
        self.config.opt_level = 5
        self.config.safe_exp = False
        self.config.quantized_patterns = [[OpType.BatchMatmul]]


class AlbertBuilder(SparsertBaseBuilder):
    def __init__(self, onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs):
        super(AlbertBuilder, self).__init__(
            onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs)

    def set_dataset_config(self):
        # model inputs info
        onnx_input_info = get_onnx_input_info(self.onnx_path)
        self.config.input_dict,self.input_shape_dict = get_model_input_info(onnx_input_info,self.batch_size)

        # calibration dataset config
        dataset = MZJBertDataset(data_path=self.dataset_dir, input_info=self.config.input_dict)
        self.config.dataloader = DataLoaderX(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.config.calib_batch = 1

        # you can also set other configs here
        self.config.do_kl = False
        self.config.opt_level = 5
        self.config.safe_exp = False
        self.config.quantized_patterns = [[OpType.BatchMatmul]]


class RobertaBuilder(SparsertBaseBuilder):
    def __init__(self, onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs):
        super(RobertaBuilder, self).__init__(
            onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs)

    def set_dataset_config(self):
        # model inputs info
        onnx_input_info = get_onnx_input_info(self.onnx_path)
        self.config.input_dict,self.input_shape_dict = get_model_input_info(onnx_input_info,self.batch_size)

        # calibration dataset config
        dataset = MZJBertDataset(data_path=self.dataset_dir, input_info=self.config.input_dict)
        self.config.dataloader = DataLoaderX(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.config.calib_batch = 1

        # you can also set other configs here
        self.config.do_kl = False
        self.config.opt_level = 5
        self.config.safe_exp = False
        self.config.quantized_patterns = [[OpType.BatchMatmul]]

class ConformerBuilder(SparsertBaseBuilder):
    def __init__(self, onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs):
        super(ConformerBuilder, self).__init__(
            onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs)

    def set_dataset_config(self):
        # model inputs info
        onnx_input_info = get_onnx_input_info(self.onnx_path)
        self.config.input_dict,self.input_shape_dict = get_model_input_info(onnx_input_info,self.batch_size)

        # calibration dataset config
        dataset = DummyDataset(input_info=self.config.input_dict)
        self.config.dataloader = DataLoaderX(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.config.calib_batch = 1

        # you can also set other configs here
        self.config.do_kl = False
        self.config.opt_level = 5
        self.config.safe_exp = False
        self.config.quantized_patterns = [[OpType.BatchMatmul]]

class GeneralBuilder(SparsertBaseBuilder):
    def __init__(self, onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs):
        super(GeneralBuilder, self).__init__(
            onnx_path, dump_dir, dataset_dir, dataset_cfg, dtype, batch_size, verify, **kwargs)

    def set_dataset_config(self):
        # model inputs info
        onnx_input_info = get_onnx_input_info(self.onnx_path)
        self.config.input_dict,self.input_shape_dict = get_model_input_info(onnx_input_info,self.batch_size)

        # calibration dataset config
        dataset = DummyDataset(input_info=self.config.input_dict)
        self.config.dataloader = DataLoaderX(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.config.calib_batch = 1

        # you can also set other configs here
        self.config.do_kl = False
        self.config.opt_level = 5
        self.config.safe_exp = False
        self.config.quantized_patterns = [[OpType.BatchMatmul]]
