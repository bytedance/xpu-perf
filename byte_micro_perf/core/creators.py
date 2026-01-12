import sys
import pathlib
import importlib
import traceback

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[1])
)


# def create_backend_instance(backend_type: str):    
#     backend_module = importlib.import_module(
#         "backends." + backend_type + ".backend_" + backend_type.lower())
#     backend_cls = getattr(backend_module, "Backend" + backend_type)

#     backend_instance = backend_cls()
#     backend_instance.backend_type = backend_type
#     backend_instance.backend_cls = backend_cls
#     backend_instance.torch_device_name = backend_instance.get_torch_device_name()
#     backend_instance.device_name = backend_instance.get_device_name(0)
#     backend_instance.device_count, backend_instance.avail_devices = backend_instance.get_device_count()
#     backend_instance.env_dict = backend_instance.get_backend_info()
#     return backend_instance



# def create_op_instance(
#     op_cls, args_dict, backend_instance, 
#     op_group=None, group_size=1
# ):
#     op_instance = op_cls(
#         args_dict, backend_instance, 
#         op_group=op_group, group_size=group_size
#     )
#     return op_instance









def get_op_info(
    backend_type: str, 
    op_type: str
):
    if op_type in OP_INFO_MAPPING:
        if "op_mapping" not in OP_INFO_MAPPING[op_type] or backend_type not in OP_INFO_MAPPING[op_type]["op_mapping"]:
            OP_INFO_MAPPING[op_type]["op_mapping"] = {}
            try:
                backend_ops = importlib.import_module(f"backends.{backend_type}.ops.{op_type}")
                OP_INFO_MAPPING[op_type]["op_mapping"][backend_type] = getattr(backend_ops, "OP_MAPPING")
            except:
                traceback.print_exc()
                OP_INFO_MAPPING[op_type]["op_mapping"][backend_type] = []
            return OP_INFO_MAPPING[op_type]["default_engine"], OP_INFO_MAPPING[op_type]["op_mapping"][backend_type]
    else:
        return "ComputeEngine", []


