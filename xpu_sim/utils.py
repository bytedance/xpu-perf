import importlib.util
import sys
from pathlib import Path


def load_dir_as_module(dir_path, module_name):
    """
    将指定文件夹动态加载为 Python 模块
    :param dir_path: 文件夹的路径 (字符串或 Path 对象)
    :param module_name: 你希望在 Python 中调用该模块的名字
    """
    # 1. 统一转为绝对路径，避免相对路径带来的困扰
    folder = Path(dir_path).resolve()
    init_file = folder / "__init__.py"
    
    if not init_file.exists():
        raise FileNotFoundError(f"在 {folder} 中找不到 __init__.py，无法作为包加载")

    # 2. 告诉 Python 这个模块的物理位置和加载方式 (Spec)
    spec = importlib.util.spec_from_file_location(module_name, str(init_file))
    
    # 3. 创建一个新的模块对象
    module = importlib.util.module_from_spec(spec)
    
    # 4. 【关键步骤】将模块注入缓存，确保模块内部的相对导入（.import）能正常工作
    sys.modules[module_name] = module
    
    # 5. 执行模块内容（如果不执行，模块只是个空壳）
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        # 如果初始化失败，清理缓存防止后续引用报错
        del sys.modules[module_name]
        raise e

    return module



def get_func_from_file(file_path, func_name):
    file_path = Path(file_path).resolve()
    module_name = file_path.stem  # 获取文件名作为模块名
    
    # 创建 Spec 和 Module
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    
    # 执行模块代码（必须执行，否则函数还没被定义）
    spec.loader.exec_module(module)
    
    # 获取并返回函数
    return getattr(module, func_name, None)