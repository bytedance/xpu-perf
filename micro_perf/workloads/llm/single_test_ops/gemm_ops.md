## gemm_ops

快速验证 `gemm` 类算子的性能

数据类型包括：
- dtype: 输入左矩阵的数据类型
- w_dtype: 输入右矩阵的数据类型
- compute_dtype: 计算`c=a@b`的数据类型
- dst_dtype: 输出矩阵的数据类型
