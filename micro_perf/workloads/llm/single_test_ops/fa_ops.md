## fa_ops

快速验证 `flash_attention` 在`prefill`和`decode`下的不同数据类型的性能。

数据类型包括：
- dtype: 输入`q`的数据类型
- cache_dtype: 输入`k_cache`和`v_cache`的数据类型
- qk_compute_dtype: 计算`q@k`的数据类型
- pv_compute_dtype: 计算`p@v`的数据类型
- dst_dtype: 输出`o`的数据类型


prefill的常用输入包括:
- batch_size = 1, cache_len = 0, q_len = 10240
- batch_size = 1, cache_len = 5120, q_len = 5120
- batch_size = 1, cache_len = 0, q_len = 32768

decode的常用输入包括:
- batch_size = 16, cache_len = 10240, q_len = 1
- batch_size = 16, cache_len = 10240, q_len = 4

