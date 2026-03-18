# XPU_SIM

## 快速上手

### 1. 启动`micro_perf`测试服务器

```
cd micro_perf
python3 ./launch.py --backend GPU
```

以上命令有以下操作：

1. 创建一个bench服务，用于在一个正确的环境提供验证后的算子性能在线测试服务。
2. 指定当前bench服务的backend为GPU，维护所有当前可以使用的device的状态。
3. 启动计算引擎和通信引擎，分别准备对应的算子测试环境。
4. 加载GPU backend的所有已支持的算子，包括基础实现和厂商自定义实现。
5. 随时接收客户端发来的测试请求。


### 2. 启动`xpu_sim`进行实测

```
cd xpu_sim
python3 ./endpoint.py --model ./model_zoo/qwen3_moe/qwen3-235b-a22b/deploys/sp_tp_ep.json --batch_size 1 --cache_len 0 --q_len 10240 --block_size 512
```

以上命令有以下操作：

1. 根据指定模型（qwen3-235b-a22b）、指定并行方式（SP8-TP8-EP8）、指定数据类型（gemm W8A8 + fa BF16_C8）以及对应的breakdown模板创建所有需要测试的算子和算子参数清单。
2. 指定当前测试内容：
    - 运行模式：prefill
    - 测试输入：均衡的固定输入，`batch_size=1, cache_len=0, q_len=10240`
    - kv_cache的配置：`blcok_size=512`
3. 将测试内容和算子、算子参数清单结合整理成测试需求发送给bench服务，得到每个具体算子测试要求的性能结果，再通过指定算子调用顺序描述将性能组装回来，从而得到最终的模型端到端性能。


参考输出如下，更多细节可以实测获取。
```bash
+-------------------------+----------------+----------+-------------+---------+----------+---------+
| instance_name           | instance_index | provider | latency(us) | tflops  | mem_bw   | comm_bw |
+-------------------------+----------------+----------+-------------+---------+----------+---------+
| add_rms_norm_0          | 0              | xpu_ops  | xxxxxxx     |         | xxxxxxx  |         |
| qkv_gemm                | 1              | xpu_ops  | xxxxxxx     | xxxxxxx | xxxxxxx  |         |
| all_to_all_0            | 2              | torch    | xxxxxxx     |         |          | xxxxxxx |
| qk_norm                 | 3              | xpu_ops  | xxxxxxx     |         | xxxxxxxx |         |
| rotary_embedding        | 4              | xpu_ops  | xxxxxx      |         | xxxxxxxx |         |
| store_kv_cache          | 5              | xpu_ops  | xxxxxx      |         | xxxxxxx  |         |
| flash_attention         | 6              | xpu_ops  | xxxxxxxx    | xxxxxx  | xxxxxx   |         |
| attn_out_quant          | 7              | xpu_ops  | xxxxxx      |         | xxxxxx   |         |
| all_to_all_1            | 8              | torch    | xxxxxxx     |         |          | xxxxx   |
| all_to_all_2            | 9              | torch    | xxxxxx      |         |          | xxxxx   |
| attn_out_gemm           | 10             | xpu_ops  | xxxxxxx     | xxxxxx  | xxxxxx   |         |
| pre_moe_norm            | 11             | xpu_ops  | xxxxxx      |         | xxxxxxx  |         |
| qwen3_moe_ag0           | 12             | torch    | xxxxxxx     |         |          | xxxxxxx |
| qwen3_moe_gating        | 13             | torch    | xxxxx       | xxxxxx  | xxxxxx   |         |
| qwen3_moe_softmax_topk  | 14             | xpu_ops  | xxxxxx      |         | xxxxx    |         |
| qwen3_moe_ag1           | 15             | torch    | xxxxxx      |         |          | xxxxx   |
| qwen3_moe_ag1           | 16             | torch    | xxxxxx      |         |          | xxxxx   |
| qwen3_moe_scatter       | 17             | xpu_ops  | xxxxxxx     |         | xxxxxxx  |         |
| qwen3_moe_moe_up_gemm   | 18             | xpu_ops  | xxxxxxx     | xxxxxx  | xxxxxx   |         |
| qwen3_moe_moe_swiglu    | 19             | xpu_ops  | xxxxxx      |         | xxxxxxxx |         |
| qwen3_moe_moe_down_gemm | 20             | xpu_ops  | xxxxxxx     | xxxxxx  | xxxxxx   |         |
+-------------------------+----------------+----------+-------------+---------+----------+---------+
+---------------------------+----------------------------------------------------+
| key                       | value                                              |
+---------------------------+----------------------------------------------------+
| batch_size                | 1                                                  |
| cache_len                 | 0                                                  |
| q_len                     | 10240                                              |
| run_mode                  | prefill                                            |
| block_size                | 512                                                |
| ------------------------- | -------------------------------------------------- |
| total_latency             | xxx.xxx ms                                         |
+---------------------------+----------------------------------------------------+
```

