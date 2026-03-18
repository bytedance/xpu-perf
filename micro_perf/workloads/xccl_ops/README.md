## ccl_ops

### [all_reduce](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)
```python
torch.distributed.all_reduce(tensor, op=<RedOpType.SUM: 0>, group=None, async_op=False)
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size, dim_size] |



### [reduce_scatter](https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter_tensor)
```python
torch.distributed.reduce_scatter_tensor(output, input, op=<RedOpType.SUM: 0>, group=None, async_op=False)[source]
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size // world_size, dim_size] |



### [all_gather](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_into_tensor)
```python
torch.distributed.all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False)
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size // world_size, dim_size] |
| output_tensor | [batch_size, dim_size] |


### [all_to_all](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single)
```python
torch.distributed.all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False)[source]
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size, dim_size] |



### [p2p](https://pytorch.org/docs/stable/distributed.html#torch.distributed.isend)
```python
torch.distributed.isend(tensor, dst, tag=0, group=None, async_op=False)
torch.distributed.irecv(tensor, src, tag=0, group=None, async_op=False)
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size, dim_size] |

Given a set of devices, we perform pairwise bandwidth tests between every two devices to evaluate the link bandwidth performance of the entire system topology. The test configuration is fixed with a tensor shape of [1024, 2097152], a data type of int8, and a total data volume of 2 GiB.






