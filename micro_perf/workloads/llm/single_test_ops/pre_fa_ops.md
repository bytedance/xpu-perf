## pre_fa_ops

### rotary_embedding
输入为`packed_qkv`形式，shape为`[num_tokens, (q_head_num + 2 * kv_head_num] * head_dim)`

其中`num_tokens`由多个`q_len`构成，由`q_lens`描述, prefill情况下q_len可能不均等。


### store_kv_cache
输入为`packed_qkv`形式，shape为`[num_tokens, (q_head_num + 2 * kv_head_num] * head_dim)`

其中`num_tokens`由多个`q_len`构成，由`q_lens`描述, decode情况下q_len一般均等。

目前假设 `cache_dtype = int8`时采用静态的`per_channel`量化，对应的`scale_shape=[kv_head_num, head_dim]`。