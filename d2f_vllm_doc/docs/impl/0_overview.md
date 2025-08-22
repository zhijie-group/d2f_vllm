# Nano-vLLM to D2F-vLLM

![D2F-vLLM](../assets/imgs/impl/0_overview/overview_of_d2f_vllm.png)

## Current Benchmark

### Configuration

| Parameter              | Value |
|------------------------|-------|
| enforce_eager          | True  |
| data_parallel_size     | 8     |
| tensor_parallel_size   | 1     |
| gpu_memory_utilization | 0.3   |
| max_num_batched_tokens | 5120  |
| max_num_seqs           | 20    |
| max_model_len          | 5120  |

### Dream

#### HumanEval

| Metric             | Value                 |
|--------------------|-----------------------|
| total_samples      | 164                   |
| total_tokens       | 36198                 |
| total_time         | 49.27073669433594     |
| TPS                | 734.6754367519178     |
| AVG Latency        | 0.300431321306926     |

#### GSM8K-CoT

| Metric             | Value                 |
|--------------------|-----------------------|
| total_samples      | 1319                  |
| total_tokens       | 336429                |
| total_time         | 380.78655076026917    |
| TPS                | 883.5107209755545     |
| AVG Latency        | 0.2886933667629031    |