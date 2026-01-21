

# 说明
对标阿里和腾讯的对deepseek-r1的推理技术的评测。

环境依赖：`sglang==0.5.7` **请确保使用该环境进行评测！！！**

## 阿里评测
```shell
python3 main.py --model /nfs/FM/gongoubo/checkpoints/Qwen/Qwen3-30B-A3B-Instruct-2507 --base_url http://192.168.16.21:18000 --eval_type decode --eval_method ali --decode_gpu_num 1 --deploy_log /nfs/FM/gongoubo/new_project/workflow/deepseek_bench/logs/test.log --output_path /nfs/FM/gongoubo/new_project/workflow/deepseek_bench/output/ali_decode.json

python3 main.py --model /nfs/FM/gongoubo/checkpoints/Qwen/Qwen3-30B-A3B-Instruct-2507 --base_url http://192.168.16.21:18000 --eval_type prefill --eval_method ali --prefill_gpu_num 1 --output_path /nfs/FM/gongoubo/new_project/workflow/deepseek_bench/output/ali_prefill.json
```
- --xxx_gpu_num：部署所用的GPU显卡数目
- --deploy_log：部署服务需要使用nohup将日志记录到log里面，多个log用英文`,`隔开。
- --output_path：结果保存的路径。

需要下载ShareGPT_V3_unfiltered_cleaned_split.json，然后指定--dataset_path。

注意：先进行deocde的评测，再进行prefill的评测。

decode结果：
```json
[
    {
        "1": 7.06,
        "2": 9.37,
        "4": 13.57,
        "8": 18.81,
        "12": 22.86,
        "16": 25.78,
        "20": 27.04,
        "24": 26.02,
        "28": 26.12,
        "32": 26.54,
        "40": 27.04,
        "48": 26.63,
        "56": 29.17,
        "64": 28.52
    },
    {
        "1": {
            "median": 141.67,
            "throughput_per_gpu": 141.67,
            "throughput_per_user": 141.67,
            "gen_througput_samples": 419
        },
        "2": {
            "median": 215.565,
            "throughput_per_gpu": 215.565,
            "throughput_per_user": 107.7825,
            "gen_througput_samples": 186
        },
        "4": {
            "median": 298.38,
            "throughput_per_gpu": 298.38,
            "throughput_per_user": 74.595,
            "gen_througput_samples": 128
        },
        "8": {
            "median": 432.87,
            "throughput_per_gpu": 432.87,
            "throughput_per_user": 54.10875,
            "gen_througput_samples": 124
        },
        "12": {
            "median": 528.825,
            "throughput_per_gpu": 528.825,
            "throughput_per_user": 44.06875,
            "gen_througput_samples": 90
        },
        "16": {
            "median": 623.81,
            "throughput_per_gpu": 623.81,
            "throughput_per_user": 38.988125,
            "gen_througput_samples": 76
        },
        "18": {
            "median": 660.02,
            "throughput_per_gpu": 660.02,
            "throughput_per_user": 36.66777777777778,
            "gen_througput_samples": 564
        },
        "17": {
            "median": 598.905,
            "throughput_per_gpu": 598.905,
            "throughput_per_user": 35.22970588235294,
            "gen_througput_samples": 212
        },
        "3": {
            "median": 235.79,
            "throughput_per_gpu": 235.79,
            "throughput_per_user": 78.59666666666666,
            "gen_througput_samples": 66
        },
        "7": {
            "median": 439.12,
            "throughput_per_gpu": 439.12,
            "throughput_per_user": 62.73142857142857,
            "gen_througput_samples": 16
        },
        "6": {
            "median": 357.59,
            "throughput_per_gpu": 357.59,
            "throughput_per_user": 59.59833333333333,
            "gen_througput_samples": 74
        },
        "11": {
            "median": 521.0999999999999,
            "throughput_per_gpu": 521.0999999999999,
            "throughput_per_user": 47.37272727272727,
            "gen_througput_samples": 66
        },
        "10": {
            "median": 491.53,
            "throughput_per_gpu": 491.53,
            "throughput_per_user": 49.153,
            "gen_througput_samples": 72
        },
        "15": {
            "median": 688.38,
            "throughput_per_gpu": 688.38,
            "throughput_per_user": 45.892,
            "gen_througput_samples": 16
        },
        "14": {
            "median": 621.69,
            "throughput_per_gpu": 621.69,
            "throughput_per_user": 44.40642857142858,
            "gen_througput_samples": 74
        },
        "19": {
            "median": 782.62,
            "throughput_per_gpu": 782.62,
            "throughput_per_user": 41.19052631578948,
            "gen_througput_samples": 292
        },
        "5": {
            "median": 291.71500000000003,
            "throughput_per_gpu": 291.71500000000003,
            "throughput_per_user": 58.343,
            "gen_througput_samples": 14
        },
        "13": {
            "median": 546.14,
            "throughput_per_gpu": 546.14,
            "throughput_per_user": 42.01076923076923,
            "gen_througput_samples": 14
        },
        "9": {
            "median": 450.96,
            "throughput_per_gpu": 450.96,
            "throughput_per_user": 50.10666666666666,
            "gen_througput_samples": 2
        }
    }
]
```
- 第一个字典是batch_size对应的TPOP（ms）。
- 第二个字典是throughput_per_gpu和throughput_per_user对应关系（tokens/s）。

prefill结果：
```json
[
    {
        "1024": 25574.94,
        "2048": 24486.29,
        "4096": 22403.85
    }
]
```
- 在1k,2k,4k设置下的Input Throughput Per GPU (tokens/s)。

## 腾讯评测

**核心原则：在限制TPOT为小于等于50ms的情况下，最大化total token/s以及QPM**

**ttft为什么耗时那么长？**

服务端的容量是有限的，在prefill和decode的时候，都会有请求进行等待。

评测脚本：

```shell
python3 main.py --model /mnt/md0/checkpoints/Deepseek-R1 --base_url http://0.0.0.0:8000  --eval_method tencent --output_path /mnt/md0/deepseek_bench/output/deepseek_r1_fp8_tencent.json --dataset_path /root/gongoubo/ShareGPT_V3_unfiltered_cleaned_split.json --batch_size 128 --request_rate 128 --max_concurrency 128

```

说明：

- 输入被限制为3500，输出被限制为1200

以下是一个样例和评测结果：

部署脚本：

```shell
python3 -m sglang.launch_server \
    --model-path /mnt/md0/checkpoints/Deepseek-R1 \
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --mem-fraction-static 0.9 \
    --cuda-graph-max-bs 128 \
    --max-running-requests 128 \
    --attention-backend flashinfer 
```

评测脚本：

```shell
python3 main.py --model /mnt/md0/checkpoints/Deepseek-R1 --base_url http://0.0.0.0:8000  --eval_method tencent --output_path /mnt/md0/deepseek_bench/output/deepseek_r1_fp8_tencent.json --dataset_path /root/gongoubo/ShareGPT_V3_unfiltered_cleaned_split.json --batch_size 128 --request_rate 128 --max_concurrency 128

```

评测结果：

```shell
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    128.0     
Max request concurrency:                 128       
Successful requests:                     128       
Benchmark duration (s):                  332.81    
Total input tokens:                      448000    
Total input text tokens:                 448000    
Total input vision tokens:               0         
Total generated tokens:                  153600    
Total generated tokens (retokenized):    153232    
Request throughput (req/s):              0.38      
Input token throughput (tok/s):          1346.12   
Output token throughput (tok/s):         461.53    
Peak output token throughput (tok/s):    585.00    
Peak concurrent requests:                128       
Total token throughput (tok/s):          1807.65   
Concurrency:                             72.22     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   187775.43 
Median E2E Latency (ms):                 187361.54 
---------------Time to First Token----------------
Mean TTFT (ms):                          144639.27 
Median TTFT (ms):                        151860.76 
P99 TTFT (ms):                           302543.20 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          35.98     
Median TPOT (ms):                        28.90     
P99 TPOT (ms):                           216.36    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           35.98     
Median ITL (ms):                         27.18     
P95 ITL (ms):                            28.74     
P99 ITL (ms):                            29.52     
Max ITL (ms):                            272786.87 
==================================================

====================================================================================================
final_metrics: [{'tpot (ms)': '35.98', 'ttft (ms)': '144639.27', 'input_throughput (tok/s)': '1346.12', 'output_throughput (tok/s)': '461.53', 'total_throughput (tok/s)': '1807.65', 'QPM (req/min)': 22.8}]

```

| batchsize/TPOT(ms)  | attention后端 | num-prompts | request-rate | max-concurrency | QPM  | Input token  throughput (tok/s) | Output token  throughput (tok/s) | Total token  throughput (tok/s) | TTFT(ms)  | TPOT(ms) | 备注              |
| ------------------- | ------------- | ----------- | ------------ | --------------- | ---- | ------------------------------- | -------------------------------- | ------------------------------- | --------- | -------- | ----------------- |
| deepseek-r1-wi4afp8 | flashinfer    | 128         | 128          | 128             | 42   | 2449.99                         | 840                              | 3289.99                         | 39079.24  | 119.49   | sgl-0.4.6-rebuild |
| deepseek-r1-fp8     | flashinfer    | 128         | 128          | 128             | 14.4 | 854.96                          | 293.13                           | 1148.09                         | 234334.82 | 54.18    | sgl-0.4.6-rebuild |
| deepseek-r1-fp8     | flashinfer    | 128         | 128          | 128             | 15   | 873.44                          | 299.47                           | 1172.91                         | 226104.28 | 48.61    | sgl-0.4.6         |
| deepseek-r1-fp8     | flashinfer    | 128         | 128          | 128             | 22.8 | 1346.64                         | 461.71                           | 1808.35                         | 144683.1  | 35.95    | sgl-0.5.7         |
| deepseek-r1-fp8     | flashinfer    | 256         | 128          | 128             | 23.4 | 1359.02                         | 465.95                           | 1824.97                         | 210335.13 | 41.46    | sgl-0.5.7         |
| deepseek-r1-fp8     | flashinfer    | 512         | 128          | 128             | 23.4 | 1359.07                         | 465.97                           | 1825.04                         | 239687.05 | 44.06    | sgl-0.5.7         |

**需要注意：**

- 这里以num-prompts=128,  request-rate=128,  max-concurrency=128为基准，确保tpot在50ms以内，如果太小，可以适当增加请求数，如果太大，可以适当减少request-rate和max-concurrency。

## 单独评测脚本

如果是评测阿里和腾讯之外的，可参考以下脚本评测：

```shell
python3 -m sglang.bench_serving \
    --backend sglang \    
    --base-url http://0.0.0.0:8000 \    
    --port 8000  \   
    --model  /mnt/md0/checkpoints/Deepseek-R1  \   
    --dataset-name random  \   
    --tokenizer /mnt/md0/checkpoints/Deepseek-R1    \ 
    --dataset-path /root/gongoubo/ShareGPT_V3_unfiltered_cleaned_split.json  \   
    --num-prompts 8     \
    --random-input-len 3500  \   
    --random-output-len 1200  \   
    --random-range-ratio 1  \   
    --flush-cache  \   
    --seed 123  \   
    --request-rate 8  \   
    --max-concurrency 8
```

- --request-rate: Number of requests per second. If this is inf, then all the requests are sent at time 0. Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.
- max-concurrency: Maximum number of concurrent requests. This can be used to help simulate an environment where a higher level component is enforcing a maximum number of concurrent requests. While the --request-rate argument controls the rate at which requests are initiated, this argument will control how many are actually allowed to execute at a time. This means that when used in combination, the actual  request rate may be lower than specified with --request-rate, if the server is not processing requests fast enough to keep up.
- --num-prompts: 并发数

