import re
import os
import json
import argparse
import subprocess
import statistics

from pprint import pprint
from tqdm import tqdm


"""
python3 main.py --model /nfs/FM/gongoubo/checkpoints/Qwen/Qwen3-30B-A3B-Instruct-2507 --base_url http://192.168.16.21:18000 --eval_type prefill --eval_method ali --prefill_gpu_num 1 --output_path /nfs/FM/gongoubo/new_project/workflow/deepseek_bench/output/ali_prefill.json


python3 main.py --model /nfs/FM/gongoubo/checkpoints/Qwen/Qwen3-30B-A3B-Instruct-2507 --base_url http://192.168.16.21:18000 --eval_type decode --eval_method ali --decode_gpu_num 1 --deploy_log /nfs/FM/gongoubo/new_project/workflow/deepseek_bench/logs/test.log --output_path /nfs/FM/gongoubo/new_project/workflow/deepseek_bench/output/ali_decode.json

python3 main.py --model /mnt/md0/checkpoints/Deepseek-R1 --base_url http://0.0.0.0:8000  --eval_method tencent --output_path /mnt/md0/deepseek_bench/output/deepseek_r1_fp8_tencent.json --dataset_path /root/gongoubo/ShareGPT_V3_unfiltered_cleaned_split.json --batch_size 128 --request_rate 128 --max_concurrency 128
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        # default="/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3-32B",
        default="",
        help="待评测的模型的路径",
    )

    parser.add_argument(
        "--base_url",
        type=str,
        # default="/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3-32B",
        default="",
        # default="MMBench_DEV_CN,SEEDBench_IMG,CCBench,MMBench,MME,OCRBench,CMMMU_VAL,HallusionBench,RealWorldQA",
        help="待评测模型的url",
    )

    parser.add_argument(
        "--eval_type",
        type=str,
        # default="/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3-32B",
        default="",
        help="评测的类型：prefill或者decode",
    )
    parser.add_argument(
        "--eval_method",
        type=str,
        default="",
        help="评测方法，ali或者tencent",
    )

    parser.add_argument(
        "--max_concurrencys",
        type=str,
        default="",
        help="腾讯评测自定义搜索最大并发",
    )

    parser.add_argument(
        "--deploy_log",
        type=str,
        default="",
        help="服务部署后的日志路径，主要是为了评测ali的decode阶段的throughout，如果是有多个decode节点，用`,`隔开",
    )

    parser.add_argument(
        "--input_len",
        type=str,
        default="",
        help="评测输入长度",
    )

    parser.add_argument(
        "--output_len",
        type=str,
        default="",
        help="评测输出长度",
    )

    parser.add_argument(
        "--prefill_gpu_num",
        type=str,
        default="",
        help="prefill使用的GPU的数目",
    )

    parser.add_argument(
        "--decode_gpu_num",
        type=str,
        default="",
        help="decode使用的GPU的数目",
    )

    parser.add_argument(
        "--batch_size",
        type=str,
        default="",
        help="并发数",
    )

    parser.add_argument(
        "--request_rate",
        type=str,
        default=128,
        help="每秒最大发送的请求数，支持多个之间用,分割",
    )

    parser.add_argument(
        "--max_concurrency",
        type=str,
        default=128,
        help="最大并发数，支持多个之间用,分割",
    )


    parser.add_argument(
        "--dataset_path",
        type=str,
        # default="/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3-32B",
        default="/nfs/FM/gongoubo/new_project/workflow/deepseek_bench/data/ShareGPT_V3_unfiltered_cleaned_split.json",
        # default="MMBench_DEV_CN,SEEDBench_IMG,CCBench,MMBench,MME,OCRBench,CMMMU_VAL,HallusionBench,RealWorldQA",
        help="评测数据集的路径",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        # default="/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3-32B",
        default="/nfs/FM/gongoubo/new_project/workflow/deepseek_bench/output/ali_prefill.json",
        # default="MMBench_DEV_CN,SEEDBench_IMG,CCBench,MMBench,MME,OCRBench,CMMMU_VAL,HallusionBench,RealWorldQA",
        help="评测数据集的路径",
    )


    return parser.parse_args()


def run_benchmark_and_extract_metrics(args, port, max_concurrency, dataset_name="random", input_len=3500, output_len=1200, num_prompts=None):
    if num_prompts is None:
        num_prompts = args.batch_size
    
    # command = """
    # python3 -m sglang.bench_serving \
    #     --backend sglang \
    #     --base-url {args.base_url} \
    #     --port {port} \
    #     --model {args.model} \
    #     --tokenizer {args.model} \
    #     --dataset-name {dataset_name} \
    #     --dataset-path {args.dataset_path} \
    #     --random-input-len {input_len} \
    #     --random-output-len {output_len} \
    #     --random-range-ratio 1 \
    #     --request-rate inf \
    #     --flush-cache \
    #     --seed 123 \
    #     --max-concurrency {max_concurrency} \
    #     --num-prompts {num_prompts}
    # """.format(args=args, port=port, max_concurrency=max_concurrency, dataset_name=dataset_name, 
    #            input_len=input_len, output_len=output_len, num_prompts=num_prompts)

    command = """
        python3 -m sglang.bench_serving \
            --backend sglang \
            --base-url {args.base_url} \
            --port {port} \
            --model {args.model} \
            --dataset-name sharegpt \
            --tokenizer {args.model} \
            --dataset-path {args.dataset_path} \
            --sharegpt-output-len 10 \
            --request-rate inf \
            --flush-cache \
            --seed 123 \
            --max-concurrency {max_concurrency} \
            --num-prompts 3000
    """.format(args=args, port=port, max_concurrency=max_concurrency)
    
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"命令执行失败，返回码: {result.returncode}")
        print(f"错误信息: {result.stderr}")
        return None, None
    
    if result.stderr:
        print(f"警告信息: {result.stderr}")
    
    # 提取 TPOT
    pattern = r"Mean TPOT \(ms\):\s+([\d.]+)"
    match = re.search(pattern, result.stdout)
    tpot = match.group(1) if match else None
    
    # 提取 TTFT
    pattern = r"Mean TTFT \(ms\):\s+([\d.]+)"
    match = re.search(pattern, result.stdout)
    ttft = match.group(1) if match else None
    
    return tpot, ttft


def main(args):
    print(vars(args))

    port = args.base_url.split(":")[-1]
    print("="*100)

    final_metrics = []

    if args.eval_method == "ali":
        if args.eval_type == "prefill":
            print("prefill")
            
            num_prompts = 512
            input_lens = [1024, 2048, 4096]

            metrics = {}

            for input_len in input_lens:
                command = """
                python3 -m sglang.bench_serving \
                    --backend sglang \
                    --base-url {args.base_url} \
                    --port {port} \
                    --model {args.model} \
                    --dataset-name random \
                    --dataset-path {args.dataset_path} \
                    --num-prompts {num_prompts} \
                    --request-rate inf \
                    --random-input-len {input_len} \
                    --random-output-len 1 \
                    --random-range-ratio 1 \
                    --flush-cache \
                    --seed 123 \
                    --output-file ./output
                """.format(args=args, port=port, input_len=input_len, num_prompts=num_prompts)
                # os.system(command)

                # 使用 subprocess 执行命令并捕获终端输出
                result = subprocess.run(command, shell=True, capture_output=True, text=True)

                # 打印终端输出
                print("标准输出:")
                print(result.stdout)
                
                # 提取数字
                pattern = r"Input token throughput \(tok/s\):\s+([\d.]+)"
                match = re.search(pattern, result.stdout)

                if match:
                    value = match.group(1)
                    print(value)
                    metrics[input_len] = float(value) / float(args.prefill_gpu_num)

            print("Input Throughput Per GPU (tokens/s)：", metrics)
            final_metrics.append(metrics)

        elif args.eval_type == "decode":
            print("decode")

            # 先测per_user_throughput和per_gpu_thrugput
            command = """
            python3 -m sglang.bench_serving \
                --backend sglang \
                --base-url {args.base_url} \
                --port {port} \
                --model {args.model} \
                --dataset-name random \
                --dataset-path {args.dataset_path} \
                --num-prompts 4096 \
                --request-rate inf \
                --random-input-len 4096 \
                --random-output-len 1536 \
                --random-range-ratio 1 \
                --flush-cache \
                --seed 123 \
                --max-concurrency 2048
            """.format(args=args, port=port)
            # os.system(command)
            print(command)
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(result.stdout)
            
            tmp = {}
            dlogs = args.deploy_log.split(",")
            for dlog in dlogs:
                with open(dlog, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        pattern = r"#running-req:\s*(?P<running_req>\d+).*?gen throughput \(token/s\):\s*(?P<gen_throughput>[\d.]+)"
                        match = re.search(pattern, line)

                        if match:
                            # print(f"running-req: {match.group('running_req')}")
                            # print(f"gen throughput: {match.group('gen_throughput')} token/s")
                            running_req = int(match.group('running_req'))
                            gen_throughput = float(match.group('gen_throughput'))
                            if running_req not in tmp:
                                tmp[running_req] = []
                            tmp[running_req].append(float(gen_throughput))
            
            # 取中位数
            metrics = {}
            for running_req, throughputs in tmp.items():
                if running_req not in metrics:
                    metrics[running_req] = {}
                median = statistics.median(throughputs)
                metrics[running_req]["median"] = median
                metrics[running_req]["throughput_per_gpu"] = median / float(args.decode_gpu_num)
                metrics[running_req]["throughput_per_user"] = median / float(running_req)
                metrics[running_req]["gen_througput_samples"] = len(throughputs)
                # metrics[running_req]["gen_througput_list"] = throughputs

            pprint(metrics)
            final_metrics.append(metrics)


            metrics = {}
            repeat_time = 10
            batch_size = [1,2,4,8,12,16,20,24,28,32,40,48,56,64]
            batch_size = [i*repeat_time for i in batch_size]
            # batch_size = [1,2,4]
            for bs in tqdm(batch_size, total=len(batch_size)):
                command = """
                python3 -m sglang.bench_serving \
                    --backend sglang \
                    --base-url {args.base_url} \
                    --port {port} \
                    --model {args.model} \
                    --dataset-name random \
                    --dataset-path {args.dataset_path} \
                    --num-prompts {bs} \
                    --request-rate inf \
                    --random-input-len 4096 \
                    --random-output-len 1536 \
                    --random-range-ratio 1 \
                    --flush-cache \
                    --seed 123 \
                    --max-concurrency {bs / repeat_time}
                """.format(args=args, port=port, bs=bs)
                # os.system(command)
                print(command)
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                # 从日志里面获取throughput
                 # 打印终端输出
                # print("标准输出:")
                # print(result.stdout)

                # 提取数字
                pattern = r"Mean TPOT \(ms\):\s+([\d.]+)"
                match = re.search(pattern, result.stdout)

                if match:
                    value = match.group(1)
                    print(value)
                    metrics[bs / repeat_time] = float(value)

            print("TPOT (ms)：", metrics)
            final_metrics.append(metrics)

    elif args.eval_method == "tencent":
        print("tencent")

        metrics = {}

        if args.max_concurrencys != "":
            max_concurrencys = args.max_concurrencys.split(",")
        else:
            max_concurrencys = [6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 128]

        tmp_metrics = []
        final_max_concurrency = -1
        for i, max_concurrency in enumerate(max_concurrencys):
            tpot, ttft = run_benchmark_and_extract_metrics(args, port, max_concurrency)
            
            print(tpot, ttft)
            tmp_metrics.append({
                "num_prompts": args.batch_size,
                "max_concurrency": max_concurrency,
                "ttft": ttft,
                "tpot": tpot
            })
            if float(ttft) < 2000:
                final_max_concurrency = max_concurrency
                continue
            else:
                best_concurrency = -1
                if i - 1 >= 0:
                    pre_max_concurrency = int(max_concurrencys[i-1])
                    current_max_concurrency = int(max_concurrency)
                    
                    # 二分查找：在 (pre_max_concurrency, current_max_concurrency) 范围内找到满足条件的最大值
                    # 注意：pre_max_concurrency 已测试且满足条件，current_max_concurrency 已测试且不满足条件
                    left = pre_max_concurrency + 1
                    right = current_max_concurrency - 1
                    best_concurrency = pre_max_concurrency
                    
                    while left <= right:
                        mid = (left + right) // 2
                        print(f"二分查找测试 concurrency: {mid}")
                        mid_tpot, mid_ttft = run_benchmark_and_extract_metrics(args, port, mid)
                        
                        print(f"mid_tpot: {mid_tpot}, mid_ttft: {mid_ttft}")
                        
                        tmp_metrics.append({
                            "num_prompts": args.batch_size,
                            "max_concurrency": mid,
                            "ttft": mid_ttft,
                            "tpot": mid_tpot,
                            "binary_search": True
                        })
                        
                        if mid_tpot is not None and mid_ttft is not None and float(mid_ttft) < 2000:
                            best_concurrency = mid
                            left = mid + 1
                        else:
                            right = mid - 1
                    
                    if best_concurrency != -1:
                        final_max_concurrency = best_concurrency
                break

        if final_max_concurrency == -1:
            print(tmp_metrics)
            raise Exception("请确认最大并发是否达到要求") 

        command = """
        python3 -m sglang.bench_serving \
            --backend sglang \
            --base-url {args.base_url} \
            --port {port} \
            --model {args.model} \
            --dataset-name sharegpt \
            --tokenizer {args.model} \
            --dataset-path {args.dataset_path} \
            --sharegpt-output-len 1200 \
            --request-rate inf \
            --flush-cache \
            --seed 123 \
            --max-concurrency {max_concurrency} \
            --num-prompts 3000
        """.format(args=args, port=port, max_concurrency=final_max_concurrency)

        print(command)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stdout)

        # 保存评估指标
        # TPOT
        pattern = r"Mean TPOT \(ms\):\s+([\d.]+)"
        match = re.search(pattern, result.stdout)

        if match:
            tpot = match.group(1)

        # TTFT
        pattern = r"Mean TTFT \(ms\):\s+([\d.]+)"
        match = re.search(pattern, result.stdout)

        if match:
            ttft = match.group(1)
        
        # input throughput
        pattern = r"Input token throughput \(tok/s\):\s+([\d.]+)"
        match = re.search(pattern, result.stdout)

        if match:
            input_throughput = match.group(1)

        # output throughput
        pattern = r"Output token throughput \(tok/s\):\s+([\d.]+)"
        match = re.search(pattern, result.stdout)

        if match:
            output_throughput = match.group(1)

        # total througput
        pattern = r"Total token throughput \(tok/s\):\s+([\d.]+)"
        match = re.search(pattern, result.stdout)

        if match:
            total_throughput = match.group(1)

        # QPM
        pattern = r"Request throughput \(req/s\):\s+([\d.]+)"
        match = re.search(pattern, result.stdout)

        if match:
            QPM = float(match.group(1)) * 60

        metrics["max_concurrency"] = final_max_concurrency
        metrics["tpot (ms)"] = tpot
        metrics["ttft (ms)"] = ttft
        metrics["input_throughput (tok/s)"] = input_throughput
        metrics["output_throughput (tok/s)"] = output_throughput
        metrics["total_throughput (tok/s)"] = total_throughput
        metrics["QPM (req/min)"] = QPM
        metrics["tmp_metrics"] = tmp_metrics

        final_metrics.append(metrics)
    else:
        raise ValueError(f"eval_method must be ali or tencent, but got {args.method}")

    print("="*100)
    print(f"final_metrics: {final_metrics}")

    with open(args.output_path, "w") as f:
        f.write(json.dumps(final_metrics, ensure_ascii=False, indent=4))

    return final_metrics

if __name__ == "__main__":
    args = get_args()
    main(args)