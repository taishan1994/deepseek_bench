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
        "--deploy_log",
        type=str,
        default="",
        help="服务部署后的日志路径，主要是为了评测ali的decode阶段的throughout",
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

            metrics = {}

            batch_size = [1,2,4,8,12,16,20,24,28,32,40,48,56,64]
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
                    --seed 123
                """.format(args=args, port=port, bs=bs)
                # os.system(command)
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
                    metrics[bs] = float(value)

            print("TPOT (ms)：", metrics)
            final_metrics.append(metrics)

            tmp = {}

            with open(args.deploy_log, "r") as f:
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

    elif args.eval_method == "tencent":
        print("tencent")
        # 腾讯的暂时不设置了
        command = """
        python3 -m sglang.bench_serving \
            --backend sglang \
            --base-url {args.base_url} \
            --port {port} \
            --model {args.model} \
            --dataset-name sharegpt \
            --dataset-path {args.dataset_path} \
            --request-rate inf \
            --flush-cache \
            --seed 123
        """.format(args=args, port=port)

    else:
        raise ValueError(f"eval_method must be ali or tencent, but got {args.method}")

    print("="*100)

    with open(args.output_path, "w") as f:
        f.write(json.dumps(final_metrics, ensure_ascii=False, indent=4))

    return final_metrics

if __name__ == "__main__":
    args = get_args()
    main(args)  

