import os
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


_DEFAULT_TASKS = [
    "gsm8k",
    "hellaswag",
    "mmlu",
    "mmlu_pro",
    "agieval",
    "ifeval",
    "openbookqa",
    "social_iqa",
    "winogrande",
    "truthfulqa",
    "squadv2",
    "arc_challenge",
    "gpqa",
    "mathqa",
    "ifeval",
    "pawsx",
    "cmmlu",
    "bbh",
    "ceval-valid",
    ]


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, help="Model path.")
    parser.add_argument("--gpu-mem-util", type=float, default=0.8, help="GPU memory usage limit for vLLM.")
    parser.add_argument("--dp", type=int, default=1, help="GPU data parallel size for mult-GPU inference.")
    parser.add_argument("--output-path", type=str, default=os.getcwd(), help="The path to the output file where the result metrics will be saved.")
    parser.add_argument("--tasks", type=str, default=",".join(_DEFAULT_TASKS), help="The tasks to run.")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Only print out the commands to be executed and quit.")
    args, remaining_args = parser.parse_known_args()

    logger.info(f"Parsed arguments: {args!s}")
    logger.info(f"Remaining arguments will be passed to lm_eval: {remaining_args!s}")

    logger.info(f"Run evaluation.")
    output_path_base = Path(args.output_path).absolute()
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_path = output_path_base / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = ["lm_eval", "--model", "vllm", \
            "--model_args", \
            f"pretrained={args.model_path.absolute()},dtype=auto,gpu_memory_utilization={args.gpu_mem_util},data_parallel_size={args.dp}", \
            "--batch_size", "auto", \
            "--trust_remote_code", \
            "--log_samples", \
            "--output_path", f"{output_path}", \
            "--tasks", args.tasks, \
            ] + remaining_args
    cmd = " ".join(cmd)
    logger.info(f"Running command: {cmd!s}")
    if not args.dry_run:
        subprocess.check_call(cmd, shell=True, cwd="/workspace/datasets")

    logger.info(f"Make results table.")
    cmd = f"python /workspace/make_table_results.py --output={output_path}/summary.txt {output_path}/*/results_*.json"
    logger.info(f"Running command: {cmd!s}")
    if not args.dry_run:
        subprocess.check_call(cmd, shell=True)
