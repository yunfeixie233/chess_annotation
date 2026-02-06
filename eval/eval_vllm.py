#!/usr/bin/env python3
"""
Evaluate chess SFT model accuracy on best-move prediction using vLLM.

Compares the fine-tuned checkpoint vs the original base model on the eval set.
Extracts the move from \\boxed{...} in model output and checks against ground truth.

Supports evaluating a single checkpoint or all checkpoints under a directory.
Runs multiple evaluation passes (default 3) and reports averaged metrics.

Usage:
    python eval_vllm.py --checkpoint /path/to/checkpoint-2000 --skip_base
    python eval_vllm.py --checkpoint /path/to/sft/  # evals all checkpoint-* dirs
"""

import argparse
import glob
import json
import os
import re
import time

print("Loading vllm (this may take a minute on NFS)...", flush=True)
from vllm import LLM, SamplingParams
print("Imports done.", flush=True)


# ---- Paths ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EVAL_FILE = os.path.join(PROJECT_ROOT, "data", "lichess_eval_sft_eval.jsonl")
CHECKPOINT_DIR = "/home/ubuntu/yunfei1/LlamaFactory/saves/qwen25-7b-chess/full/sft"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Matches the training instruction from generate_llamafactory_sft_data.py (v1 variant 0).
# LlamaFactory alpaca converter joins instruction + "\n" + input as the user message.
# No system prompt was used during training.
EVAL_INSTRUCTION = "Analyze this chess position and determine the best move. Explain your reasoning by examining candidate moves and their consequences."


def load_eval_data(eval_file, num_samples=-1):
    """Load evaluation data from JSONL."""
    samples = []
    with open(eval_file) as f:
        for line in f:
            samples.append(json.loads(line))
    if num_samples > 0:
        samples = samples[:num_samples]
    print(f"Loaded {len(samples)} eval samples")
    return samples


def build_prompt(fen):
    """Build the user message content matching LlamaFactory alpaca format.

    During training, user content = instruction + "\\n" + input (FEN).
    No system prompt was used.
    """
    return EVAL_INSTRUCTION + "\n" + fen


def extract_move(text):
    """Extract the move from \\boxed{...} in model output.

    Returns (move_str, has_format) where has_format indicates whether
    the output contained a recognized format (\\boxed{} or <answer>).
    """
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip(), True

    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        inner = match.group(1).strip()
        boxed = re.search(r"\\boxed\{([^}]+)\}", inner)
        if boxed:
            return boxed.group(1).strip(), True
        return inner, True

    match = re.search(r"best move is\s+(\S+)", text)
    if match:
        return match.group(1).strip().rstrip(",."), False

    return "", False


def run_inference(llm, samples, max_new_tokens=2048, desc="Evaluating"):
    """Run batched inference using vLLM and return predictions."""
    conversations = []
    for s in samples:
        conversations.append([
            {
                "role": "user",
                "content": build_prompt(s["input"]),
            },
        ])

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=max_new_tokens,
    )

    print(f"Running {desc} on {len(samples)} samples...", flush=True)
    outputs = llm.chat(conversations, sampling_params, use_tqdm=True)

    predictions = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        predicted_move, has_format = extract_move(generated_text)
        predictions.append({
            "fen": samples[i]["input"],
            "best_move": samples[i]["best_move"],
            "predicted_move": predicted_move,
            "has_format": has_format,
            "correct": predicted_move == samples[i]["best_move"],
            "raw_output": generated_text[-500:],
        })

    return predictions


def compute_metrics(predictions):
    """Compute accuracy metrics.

    - format_accuracy: fraction of outputs that contained \\boxed{} or <answer> format
    - answer_accuracy: among correctly formatted outputs, fraction with correct move
    - accuracy: overall fraction with correct move (regardless of format)
    """
    total = len(predictions)
    correct = sum(1 for p in predictions if p["correct"])
    has_format = sum(1 for p in predictions if p["has_format"])
    no_answer = sum(1 for p in predictions if p["predicted_move"] == "")
    correct_among_formatted = sum(1 for p in predictions if p["has_format"] and p["correct"])

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "has_format": has_format,
        "format_accuracy": has_format / total if total > 0 else 0,
        "answer_accuracy": correct_among_formatted / has_format if has_format > 0 else 0,
        "no_answer": no_answer,
        "no_answer_rate": no_answer / total if total > 0 else 0,
    }


def average_metrics(metrics_list):
    """Average a list of metrics dicts. Returns dict with mean and std for each key."""
    keys = [k for k in metrics_list[0] if isinstance(metrics_list[0][k], (int, float))]
    n = len(metrics_list)
    avg = {}
    for k in keys:
        values = [m[k] for m in metrics_list]
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = variance ** 0.5
        avg[k] = mean
        avg[k + "_std"] = std
    return avg


def resolve_checkpoints(checkpoint_path):
    """Resolve checkpoint path to a list of checkpoint directories.

    If checkpoint_path points to a specific checkpoint dir (contains model files),
    return [checkpoint_path].
    If it points to a parent dir containing checkpoint-* subdirs,
    return all checkpoint dirs sorted by step number.
    """
    if os.path.isfile(os.path.join(checkpoint_path, "config.json")):
        return [checkpoint_path]

    pattern = os.path.join(checkpoint_path, "checkpoint-*")
    ckpt_dirs = sorted(
        glob.glob(pattern),
        key=lambda d: int(os.path.basename(d).split("-")[-1]),
    )
    ckpt_dirs = [d for d in ckpt_dirs if os.path.isfile(os.path.join(d, "config.json"))]
    return ckpt_dirs


def save_eval_result(output_dir, model_name, avg_metrics, per_run_metrics, total_time, last_predictions, checkpoint=None):
    """Save a single eval result as a JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    safe_name = model_name.replace("/", "_").replace("\\", "_")
    output_file = os.path.join(output_dir, f"eval_{safe_name}.json")

    result = {
        "model": model_name,
        "num_runs": len(per_run_metrics),
        "avg_metrics": avg_metrics,
        "per_run_metrics": per_run_metrics,
        "total_time": total_time,
    }
    if checkpoint:
        result["checkpoint"] = checkpoint

    result["predictions"] = [
        {k: v for k, v in p.items() if k != "raw_output"}
        for p in last_predictions
    ]

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Results saved to {output_file}", flush=True)
    return output_file


def print_metrics(metrics, prefix=""):
    """Print metrics in a readable format."""
    print(f"{prefix}  Accuracy:        {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})", flush=True)
    print(f"{prefix}  Format accuracy: {metrics['format_accuracy']:.2%} ({metrics['has_format']}/{metrics['total']})", flush=True)
    print(f"{prefix}  Answer accuracy: {metrics['answer_accuracy']:.2%} (correct/formatted)", flush=True)
    print(f"{prefix}  No answer rate:  {metrics['no_answer_rate']:.2%} ({metrics['no_answer']})", flush=True)


def print_avg_metrics(avg_metrics, prefix=""):
    """Print averaged metrics with std."""
    print(f"{prefix}  Accuracy:        {avg_metrics['accuracy']:.2%} +/- {avg_metrics['accuracy_std']:.2%}", flush=True)
    print(f"{prefix}  Format accuracy: {avg_metrics['format_accuracy']:.2%} +/- {avg_metrics['format_accuracy_std']:.2%}", flush=True)
    print(f"{prefix}  Answer accuracy: {avg_metrics['answer_accuracy']:.2%} +/- {avg_metrics['answer_accuracy_std']:.2%}", flush=True)
    print(f"{prefix}  No answer rate:  {avg_metrics['no_answer_rate']:.2%} +/- {avg_metrics['no_answer_rate_std']:.2%}", flush=True)


def eval_single_model(model_path, samples, args, desc, output_dir):
    """Load a vLLM engine, run eval num_runs times, save averaged results. Returns avg metrics."""
    print(f"\nLoading model from {model_path}...", flush=True)

    # Use base model tokenizer for fine-tuned checkpoints because LlamaFactory
    # saved extra_special_tokens as a list in tokenizer_config.json, which is
    # incompatible with newer transformers (expects a dict).
    # The SFT did not change the vocabulary so the base tokenizer is correct.
    is_local_checkpoint = os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, "config.json"))
    tokenizer_to_use = BASE_MODEL if is_local_checkpoint else model_path

    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_to_use,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print("Model loaded successfully", flush=True)

    per_run_metrics = []
    last_predictions = None
    total_t0 = time.time()

    for run_idx in range(args.num_runs):
        print(f"\n  --- Run {run_idx + 1}/{args.num_runs} ---", flush=True)
        t0 = time.time()
        predictions = run_inference(
            llm, samples,
            max_new_tokens=args.max_new_tokens,
            desc=f"{desc} run {run_idx + 1}",
        )
        elapsed = time.time() - t0
        metrics = compute_metrics(predictions)
        metrics["time"] = elapsed
        per_run_metrics.append(metrics)
        last_predictions = predictions

        print_metrics(metrics, prefix="  ")
        print(f"    Time: {elapsed:.1f}s", flush=True)

    total_time = time.time() - total_t0

    avg_metrics = average_metrics(per_run_metrics)

    print(f"\n--- {desc} Averaged Results ({args.num_runs} runs) ---", flush=True)
    print_avg_metrics(avg_metrics, prefix="")

    model_name = os.path.basename(model_path) if os.path.isdir(model_path) else model_path
    save_eval_result(output_dir, model_name, avg_metrics, per_run_metrics, total_time, last_predictions, checkpoint=model_path)

    del llm

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate chess SFT model with vLLM")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of eval samples (-1 for all)")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of eval runs to average over")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint dir or parent dir with checkpoint-* subdirs")
    parser.add_argument("--skip_base", action="store_true", help="Skip base model evaluation")
    parser.add_argument("--skip_finetuned", action="store_true", help="Skip fine-tuned model evaluation")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for eval result JSONs")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    args = parser.parse_args()

    samples = load_eval_data(EVAL_FILE, args.num_samples)
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "eval_results")

    all_results = {}

    # ---- Evaluate fine-tuned model(s) ----
    if not args.skip_finetuned:
        ckpt_path = args.checkpoint if args.checkpoint else CHECKPOINT_DIR
        checkpoints = resolve_checkpoints(ckpt_path)

        if not checkpoints:
            print(f"No checkpoints found at {ckpt_path}", flush=True)
        else:
            print(f"\nFound {len(checkpoints)} checkpoint(s) to evaluate:", flush=True)
            for c in checkpoints:
                print(f"  {c}", flush=True)

            for ckpt_dir in checkpoints:
                ckpt_name = os.path.basename(ckpt_dir)
                print("\n" + "=" * 60, flush=True)
                print(f"Evaluating FINE-TUNED model: {ckpt_name}", flush=True)
                print("=" * 60, flush=True)

                avg_metrics = eval_single_model(
                    ckpt_dir, samples, args,
                    desc=ckpt_name,
                    output_dir=output_dir,
                )
                all_results[ckpt_name] = avg_metrics

    # ---- Evaluate base model ----
    if not args.skip_base:
        print("\n" + "=" * 60, flush=True)
        print(f"Evaluating BASE model ({BASE_MODEL})", flush=True)
        print("=" * 60, flush=True)

        avg_metrics = eval_single_model(
            BASE_MODEL, samples, args,
            desc="base",
            output_dir=output_dir,
        )
        all_results["base"] = avg_metrics

    # ---- Summary ----
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY (averaged over runs)", flush=True)
    print("=" * 60, flush=True)
    header = f"  {'model':30s}  {'accuracy':>10s}  {'format_acc':>10s}  {'answer_acc':>10s}"
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)
    for name, m in all_results.items():
        print(f"  {name:30s}  {m['accuracy']:>9.2%}  {m['format_accuracy']:>10.2%}  {m['answer_accuracy']:>10.2%}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
