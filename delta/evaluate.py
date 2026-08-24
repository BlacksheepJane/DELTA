"""PPL and lm-evaluation-harness evaluation for DELTA checkpoints."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .checkpoint import load_delta_checkpoint
from .data import get_ppl_dataloader
from .quantization import SINQLinearFactor


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


DEFAULT_PPL_DATASETS = ("wikitext2", "c4")
DEFAULT_DOWNSTREAM_TASKS = (
    "openbookqa",
    "winogrande",
    "hellaswag",
    "arc_easy",
    "piqa",
    "mathqa",
)
SCRIPTED_DATASETS = {
    "piqa": (
        "piqa",
        "piqa.py",
        "2e8ac2dffd59bac8c3c6714948f4c551a0848bb0",
    ),
    "mathqa": (
        "math_qa",
        "math_qa.py",
        "c4f1cc784c04c4957b50c97858f23893b633eea6",
    ),
}


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _model_storage_bytes(model) -> int:
    seen = set()
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor is None:
            continue
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key in seen:
            continue
        seen.add(key)
        total += storage.nbytes()
    return total


def _cache_dequantized_factors(model, device: str) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, SINQLinearFactor):
            module.cache_dequantized(device=torch.device(device), dtype=torch.float32)
            count += 1
    return count


def _prepare_lm_eval_task(
    task: str,
    task_manager: Any,
    cache_dir: Optional[str],
) -> Any:
    """Return an lm-eval task spec compatible with scripted Hub datasets.

    HfFileSystem may expose old-style builder scripts as gzip-compressed bytes
    to datasets 2.16.1. Downloading the same official pinned script through the
    Hub client first gives datasets a normal local Python file without changing
    the task definition or data.
    """
    if task not in SCRIPTED_DATASETS:
        return task

    from huggingface_hub import hf_hub_download

    repo_id, filename, revision = SCRIPTED_DATASETS[task]
    script_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        cache_dir=cache_dir,
    )
    task_config = task_manager._get_config(task)
    task_config["dataset_path"] = script_path
    return task_config


@torch.no_grad()
def evaluate_ppl(
    model,
    tokenizer,
    datasets: List[str],
    device: str,
    seqlen: int,
    batch_size: int,
    max_samples: Optional[int],
    cache_dir: Optional[str],
    wikitext_path: Optional[str],
    c4_path: Optional[str],
) -> Dict[str, float]:
    model.to(device)
    model.eval()
    cached_factors = _cache_dequantized_factors(model, device)
    print(
        "Cached {} dequantized DELTA factors on {}".format(
            cached_factors, device
        ),
        flush=True,
    )
    results = {}
    for dataset_name in datasets:
        loader = get_ppl_dataloader(
            name=dataset_name,
            tokenizer=tokenizer,
            seqlen=seqlen,
            batch_size=batch_size,
            max_samples=max_samples,
            cache_dir=cache_dir,
            wikitext_path=wikitext_path,
            c4_path=c4_path,
        )
        total_nll = 0.0
        total_tokens = 0
        for input_ids in tqdm(loader, desc="PPL {}".format(dataset_name)):
            input_ids = input_ids.to(device)
            logits = model(input_ids=input_ids, use_cache=False).logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                reduction="sum",
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss while evaluating {}".format(dataset_name))
            total_nll += float(loss.item())
            total_tokens += int(shift_labels.numel())
        if total_tokens == 0:
            raise RuntimeError("No tokens evaluated for {}".format(dataset_name))
        results[dataset_name] = math.exp(total_nll / total_tokens)
    return results


def evaluate_downstream(
    model,
    tokenizer,
    tasks: List[str],
    device: str,
    batch_size: str,
    num_fewshot: int,
    limit: Optional[float],
    cache_dir: Optional[str] = None,
    on_task_result: Optional[
        Callable[[str, Dict[str, Any]], None]
    ] = None,
) -> Dict[str, Dict[str, Any]]:
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
    except ImportError as error:
        raise ImportError("lm_eval==0.4.7 is required for downstream evaluation") from error

    model.to(device)
    model.eval()
    cached_factors = _cache_dequantized_factors(model, device)
    print(
        "Cached {} dequantized DELTA factors on {}".format(
            cached_factors, device
        ),
        flush=True,
    )
    harness_model = HFLM(pretrained=model, tokenizer=tokenizer)
    task_manager = TaskManager()
    outputs = {}
    for task in tasks:
        task_spec = _prepare_lm_eval_task(task, task_manager, cache_dir)
        result = evaluator.simple_evaluate(
            model=harness_model,
            tasks=[task_spec],
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=limit,
            task_manager=task_manager,
        )
        outputs[task] = result["results"][task]
        if on_task_result is not None:
            on_task_result(task, outputs[task])
    return outputs


def _write_ppl_results(
    output_dir: Path,
    checkpoint: str,
    ppls: Dict[str, float],
    storage_bytes: int,
) -> None:
    payload = {
        "checkpoint": checkpoint,
        "ppl": ppls,
        "model_storage_bytes": storage_bytes,
    }
    _write_json(output_dir / "ppl.json", payload)
    with (output_dir / "ppl.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["checkpoint", "dataset", "ppl"])
        writer.writeheader()
        for dataset_name, value in ppls.items():
            writer.writerow(
                {"checkpoint": checkpoint, "dataset": dataset_name, "ppl": value}
            )


def _write_downstream_results(
    output_dir: Path,
    checkpoint: str,
    results: Dict[str, Dict[str, Any]],
) -> None:
    rows = []
    for task, task_result in results.items():
        _write_json(
            output_dir / "downstream_{}.json".format(task),
            {"checkpoint": checkpoint, "task": task, "results": task_result},
        )
        task_rows = []
        for metric, value in task_result.items():
            if isinstance(value, (int, float, np.generic)):
                task_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "task": task,
                        "metric": metric,
                        "value": _jsonable(value),
                    }
                )
        rows.extend(task_rows)
        with (output_dir / "downstream_{}.csv".format(task)).open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["checkpoint", "task", "metric", "value"]
            )
            writer.writeheader()
            writer.writerows(task_rows)
    if len(results) > 1:
        _write_json(
            output_dir / "downstream_summary.json",
            {"checkpoint": checkpoint, "tasks": results},
        )
        with (output_dir / "downstream_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["checkpoint", "task", "metric", "value"]
            )
            writer.writeheader()
            writer.writerows(rows)


def _parse_limit(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a DELTA directory checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("ppl", "downstream"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", default=",".join(DEFAULT_PPL_DATASETS))
    parser.add_argument("--tasks", default=",".join(DEFAULT_DOWNSTREAM_TASKS))
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", default="4")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--wikitext-path", default=None)
    parser.add_argument("--c4-path", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading DELTA checkpoint from {}".format(args.checkpoint), flush=True)
    model, tokenizer = load_delta_checkpoint(args.checkpoint)
    if args.mode == "ppl":
        datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
        ppls = evaluate_ppl(
            model=model,
            tokenizer=tokenizer,
            datasets=datasets,
            device=args.device,
            seqlen=args.seq_len,
            batch_size=int(args.batch_size),
            max_samples=args.max_eval_samples,
            cache_dir=args.cache_dir,
            wikitext_path=args.wikitext_path,
            c4_path=args.c4_path,
        )
        _write_ppl_results(
            output_dir, args.checkpoint, ppls, _model_storage_bytes(model)
        )
        print(json.dumps(ppls, sort_keys=True), flush=True)
        return

    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    unknown = sorted(set(tasks) - set(DEFAULT_DOWNSTREAM_TASKS))
    if unknown:
        raise SystemExit("Unsupported downstream tasks: {}".format(unknown))
    results = evaluate_downstream(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        device=args.device,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        limit=_parse_limit(args.limit),
        cache_dir=args.cache_dir,
        on_task_result=lambda task, task_result: _write_downstream_results(
            output_dir, args.checkpoint, {task: task_result}
        ),
    )
    _write_downstream_results(output_dir, args.checkpoint, results)
    print("Wrote downstream results to {}".format(output_dir), flush=True)


if __name__ == "__main__":
    main()
