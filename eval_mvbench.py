#!/usr/bin/env python3
"""Evaluate vanilla LLaVA and VisionZip on MVBench."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    from decord import VideoReader, cpu
except Exception as exc:  # pragma: no cover
    raise RuntimeError("decord is required. Install via: pip install decord") from exc

CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def _first_existing(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def normalize_options(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw]
    if isinstance(raw, dict):
        return [str(raw[k]).strip() for k in sorted(raw)]
    if isinstance(raw, str):
        return [p.strip() for p in re.split(r"\n+|\s*\|\s*", raw) if p.strip()]
    return [str(raw).strip()]


def normalize_answer(raw: Any, options: Sequence[str]) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    up = text.upper()
    if len(up) == 1 and up in CHOICE_LETTERS:
        return up
    m = re.search(r"\b([A-Z])\b", up)
    if m and m.group(1) in CHOICE_LETTERS:
        return m.group(1)
    for i, opt in enumerate(options):
        if text.lower() == opt.lower():
            return CHOICE_LETTERS[i]
    return up


def extract_choice_letter(pred_text: str, options: Sequence[str]) -> str:
    up = pred_text.strip().upper()
    m = re.search(r"\b([A-Z])\b", up)
    if m and m.group(1) in CHOICE_LETTERS[: len(options)]:
        return m.group(1)
    for i, opt in enumerate(options):
        if opt.lower() in pred_text.lower():
            return CHOICE_LETTERS[i]
    return ""


@dataclass
class Sample:
    sample_id: str
    task_type: str
    question: str
    options: List[str]
    answer_letter: str
    video_path: Path


def load_mvbench_samples(dataset_root: Path, annotation_file: Optional[Path] = None) -> List[Sample]:
    if annotation_file is None:
        candidates = sorted(dataset_root.glob("*.json")) + sorted((dataset_root / "json").glob("*.json"))
        if not candidates:
            raise FileNotFoundError("No annotation JSON found. Pass --annotation-file.")
        annotation_file = candidates[0]

    content = json.loads(annotation_file.read_text(encoding="utf-8"))
    if isinstance(content, dict):
        if isinstance(content.get("data"), list):
            rows = content["data"]
        else:
            rows = []
            for v in content.values():
                if isinstance(v, list):
                    rows.extend(v)
    elif isinstance(content, list):
        rows = content
    else:
        raise ValueError("Unsupported annotation JSON format")

    samples: List[Sample] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        q = _first_existing(row, ["question", "query", "prompt", "instruction"])
        if not q:
            continue
        options = normalize_options(_first_existing(row, ["options", "candidates", "choices", "answer_choices"]))
        ans = normalize_answer(_first_existing(row, ["answer", "gt", "label", "correct_answer"]), options)
        task = str(_first_existing(row, ["task_type", "task", "category", "sub_task"]) or "unknown")
        sid = str(_first_existing(row, ["id", "uid", "sample_id", "question_id"]) or idx)
        video_rel = _first_existing(row, ["video", "video_path", "video_name", "video_file", "vid", "path", "data_path"])
        if video_rel is None:
            continue
        video_path = Path(str(video_rel))
        if not video_path.is_absolute():
            maybe = dataset_root / video_path
            video_path = maybe if maybe.exists() else next(dataset_root.rglob(video_path.name), Path(""))
        if not video_path.exists() or video_path.suffix.lower() not in VIDEO_EXTS:
            continue

        samples.append(Sample(sid, task, str(q).strip(), options, ans, video_path))

    if not samples:
        raise RuntimeError("No valid samples parsed from annotation file.")
    return samples


def sample_video_frames(video_path: Path, num_frames: int) -> List[Image.Image]:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total = len(vr)
    if total == 0:
        raise ValueError(f"No frames in {video_path}")
    k = min(total, num_frames)
    indices = np.linspace(0, total - 1, k, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(x) for x in frames]


def build_prompt(question: str, options: Sequence[str]) -> str:
    p = question.strip()
    if options:
        p += "\nOptions:\n"
        for i, opt in enumerate(options):
            p += f"{CHOICE_LETTERS[i]}. {opt}\n"
        p += "Answer with the option letter only."
    return p


class LlavaRunner:
    def __init__(self, model_path: str, model_base: Optional[str], conv_mode: str, device: str, dtype: str,
                 visionzip_cfg: Optional[Tuple[int, int]]) -> None:
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model

        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.conv_templates = conv_templates
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=get_model_name_from_path(model_path),
        )

        if visionzip_cfg is not None:
            from visionzip import visionzip
            model = visionzip(model, dominant=visionzip_cfg[0], contextual=visionzip_cfg[1])

        tdtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        self.tokenizer = tokenizer
        self.model = model.to(torch.device(device), dtype=tdtype).eval()
        self.image_processor = image_processor
        self.conv_mode = conv_mode
        self.device = torch.device(device)
        self.dtype = tdtype

    @torch.inference_mode()
    def infer_choice(self, frames: List[Image.Image], prompt: str, max_new_tokens: int) -> str:
        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], self.DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            full_prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        image_tensor = self.process_images(frames, self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [x.to(self.device, dtype=self.dtype) for x in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=self.dtype)

        out_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[img.size for img in frames],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        return self.tokenizer.batch_decode(out_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def evaluate_model(runner: LlavaRunner, samples: Sequence[Sample], num_frames: int,
                   max_new_tokens: int, limit: Optional[int], seed: int):
    rows = list(samples)
    if limit is not None and limit < len(rows):
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:limit]

    results = []
    totals: Dict[str, List[int]] = {}
    correct, total = 0, 0

    for s in tqdm(rows, desc="Evaluating", ncols=100):
        try:
            frames = sample_video_frames(s.video_path, num_frames)
            pred_text = runner.infer_choice(frames, build_prompt(s.question, s.options), max_new_tokens)
            pred = extract_choice_letter(pred_text, s.options)
            ok = int(pred == s.answer_letter)
            correct += ok
            total += 1
            totals.setdefault(s.task_type, [0, 0])
            totals[s.task_type][0] += ok
            totals[s.task_type][1] += 1
            results.append({
                "id": s.sample_id,
                "task_type": s.task_type,
                "video": str(s.video_path),
                "question": s.question,
                "options": s.options,
                "gt": s.answer_letter,
                "pred": pred,
                "pred_text": pred_text,
                "correct": bool(ok),
            })
        except Exception as exc:
            results.append({"id": s.sample_id, "task_type": s.task_type, "video": str(s.video_path), "error": str(exc)})

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "per_task": {
            k: {"accuracy": (v[0] / v[1]) if v[1] else 0.0, "correct": v[0], "total": v[1]}
            for k, v in sorted(totals.items())
        },
    }
    return results, summary


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate VisionZip vs vanilla LLaVA on MVBench")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--annotation-file", type=Path, default=None)
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--conv-mode", type=str, default="llava_v1")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/mvbench_eval"))
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--dominant", type=int, default=54)
    p.add_argument("--contextual", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-visionzip", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_mvbench_samples(args.dataset_root, args.annotation_file)
    print(f"Loaded {len(samples)} samples")

    if not args.skip_baseline:
        print("[1/2] Baseline LLaVA")
        baseline = LlavaRunner(args.model_path, args.model_base, args.conv_mode, args.device, args.dtype, None)
        results, summary = evaluate_model(baseline, samples, args.num_frames, args.max_new_tokens, args.limit, args.seed)
        dump_json(args.output_dir / "baseline_results.json", results)
        dump_json(args.output_dir / "baseline_summary.json", summary)
        print(f"Baseline accuracy: {summary['accuracy']:.4f} ({summary['correct']}/{summary['total']})")

    if not args.skip_visionzip:
        print("[2/2] VisionZip")
        vz = LlavaRunner(
            args.model_path,
            args.model_base,
            args.conv_mode,
            args.device,
            args.dtype,
            (args.dominant, args.contextual),
        )
        results, summary = evaluate_model(vz, samples, args.num_frames, args.max_new_tokens, args.limit, args.seed)
        dump_json(args.output_dir / "visionzip_results.json", results)
        dump_json(args.output_dir / "visionzip_summary.json", summary)
        print(f"VisionZip accuracy: {summary['accuracy']:.4f} ({summary['correct']}/{summary['total']})")

    print(f"Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
