#!/usr/bin/env python3
"""Audit qualified splits and freeze reproducible P2 Train/Dev subsets."""
import argparse
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import math
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from omni_av.data import Observation, file_hash, gt_text_plan, write_json_new


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/private/liqi_workspace/data/seamless/subset_300h_pure"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-count", type=int, default=91009)
    p.add_argument("--dev-count", type=int, default=48)
    p.add_argument("--budget", type=float, default=24)
    p.add_argument("--extend-dev", type=Path, help="Retain all previously selected Dev IDs")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=45)
    a = p.parse_args()
    if min(a.train_count, a.dev_count) < 1 or not 0 < a.budget <= 30:
        p.error("Positive counts and budget <=30 required")
    paths = {"train": [f"train_gt_qualified_20260920/train_part{i}_manifest.json" for i in (0, 1)],
             "dev": ["manifests_gt_qualified_20260920/dev_manifest.json"],
             "test": ["manifests_gt_qualified_20260920/test_manifest.json"]}
    ids, sources, rows_by_split, report = {}, {}, {}, {}
    for split, names in paths.items():
        rows = []
        for part, name in enumerate(names):
            path = a.data_root / name
            sources[str(path)] = file_hash(path)
            rows.extend((row, part) for row in json.loads(path.read_text()))
        ids[split] = {r["clip_id"] for r, _ in rows}
        if len(ids[split]) != len(rows) or any(r["split"] != split for r, _ in rows):
            raise ValueError("Duplicate IDs or wrong split in source manifest")
        meta = a.data_root / "ltx23_precomputed_720p_area_qualified_20260920" / f"metadata_{split}.jsonl"
        m = [json.loads(line) for line in meta.open()]
        if len(m) != len(rows) or {r["sample_id"] for r in m} != ids[split] or any(r["split"] != split for r in m):
            raise ValueError(f"Qualified VAE metadata mismatch: {split}")
        sources[str(meta)] = file_hash(meta)
        report[split] = dict(qualified=len(rows), within_budget=sum(r["duration_s"] <= a.budget for r, _ in rows))
        rows_by_split[split] = rows
    if ids["train"] & ids["dev"] or ids["train"] & ids["test"] or ids["dev"] & ids["test"]:
        raise ValueError("Train/Dev/Test overlap")
    selected = {}
    previous = json.loads(a.extend_dev.read_text()) if a.extend_dev else []
    previous_ids = [x["observation"]["sample_id"] for x in previous]
    if len(set(previous_ids)) != len(previous_ids) or not set(previous_ids) <= ids["dev"] or len(previous_ids) > a.dev_count:
        raise ValueError("Invalid previous Dev selection")
    def convert(item):
        split, row, part = item
        duration, end = float(row["duration_s"]), float(row["speaker_speech_duration_s"])
        if not 0 < duration <= a.budget or not 0 < end <= duration:
            raise ValueError(f"Invalid full-data timing: {row['clip_id']}")
        sid, files = row["clip_id"], row["files"]
        gt = a.data_root / (f"omni_gt_train_part{part}_30fps" if split == "train" else "omni_gt_dev") / sid / "avatar_plan.validated.json"
        payload = json.loads(gt.read_text())
        if not payload.get("annotation_quality", {}).get("usable_for_ltx_training", False):
            raise ValueError(f"Qualified GT is no longer usable: {sid}")
        for key in ("speaker_audio", "listener_audio", "target_video", "first_frame_jpg"):
            if not Path(files[key]).is_file():
                raise FileNotFoundError(files[key])
        budget = min(a.budget, math.ceil(duration / 8) * 8)
        obs = Observation(sid, files["speaker_audio"], files["first_frame_jpg"], 0, end, budget)
        return dict(observation=asdict(obs), target_video=files["target_video"],
            target_audio=files["listener_audio"], duration_s=duration, text_plan=asdict(gt_text_plan(payload, sid)),
            gt_path=str(gt), split=split, target_height=row["target_size"][0], target_width=row["target_size"][1])
    for split, count in (("train", a.train_count), ("dev", a.dev_count)):
        rows = sorted(rows_by_split[split], key=lambda r: hashlib.sha256(f"{a.seed}:{r[0]['clip_id']}".encode()).hexdigest())
        if split == "dev" and previous_ids:
            by_id = {row["clip_id"]: (row, part) for row, part in rows}
            rows = [by_id[sid] for sid in previous_ids] + [x for x in rows if x[0]["clip_id"] not in set(previous_ids)]
        if len(rows) < count:
            raise ValueError(f"Requested {count} {split}, only {len(rows)} exist")
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            examples = []
            for example in pool.map(convert, [(split, row, part) for row, part in rows[:count]]):
                examples.append(example)
                if len(examples) % 10000 == 0:
                    print(json.dumps(dict(event="dataset_audit", split=split, checked=len(examples))), flush=True)
        selected[split] = examples
        report[split]["selected"] = len(examples)
    a.output.mkdir(parents=True, exist_ok=False)
    for split, examples in selected.items():
        write_json_new(a.output / f"{split}.json", examples)
    write_json_new(a.output / "audit.json", dict(splits=report, source_sha256=sources,
        disjoint=True, budget=a.budget, seed=a.seed, selection="sha256(seed:sample_id)",
        test_used_for_tuning=False, scope="full-qualified-train" if a.train_count == len(ids["train"]) else "subset",
        budget_policy="ceil duration to 8/16/24 seconds; no temporal or spatial crop",
        retained_dev_ids=previous_ids))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
