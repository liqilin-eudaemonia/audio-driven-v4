#!/usr/bin/env python3
"""Count speaker-listener pairs from adjacent conversational floors.

This is the loose interview split: each time the other person takes the floor,
the previous floor is the speaker and the new floor is the listener. The
listener floor becomes the speaker of the next pair. Short replies, overlap,
and long turns stay. The v4+ word-count / gap / duration gates are only used
as a comparison counter, not as a drop rule.
"""

from __future__ import annotations

import argparse
import json
import random
import tarfile
import time
from collections import defaultdict
from pathlib import Path

INDEX_PATH = Path("/private/liqi_workspace/data/pipeline/seamless/outputs_demo100/pair_index_all.json")
OUT_DIR = Path("/private/liqi_workspace/data/pipeline/seamless/outputs_alternate_pairs")

WORD_GAP = 0.60
MIN_UTT_S = 0.15


def parse_words(meta: dict) -> list[tuple[float, float, str]]:
    transcript = meta.get("metadata:transcript") or []
    if isinstance(transcript, dict):
        transcript = transcript.get("words") or transcript.get("segments") or []
    words: list[tuple[float, float, str]] = []
    for seg in transcript:
        if not isinstance(seg, dict):
            continue
        inner = seg.get("words")
        items = inner if isinstance(inner, list) else [seg]
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("start") is None or item.get("end") is None:
                continue
            tok = str(item.get("word") or "").strip()
            if not tok:
                continue
            words.append((float(item["start"]), float(item["end"]), tok))
    words.sort(key=lambda row: (row[0], row[1]))
    return words


def utterances(
    words: list[tuple[float, float, str]],
    max_word_s: float | None = None,
) -> list[tuple[float, float, str, int]]:
    if max_word_s is not None and max_word_s > 0:
        capped: list[tuple[float, float, str]] = []
        for start, end, tok in words:
            end = min(end, start + max_word_s)
            if end > start:
                capped.append((start, end, tok))
        words = capped
    if not words:
        return []
    out: list[tuple[float, float, str, int]] = []
    start, end, toks = words[0][0], words[0][1], [words[0][2]]
    for begin, finish, tok in words[1:]:
        if begin - end <= WORD_GAP:
            end = max(end, finish)
            toks.append(tok)
        else:
            if (end - start) >= MIN_UTT_S and toks:
                out.append((start, end, " ".join(toks), len(toks)))
            start, end, toks = begin, finish, [tok]
    if (end - start) >= MIN_UTT_S and toks:
        out.append((start, end, " ".join(toks), len(toks)))
    return out


def process_tar(task: tuple) -> tuple[str, dict[str, list[list]], list[str]]:
    archive, stems = task[0], task[1]
    max_word_s = task[2] if len(task) > 2 else None
    wanted = set(stems)
    found: dict[str, list[list]] = {}
    try:
        with tarfile.open(archive, "r:") as tar:
            members = {}
            for info in tar:
                if not info.isfile():
                    continue
                name = Path(info.name).name
                if not name.endswith(".json"):
                    continue
                stem = name[:-5]
                if stem in wanted:
                    members[stem] = info
            for stem, info in members.items():
                handle = tar.extractfile(info)
                if handle is None:
                    continue
                meta = json.loads(handle.read().decode("utf-8"))
                utts = utterances(parse_words(meta), max_word_s=max_word_s)
                found[stem] = [[s, e, t[:240], n] for s, e, t, n in utts]
    except Exception as exc:  # noqa: BLE001
        return archive, found, [f"ERR {type(exc).__name__}: {exc}"]
    missing = sorted(wanted - set(found))
    return archive, found, missing


def _clip(text: str, limit: int = 220) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def build_pairs(person_a: dict, person_b: dict, utts_a: list[list], utts_b: list[list]) -> list[dict]:
    events: list[tuple[str, float, float, str, int]] = []
    for start, end, text, nwords in utts_a:
        events.append(("a", float(start), float(end), str(text), int(nwords)))
    for start, end, text, nwords in utts_b:
        events.append(("b", float(start), float(end), str(text), int(nwords)))
    events.sort(key=lambda row: (row[1], row[2], row[0]))

    floors: list[dict] = []
    for who, start, end, text, nwords in events:
        if end <= start:
            continue
        if not floors:
            floors.append(
                {"who": who, "start": start, "end": end, "text": text, "nwords": nwords, "overlaps": []}
            )
            continue
        cur = floors[-1]
        if who == cur["who"]:
            cur["end"] = max(cur["end"], end)
            cur["nwords"] += nwords
            if len(cur["text"]) < 280:
                cur["text"] = _clip(f"{cur['text']} {text}", 280)
            continue
        if end <= cur["end"] + 0.05:
            cur["overlaps"].append({"start": start, "end": end, "text": _clip(text, 120), "nwords": nwords})
            continue
        floors.append({"who": who, "start": start, "end": end, "text": text, "nwords": nwords, "overlaps": []})

    people = {"a": person_a, "b": person_b}
    pairs: list[dict] = []
    for index in range(len(floors) - 1):
        spk = floors[index]
        lis = floors[index + 1]
        duration = lis["end"] - spk["start"]
        if duration <= 0.2:
            continue
        spk_person = people[spk["who"]]
        lis_person = people[lis["who"]]
        gap = lis["start"] - spk["end"]
        pairs.append(
            {
                "kind": "floor_change",
                "start_s": round(spk["start"], 3),
                "end_s": round(lis["end"], 3),
                "duration_s": round(duration, 3),
                "gap_s": round(gap, 3),
                "speaker_id": spk_person["id"],
                "listener_id": lis_person["id"],
                "speaker_archive": spk_person["archive"],
                "listener_archive": lis_person["archive"],
                "speaker_utt": {
                    "start_s": round(spk["start"], 3),
                    "end_s": round(spk["end"], 3),
                    "text": _clip(spk["text"]),
                    "nwords": spk["nwords"],
                },
                "listener_utt": {
                    "start_s": round(lis["start"], 3),
                    "end_s": round(lis["end"], 3),
                    "text": _clip(lis["text"]),
                    "nwords": lis["nwords"],
                },
                "n_overlap": len(spk["overlaps"]),
                "overlap_text": _clip(" | ".join(item["text"] for item in spk["overlaps"][:2]), 160),
            }
        )
    return pairs


def passes_old_gate(pair: dict) -> bool:
    spk = pair["speaker_utt"]
    lis = pair["listener_utt"]
    spk_dur = spk["end_s"] - spk["start_s"]
    lis_dur = lis["end_s"] - lis["start_s"]
    gap = pair["gap_s"]
    clip = pair["duration_s"]
    if spk["nwords"] < 5 or spk_dur < 1.20:
        return False
    if lis["nwords"] < 5 or lis_dur < 1.20:
        return False
    if lis["start_s"] < spk["end_s"] - 0.20:
        return False
    if gap < 0.12 or gap > 2.50:
        return False
    if spk_dur > 12.0 or lis_dur < 0.80 or lis_dur > 20.0:
        return False
    if clip < 2.0 or clip > 24.0:
        return False
    return True


def reject_bucket(pair: dict) -> str:
    if passes_old_gate(pair):
        return "kept_like_old"
    spk = pair["speaker_utt"]
    lis = pair["listener_utt"]
    if lis["start_s"] < spk["end_s"] - 0.20:
        return "overlap_reply"
    if pair["duration_s"] > 24.0 or (spk["end_s"] - spk["start_s"]) > 12.0:
        return "long_turn"
    if pair["gap_s"] > 2.50:
        return "long_gap"
    if 0.0 <= pair["gap_s"] < 0.12:
        return "short_gap"
    if lis["nwords"] < 5 or (lis["end_s"] - lis["start_s"]) < 1.20:
        return "short_reply"
    return "other_loose"


def merge_intervals(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    cur_s, cur_e = ordered[0]
    total = 0.0
    for start, end in ordered[1:]:
        if start <= cur_e:
            cur_e = max(cur_e, end)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = start, end
    total += cur_e - cur_s
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--per-bucket", type=int, default=8)
    args = parser.parse_args()

    started = time.time()
    pairs_index = json.loads(args.index.read_text(encoding="utf-8"))
    if args.limit_pairs:
        pairs_index = pairs_index[: args.limit_pairs]
    by_archive: dict[str, set[str]] = defaultdict(set)
    for row in pairs_index:
        by_archive[row["a"]["archive"]].add(row["a"]["id"])
        by_archive[row["b"]["archive"]].add(row["b"]["id"])
    tasks = [(archive, sorted(stems)) for archive, stems in by_archive.items()]
    print(f"pairs={len(pairs_index)} archives={len(tasks)} workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    utterances_by_id: dict[str, list[list]] = {}
    missing = 0
    errors = 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_tar, task) for task in tasks]
        for fut in as_completed(futures):
            _archive, found, missing_stems = fut.result()
            utterances_by_id.update(found)
            missing += sum(1 for item in missing_stems if not item.startswith("ERR "))
            errors += sum(1 for item in missing_stems if item.startswith("ERR "))
            done += 1
            if done % 500 == 0 or done == len(tasks):
                print(
                    f"archives {done}/{len(tasks)} people={len(utterances_by_id)} missing={missing} "
                    f"{done / max(time.time() - started, 1e-6):.1f} tar/s",
                    flush=True,
                )

    rng = random.Random(args.seed)
    chain_rng = random.Random(args.seed + 17)
    buckets: dict[str, list[dict]] = defaultdict(list)
    chains: list[dict] = []
    n_chains = 0
    shared_sec = 0.0
    per_bucket = args.per_bucket
    sum_sec = 0.0
    old_sec = 0.0
    n_pairs = 0
    n_old = 0
    n_empty = 0
    by_label_sec: dict[str, float] = defaultdict(float)
    by_label_n: dict[str, int] = defaultdict(int)
    by_split_sec: dict[str, float] = defaultdict(float)
    by_split_n: dict[str, int] = defaultdict(int)
    bucket_sec: dict[str, float] = defaultdict(float)
    bucket_n: dict[str, int] = defaultdict(int)
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    duration_hist = defaultdict(int)
    gap_hist = defaultdict(int)

    def add_hist(table: dict[str, int], key: str) -> None:
        table[key] += 1

    for row in pairs_index:
        utt_a = utterances_by_id.get(row["a"]["id"])
        utt_b = utterances_by_id.get(row["b"]["id"])
        if not utt_a or not utt_b:
            n_empty += 1
            continue
        built = build_pairs(row["a"], row["b"], utt_a, utt_b)
        if not built:
            n_empty += 1
            continue
        for offset, pair in enumerate(built):
            n_pairs += 1
            duration = float(pair["duration_s"])
            sum_sec += duration
            label = row.get("label") or "unknown"
            split = row.get("split") or "unknown"
            by_label_sec[label] += duration
            by_label_n[label] += 1
            by_split_sec[split] += duration
            by_split_n[split] += 1
            intervals[row["interaction"]].append((pair["start_s"], pair["end_s"]))
            bucket = reject_bucket(pair)
            bucket_sec[bucket] += duration
            bucket_n[bucket] += 1
            if bucket == "kept_like_old":
                n_old += 1
                old_sec += duration
            if duration < 2:
                add_hist(duration_hist, "<2s")
            elif duration < 8:
                add_hist(duration_hist, "2-8s")
            elif duration <= 12:
                add_hist(duration_hist, "8-12s")
            elif duration <= 24:
                add_hist(duration_hist, "12-24s")
            elif duration <= 60:
                add_hist(duration_hist, "24-60s")
            else:
                add_hist(duration_hist, ">60s")
            gap = float(pair["gap_s"])
            if gap < 0:
                add_hist(gap_hist, "<0 overlap")
            elif gap < 0.12:
                add_hist(gap_hist, "0-0.12s")
            elif gap <= 2.5:
                add_hist(gap_hist, "0.12-2.5s")
            elif gap <= 6:
                add_hist(gap_hist, "2.5-6s")
            else:
                add_hist(gap_hist, ">6s")

            sample = dict(pair)
            sample["interaction"] = row["interaction"]
            sample["label"] = label
            sample["split"] = split
            sample["bucket"] = bucket
            sample["clip_id"] = f"{row['interaction']}_{offset:04d}_{pair['listener_id'].split('_')[-1]}"
            pair["clip_id"] = sample["clip_id"]
            pair["interaction"] = row["interaction"]
            pair["label"] = label
            pair["split"] = split
            pair["bucket"] = bucket
            held = buckets[bucket]
            if len(held) < per_bucket:
                held.append(sample)
            else:
                slot = rng.randrange(bucket_n[bucket])
                if slot < per_bucket:
                    held[slot] = sample
        for left, right in zip(built, built[1:]):
            if left.get("listener_id") != right.get("speaker_id"):
                continue
            shared = left["listener_utt"]
            if abs(shared["start_s"] - right["speaker_utt"]["start_s"]) > 0.05:
                continue
            n_chains += 1
            shared_len = max(0.0, shared["end_s"] - shared["start_s"])
            shared_sec += shared_len
            item = {
                "chain_id": f"{row['interaction']}_{left['clip_id'].rsplit('_', 1)[-1]}_{right['clip_id'].rsplit('_', 1)[-1]}",
                "interaction": row["interaction"],
                "label": row.get("label") or "unknown",
                "split": row.get("split") or "unknown",
                "shared_text": shared.get("text") or "",
                "shared_words": shared.get("nwords") or 0,
                "shared_start_s": shared["start_s"],
                "shared_end_s": shared["end_s"],
                "shared_s": round(shared_len, 3),
                "pair_a": left,
                "pair_b": right,
            }
            if len(chains) < per_bucket:
                chains.append(item)
            else:
                slot = chain_rng.randrange(n_chains)
                if slot < per_bucket:
                    chains[slot] = item

    unique_sec = 0.0
    for spans in intervals.values():
        unique_sec += merge_intervals(spans)

    samples = []
    for bucket, rows in sorted(buckets.items()):
        samples.extend(rows)

    summary = {
        "definition": (
            "词间隔<=0.6s 并成一句；对方整句在本轮结束前说完的算话中反应，不切段；"
            "对方把话说到本轮结束之后，就换成新的一轮。每一对 = 上一轮 speaker + 下一轮 listener，"
            "时长 = listener 结束 - speaker 开始。下一对里，这一轮 listener 变成 speaker。"
        ),
        "pairs_in_index": len(pairs_index),
        "interactions_with_pairs": len(intervals),
        "interactions_without_pairs": n_empty,
        "missing_transcripts": missing,
        "archive_errors": errors,
        "n_pairs": n_pairs,
        "sum_hours": round(sum_sec / 3600.0, 3),
        "unique_coverage_hours": round(unique_sec / 3600.0, 3),
        "adjacent_reuse": {
            "links": n_chains,
            "shared_hours": round(shared_sec / 3600.0, 3),
        },
        "old_text_time_gate_pairs": n_old,
        "old_text_time_gate_hours": round(old_sec / 3600.0, 3),
        "by_label": {
            key: {"pairs": by_label_n[key], "hours": round(by_label_sec[key] / 3600.0, 3)}
            for key in sorted(by_label_n)
        },
        "by_split": {
            key: {"pairs": by_split_n[key], "hours": round(by_split_sec[key] / 3600.0, 3)}
            for key in sorted(by_split_n)
        },
        "by_bucket": {
            key: {"pairs": bucket_n[key], "hours": round(bucket_sec[key] / 3600.0, 3)}
            for key in sorted(bucket_n)
        },
        "duration_hist_pairs": dict(duration_hist),
        "gap_hist_pairs": dict(gap_hist),
        "elapsed_sec": round(time.time() - started, 1),
        "note_old_gate": (
            "kept_like_old 只复现了词数、间隔和时长门，没有复现可见性、抢话词数和 v4+ 的扩段逻辑，"
            "所以小时数不会等于已导出的 482.16h。"
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "preview_samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "preview_chains.jsonl").open("w", encoding="utf-8") as handle:
        for row in chains:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
