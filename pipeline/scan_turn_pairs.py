#!/usr/bin/env python3
"""Cut the full Seamless set into turns.

Rules and the three constants live here and in NOTES_TURN_CUT.md.
Change the constants, then rerun this scan and render_turn_cut.py.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from scan_alternate_pairs import INDEX_PATH, OUT_DIR, merge_intervals, process_tar

# Revisable knobs. See NOTES_TURN_CUT.md before changing them.
BACKCHANNEL_MAX_WORDS = 3
RESUME_WITHIN_S = 1.0
END_SLACK_S = 0.05
# One ASR word longer than this is trimmed before utterances are built.
# Stretched tokens such as a 7s "Okay." otherwise cover the other person's reply.
MAX_WORD_S = 1.0
# Case 2: overlap = speaker floor end - listener floor start.
# At most this many seconds counts as an immediate reply and stays.
# Longer overlap is set aside.
CASE2_KEEP_OVERLAP_S = 0.5

TURN_DIR = OUT_DIR / "turn_cut"
BUCKETS = (
    "with_backchannel",
    "plain",
    "case2_immediate",
    "case2_dropped",
)
KEPT_BUCKETS = ("with_backchannel", "plain", "case2_immediate")


def _events(groups: dict[str, list]) -> list[dict]:
    events = []
    for who, rows in groups.items():
        for row in rows:
            start, end, text, nwords = float(row[0]), float(row[1]), str(row[2]), int(row[3])
            if end <= start:
                continue
            events.append({
                "who": who,
                "start": start,
                "end": end,
                "text": text,
                "nwords": nwords,
            })
    events.sort(key=lambda row: (row["start"], row["end"], row["who"]))
    return events


def _next_start(events: list[dict], begin: int, who: str) -> float | None:
    for event in events[begin:]:
        if event["who"] == who:
            return event["start"]
    return None


def _new_floor(utt: dict, case2: bool) -> dict:
    return {
        "who": utt["who"],
        "start": utt["start"],
        "end": utt["end"],
        "nwords": utt["nwords"],
        "parts": [utt],
        "backchannels": [],
        "case2": case2,
    }


def build_floors(groups: dict[str, list]) -> list[dict]:
    """One floor per held turn. Backchannels stay inside the current floor."""
    events = _events(groups)
    floors: list[dict] = []
    for index, utt in enumerate(events):
        if not floors:
            floors.append(_new_floor(utt, False))
            continue
        floor = floors[-1]
        if utt["who"] == floor["who"]:
            floor["end"] = max(floor["end"], utt["end"])
            floor["nwords"] += utt["nwords"]
            floor["parts"].append(utt)
            continue
        if utt["nwords"] <= BACKCHANNEL_MAX_WORDS and utt["end"] <= floor["end"] + END_SLACK_S:
            floor["backchannels"].append({**utt, "kind": "ended_before_speaker"})
            continue
        if utt["start"] < floor["end"] - END_SLACK_S and utt["end"] > floor["end"] + END_SLACK_S:
            floors.append(_new_floor(utt, True))
            continue
        resume = _next_start(events, index + 1, floor["who"])
        if (
            utt["nwords"] <= BACKCHANNEL_MAX_WORDS
            and resume is not None
            and resume <= utt["end"] + RESUME_WITHIN_S
        ):
            floor["backchannels"].append({**utt, "kind": "short_then_speaker_resumes"})
            continue
        floors.append(_new_floor(utt, False))
    return floors


def _pack_part(part: dict) -> dict:
    return {
        "start_s": round(part["start"], 3),
        "end_s": round(part["end"], 3),
        "text": part["text"],
        "nwords": part["nwords"],
        "who": part["who"],
    }


def _bucket(speaker: dict, listener: dict) -> str:
    if listener["case2"]:
        overlap = speaker["end"] - listener["start"]
        if overlap > CASE2_KEEP_OVERLAP_S:
            return "case2_dropped"
        return "case2_immediate"
    if speaker["backchannels"] or listener["backchannels"]:
        return "with_backchannel"
    return "plain"


def turn_cut_pairs(person_a: dict, person_b: dict, utts_a: list, utts_b: list) -> list[dict]:
    floors = build_floors({"a": utts_a, "b": utts_b})
    people = {"a": person_a, "b": person_b}
    pairs = []
    for index in range(len(floors) - 1):
        speaker = floors[index]
        listener = floors[index + 1]
        word_end = max(speaker["end"], listener["end"])
        nxt = floors[index + 2] if index + 2 < len(floors) else None
        end = nxt["start"] if nxt is not None and nxt["start"] > word_end else word_end
        if end - speaker["start"] <= 0.2:
            continue
        backchannels = []
        for item in speaker["backchannels"] + listener["backchannels"]:
            if item["end"] < speaker["start"] - 0.02 or item["start"] > end + 0.02:
                continue
            packed = _pack_part(item)
            packed["kind"] = item["kind"]
            # After the listener turn starts, the speaker's extra inserts are silenced.
            # The listener's own short reactions, such as "Hmm.", stay.
            # Those silenced inserts are not the listener turn, so the next pair
            # still uses the listener turn as its speaker.
            if item["who"] == speaker["who"] and item["start"] >= listener["start"] - END_SLACK_S:
                packed["audio"] = "mute"
            else:
                packed["audio"] = "keep"
            backchannels.append(packed)
        spk_person = people[speaker["who"]]
        lis_person = people[listener["who"]]
        pairs.append({
            "start_s": round(speaker["start"], 3),
            "end_s": round(end, 3),
            "word_end_s": round(word_end, 3),
            "duration_s": round(end - speaker["start"], 3),
            "speaker_id": spk_person["id"],
            "listener_id": lis_person["id"],
            "speaker_archive": spk_person["archive"],
            "listener_archive": lis_person["archive"],
            "speaker_parts": [_pack_part(part) for part in speaker["parts"]],
            "listener_parts": [_pack_part(part) for part in listener["parts"]],
            "backchannels": backchannels,
            "overlap_s": round(speaker["end"] - listener["start"], 3),
            "bucket": _bucket(speaker, listener),
        })
    return pairs


def _hist(edges: list[float], value: float, counts: list[int]) -> None:
    for index, edge in enumerate(edges[1:]):
        if value < edge:
            counts[index] += 1
            return
    counts[-1] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--out-dir", type=Path, default=TURN_DIR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--per-bucket", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    started = time.time()
    pairs_index = json.loads(args.index.read_text(encoding="utf-8"))
    by_archive: dict[str, set[str]] = defaultdict(set)
    for row in pairs_index:
        by_archive[row["a"]["archive"]].add(row["a"]["id"])
        by_archive[row["b"]["archive"]].add(row["b"]["id"])
    tasks = [(archive, sorted(stems), MAX_WORD_S) for archive, stems in by_archive.items()]
    print(f"pairs={len(pairs_index)} archives={len(tasks)}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    utterances_by_id: dict[str, list] = {}
    missing = 0
    errors = 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_tar, task) for task in tasks]
        for fut in as_completed(futures):
            _archive, found, missing_stems = fut.result()
            utterances_by_id.update(found)
            missing += sum(1 for item in missing_stems if not str(item).startswith("ERR "))
            errors += sum(1 for item in missing_stems if str(item).startswith("ERR "))
            done += 1
            if done % 500 == 0 or done == len(tasks):
                print(f"archives {done}/{len(tasks)} people={len(utterances_by_id)}", flush=True)

    rng = random.Random(args.seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    bucket_n: dict[str, int] = defaultdict(int)
    bucket_sec: dict[str, float] = defaultdict(float)
    kept_sec = 0.0
    kept_n = 0
    n_empty = 0
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    duration_edges = [0, 8, 20, 45, 90, 180, 10**9]
    duration_counts = [0] * (len(duration_edges) - 1)
    per_bucket = args.per_bucket

    for row in pairs_index:
        utt_a = utterances_by_id.get(row["a"]["id"])
        utt_b = utterances_by_id.get(row["b"]["id"])
        if not utt_a or not utt_b:
            n_empty += 1
            continue
        built = turn_cut_pairs(row["a"], row["b"], utt_a, utt_b)
        if not built:
            n_empty += 1
            continue
        for offset, pair in enumerate(built):
            bucket = pair["bucket"]
            duration = float(pair["duration_s"])
            bucket_n[bucket] += 1
            bucket_sec[bucket] += duration
            if bucket in KEPT_BUCKETS:
                kept_n += 1
                kept_sec += duration
                _hist(duration_edges, duration, duration_counts)
                intervals[row["interaction"]].append((pair["start_s"], pair["end_s"]))
            sample = dict(pair)
            sample["interaction"] = row["interaction"]
            sample["label"] = row.get("label") or "unknown"
            sample["split"] = row.get("split") or "unknown"
            sample["clip_id"] = f"{row['interaction']}_{offset:04d}_{pair['listener_id'].split('_')[-1]}"
            held = buckets[bucket]
            seen = bucket_n[bucket]
            if len(held) < per_bucket:
                held.append(sample)
            else:
                slot = rng.randrange(seen)
                if slot < per_bucket:
                    held[slot] = sample

    def _pin(interaction: str, note: str, match) -> None:
        for row in pairs_index:
            if row["interaction"] != interaction:
                continue
            built = turn_cut_pairs(
                row["a"], row["b"],
                utterances_by_id.get(row["a"]["id"]) or [],
                utterances_by_id.get(row["b"]["id"]) or [],
            )
            for offset, pair in enumerate(built):
                if not match(pair):
                    continue
                sample = dict(pair)
                sample["interaction"] = row["interaction"]
                sample["label"] = row.get("label") or "unknown"
                sample["split"] = row.get("split") or "unknown"
                sample["clip_id"] = f"{row['interaction']}_{offset:04d}_{pair['listener_id'].split('_')[-1]}_pin"
                sample["pinned"] = True
                sample["pin_note"] = note
                buckets[pair["bucket"]].insert(0, sample)
                return

    _pin(
        "V01_S0083_I00000140",
        "这是 P1158 那条例子。Are they still together? 有 4 个词，已经单独成轮。",
        lambda pair: any("Are they still together?" in part["text"] for part in pair["listener_parts"]),
    )
    _pin(
        "V00_S0895_I00000817",
        "这是 P0825A 那条例子。超过 3 个词的回答已经单独成轮。",
        lambda pair: any("There is no down" in part["text"] for part in pair["listener_parts"]),
    )

    unique_sec = sum(merge_intervals(spans) for spans in intervals.values())
    labels = ["0-8s", "8-20s", "20-45s", "45-90s", "90-180s", ">180s"]
    summary = {
        "rule_version": "2026-10-16-turn-cut-v3",
        "notes": "NOTES_TURN_CUT.md",
        "constants": {
            "backchannel_max_words": BACKCHANNEL_MAX_WORDS,
            "resume_within_s": RESUME_WITHIN_S,
            "end_slack_s": END_SLACK_S,
            "case2_keep_overlap_s": CASE2_KEEP_OVERLAP_S,
            "max_word_s": MAX_WORD_S,
            "utterance_word_gap_s": 0.6,
            "min_utterance_s": 0.15,
        },
        "definition": (
            "超过 3 个词的句子单独成轮，即使它在对方这轮的时间结束前停了。不超过 3 个词、并在对方说完前结束的，仍留在当前轮。单个词最长 1 秒，拉长的词尾先裁掉。"
            "听众自己的短反应保留。听众开口之后，说话人又插进来的句子从这一对的音轨去掉；下一对仍把听众这一整轮当成说话人。"
            "插话一直说到对方结束之后时，重叠不超过 0.5 秒的算紧接着，先保留；超过 0.5 秒的明显重叠先拿掉。"
        ),
        "interactions_without_pairs": n_empty,
        "missing_transcripts": missing,
        "archive_errors": errors,
        "kept_pairs": kept_n,
        "kept_hours": round(kept_sec / 3600.0, 3),
        "kept_unique_hours": round(unique_sec / 3600.0, 3),
        "by_bucket": {
            key: {"pairs": bucket_n[key], "hours": round(bucket_sec[key] / 3600.0, 3)}
            for key in BUCKETS
        },
        "duration_hist": {label: duration_counts[index] for index, label in enumerate(labels)},
        "elapsed_sec": round(time.time() - started, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.out_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for bucket in BUCKETS:
            for row in buckets.get(bucket, []):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
