#!/usr/bin/env python3
"""Apply the original turn gates, micro tags, and static crop to plain turn-cut pairs.

Audio bleed checks are done when the preview is rendered. This pass records the
keep intervals and the listener VAD shorts that fall inside the speaker turn.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from common import (
    merge_runs,
    probe_mp4_in_tar,
    resolve_person_members,
    stable_official_crop,
    time_to_frame,
)
from scan_alternate_pairs import INDEX_PATH, OUT_DIR, parse_words, process_tar
from scan_turn_pairs import MAX_WORD_S, turn_cut_pairs

OUT = OUT_DIR / "plain_screen"
REASONS = (
    "speaker_short",
    "listener_short",
    "gap",
    "listener_insert",
    "speaker_barge",
    "duration",
    "visibility",
    "no_crop",
    "media",
)


def _span(parts: list[dict]) -> tuple[float, float, int]:
    start = min(float(part["start_s"]) for part in parts)
    end = max(float(part["end_s"]) for part in parts)
    words = sum(int(part["nwords"]) for part in parts)
    return start, end, words


def _words_in(words: list[tuple[float, float, str]], t0: float, t1: float) -> int:
    if t1 <= t0:
        return 0
    return sum(1 for a, b, _tok in words if t0 <= 0.5 * (a + b) <= t1)


def structure_reason(pair: dict) -> str | None:
    spk_start, spk_end, spk_n = _span(pair["speaker_parts"])
    lis_start, lis_end, lis_n = _span(pair["listener_parts"])
    if spk_n < 5 or (spk_end - spk_start) < 1.20:
        return "speaker_short"
    if lis_n < 5 or (lis_end - lis_start) < 1.20:
        return "listener_short"
    gap = lis_start - spk_end
    overlap = max(0.0, spk_end - lis_start)
    if gap < 0.12 or gap > 2.50 or overlap > 0.20:
        return "gap"
    if (lis_end - lis_start) < 0.80 or (lis_end - lis_start) > 20.0 or (spk_end - spk_start) > 12.0:
        return "duration"
    clip_start = max(0.0, spk_start - 0.20)
    if (lis_end + 0.30) - clip_start < 2.0 or (lis_end + 0.30) - clip_start > 24.0:
        return "duration"
    return None


def lex_reason(pair: dict, speaker_words: list, listener_words: list) -> str | None:
    reason = structure_reason(pair)
    if reason:
        return reason
    spk_start, spk_end, _spk_n = _span(pair["speaker_parts"])
    lis_start, lis_end, _lis_n = _span(pair["listener_parts"])
    if _words_in(listener_words, spk_start + 0.25, min(spk_end, lis_start) - 0.05) >= 4:
        return "listener_insert"
    if _words_in(speaker_words, lis_start + 0.20, lis_end - 0.10) >= 3:
        return "speaker_barge"
    return None


def _shorts(vad: list[tuple[float, float]]) -> list[tuple[float, float]]:
    runs = merge_runs(vad, gap=0.12, min_len=0.08)
    return [(a, b) for a, b in runs if (b - a) <= 0.70]


def _load_track(archive: str, stem: str) -> dict:
    members = resolve_person_members(Path(archive), stem)
    import tarfile

    with tarfile.open(archive, "r:") as tar:
        json_member = None
        npz_member = None
        for info in tar:
            if not info.isfile():
                continue
            name = Path(info.name).name
            if name == Path(members["json"]).name:
                json_member = info
            elif name == Path(members["npz"]).name:
                npz_member = info
        if json_member is None or npz_member is None:
            raise FileNotFoundError(stem)
        import io
        meta = json.loads(tar.extractfile(json_member).read().decode("utf-8"))
        npz = np.load(io.BytesIO(tar.extractfile(npz_member).read()), allow_pickle=True)
        official = npz["boxes_and_keypoints:box"].astype(np.float32)
        valid = npz["boxes_and_keypoints:is_valid_box"].astype(np.uint8)
        keypoints = npz["boxes_and_keypoints:keypoints"].astype(np.float32)
    video = probe_mp4_in_tar(Path(archive), members["mp4"])
    fps = float(video.get("fps") or 0.0)
    if fps <= 1e-3:
        fps = 30.0
    width = int(video.get("width") or 1080)
    height = int(video.get("height") or 1920)
    vad = [(float(item["start"]), float(item["end"])) for item in (meta.get("metadata:vad") or [])]
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "n_frames": int(official.shape[0]),
        "duration": float(video.get("duration_s") or (official.shape[0] / fps)),
        "official": official,
        "valid": valid,
        "keypoints": keypoints,
        "words": parse_words(meta),
        "vad": vad,
        "shorts": _shorts(vad),
    }


def _visible(track: dict, t0: float, t1: float) -> float:
    f0 = max(0, time_to_frame(t0, track["fps"]))
    f1 = min(track["n_frames"], max(f0 + 1, time_to_frame(t1, track["fps"])))
    sl = track["valid"][f0:f1]
    return float(sl.mean()) if sl.size else 0.0


def _nearby_text(words: list, t0: float, t1: float) -> str:
    toks = [tok for a, b, tok in words if t0 - 0.10 <= 0.5 * (a + b) <= t1 + 0.10]
    return " ".join(toks).strip()


def finish_pair(pair: dict, speaker: dict, listener: dict) -> tuple[str | None, dict | None]:
    reason = lex_reason(pair, speaker["words"], listener["words"])
    if reason:
        return reason, None
    spk_start, spk_end, _spk_n = _span(pair["speaker_parts"])
    lis_start, lis_end, _lis_n = _span(pair["listener_parts"])
    clip_start = max(0.0, spk_start - 0.20)
    clip_end = min(speaker["duration"], listener["duration"], lis_end + 0.30)
    if clip_end < lis_end - 0.03:
        return "duration", None
    dur = clip_end - clip_start
    if dur < 2.0 or dur > 24.0:
        return "duration", None
    lis_vis = _visible(listener, clip_start, clip_end)
    spk_vis = _visible(speaker, clip_start, clip_end)
    if lis_vis < 0.75 or spk_vis < 0.40:
        return "visibility", None
    f0 = max(0, min(listener["n_frames"] - 1, time_to_frame(clip_start, listener["fps"])))
    f1 = max(f0 + 1, min(listener["n_frames"], time_to_frame(clip_end, listener["fps"])))
    box = stable_official_crop(
        listener["official"][f0:f1],
        listener["valid"][f0:f1],
        listener["keypoints"][f0:f1],
        width=listener["width"],
        height=listener["height"],
    )
    if not box:
        return "no_crop", None
    micros = []
    for a0, a1 in listener["shorts"]:
        if a0 < clip_start - 0.05 or a1 > clip_end + 0.05:
            continue
        if min(a1, spk_end) - max(a0, spk_start) < 0.04:
            continue
        if a1 > lis_start + 0.12:
            continue
        text = _nearby_text(listener["words"], a0, a1)
        if len(text.split()) >= 5:
            continue
        micros.append({
            "start_s": round(a0, 3),
            "end_s": round(a1, 3),
            "text": text or "没有转写",
        })
    speaker_keep = [
        (round(max(clip_start, float(part["start_s"]) - 0.05), 3), round(min(clip_end, float(part["end_s"]) + 0.08), 3))
        for part in pair["speaker_parts"]
    ]
    listener_keep = [
        (round(max(clip_start, float(part["start_s"]) - 0.08), 3), round(min(clip_end, float(part["end_s"]) + 0.12), 3))
        for part in pair["listener_parts"]
    ]
    listener_keep.extend(
        (round(max(clip_start, item["start_s"] - 0.06), 3), round(min(clip_end, item["end_s"] + 0.10), 3))
        for item in micros
    )
    row = dict(pair)
    row.update({
        "clip_start_s": round(clip_start, 3),
        "clip_end_s": round(clip_end, 3),
        "clip_duration_s": round(dur, 3),
        "gap_s": round(lis_start - spk_end, 3),
        "speaker_span_s": round(spk_end - spk_start, 3),
        "listener_span_s": round(lis_end - lis_start, 3),
        "listener_visible": round(lis_vis, 3),
        "speaker_visible": round(spk_vis, 3),
        "fps": listener["fps"],
        "width": listener["width"],
        "height": listener["height"],
        "static_official": box,
        "micros": micros,
        "keep_speaker_audio": speaker_keep,
        "keep_listener_audio": merge_runs(listener_keep, gap=0.12, min_len=0.06),
    })
    return None, row


def _prefilter(pairs_index: list[dict], utterances_by_id: dict[str, list]) -> tuple[dict[str, list], dict[str, int]]:
    grouped: dict[str, list] = defaultdict(list)
    reasons = defaultdict(int)
    for row in pairs_index:
        utt_a = utterances_by_id.get(row["a"]["id"])
        utt_b = utterances_by_id.get(row["b"]["id"])
        if not utt_a or not utt_b:
            continue
        words = {"a": [], "b": []}
        for pair in turn_cut_pairs(row["a"], row["b"], utt_a, utt_b):
            if pair["bucket"] != "plain":
                continue
            reason = structure_reason(pair)
            if reason:
                reasons[reason] += 1
                continue
            grouped[row["interaction"]].append(pair)
    return grouped, reasons


def _screen_task(payload: tuple[dict, list[dict]]) -> dict:
    row, pairs = payload
    reasons = defaultdict(int)
    kept = []
    try:
        tracks = {
            "a": _load_track(row["a"]["archive"], row["a"]["id"]),
            "b": _load_track(row["b"]["archive"], row["b"]["id"]),
        }
        for pair in pairs:
            who = "a" if pair["speaker_id"] == row["a"]["id"] else "b"
            other = "b" if who == "a" else "a"
            reason, done = finish_pair(pair, tracks[who], tracks[other])
            if reason:
                reasons[reason] += 1
            else:
                done["interaction"] = row["interaction"]
                done["label"] = row.get("label") or "unknown"
                done["split"] = row.get("split") or "unknown"
                kept.append(done)
    except Exception as exc:  # noqa: BLE001
        return {"reasons": {"media": len(pairs)}, "kept": [], "error": f"{type(exc).__name__}: {exc}"}
    return {"reasons": dict(reasons), "kept": kept, "error": None}


def main() -> None:
    started = time.time()
    pairs_index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    by_archive: dict[str, set[str]] = defaultdict(set)
    for row in pairs_index:
        by_archive[row["a"]["archive"]].add(row["a"]["id"])
        by_archive[row["b"]["archive"]].add(row["b"]["id"])
    tasks = [(archive, sorted(stems), MAX_WORD_S) for archive, stems in by_archive.items()]
    print(f"archives={len(tasks)}", flush=True)
    utterances_by_id: dict[str, list] = {}
    with ProcessPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(process_tar, task) for task in tasks]
        done = 0
        for fut in as_completed(futures):
            _archive, found, _missing = fut.result()
            utterances_by_id.update(found)
            done += 1
            if done % 4000 == 0 or done == len(tasks):
                print(f"utterances {done}/{len(tasks)}", flush=True)
    grouped, pre_reasons = _prefilter(pairs_index, utterances_by_id)
    print(f"interactions_after_lex={len(grouped)} pre={dict(pre_reasons)}", flush=True)
    index_by_id = {row["interaction"]: row for row in pairs_index}
    jobs = [(index_by_id[key], pairs) for key, pairs in grouped.items()]
    reasons = defaultdict(int, pre_reasons)
    OUT.mkdir(parents=True, exist_ok=True)
    kept_path = OUT / "kept.jsonl"
    done_ids: set[str] = set()
    if kept_path.exists() and kept_path.stat().st_size:
        kept_lines = []
        last_id = None
        with kept_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                last_id = json.loads(line)["interaction"]
                kept_lines.append(line)
        if last_id is not None:
            kept_lines = [line for line in kept_lines if json.loads(line)["interaction"] != last_id]
        done_ids = {json.loads(line)["interaction"] for line in kept_lines}
        kept_path.write_text("".join(kept_lines), encoding="utf-8")
        jobs = [job for job in jobs if job[0]["interaction"] not in done_ids]
        print(f"resume interactions_done={len(done_ids)} remaining={len(jobs)}", flush=True)
    kept_n = 0
    with_micro = 0
    hours = 0.0
    errors = 0
    with ProcessPoolExecutor(max_workers=8) as pool, kept_path.open("a", encoding="utf-8") as handle:
        futures = [pool.submit(_screen_task, job) for job in jobs]
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            for key, count in result["reasons"].items():
                reasons[key] += count
            for row in result["kept"]:
                row.pop("backchannels", None)
                listener = row["listener_id"].split("_")[-1]
                row["clip_id"] = f"{row['interaction']}_{int(round(row['clip_start_s'] * 1000)):08d}_{listener}"
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                kept_n += 1
                hours += float(row["clip_duration_s"])
                if row["micros"]:
                    with_micro += 1
            if result["error"]:
                errors += 1
            done += 1
            if done % 500 == 0 or done == len(jobs):
                print(f"screen {done}/{len(jobs)} kept={kept_n}", flush=True)
    full_n = 0
    full_hours = 0.0
    full_micro = 0
    with kept_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            full_n += 1
            full_hours += float(row["clip_duration_s"])
            if row.get("micros"):
                full_micro += 1
    prior_path = OUT / "summary_run1.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {}
    summary = {
        "source": "plain turn-cut pairs",
        "source_pairs": 554831,
        "kept_pairs": full_n,
        "kept_hours": round(full_hours / 3600.0, 3),
        "with_micro": full_micro,
        "manifest": "kept.jsonl",
        "reject": prior.get("reject") if full_n == prior.get("kept_pairs") else {key: int(reasons.get(key, 0)) for key in REASONS},
        "media_errors": prior.get("media_errors", errors) if full_n == prior.get("kept_pairs") else errors,
        "elapsed_sec": round(time.time() - started, 1),
        "resumed_new_pairs": kept_n,
        "note": "samples.jsonl and index.html are the earlier 8-clip preview and are not rewritten by this manifest pass.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
