#!/usr/bin/env python3
"""Shared helpers for the Seamless listener-response demo pipeline."""

from __future__ import annotations

import json
import struct
import shutil
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

DATA_ROOT = Path("/public/0_DATA/2_Avatar/seamless_interaction")
PIPELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = PIPELINE_ROOT / "outputs_demo100"
SEED = 20260831
FPS_FALLBACK = 30.0
FRAME_W = 1080
FRAME_H = 1920
PERSON_EXTS = ("mp4", "wav", "json", "npz")
NPZ_VIDEO_FRAME_TOL = 2
NPZ_VIDEO_TIME_TOL = 0.08
ALL_SPLITS = (
    ("improvised", "dev"),
    ("improvised", "test"),
    ("improvised", "train"),
    ("naturalistic", "dev"),
    ("naturalistic", "test"),
    ("naturalistic", "train"),
)
PREVIEW_SPLITS = (("improvised", "dev"), ("improvised", "test"), ("naturalistic", "dev"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_tars(root: Path, splits: tuple[tuple[str, str], ...] = ALL_SPLITS) -> list[Path]:
    tars: list[Path] = []
    for label, split in splits:
        tars.extend(sorted((root / label / split).glob("[0-9][0-9][0-9][0-9]/*.tar")))
    return tars


def build_pair_index(
    root: Path,
    *,
    splits: tuple[tuple[str, str], ...] = ALL_SPLITS,
) -> list[dict]:
    """Complete dyads: same V_S_I, two participants, each has mp4/wav/json/npz."""
    inter: dict[str, list[dict]] = defaultdict(list)
    tars = iter_tars(root, splits)
    for tar_path in tars:
        with tarfile.open(tar_path, "r:") as tar:
            names = tar.getnames()
        by: dict[str, set[str]] = defaultdict(set)
        for name in names:
            if "." not in name:
                continue
            stem, ext = name.rsplit(".", 1)
            by[stem].add(ext)
        label = tar_path.parts[-4]
        split = tar_path.parts[-3]
        for stem, exts in by.items():
            parts = stem.split("_")
            if len(parts) != 4:
                continue
            _v, _s, _i, p = parts
            inter[f"{parts[0]}_{parts[1]}_{parts[2]}"].append(
                {
                    "id": stem,
                    "p": p,
                    "archive": str(tar_path),
                    "label": label,
                    "split": split,
                    "complete": {"mp4", "wav", "json", "npz"} <= exts,
                }
            )
    pairs = []
    for key, rows in inter.items():
        uniq: dict[str, dict] = {}
        for row in rows:
            if row["complete"]:
                uniq.setdefault(row["p"], row)
        ok = list(uniq.values())
        if len(ok) >= 2:
            pairs.append(
                {
                    "interaction": key,
                    "a": ok[0],
                    "b": ok[1],
                    "label": ok[0]["label"],
                    "split": ok[0]["split"],
                }
            )
    return pairs


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {err}")


def ffprobe_json(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(proc.stdout.decode("utf-8"))


def _parse_rate(txt: str | None) -> float | None:
    if not txt or txt in {"0/0", "0", "N/A"}:
        return None
    if "/" in txt:
        num_s, den_s = txt.split("/", 1)
        den = float(den_s)
        if den == 0:
            return None
        return float(num_s) / den
    try:
        val = float(txt)
    except ValueError:
        return None
    return val if val > 0 else None


def video_meta_from_ffprobe(info: dict) -> dict:
    """Parse ffprobe JSON into a stable video-meta dict."""
    streams = info.get("streams") or []
    vstream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if vstream is None:
        raise ValueError("ffprobe: no video stream")
    fps = _parse_rate(vstream.get("avg_frame_rate")) or _parse_rate(vstream.get("r_frame_rate"))
    if fps is None:
        fps = FPS_FALLBACK
    width = int(vstream.get("width") or FRAME_W)
    height = int(vstream.get("height") or FRAME_H)
    n_frames = None
    for key in ("nb_frames", "nb_read_frames"):
        raw = vstream.get(key)
        if raw not in (None, "N/A", ""):
            n_frames = int(raw)
            break
    duration = None
    for raw in (vstream.get("duration"), (info.get("format") or {}).get("duration")):
        if raw not in (None, "N/A", ""):
            duration = float(raw)
            break
    if duration is None and n_frames is not None and fps:
        duration = n_frames / fps
    if n_frames is None and duration is not None and fps:
        n_frames = int(round(duration * fps))
    return {
        "fps": float(fps),
        "width": width,
        "height": height,
        "n_frames": int(n_frames) if n_frames is not None else None,
        "duration_s": float(duration) if duration is not None else None,
        "source": "ffprobe",
    }


_TAR_NAME_CACHE: dict[str, list[str]] = {}


def list_tar_names(archive: Path) -> list[str]:
    """Process-local cache of tar member names. Same archive is reused across pairs."""
    key = str(archive)
    hit = _TAR_NAME_CACHE.get(key)
    if hit is None:
        with tarfile.open(archive, "r:") as tar:
            hit = tar.getnames()
        _TAR_NAME_CACHE[key] = hit
    return hit


def resolve_person_members(archive: Path, stem: str) -> dict[str, str]:
    """Map mp4/wav/json/npz to the exact tar member name for one person stem."""
    wanted = {ext: f"{stem}.{ext}" for ext in PERSON_EXTS}
    found: dict[str, str] = {}
    names = list_tar_names(archive)
    by_base = {Path(name).name: name for name in names}
    for ext, expect in wanted.items():
        if expect in names:
            found[ext] = expect
        elif expect in by_base:
            found[ext] = by_base[expect]
        else:
            hits = [n for n in names if Path(n).name == expect or n.endswith("/" + expect)]
            if len(hits) == 1:
                found[ext] = hits[0]
    missing = [ext for ext in PERSON_EXTS if ext not in found]
    if missing:
        raise FileNotFoundError(f"{stem} missing {missing} in {archive}")
    return found


def _tar_member_info(archive: Path, member: str) -> tarfile.TarInfo:
    with tarfile.open(archive, "r:") as tar:
        try:
            return tar.getmember(member)
        except KeyError:
            info = tar.getmember(Path(member).name)
            return info


def read_tar_slice(archive: Path, member: str, offset: int, length: int) -> bytes:
    """Read a byte slice from an uncompressed tar member without extracting it."""
    info = _tar_member_info(archive, member)
    start = max(0, int(offset))
    length = max(0, int(length))
    if start >= info.size or length == 0:
        return b""
    length = min(length, info.size - start)
    with open(archive, "rb") as handle:
        handle.seek(info.offset_data + start)
        return handle.read(length)


def _iter_mp4_boxes(buf: bytes, start: int = 0, end: int | None = None):
    i = start
    limit = len(buf) if end is None else end
    while i + 8 <= limit:
        size = struct.unpack(">I", buf[i : i + 4])[0]
        typ = buf[i + 4 : i + 8]
        hdr = 8
        if size == 1:
            if i + 16 > limit:
                break
            size = struct.unpack(">Q", buf[i + 8 : i + 16])[0]
            hdr = 16
        elif size == 0:
            size = limit - i
        if size < hdr:
            break
        if i + size > limit:
            yield typ, i, hdr, limit - i, True
            break
        yield typ, i, hdr, size, False
        i += size


_MP4_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"}


def _walk_mp4(buf: bytes, start: int = 0, end: int | None = None, path: str = ""):
    for typ, off, hdr, size, truncated in _iter_mp4_boxes(buf, start, end):
        name = typ.decode("latin1", errors="replace")
        cur = f"{path}/{name}" if path else name
        payload = off + hdr
        stop = off + size
        yield cur, typ, off, hdr, size, truncated
        if (not truncated) and typ in _MP4_CONTAINERS:
            yield from _walk_mp4(buf, payload, stop, cur)


def _parse_mdhd(body: bytes) -> tuple[int, int]:
    ver = body[0]
    if ver == 1:
        timescale = struct.unpack(">I", body[20:24])[0]
        duration = struct.unpack(">Q", body[24:32])[0]
    else:
        timescale = struct.unpack(">I", body[12:16])[0]
        duration = struct.unpack(">I", body[16:20])[0]
    return int(timescale), int(duration)


def _parse_tkhd_wh(body: bytes) -> tuple[float, float]:
    ver = body[0]
    if ver == 1:
        width = struct.unpack(">I", body[88:92])[0] / 65536.0
        height = struct.unpack(">I", body[92:96])[0] / 65536.0
    else:
        width = struct.unpack(">I", body[76:80])[0] / 65536.0
        height = struct.unpack(">I", body[80:84])[0] / 65536.0
    return width, height


def _parse_stts(body: bytes) -> tuple[int, int]:
    n_ent = struct.unpack(">I", body[4:8])[0]
    frames = 0
    ticks = 0
    pos = 8
    for _ in range(n_ent):
        if pos + 8 > len(body):
            break
        count, delta = struct.unpack(">II", body[pos : pos + 8])
        frames += int(count)
        ticks += int(count) * int(delta)
        pos += 8
    return frames, ticks


def _parse_stsz_count(body: bytes) -> int | None:
    if len(body) < 12:
        return None
    return int(struct.unpack(">I", body[8:12])[0])


def _parse_stsd_visual_wh(body: bytes) -> tuple[int, int] | None:
    """Coded pixel size from the first visual sample entry (avc1/hev1/...)."""
    if len(body) < 16:
        return None
    n_ent = struct.unpack(">I", body[4:8])[0]
    if n_ent < 1 or len(body) < 16:
        return None
    entry = body[8:]
    if len(entry) < 36:
        return None
    fmt = entry[4:8]
    if fmt in {b"mp4a", b"ec-3", b"ac-3", b"samr", b"sawb"}:
        return None
    width = struct.unpack(">H", entry[32:34])[0]
    height = struct.unpack(">H", entry[34:36])[0]
    if width < 16 or height < 16:
        return None
    return int(width), int(height)


def parse_mp4_moov(moov: bytes) -> dict:
    tracks: list[dict] = []
    cur: dict | None = None
    for _path, typ, off, hdr, size, truncated in _walk_mp4(moov):
        if truncated:
            continue
        body = moov[off + hdr : off + size]
        if typ == b"trak":
            cur = {}
            tracks.append(cur)
        if cur is None:
            continue
        if typ == b"hdlr" and len(body) >= 12:
            cur["hdlr"] = body[8:12]
        elif typ == b"mdhd" and len(body) >= 20:
            ts, dur = _parse_mdhd(body)
            cur["timescale"] = ts
            cur["media_duration"] = dur
        elif typ == b"tkhd" and len(body) >= 84:
            w, h = _parse_tkhd_wh(body)
            cur["width"] = w
            cur["height"] = h
        elif typ == b"stts" and len(body) >= 8:
            frames, ticks = _parse_stts(body)
            cur["n_frames"] = frames
            cur["stts_ticks"] = ticks
        elif typ == b"stsz" and len(body) >= 12:
            cur["stsz_n"] = _parse_stsz_count(body)
        elif typ == b"stsd" and len(body) >= 16:
            wh = _parse_stsd_visual_wh(body)
            if wh is not None:
                cur["coded_width"], cur["coded_height"] = wh

    video = None
    for track in tracks:
        if track.get("hdlr") == b"vide":
            video = track
            break
    if video is None:
        for track in tracks:
            if (track.get("width") or 0) >= 16 and (track.get("height") or 0) >= 16:
                video = track
                break
    if video is None:
        raise ValueError("mp4 moov: no video track")

    timescale = int(video.get("timescale") or 0)
    media_duration = int(video.get("media_duration") or 0)
    n_frames = int(video.get("n_frames") or video.get("stsz_n") or 0)
    ticks = int(video.get("stts_ticks") or 0)
    if timescale and ticks and n_frames:
        fps = timescale / (ticks / n_frames)
        duration_s = ticks / timescale
    elif timescale and n_frames and media_duration:
        duration_s = media_duration / timescale
        fps = n_frames / duration_s if duration_s else FPS_FALLBACK
    elif timescale and media_duration:
        duration_s = media_duration / timescale
        fps = FPS_FALLBACK
    else:
        duration_s = None
        fps = FPS_FALLBACK
    if video.get("coded_width") and video.get("coded_height"):
        width = int(video["coded_width"])
        height = int(video["coded_height"])
    else:
        width = int(round(float(video.get("width") or FRAME_W)))
        height = int(round(float(video.get("height") or FRAME_H)))
    return {
        "fps": float(fps),
        "width": width,
        "height": height,
        "n_frames": int(n_frames) if n_frames else None,
        "duration_s": float(duration_s) if duration_s is not None else None,
        "source": "mp4_moov",
    }


def _moov_candidates(blob: bytes) -> list[bytes]:
    found: list[bytes] = []
    for typ, off, hdr, size, truncated in _iter_mp4_boxes(blob):
        if typ == b"moov" and not truncated and size >= 256:
            found.append(blob[off : off + size])
    start = 0
    while True:
        idx = blob.find(b"moov", start)
        if idx < 0:
            break
        start = idx + 4
        if idx < 4:
            continue
        size = struct.unpack(">I", blob[idx - 4 : idx])[0]
        begin = idx - 4
        if size < 256 or begin + size > len(blob):
            continue
        chunk = blob[begin : begin + size]
        if chunk not in found:
            found.append(chunk)
    return found


def _parse_first_valid_moov(blob: bytes) -> dict | None:
    for moov in _moov_candidates(blob):
        try:
            return parse_mp4_moov(moov)
        except Exception:  # noqa: BLE001
            continue
    return None


def probe_mp4_in_tar(archive: Path, member: str, *, tail_bytes: int = 16 << 20) -> dict:
    """Read MP4 fps/size/duration from an uncompressed tar member. No full extract."""
    info = _tar_member_info(archive, member)
    head = read_tar_slice(archive, member, 0, 64)
    if head[4:8] not in {b"ftyp", b"mdat", b"moov", b"free", b"wide"}:
        raise ValueError(f"{member}: not an mp4 (magic={head[:12]!r})")
    tail_n = min(int(tail_bytes), info.size)
    tail = read_tar_slice(archive, member, info.size - tail_n, tail_n)
    meta = _parse_first_valid_moov(tail)
    if meta is None:
        meta = _parse_first_valid_moov(read_tar_slice(archive, member, 0, min(info.size, 4 << 20)))
    if meta is None and info.size > tail_n:
        bigger = min(info.size, 64 << 20)
        meta = _parse_first_valid_moov(read_tar_slice(archive, member, info.size - bigger, bigger))
    if meta is None:
        raise ValueError(f"{member}: moov atom not found in {archive}")
    meta["member"] = member
    meta["bytes"] = int(info.size)
    return meta


def probe_wav_in_tar(archive: Path, member: str) -> dict:
    """Parse WAV duration from the header of an uncompressed tar member."""
    info = _tar_member_info(archive, member)
    head = read_tar_slice(archive, member, 0, min(info.size, 4096))
    if len(head) < 44 or head[0:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError(f"{member}: not a WAV")
    byte_rate = None
    data_bytes = None
    pos = 12
    while pos + 8 <= len(head):
        chunk = head[pos : pos + 4]
        size = struct.unpack("<I", head[pos + 4 : pos + 8])[0]
        payload = pos + 8
        if chunk == b"fmt " and payload + 16 <= len(head):
            byte_rate = struct.unpack("<I", head[payload + 8 : payload + 12])[0]
        elif chunk == b"data":
            data_bytes = size
            break
        pos = payload + size + (size & 1)
    if byte_rate is None:
        byte_rate = struct.unpack("<I", head[28:32])[0]
    if data_bytes is None:
        data_bytes = max(0, info.size - 44)
    duration_s = (data_bytes / byte_rate) if byte_rate else None
    return {
        "member": member,
        "bytes": int(info.size),
        "duration_s": float(duration_s) if duration_s is not None else None,
        "byte_rate": int(byte_rate) if byte_rate else None,
        "source": "wav_header",
    }


def check_npz_video_align(
    *,
    npz_n_frames: int,
    fps: float,
    video_n_frames: int | None,
    video_duration_s: float | None,
    wav_duration_s: float | None = None,
    frame_tol: int = NPZ_VIDEO_FRAME_TOL,
    time_tol: float = NPZ_VIDEO_TIME_TOL,
) -> dict:
    """Compare NPZ track length against MP4 (and optional WAV) duration."""
    npz_duration_s = (npz_n_frames / fps) if fps else None
    frame_delta = None
    if video_n_frames is not None:
        frame_delta = int(npz_n_frames) - int(video_n_frames)
    time_delta = None
    if npz_duration_s is not None and video_duration_s is not None:
        time_delta = float(npz_duration_s) - float(video_duration_s)
    wav_delta = None
    if video_duration_s is not None and wav_duration_s is not None:
        wav_delta = float(wav_duration_s) - float(video_duration_s)
    ok_frames = frame_delta is None or abs(frame_delta) <= frame_tol
    ok_time = time_delta is None or abs(time_delta) <= time_tol
    ok_wav = wav_delta is None or abs(wav_delta) <= max(time_tol, 0.05)
    return {
        "npz_n_frames": int(npz_n_frames),
        "video_n_frames": int(video_n_frames) if video_n_frames is not None else None,
        "video_duration_s": float(video_duration_s) if video_duration_s is not None else None,
        "npz_duration_s": float(npz_duration_s) if npz_duration_s is not None else None,
        "wav_duration_s": float(wav_duration_s) if wav_duration_s is not None else None,
        "frame_delta": frame_delta,
        "time_delta_s": None if time_delta is None else round(time_delta, 6),
        "wav_delta_s": None if wav_delta is None else round(wav_delta, 6),
        "ok": bool(ok_frames and ok_time),
        "ok_wav": bool(ok_wav),
    }


def extract_members(archive: Path, jobs: list[tuple[str, Path]]) -> dict[str, Path]:
    """Extract several members from one tar in a single open."""
    out: dict[str, Path] = {}
    pending: list[tuple[str, Path]] = []
    for member, dest in jobs:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            out[member] = dest
        else:
            pending.append((member, dest))
    if not pending:
        return out
    with tarfile.open(archive, "r:") as tar:
        for member, dest in pending:
            handle = tar.extractfile(member)
            if handle is None:
                handle = tar.extractfile(Path(member).name)
            if handle is None:
                raise FileNotFoundError(f"{member} not in {archive}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            with handle, tmp.open("wb") as buf:
                shutil.copyfileobj(handle, buf, 8 << 20)
            tmp.replace(dest)
            out[member] = dest
    return out


def extract_member(archive: Path, member: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tarfile.open(archive, "r:") as tar:
        handle = tar.extractfile(member)
        if handle is None:
            handle = tar.extractfile(Path(member).name)
        if handle is None:
            raise FileNotFoundError(f"{member} not in {archive}")
        with handle, tmp.open("wb") as out:
            shutil.copyfileobj(handle, out, 8 << 20)
    tmp.replace(dest)
    return dest


def merge_runs(intervals: list[tuple[float, float]], *, gap: float, min_len: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_s, prev_e = merged[-1]
        if start - prev_e <= gap:
            merged[-1] = (prev_s, max(prev_e, end))
        else:
            merged.append((start, end))
    return [(s, e) for s, e in merged if (e - s) >= min_len]


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def face_box_from_kpts(kpt: np.ndarray, *, conf: float = 0.25) -> np.ndarray | None:
    face = kpt[23:91]
    good = face[:, 2] >= conf
    if int(good.sum()) < 8:
        return None
    xs = face[good, 0]
    ys = face[good, 1]
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def upper_body_box(
    official: np.ndarray,
    kpt: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Head + shoulders + chest. Official Seamless box is often full-body."""
    x1, y1, x2, y2 = [float(v) for v in official]
    face = face_box_from_kpts(kpt)
    sh = kpt[[5, 6]]
    hp = kpt[[11, 12]]
    sh_ok = sh[:, 2] >= 0.25
    hp_ok = hp[:, 2] >= 0.20
    if face is not None:
        fx1, fy1, fx2, fy2 = [float(v) for v in face]
        fh = max(fy2 - fy1, 8.0)
        fw = max(fx2 - fx1, 8.0)
        y1 = min(y1, fy1 - 0.35 * fh)
        x1 = min(x1, fx1 - 0.55 * fw)
        x2 = max(x2, fx2 + 0.55 * fw)
        chest = fy2 + 2.15 * fh
        if sh_ok.any():
            chest = max(chest, float(sh[sh_ok, 1].max()) + 1.15 * fh)
        if hp_ok.any():
            mid = 0.55 * float(sh[sh_ok, 1].mean()) + 0.45 * float(hp[hp_ok, 1].mean()) if sh_ok.any() else float(hp[hp_ok, 1].min())
            chest = min(chest, mid)
        y2 = min(y2, chest)
    else:
        # fallback: keep top 58% of official person box
        y2 = y1 + 0.58 * max(y2 - y1, 8.0)
    out = np.asarray([x1, y1, x2, y2], dtype=np.float32)
    return np.clip(out, [0, 0, 0, 0], [width, height, width, height])


def pack_boxes(boxes: np.ndarray, valid: np.ndarray, width: int, height: int) -> list[int]:
    n = int(boxes.shape[0])
    out = np.full((n, 4), -1, dtype=np.int16)
    good = (valid > 0) & np.isfinite(boxes).all(axis=1)
    if np.any(good):
        pix = np.rint(boxes[good]).astype(np.int32)
        pix[:, 0] = np.clip(pix[:, 0], 0, width)
        pix[:, 2] = np.clip(pix[:, 2], 0, width)
        pix[:, 1] = np.clip(pix[:, 1], 0, height)
        pix[:, 3] = np.clip(pix[:, 3], 0, height)
        out[good] = np.clip(pix, -1, 32767).astype(np.int16)
    return out.reshape(-1).tolist()


def static_from_track(boxes: np.ndarray, valid: np.ndarray) -> list[float] | None:
    good = (valid > 0) & np.isfinite(boxes).all(axis=1)
    if not np.any(good):
        return None
    sl = boxes[good]
    mean = sl.mean(axis=0)
    enc = np.array([sl[:, 0].min(), sl[:, 1].min(), sl[:, 2].max(), sl[:, 3].max()], dtype=np.float32)
    mean_a = max((mean[2] - mean[0]) * (mean[3] - mean[1]), 1.0)
    enc_a = max((enc[2] - enc[0]) * (enc[3] - enc[1]), 1.0)
    box = mean if enc_a / mean_a < 1.5 else enc
    return [round(float(x), 2) for x in box]


# COCO-WholeBody: wrists 9/10, hands 91–132
_HAND_IDX = np.asarray([9, 10, *range(91, 133)], dtype=np.int32)
PREVIEW_W = 540
PREVIEW_H = 960
STABLE_OFFICIAL_PAD = 0.07
HAND_KPT_CONF = 0.25


def stable_official_crop(
    official: np.ndarray,
    valid: np.ndarray,
    keypoints: np.ndarray | None,
    *,
    width: int,
    height: int,
    pad: float = STABLE_OFFICIAL_PAD,
    kpt_conf: float = HAND_KPT_CONF,
) -> list[float] | None:
    """One static box: official enclosing ∪ wrists/hands, then pad.

    Used as the locked-off training crop so per-frame official jitter
    does not shake the video, and outstretched hands are less often cut.
    """
    good = (valid > 0) & np.isfinite(official).all(axis=1)
    if not np.any(good):
        return None
    sl = official[good]
    x1 = float(sl[:, 0].min())
    y1 = float(sl[:, 1].min())
    x2 = float(sl[:, 2].max())
    y2 = float(sl[:, 3].max())
    if keypoints is not None and len(keypoints) == len(official):
        k = keypoints[good]
        if k.ndim == 3 and k.shape[1] > int(_HAND_IDX.max()):
            pts = k[:, _HAND_IDX, :]
            mask = pts[:, :, 2] >= kpt_conf
            if np.any(mask):
                xy = pts[:, :, :2][mask]
                x1 = min(x1, float(xy[:, 0].min()))
                y1 = min(y1, float(xy[:, 1].min()))
                x2 = max(x2, float(xy[:, 0].max()))
                y2 = max(y2, float(xy[:, 1].max()))
    pw = pad * max(x2 - x1, 8.0)
    ph = pad * max(y2 - y1, 8.0)
    box = np.clip(
        np.asarray([x1 - pw, y1 - ph, x2 + pw, y2 + ph], dtype=np.float32),
        [0, 0, 0, 0],
        [width, height, width, height],
    )
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        return None
    return [round(float(v), 2) for v in box]


def preview_crop_rect(
    box: list[float] | np.ndarray,
    *,
    src_w: int,
    src_h: int,
    prev_w: int = PREVIEW_W,
    prev_h: int = PREVIEW_H,
) -> tuple[int, int, int, int]:
    """Map a source-pixel box onto the 540×960 preview and make even crop=w:h:x:y."""
    x1, y1, x2, y2 = [float(v) for v in box]
    sx = prev_w / max(src_w, 1)
    sy = prev_h / max(src_h, 1)
    x = max(0, min(prev_w - 2, int(np.floor(x1 * sx)) // 2 * 2))
    y = max(0, min(prev_h - 2, int(np.floor(y1 * sy)) // 2 * 2))
    w = int(np.ceil(x2 * sx) - x) // 2 * 2
    h = int(np.ceil(y2 * sy) - y) // 2 * 2
    w = max(2, min(w, (prev_w - x) // 2 * 2))
    h = max(2, min(h, (prev_h - y) // 2 * 2))
    return x, y, w, h


def write_preview_crop(
    src_mp4: Path,
    crop_mp4: Path,
    first_jpg: Path,
    box: list[float],
    *,
    src_w: int,
    src_h: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = preview_crop_rect(box, src_w=src_w, src_h=src_h)
    vf = f"crop={w}:{h}:{x}:{y}"
    common = [
        "-i",
        str(src_mp4),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "26",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(crop_mp4),
    ]
    try:
        run_ffmpeg(
            [
                *common[:-1],
                "-c:a",
                "aac",
                "-ac",
                "1",
                "-ar",
                "44100",
                "-b:a",
                "96k",
                common[-1],
            ]
        )
    except RuntimeError:
        run_ffmpeg(["-an", *common])
    run_ffmpeg(["-i", str(crop_mp4), "-frames:v", "1", "-q:v", "3", str(first_jpg)])
    return x, y, w, h


def time_to_frame(t: float, fps: float) -> int:
    return int(round(float(t) * float(fps)))


def slice_pack(full: list[int], f0: int, f1: int) -> list[int]:
    return full[max(0, f0) * 4 : max(0, f1) * 4]


def read_pcm16(path: Path, *, start_s: float, duration_s: float, sr: int = 16000) -> np.ndarray:
    """Accurate decode: seek after opening the input so timestamps match ASR."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ss",
        f"{max(0.0, start_s):.3f}",
        "-t",
        f"{max(0.02, duration_s):.3f}",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "s16le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"ffmpeg pcm failed: {err}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def write_pcm16_wav(path: Path, samples: np.ndarray, *, sr: int = 16000) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-i",
        "pipe:0",
        "-c:a",
        "pcm_s16le",
        str(path),
    ]
    proc = subprocess.run(cmd, input=pcm.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"ffmpeg write wav failed: {err}")


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def gate_samples(samples: np.ndarray, *, start_s: float, sr: int, keep: list[tuple[float, float]]) -> np.ndarray:
    out = np.zeros_like(samples)
    n = int(samples.size)
    for a, b in keep:
        i0 = max(0, int(round((a - start_s) * sr)))
        i1 = min(n, int(round((b - start_s) * sr)))
        if i1 > i0:
            out[i0:i1] = samples[i0:i1]
    return out
