"""Explicit multi-node online training, with idle-GPU and source/artifact agreement."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def source_signature():
    from omni_av_p3.distributed import sha
    files = [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/pipeline3_online_*.py")]
    files += [ROOT / "scripts/pipeline3_cluster.py", ROOT / "scripts/pipeline2_cluster.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def network_environment(config):
    if any(not key.startswith(("NCCL_", "GLOO_", "TORCH_NCCL_")) for key in config):
        raise ValueError("Only communication environment variables are allowed")
    return {key: str(value) for key, value in config.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cluster", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--probe", type=int, help=argparse.SUPPRESS)
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    a = p.parse_args()
    from omni_av_p3.launch import cluster_nodes
    from omni_av_p3.distributed import sha
    from omni_av_p2.checkpoint import atomic_json_new
    from pipeline3_staged_worker import probe
    cfg = json.loads(a.cluster.read_text())
    nodes = cluster_nodes(cfg)
    extra = a.arguments[1:] if a.arguments[:1] == ["--"] else a.arguments
    def argument(name, default=None):
        for i, value in enumerate(extra):
            if value == name:
                return extra[i + 1]
            if value.startswith(name + "="):
                return value.split("=", 1)[1]
        return default
    if not argument("--config"):
        extra += ["--config", str(ROOT / "configs/pipeline3_bernini_online.json")]
    if a.probe is not None:
        node = nodes[a.probe]
        if any(g["pids"] or g["memory_mib"] >= 1000 or g["util"] > 5 for g in probe(node["gpus"])["gpus"]):
            raise RuntimeError("Selected GPU is occupied")
        output, control = Path(argument("--output")), Path(argument("--control-root"))
        if output.exists() or (control / "STOP").exists():
            raise RuntimeError("Output exists or STOP is set")
        artifacts = {}
        for flag in ("--manifest", "--config", "--initialize", "--resume", "--engineering-indices"):
            value = argument(flag)
            if value:
                artifacts[flag] = sha(value)
        seal = Path(argument("--integrity-root")) / "complete.json"
        if seal.exists():
            artifacts["input_seal"] = sha(seal)
        elif not argument("--engineering-indices"):
            raise ValueError("Production requires input seal")
        if cfg.get("network_config"):
            artifacts["network_config"] = sha(cfg["network_config"])
        print(json.dumps(dict(source=source_signature(), artifacts=artifacts)))
        return
    a.log_dir.mkdir(parents=True, exist_ok=False)
    hosts = {"localhost", "127.0.0.1", socket.gethostname(),
             *subprocess.check_output(["hostname", "-I"], text=True).split()}
    network = network_environment(json.loads(Path(cfg["network_config"]).read_text())) if cfg.get("network_config") else {}
    def transport(node, command):
        shell = "cd " + shlex.quote(str(ROOT)) + " && exec " + shlex.join(command)
        if node["host"] in hosts:
            return ["bash", "-c", shell]
        return ["ssh", "-i", cfg["ssh_key"], "-p", str(node.get("ssh_port", 22)),
            "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15",
            f'{node.get("ssh_user", "root")}@{node["host"]}', shell]
    def preflight(pair):
        index, node = pair
        command = [cfg["python"], str(Path(__file__).resolve()), "--cluster", str(a.cluster.resolve()),
            "--log-dir", str(a.log_dir.resolve()), "--probe", str(index), "--", *extra]
        result = subprocess.run(transport(node, command), capture_output=True, text=True,
                                timeout=cfg.get("preflight_timeout", 900))
        (a.log_dir / f'preflight_{node["name"]}.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("Preflight failed: " + node["name"])
        return json.loads(result.stdout.strip().splitlines()[-1])
    with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        checks = list(pool.map(preflight, enumerate(nodes)))
    if any(c != checks[0] for c in checks):
        raise ValueError("Nodes disagree on source or input artifacts")
    atomic_json_new(a.log_dir / "preflight.json", dict(nodes=nodes, checks=checks))
    if a.preflight_only:
        return
    workers, logs = [], []
    try:
        for index, node in enumerate(nodes):
            command = [cfg["python"], "-m", "torch.distributed.run", "--max_restarts=0",
                "--nnodes", str(len(nodes)), "--nproc_per_node", str(len(node["gpus"])),
                "--node_rank", str(index), "--master_addr", nodes[0]["host"],
                "--master_port", str(cfg.get("master_port", 29937)),
                str(ROOT / "scripts/pipeline3_online_train.py"), *extra]
            guarded = ["env", "CUDA_VISIBLE_DEVICES=" + ",".join(map(str, node["gpus"])),
                "OMP_NUM_THREADS=2", "PYTHONUNBUFFERED=1", *[k + "=" + v for k, v in network.items()],
                cfg["python"], str(ROOT / "scripts/pipeline3_cluster.py"), "--guard", *command]
            log = (a.log_dir / f'{node["name"]}.log').open("x")
            logs.append(log)
            workers.append(subprocess.Popen(transport(node, guarded), stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT))
        while any(w.poll() is None for w in workers):
            if any(w.poll() not in (None, 0) for w in workers):
                raise RuntimeError("Online node failed; close only this launcher's owned workers")
            time.sleep(.5)
        if any(w.returncode for w in workers):
            raise RuntimeError("Online worker failed")
        atomic_json_new(a.log_dir / "completed.json", dict(returncodes=[w.returncode for w in workers]))
    finally:
        for worker in workers:
            worker.stdin.close()
        for worker in workers:
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.terminate()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
