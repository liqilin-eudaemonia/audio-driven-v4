#!/usr/bin/env python3
"""Configured local/SSH torchrun, remote master support, all-node preflight."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from omni_av.data import write_json_new
from omni_av_p3.launch import cluster_nodes


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--guard":
        # Parent-owned process group only; stdin EOF terminates this run's descendants.
        from pipeline2_cluster import guard
        raise SystemExit(guard(sys.argv[2:]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = json.loads(args.cluster.read_text())
    nodes = cluster_nodes(cfg)
    extra = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    reserved = {"--nnodes", "--node-rank", "--master-addr", "--master-port", "--gpus", "--worker", "--preflight-only"}
    if any(x.split("=")[0] in reserved for x in extra):
        parser.error("Topology/worker flags belong to cluster config, not forwarded arguments")
    args.log_dir.mkdir(parents=True, exist_ok=False)
    python = cfg["python"]
    local_hosts = {"localhost", "127.0.0.1", socket.gethostname(),
                   *subprocess.check_output(["hostname", "-I"], text=True).split()}
    if any(n.get("local", False) and n["host"] not in local_hosts for n in nodes):
        raise ValueError("A node marked local does not match this machine; remove the stale local flag")
    key = cfg.get("ssh_key", f"/private/ssh_keys/root_{socket.gethostname()}_ed25519")
    network = ["--network-config", cfg["network_config"]] if cfg.get("network_config") else []
    def transport(node, command):
        shell = "cd " + shlex.quote(str(ROOT)) + " && exec " + shlex.join(command)
        if node["host"] in local_hosts:
            return ["bash", "-c", shell]
        multiplex = ["-o", "ControlMaster=auto", "-o", "ControlPersist=600",
                     "-o", "ControlPath=" + str(Path(cfg["ssh_control_dir"]) / "%C")] if cfg.get("ssh_control_dir") else []
        return ["ssh", *multiplex, "-i", key, "-p", str(node.get("ssh_port", 22)), "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
                f'{node.get("ssh_user", "root")}@{node["host"]}', shell]
    commands = [[python, str(ROOT / "scripts/pipeline3_parallel.py"), *extra, *network,
        "--nnodes", str(len(nodes)), "--node-rank", str(rank), "--master-addr", nodes[0]["host"],
        "--master-port", str(cfg.get("master_port", 29873)), "--gpus", ",".join(map(str, node["gpus"]))]
        for rank, node in enumerate(nodes)]
    def check(item):
        node, command = item
        for attempt in range(6):
            result = subprocess.run(transport(node, command + ["--preflight-only"]), text=True,
                                    capture_output=True, timeout=cfg.get("preflight_timeout", 180))
            if result.returncode != 255 or attempt == 5:
                break
            time.sleep(2 * (attempt + 1))
        with (args.log_dir / f'preflight_{node["name"]}.log').open("x") as stream:
            stream.write(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'Preflight failed on {node["name"]}: {result.stderr}')
        return json.loads(result.stdout.strip().splitlines()[-1])
    with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        checks = list(pool.map(check, zip(nodes, commands)))
    if any(c["source"] != checks[0]["source"] or c["artifacts"] != checks[0]["artifacts"] for c in checks):
        raise RuntimeError("Node source/cache/checkpoint mismatch")
    write_json_new(args.log_dir / "preflight.json", dict(nodes=nodes, checks=checks, commands=commands))
    if args.preflight_only:
        print(json.dumps(dict(preflight="passed", nodes=len(nodes))), flush=True)
        return
    processes, streams = [], []
    try:
        for node, command in zip(nodes, commands):
            stream = (args.log_dir / f'{node["name"]}.log').open("x")
            streams.append(stream)
            guarded = [python, str(Path(__file__).resolve()), "--guard", *command]
            processes.append(subprocess.Popen(transport(node, guarded), stdin=subprocess.PIPE,
                                             stdout=stream, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("A node failed; closing only this launcher's workers")
            time.sleep(.5)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("One or more nodes failed")
        write_json_new(args.log_dir / "completed.json", dict(returncodes=[p.returncode for p in processes], nodes=nodes))
    finally:
        for process in processes:
            process.stdin.close()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
