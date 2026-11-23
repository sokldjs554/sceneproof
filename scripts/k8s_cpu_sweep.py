"""Measure CPU limits with sequential real-time Jobs on kind-sceneproof-eval."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from uuid import uuid4

import yaml

from compare_event_eval import different_keys

ROOT = Path(__file__).resolve().parents[1]
RECORDED = Path("docs/experiments/rerun-20261121/session_base.json")
HOST_OUT = Path("/tmp/sceneproof-eval-out")
KUBECTL = ["kubectl", "--context", "kind-sceneproof-eval", "-n", "sceneproof-eval", "--request-timeout=30s"]


def parse_limits(value):
    try:
        limits = [Decimal(item) for item in value.split()]
        if (not limits or len(limits) != len(set(limits))
                or any(not n.is_finite() or n <= 0 or n * 1000 != int(n * 1000) for n in limits)):
            raise ValueError
        return limits
    except (ValueError, InvalidOperation, OverflowError) as exc:
        raise argparse.ArgumentTypeError("use distinct positive CPU counts in whole millicores, e.g. '0.5 1 2'") from exc


def wait_for_job(name, deadline):
    until = time.monotonic() + deadline + 60
    while time.monotonic() < until:
        output = subprocess.check_output([*KUBECTL, "get", "job", name, "-o", "json"])
        conditions = json.loads(output).get("status", {}).get("conditions", [])
        for condition in conditions:
            if condition["status"] != "True":
                continue
            if condition["type"] == "Complete":
                return
            if condition["type"] == "Failed":
                raise RuntimeError(f"{name}: {condition.get('message', 'Job failed')}")
        time.sleep(3)
    raise TimeoutError(f"{name} did not finish within its deadline")


def measure(limit, threads, cameras, destination):
    job = yaml.safe_load((ROOT / "deploy/k8s/eval/20-job.yaml").read_text())
    name = f"eval-rt-{uuid4().hex[:12]}"
    job["metadata"]["name"] = name
    container = job["spec"]["template"]["spec"]["containers"][0]
    millicores = int(limit * 1000)
    container["resources"]["limits"]["cpu"] = f"{millicores}m"
    container["resources"]["requests"]["cpu"] = f"{min(250, millicores)}m"
    filename = f"{name}.json"
    container["args"] = ["--realtime", "--threads", str(threads), "--out", f"/out/{filename}"]
    if cameras:
        container["args"].extend(["--cameras", cameras])
    subprocess.run([*KUBECTL, "create", "-f", "-"], input=json.dumps(job), text=True, check=True)
    try:
        wait_for_job(name, job["spec"]["activeDeadlineSeconds"])
        source = HOST_OUT / filename
        shutil.copyfile(source, destination)
        source.unlink()
    except (RuntimeError, TimeoutError):
        subprocess.run([*KUBECTL, "describe", "job", name], check=False)
        subprocess.run([*KUBECTL, "logs", f"job/{name}", "--tail=30"], check=False)
        raise
    finally:
        subprocess.run([*KUBECTL, "delete", "job", name, "--wait=true", "--timeout=90s"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limits", type=parse_limits, required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cameras", default="")
    parser.add_argument("--resume-from", type=Path, metavar="BATCH_DIR",
                        help="reuse completed results from a previous batch under this output directory")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    recorded = json.loads((ROOT / RECORDED).read_text())
    available = set(recorded["per_video"])
    cameras = set(args.cameras.split(",")) if args.cameras else available
    if not cameras <= available:
        parser.error(f"unknown cameras: {sorted(cameras - available)}")

    # Keep an earlier summary and its files usable until every new Job has finished.
    output_path = args.out.parent.resolve() / args.out.name
    directory = output_path.parent / output_path.stem
    previous = args.resume_from.resolve() if args.resume_from else None
    if previous is not None and (previous.parent != directory.resolve() or not previous.is_dir()):
        parser.error(f"--resume-from must be an existing batch directory inside {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    batch = None
    commit = None
    rows = []
    for limit in args.limits:
        threads = max(1, int(limit))
        for run in range(1, args.runs + 1):
            filename = f"cpu{limit:f}-run{run}.json"
            print(f"CPU {limit}, threads {threads}, run {run}/{args.runs}", flush=True)
            if previous is not None and (previous / filename).is_file():
                destination = directory / previous.name / filename
                print(f"Using saved result {destination}", flush=True)
            else:
                if batch is None:
                    batch = Path(mkdtemp(dir=directory, prefix="sweep-"))
                destination = batch / filename
                measure(limit, threads, args.cameras, destination)
            result = json.loads(destination.read_text())
            videos = result["per_video"]
            dropped = sum(video["run"]["dropped"] for video in videos.values())
            if (result["realtime"] is not True or result["environment"]["cpu_limit"] != float(limit)
                    or result["config"]["detector"]["intra_op_threads"] != threads or not result["git"]
                    or set(videos) != cameras or dropped != result["dropped_frames"]):
                raise ValueError(f"{destination}: recorded run conditions or drop count disagree with the Job")
            if commit is not None and result["git"] != commit:
                raise ValueError(f"{destination}: image commit {result['git']} differs from {commit}")
            commit = result["git"]
            rows.append({
                "cpu_limit": float(limit), "threads": threads, "run": run,
                "result": str(destination.relative_to(output_path.parent)),
                "frames": sum(video["run"]["frames"] for video in videos.values()),
                "dropped_frames": dropped, "values_different": len(different_keys(recorded, result)[0]),
            })

    with NamedTemporaryFile(mode="w", dir=output_path.parent, delete=False) as summary:
        json.dump({"recorded": str(RECORDED), "runs": rows}, summary, indent=2)
        summary.write("\n")
    Path(summary.name).replace(output_path)
    print(args.out)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
