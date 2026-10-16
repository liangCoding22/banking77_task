"""Deploy a promoted SFT model, score it with `banking77 chat-eval`, then delete the deployment.

    PYTHONPATH=src .venv-sft/bin/python scripts/eval_promoted.py \\
        --promoted-model accounts/<acct>/models/b77-lr1em4-r16 --deployment-id b77-eval \\
        --eval-split test --prompt-version v3

Run `banking77 prepare --model <promoted-model> --prompt-version v3 --eval-split test` first so the
input exists. Needs the fireworks SDK (the cookbook venv) and FIREWORKS_API_KEY. Creating the
deployment costs GPU time; it is deleted at the end unless --keep-deployment.
"""

import argparse
import os
import subprocess
import sys

from fireworks.training.sdk.deployment import DeploymentConfig, DeploymentManager

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHAPE = "accounts/fireworks/deploymentShapes/rft-qwen3-4b"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--promoted-model", required=True)
    p.add_argument("--deployment-id", required=True)
    p.add_argument("--prompt-version", default="v3", choices=("v1", "v2", "v3"))
    p.add_argument("--eval-split", default="test", choices=("dev", "test"))
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--only-failed", action="store_true", help="rerun only rows that errored last time")
    p.add_argument("--keep-deployment", action="store_true")
    args = p.parse_args()

    mgr = DeploymentManager(api_key=os.environ["FIREWORKS_API_KEY"])
    info = mgr.create_or_get(DeploymentConfig(
        deployment_id=args.deployment_id, base_model=args.promoted_model, deployment_shape=SHAPE,
        expected_deployment_shape=SHAPE, enable_hot_load=False, hot_load_bucket_type=None, for_training=False,
    ))
    print("created:", info.name, info.state, flush=True)
    try:
        ready = mgr.wait_for_ready(args.deployment_id)
        model = f"{args.promoted_model}#{ready.inference_model}"  # "<model>#<deployment>" addresses the tuned model
        print("serving as", model, flush=True)
        cmd = [sys.executable, "-m", "banking77_task.cli", "chat-eval", "--model", args.promoted_model,
               "--prompt-version", args.prompt_version, "--eval-split", args.eval_split,
               "--concurrency", str(args.concurrency)]
        if args.only_failed:
            cmd.append("--only-failed")
        # run_dir is keyed on --model, so score under the promoted-model name but call the served name.
        env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"), B77_SERVED_MODEL=model)
        subprocess.run(cmd, cwd=REPO, env=env, check=True)
    finally:
        if args.keep_deployment:
            print(f"kept deployment {args.deployment_id}", flush=True)
        else:
            print(f"deleting deployment {args.deployment_id}", flush=True)
            mgr.delete(args.deployment_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
