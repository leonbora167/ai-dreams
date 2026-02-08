import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data_training")
    parser.add_argument("--manifest", type=str, default="data_training/manifest.csv")
    parser.add_argument("--aux-dir", type=str, default="data_training/aux")
    parser.add_argument("--config", type=str, default="configs/train_example.yaml")
    parser.add_argument("--skip-precompute", action="store_true")
    parser.add_argument("--frames-per-clip", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    project_root_path = Path(__file__).resolve().parents[1]
    project_root = str(project_root_path)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = project_root if not existing_pythonpath else f"{project_root};{existing_pythonpath}"

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_root_path / data_root
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = project_root_path / manifest_path
    aux_dir = Path(args.aux_dir)
    if not aux_dir.is_absolute():
        aux_dir = project_root_path / aux_dir
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root_path / config_path

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    aux_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "scripts/build_manifest.py",
            "--data-root",
            str(data_root),
            "--output",
            str(manifest_path),
        ],
        env=env,
    )
    if not args.skip_precompute:
        run(
            [
                sys.executable,
                "scripts/precompute_aux.py",
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(aux_dir),
                "--frames-per-clip",
                str(args.frames_per_clip),
                "--image-size",
                str(args.image_size),
                "--device",
                args.device,
            ],
            env=env,
        )
    run([sys.executable, "-m", "crowd_action.train", "--config", str(config_path)], env=env)


if __name__ == "__main__":
    main()
