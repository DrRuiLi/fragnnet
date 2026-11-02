"""Evaluate a held-out nist23 test split with the best training checkpoint.

Does not call trainer.fit. Refuses to start while run_pl_model_fit.py is still running.
"""

import argparse
import logging
import os
import subprocess
from pathlib import Path

import lightning as L
import torch
import yaml
from pandarallel import pandarallel

from fragnnet.pl_model.fragnnet_pl import FraGNNetPL
from fragnnet.runner import init_dataloader, init_dataset, load_config
from fragnnet.utils.misc_utils import get_core_count
from fragnnet.utils.pl_utils import ConsoleLogger


ROOT = Path("/home/lirui/package/fragnnet")


def best_checkpoint(ckpt_dir: Path) -> Path:
    last_fp = ckpt_dir / "last.ckpt"
    if last_fp.is_file():
        data = torch.load(last_fp, map_location="cpu", weights_only=False)
        for key, value in data.get("callbacks", {}).items():
            if not str(key).startswith("ModelCheckpoint"):
                continue
            best_fp = Path(value.get("best_model_path") or "")
            if best_fp.is_file():
                return best_fp
    candidates = sorted(ckpt_dir.glob("model-*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint in {ckpt_dir}")
    return candidates[-1]


def training_still_running() -> bool:
    proc = subprocess.run(
        ["pgrep", "-f", "scripts/run_pl_model_fit.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--custom",
        default="config/nist23_inchikey/fragnnet_d3/s0.yml",
    )
    parser.add_argument("--ckpt-dir", type=Path, default=ROOT / "tmp_ckpt")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "logs" / "nist23_inchikey_fragnnet_d3_s0_test.yaml",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.chdir(ROOT)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if training_still_running():
        raise RuntimeError("run_pl_model_fit.py is still running; refusing to load the test set")

    config = load_config("config/template.yml", args.custom)
    ckpt_fp = best_checkpoint(args.ckpt_dir)
    logging.info("test checkpoint %s", ckpt_fp)
    L.seed_everything(config["seed"], workers=True)
    pandarallel.initialize(
        progress_bar=config["pandarallel_enable_progress_bar"],
        nb_workers=get_core_count(),
    )
    model = FraGNNetPL.load_from_checkpoint(str(ckpt_fp), map_location="cpu")
    test_ds = init_dataset(config, splits=["test"])[0]
    test_dl = init_dataloader(test_ds, config)
    trainer = L.Trainer(
        logger=ConsoleLogger(),
        accelerator=config["accelerator"],
        devices=config["devices"],
        precision=config["precision"],
        enable_checkpointing=False,
        enable_progress_bar=config["pl_enable_progress_bar"],
        num_sanity_val_steps=0,
    )
    results = trainer.test(model, dataloaders=test_dl)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(results, sort_keys=True))
    logging.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
