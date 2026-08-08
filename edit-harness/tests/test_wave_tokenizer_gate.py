"""CPU-only integration tests for the wave tokenizer preflight gate."""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

HARNESS = Path(__file__).resolve().parents[1]
PREPARE = HARNESS / "engine" / "box_prepare_wave.sh"
PREFLIGHT = HARNESS / "engine" / "box_preflight.sh"
ASSERT_CLI = HARNESS / "experiments" / "assert_targets_distinguishable.py"
COUNTERFACT_SHA = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
MODEL_NAMES = ("Mistral-7B-v0.3", "Qwen2.5-7B", "Llama-3.1-8B")
TARGETS = ("Paris", "London", "English", "Michael", "I", "cannot", "answer")


def _save_tokenizer(path: Path, collision: bool = False) -> None:
    vocab = {"[UNK]": 0}
    if not collision:
        vocab.update({target: index for index, target in enumerate(TARGETS, start=1)})
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.normalizer = normalizers.Strip()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
    tokenizer.save_pretrained(path)


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class TestWaveTokenizerGate(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "harness"
        (self.root / "engine").mkdir(parents=True)
        (self.root / "experiments" / "tools").mkdir(parents=True)
        (self.root / "data" / "models").mkdir(parents=True)
        (self.root / "docs" / "plans").mkdir(parents=True)
        (self.root / "results" / "matrices").mkdir(parents=True)
        (self.root / "bin").mkdir(parents=True)

        shutil.copy2(PREPARE, self.root / "engine" / PREPARE.name)
        shutil.copy2(PREFLIGHT, self.root / "engine" / PREFLIGHT.name)
        (self.root / "engine" / PREFLIGHT.name).chmod(
            (self.root / "engine" / PREFLIGHT.name).stat().st_mode | stat.S_IXUSR
        )
        shutil.copy2(ASSERT_CLI, self.root / "experiments" / ASSERT_CLI.name)
        shutil.copy2(HARNESS / "metrics.py", self.root / "metrics.py")
        (self.root / "requirements-box-waves.txt").write_text("", encoding="utf-8")
        (self.root / "run_deletion_wave1.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (self.root / "experiments" / "killgate_keygeom.py").write_text(
            "import argparse\nargparse.ArgumentParser().parse_args()\n", encoding="utf-8"
        )
        _write_executable(
            self.root / "experiments" / "tools" / "integrity_check.py",
            "#!/usr/bin/env python3\nraise SystemExit(0)\n",
        )

        prereg = self.root / "docs" / "plans" / "PREREG-DELETION-PREDICTOR-2026-07-26.md"
        prereg.write_text("STATUS: RATIFIED\n", encoding="utf-8")
        for receipt in (
            "DELETION_PHASEL_GD1_PASS.ok",
            "DELETION_PHASEL_GD2_PASS.ok",
            "DELETION_PHASEL_TEXT_PASS.ok",
        ):
            (self.root / "engine" / receipt).touch()
        for tag in ("gate_mistral7b_rome_cf_L24", "gate_llama8b_rome_cf_L24"):
            for seed in range(3):
                (self.root / "results" / "matrices" / f"{tag}_s{seed}.npz").touch()

        dataset = [
            {
                "requested_rewrite": {
                    "target_new": {"str": "Paris"},
                    "target_true": {"str": "London"},
                }
            },
            {
                "requested_rewrite": {
                    "target_new": {"str": "English"},
                    "target_true": {"str": "Michael"},
                }
            },
        ]
        data_path = self.root / "data" / "counterfact.json"
        data_path.write_text(json.dumps(dataset), encoding="utf-8")
        self._pad_to_counterfact_sha(data_path)

        for name in MODEL_NAMES:
            model_dir = self.root / "data" / "models" / name
            model_dir.mkdir()
            _save_tokenizer(model_dir)

        _write_executable(
            self.root / "bin" / "nvidia-smi",
            "#!/bin/sh\n"
            "case \"$*\" in\n"
            "  *index,name,memory.total*) "
            "printf '0, Test 4090D, 24564\\n1, Test 4090D, 24564\\n' ;;\n"
            "  *memory.total*) printf '24564\\n24564\\n' ;;\n"
            "  *) exit 0 ;;\n"
            "esac\n",
        )
        _write_executable(
            self.root / "bin" / "python-wave-gate",
            "#!/bin/sh\n"
            "if [ \"${1:-}\" = - ]; then\n"
            "  script=$(cat)\n"
            "  case \"$script\" in\n"
            "    *torch.cuda.is_available*) printf 'torch CUDA devices 2\\n'; exit 0 ;;\n"
            "  esac\n"
            "  if printf '%s\\n' \"$script\" | grep -q 'bitsandbytes'; then\n"
            "    printf 'numpy=fixture scipy=fixture transformers=fixture huggingface_hub=fixture bitsandbytes=fixture\\n'\n"
            "    exit 0\n"
            "  fi\n"
            "  printf '%s\\n' \"$script\" | exec python3 -\n"
            "else\n"
            "  exec python3 \"$@\"\n"
            "fi\n",
        )
        self.env = os.environ.copy()
        self.env.update(
            {
                "HARNESS": str(self.root),
                "CLOUD_PY": str(self.root / "bin" / "python-wave-gate"),
                "DATA_DISK": str(self.root / "data"),
                "HOME": str(self.root / "home"),
                "HF_HOME": str(self.root / "home" / "hf-cache"),
                "PATH": f"{self.root / 'bin'}:{self.env['PATH']}",
            }
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _pad_to_counterfact_sha(path: Path) -> None:
        """Use a sha256 wrapper in the integration fixture; production still uses sha256sum."""
        # The production hash is intentionally immutable. The fixture's tiny JSON cannot
        # reproduce it, so its sha256sum wrapper returns the canonical value only for the
        # fixture dataset and delegates every other call to the system binary.
        wrapper = path.parents[1] / "bin" / "sha256sum"
        _write_executable(
            wrapper,
            "#!/bin/sh\n"
            f"if [ \"$1\" = \"{path}\" ]; then printf '{COUNTERFACT_SHA}  %s\\n' \"$1\"; "
            "else exec /usr/bin/sha256sum \"$@\"; fi\n",
        )

    def _run_check(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.root / "engine" / "box_prepare_wave.sh"), "deletion-wave1", "check"],
            cwd=self.root,
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_planted_collision_blocks_wave_and_names_model(self) -> None:
        _save_tokenizer(self.root / "data" / "models" / "Qwen2.5-7B", collision=True)

        result = self._run_check()
        output = result.stdout + result.stderr

        self.assertNotEqual(result.returncode, 0, output)
        self.assertFalse((self.root / "engine" / "BOX_READY_deletion-wave1.ok").exists())
        self.assertIn("TOKENIZER-GATE PASS Mistral-7B-v0.3", output)
        self.assertIn("TOKENIZER-GATE FAIL Qwen2.5-7B", output)
        self.assertIn("ratio", output)
        self.assertIn("Per-edit efficacy would be unmeasurable", output)
        self.assertIn("TOKENIZER-GATE PASS Llama-3.1-8B", output)
        self.assertIn("BLOCKED: fix every FAIL line before launching science", output)

    def test_distinguishable_targets_allow_ready_receipt(self) -> None:
        result = self._run_check()
        output = result.stdout + result.stderr

        self.assertEqual(result.returncode, 0, output)
        for name in MODEL_NAMES:
            self.assertIn(f"TOKENIZER-GATE PASS {name}", output)
        self.assertTrue((self.root / "engine" / "BOX_READY_deletion-wave1.ok").is_file())


if __name__ == "__main__":
    unittest.main()
