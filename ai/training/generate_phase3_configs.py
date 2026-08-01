"""Generate all phase 3 YAML configs in one shot.

Outputs:
  configs/phase3/control_token/<genre>_ctrl_seed{42,123,7}.yaml     (33 files)
  configs/phase3/phase0_ablation/<genre>_phase0_full.yaml           (11 files)
  configs/phase3/phase0_ablation/<genre>_phase0_lora.yaml           (11 files)
  configs/phase3/stage_a/<genre>_{lora,bitfit,ia3,full}_funksize.yaml (40 files)
  configs/phase3/stage_b/<genre>_{lora,full}_ep{3,5,8,12}.yaml      (24 files)

Total: 119 YAML configs.
"""
from __future__ import annotations
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "phase3"

GENRES_ALL = ["blues", "bossa", "classical", "country", "electronic", "folk",
              "funk", "gospel", "hip_hop", "rnb_soul", "rock"]
GENRES_FUNK_SIZED = ["blues", "bossa", "country", "electronic", "folk",
                     "funk", "gospel", "hip_hop", "rnb_soul", "rock"]  # classical excluded
GENRES_DATA_RICH = ["rock", "country", "folk"]

LORA_RANK = {
    "blues": 32, "bossa": 16, "classical": 64, "country": 4,
    "electronic": 8, "folk": 32, "funk": 4, "gospel": 4,
    "hip_hop": 4, "rnb_soul": 4, "rock": 16,
}

F1_BASE = "checkpoints/ft_jazz_pop80/best.pt"
PHASE0_BASE = "checkpoints/phase0_pop_baseline/best.pt"


def common_hparams(lr: float = 1e-5, epochs: int = 8) -> str:
    """Shared block: model arch + training hyperparams."""
    return f"""d_model: 512
n_heads: 8
d_ff: 2048
n_layers: 8
max_seq_len: 256
dropout: 0.1

epochs: {epochs}
batch_size: 32
gradient_accumulation_steps: 2
lr: {lr:.1e}
weight_decay: 0.01
warmup_epochs: 1
max_grad_norm: 1.0
use_amp: true
num_workers: 4"""


# ===========================================================================
# E1: Control-token baseline (embedding_only peft_method)
# ===========================================================================

def write_e1():
    out_dir = CONFIG_DIR / "control_token"
    out_dir.mkdir(parents=True, exist_ok=True)
    for genre in GENRES_ALL:
        for seed in (42, 123, 7):
            path = out_dir / f"{genre}_ctrl_seed{seed}.yaml"
            path.write_text(f"""# E1 Control-token baseline: only new genre token embeddings trainable.
# Tests "why not just genre tokens?" — adapter-less symbolic conditioning.

run_name: ft_f1_control_token_{genre}_seed{seed}

base_checkpoint: {F1_BASE}
target_genre: {genre}
sources: ["chordonomicon_{genre}"]

peft_method: embedding_only

seed: {seed}

{common_hparams(lr=3e-4)}
""")
    print(f"E1: 33 configs → {out_dir}")


# ===========================================================================
# E2: Phase 0 + Full FT × 11 genres
# ===========================================================================

def write_e2():
    out_dir = CONFIG_DIR / "phase0_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    for genre in GENRES_ALL:
        path = out_dir / f"{genre}_phase0_full.yaml"
        path.write_text(f"""# E2 Phase 0 + Full FT: F1 base justification ablation.
# Same as B2_full but starting from Phase 0 (no-jazz baseline) instead of F1.

run_name: phase0_full_ft_{genre}

base_checkpoint: {PHASE0_BASE}
target_genre: {genre}
sources: ["chordonomicon_{genre}"]

peft_method: full

seed: 42

{common_hparams(lr=1e-5)}
""")
    print(f"E2: 11 configs → {out_dir}")


# ===========================================================================
# E3: Phase 0 + LoRA × 11 genres (selected rank)
# ===========================================================================

def write_e3():
    out_dir = CONFIG_DIR / "phase0_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    for genre in GENRES_ALL:
        r = LORA_RANK[genre]
        alpha = r * 2
        path = out_dir / f"{genre}_phase0_lora.yaml"
        path.write_text(f"""# E3 Phase 0 + LoRA: F1 base justification ablation (LoRA case).
# Same as B3_lora_seed42 but starting from Phase 0 instead of F1.

run_name: phase0_lora_{genre}

base_checkpoint: {PHASE0_BASE}
target_genre: {genre}
sources: ["chordonomicon_{genre}"]

peft_method: lora
lora_r: {r}
lora_alpha: {alpha}
lora_dropout: 0.05
lora_target_modules: ["w_q", "w_k", "w_v"]

seed: 42

{common_hparams(lr=3e-4)}
""")
    print(f"E3: 11 configs → {out_dir}")


# ===========================================================================
# F1: Stage A controlled funk-size sweep (10 genres × 4 methods)
# ===========================================================================

def write_f1():
    out_dir = CONFIG_DIR / "stage_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    funk_size_source_tag = "funksize_subsample"  # subsample_funk_size.py creates these splits
    for genre in GENRES_FUNK_SIZED:
        for method in ("lora", "bitfit", "ia3", "full"):
            cfg_lines = [
                f"# F1 Stage A controlled sweep: {genre} sub-sampled to funk-size",
                f"# Method: {method}",
                "",
                f"run_name: stageA_{method}_{genre}_funksize",
                "",
                f"base_checkpoint: {F1_BASE}",
                f"target_genre: {genre}",
                f'sources: ["{funk_size_source_tag}_{genre}"]',
                "",
            ]
            if method == "lora":
                r = LORA_RANK[genre]
                cfg_lines += [
                    "peft_method: lora",
                    f"lora_r: {r}",
                    f"lora_alpha: {r*2}",
                    "lora_dropout: 0.05",
                    'lora_target_modules: ["w_q", "w_k", "w_v"]',
                ]
                lr = 3e-4
            elif method == "bitfit":
                cfg_lines.append("peft_method: bitfit")
                lr = 3e-4
            elif method == "ia3":
                cfg_lines += [
                    "peft_method: ia3",
                    'ia3_target_modules: ["w_k", "w_v"]',
                ]
                lr = 3e-4
            else:  # full
                cfg_lines.append("peft_method: full")
                lr = 1e-5
            cfg_lines += [
                "",
                "seed: 42",
                "",
                common_hparams(lr=lr, epochs=5),  # Stage A uses 5 epochs (smaller data)
                "",
            ]
            path = out_dir / f"{genre}_{method}_funksize.yaml"
            path.write_text("\n".join(cfg_lines))
    print(f"F1: 40 configs → {out_dir}")


# ===========================================================================
# F2: Stage B epoch sensitivity (3 genres × 2 methods × 4 epoch points)
# ===========================================================================

def write_f2():
    out_dir = CONFIG_DIR / "stage_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    for genre in GENRES_DATA_RICH:
        for method in ("lora", "full"):
            for n_ep in (3, 5, 8, 12):
                cfg_lines = [
                    f"# F2 Stage B epoch sensitivity: {genre} {method} ep={n_ep}",
                    "",
                    f"run_name: stageB_{method}_{genre}_ep{n_ep}",
                    "",
                    f"base_checkpoint: {F1_BASE}",
                    f"target_genre: {genre}",
                    f'sources: ["chordonomicon_{genre}"]',
                    "",
                ]
                if method == "lora":
                    r = LORA_RANK[genre]
                    cfg_lines += [
                        "peft_method: lora",
                        f"lora_r: {r}",
                        f"lora_alpha: {r*2}",
                        "lora_dropout: 0.05",
                        'lora_target_modules: ["w_q", "w_k", "w_v"]',
                    ]
                    lr = 3e-4
                else:
                    cfg_lines.append("peft_method: full")
                    lr = 1e-5
                cfg_lines += [
                    "",
                    "seed: 42",
                    "",
                    common_hparams(lr=lr, epochs=n_ep),
                    "",
                ]
                path = out_dir / f"{genre}_{method}_ep{n_ep}.yaml"
                path.write_text("\n".join(cfg_lines))
    print(f"F2: 24 configs → {out_dir}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    write_e1()
    write_e2()
    write_e3()
    write_f1()
    write_f2()
    print(f"\nTotal: 33 + 11 + 11 + 40 + 24 = 119 phase3 YAML configs generated")


if __name__ == "__main__":
    main()
