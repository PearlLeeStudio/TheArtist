---
base_model: PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80
language:
- en
library_name: peft
license: cc-by-nc-4.0
pipeline_tag: text-generation
tags:
- music
- music-generation
- chord-generation
- symbolic-music
- music-transformer
- lora
- peft
- gospel
---

# TheArtist Music Transformer — LoRA Adapter (Gospel)

LoRA adapter that steers the F1 base toward **gospel chord progressions**: gospel cadences, plagal motion, and gospel substitutions. It is one of eleven per-genre adapters from the paper [How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity?](https://huggingface.co/papers/2606.07334) (Lee, 2026), all trained on and loaded over the same F1 base. Use it to generate gospel progressions or to compare genres over one base.

[Paper](https://huggingface.co/papers/2606.07334) · [Code](https://github.com/PearlLeeStudio/TheArtist) · [Web app](https://artist.pearlleestudio.co) · [Demo](https://youtu.be/iSDj4VK9nKc) · [All models](https://huggingface.co/PearlLeeStudio)

## Model details

| Field | Value |
|---|---|
| Base model | [PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80) (F1). The released F1 is weight-identical to the Phase-0 pop baseline (a checkpoint-selection artifact). |
| Adapter | LoRA on the Q/K/V projections (`w_q`, `w_k`, `w_v`), r=4 (best of r ∈ {4, 8, 16, 32, 64}), α=8, dropout 0.05 |
| Trainable parameters | 465,920 (98,304 LoRA + 367,616 embedding and output), 1.8% of the 25,849,344 that full fine-tuning updates |
| Vocabulary | 359 tokens (351 base + 8 new genre tokens) |
| Files | `adapter/` (LoRA weights, 0.4 MB) and `embedding_extension.pt` (the trained embedding and output matrices) |

## Usage

Requires `torch`, `huggingface_hub`, `peft`, `safetensors`. Both repos bundle `model.py` and `tokenizer.py`, so nothing needs to be cloned from GitHub.

```python
import sys
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from peft import PeftModel

# 1. Download the base + LoRA repos. Both bundle model.py and tokenizer.py.
base_dir = snapshot_download(repo_id="PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80")
lora_dir = snapshot_download(repo_id="PearlLeeStudio/TheArtist-MusicTransformer-lora-gospel")
sys.path.insert(0, base_dir)  # so the next two imports resolve

from model import MusicTransformer
from tokenizer import ChordTokenizer

# 2. Extended tokenizer (351 base + 8 new genre tokens = 359). The PAD id
#    is unchanged across base and extended tokenizers.
tokenizer = ChordTokenizer(include_extra_genres=True)

# 3. Build the model at the BASE vocab size (351) so F1's state_dict loads
#    cleanly; we grow the embedding rows immediately after. Passing the
#    extended tokenizer's pad_id is safe because PAD is shared (see step 2).
BASE_VOCAB = 351
model = MusicTransformer(
    vocab_size=BASE_VOCAB,
    d_model=512, n_heads=8, d_ff=2048, n_layers=8,
    max_seq_len=256, dropout=0.0, pad_id=tokenizer.pad_id,
)
ckpt = torch.load(f"{base_dir}/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

# 4. Grow token_emb + out_proj from 351 -> 359 (new rows init from
#    [GENRE:none]), then overlay the LoRA's trained extension rows.
def _grow_to_extended_vocab(m, new_vocab, none_id):
    d = m.token_emb.embedding_dim
    new_emb = nn.Embedding(new_vocab, d, padding_idx=m.token_emb.padding_idx)
    with torch.no_grad():
        new_emb.weight[:m.token_emb.num_embeddings] = m.token_emb.weight
        for i in range(m.token_emb.num_embeddings, new_vocab):
            new_emb.weight[i] = m.token_emb.weight[none_id]
    m.token_emb = new_emb
    new_out = nn.Linear(d, new_vocab, bias=False)
    with torch.no_grad():
        new_out.weight[:m.out_proj.out_features] = m.out_proj.weight
        for i in range(m.out_proj.out_features, new_vocab):
            new_out.weight[i] = m.out_proj.weight[none_id]
    m.out_proj = new_out

_grow_to_extended_vocab(model, tokenizer.vocab_size, tokenizer.encode_genre("none"))

ext = torch.load(f"{lora_dir}/embedding_extension.pt",
                 map_location="cpu", weights_only=False)
model.token_emb.load_state_dict(ext["token_emb_state"])
model.out_proj.load_state_dict(ext["out_proj_state"])

# 5. Apply the LoRA adapter (the adapter files live at lora_dir/adapter/).
model = PeftModel.from_pretrained(model, f"{lora_dir}/adapter")
model.eval()

# 6. Generate a gospel continuation. With LoRA injected,
#    PeftModel.forward routes through the adapted attention layers.
song = {
    "key": "Cmaj", "time_signature": "4/4", "genre": "gospel",
    "bars": [["Cmaj7"], ["Fmaj7"]],
}
prompt_ids = tokenizer.encode_sequence(song)[:-1]
ids = torch.tensor([prompt_ids])
with torch.no_grad():
    for _ in range(32):
        logits = model(ids)  # routed through LoRA via PeftModel
        next_id = torch.multinomial(
            torch.softmax(logits[:, -1, :] / 0.8, dim=-1), 1,
        )
        ids = torch.cat([ids, next_id], dim=-1)
        if next_id.item() == tokenizer.eos_id:
            break
print(tokenizer.decode(ids[0].tolist()))
```

## Evaluation

All tables report teacher-forced next-token metrics. Top-1 counts every target, including the three header tokens that open each sequence (key, time signature, genre marker).

On this genre's val split (374 sequences, no key augmentation) both columns use the same dataloader and the same `[GENRE:none]`-initialised embedding extension, so only the adapter weights and the trained embedding rows differ.

| Metric | F1 base | F1 + this LoRA (r=4) | Δ |
|---|---:|---:|---:|
| Top-1 accuracy (%) | 79.34 | **83.20** | **+3.86** |
| Top-5 accuracy (%) | 94.73 | 96.63 | +1.90 |
| Cross-entropy loss | 0.8813 | 0.6092 | -0.2721 |

Rank sweep the released adapter was selected from (minimum val loss, top-1 tiebreak):

| Rank | val_loss | Top-1 (%) | Δ Top-1 vs F1 |
|---|---:|---:|---:|
| **r=4** | 0.6092 | 83.20 | +3.86 ← released |
| r=8 | 0.6116 | 82.29 | +2.95 |
| r=16 | 0.6115 | 82.31 | +2.97 |
| r=32 | 0.6116 | 82.17 | +2.83 |
| r=64 | 0.6131 | 82.26 | +2.92 |

Real-song check, mean over **10 held-out gospel songs** of 24–85 bars, this genre's slice of a 130-song eval set (10 per genre × 13 genres, seed 42) drawn only from the held-out val/test partitions. Their titles are Spotify track IDs by upstream Chordonomicon policy, but the progressions are real songs.

| Model | Top-1 (%) | Top-5 (%) | Loss |
|---|---:|---:|---:|
| F1 base | 79.79 | 96.85 | 0.7367 |
| **F1 + this LoRA** | **82.85** | 96.64 | 0.6071 |
| Δ | +3.06 | -0.21 | -0.1295 |

## Training data

3,747 sequences (2,856 songs) from the **gospel** subset of the [Chordonomicon dataset](https://huggingface.co/datasets/ailsntua/Chordonomicon), with a song-level 80/10/10 split (seed 42) and 12-key augmentation on train. The adapter trains on this subset alone, and the best checkpoint has the minimum val loss.

## Limitations

- **Chord symbols, automatic scores.** Chord symbols only (no melody, rhythm, or audio), and every score here is automatic, not a listening result.
- **Header tokens.** Part of each Δ comes from header-token predictions: F1 predicts neither the `[GENRE:gospel]` marker nor the placeholder C-major key, and this adapter predicts both, the key alone adding about one key-token step to the Δ. In the rank sweep, a rank about one key-token step above the others is a run that predicted the C-major key.
- **The gain is mostly a usable genre marker.** F1 has never read `[GENRE:gospel]` as an input in training (the eight new markers start as copies of the equally unused `[GENRE:none]` row). Scored without the header targets and with F1 given `[GENRE:pop]` instead, the released matched adapters gain −0.02 to +0.12 pp in the ten non-chorale genres ([arXiv:2606.07334](https://arxiv.org/abs/2606.07334) v5), so outside Bach chorales this adapter mainly adds a genre marker the model can use.

## License

**CC BY-NC 4.0** (matching Chordonomicon, the upstream training corpus). Research, paper replication, portfolio, and demo use are permitted; commercial use is not.

## Citation

```bibtex
@misc{lee2026chordmix,
  title         = {Empirical Study of Pop and Jazz Mix Ratios for Genre-Adaptive Chord Generation},
  author        = {Lee, Jinju},
  year          = {2026},
  eprint        = {2605.04998},
  archivePrefix = {arXiv}
}

@misc{lee2026chordtimeseries,
  title         = {How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity?},
  author        = {Lee, Jinju},
  year          = {2026},
  eprint        = {2606.07334},
  archivePrefix = {arXiv}
}

```
