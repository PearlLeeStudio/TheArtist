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
- country
---

# TheArtist Music Transformer — LoRA Adapter (Country)

LoRA adapter (r=4) that conditions the F1 base ([PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80)) toward **country chord progressions** — country pop, country rock, and modern country idioms. One of eleven per-genre adapters from the paper [How Far Can Chord-Symbol Time-Series Adaptation Carry Genre Identity?](https://huggingface.co/papers/2606.07334) (Lee, 2026). The released snapshot is the best rank of a 5-point sweep (r ∈ {4, 8, 16, 32, 64}); the vocabulary is extended 351 → 359 with `[GENRE:X]` tokens shipped in `embedding_extension.pt`.

[Paper](https://huggingface.co/papers/2606.07334) · [Code](https://github.com/PearlLeeStudio/TheArtist) · [Demo](https://youtu.be/iSDj4VK9nKc) · [All models](https://huggingface.co/PearlLeeStudio)

> **Base-weights note.** The released F1 base is weight-identical to the Phase-0 pop baseline (a checkpoint-selection artifact — see the note on the [base card](https://huggingface.co/PearlLeeStudio/TheArtist-MusicTransformer-ft-pop80)). Every "F1 base" column below was measured against those exact weights, so the Δ shown is the adapter's gain over a pure-pop harmonic prior.

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
lora_dir = snapshot_download(repo_id="PearlLeeStudio/TheArtist-MusicTransformer-lora-country")
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

# 6. Generate a country continuation. With LoRA injected,
#    PeftModel.forward routes through the adapted attention layers.
song = {
    "key": "Cmaj", "time_signature": "4/4", "genre": "country",
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

Teacher-forced token-level metrics on the country val split (6,173 sequences, no key augmentation). Both columns use the same dataloader and the same `[GENRE:none]`-initialised embedding extension — only the adapter weights and trained embedding rows differ.

| Metric | F1 base | F1 + this LoRA (r=4) | Δ |
|---|---:|---:|---:|
| Top-1 accuracy (%) | 82.45 | **86.01** | **+3.56** |
| Top-5 accuracy (%) | 96.22 | 97.90 | +1.68 |
| Cross-entropy loss | 0.7402 | 0.4970 | -0.2432 |

Rank sweep the released adapter was selected from (minimum val loss, top-1 tiebreak):

| Rank | val_loss | Top-1 (%) | Δ Top-1 vs F1 |
|---|---:|---:|---:|
| **r=4** | 0.4970 | 86.01 | +3.56 ← released |
| r=8 | 0.4977 | 85.08 | +2.63 |
| r=16 | 0.4982 | 85.06 | +2.61 |
| r=32 | 0.4972 | 85.98 | +3.53 |
| r=64 | 0.4977 | 85.98 | +3.53 |

Real-song check — mean over **10 held-out country songs** (30–81 bars each; titles are Spotify track IDs by upstream Chordonomicon policy — the progressions are real songs):

| Model | Top-1 (%) | Top-5 (%) | Loss |
|---|---:|---:|---:|
| F1 base | 85.90 | 98.44 | 0.5152 |
| **F1 + this LoRA** | **87.41** | 98.09 | 0.4231 |
| Δ | +1.51 | -0.36 | -0.0921 |

The 10 songs are this genre's slice of a 130-song eval set (10 per genre × 13 genres, seed 42) drawn from the held-out val/test partitions only — pop from McGill Billboard (CC0), jazz from public standards corpora, classical from Bach chorales, the other ten genres from the matching Chordonomicon subsets (CC BY-NC 4.0).

## Training data

61,736 sequences from 53,236 songs — the **country** subset of the [Chordonomicon dataset](https://huggingface.co/datasets/ailsntua/Chordonomicon), song-level 80/10/10 split (seed 42), 12-key augmentation on train. Chordonomicon is licensed CC BY-NC 4.0; see the dataset card for full terms.

Adapter: LoRA on the Q/K/V projections (`w_q`, `w_k`, `w_v`), r=4, α=8, dropout 0.05. The adapter file holds 98,304 LoRA parameters (0.4 MB). Training also updates the token-embedding and output matrices (367,616 parameters, shipped in `embedding_extension.pt`), so the trained set totals 465,920 parameters, 1.8% of the 25,665,536 that full fine-tuning updates. Best checkpoint by minimum val loss.

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
