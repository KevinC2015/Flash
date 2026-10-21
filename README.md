<div align="center">
  <img src="./asset/flash_logo.png" width="700"/>

[![Python](https://img.shields.io/badge/Python-3-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-required-ee4c2c)](https://pytorch.org/) [![License](https://img.shields.io/badge/license-CC--BY--NC%204.0-green)](#license) [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](../../issues)

  <h4>FLASH = Fast LLM-Aligned SimHash for Generative Recommendation</h4>
</div>

------

FLASH tokenizes each item into an unordered set of discrete semantic codes via SimHash on frozen LLM text embeddings, then predicts all code positions in parallel. A semantic alignment objective further anchors the learned item representations to the original LLM embedding space, improving recommendation performance and generalization.

## Paper

Official implementation of **[Rethinking Semantic ID Construction for Generative Recommendation: SimHash with Parallel Decoding and Semantic Alignment](https://arxiv.org/abs/2610.07402)**, accepted at **NeurIPS 2026**.

## Highlights

- **Training-free semantic ID construction.** FLASH uses SimHash with fixed random projections and requires **no tokenizer training**, unlike learned quantization methods such as RQ-VAE.
- **Fast tokenization.** Semantic IDs are constructed in **less than 0.5 seconds on CPU** across all four evaluated Amazon datasets. In our tokenization benchmarks, SimHash is **146×–2448× faster than RQ-VAE**, depending on the dataset and codebook configuration, even though RQ-VAE is measured on GPU.
- **Low code collisions.** SimHash exhibits consistently low collision counts across datasets, and collisions rapidly decrease as the number of codebooks increases. Under matched configurations, its collision count is **comparable to or lower than RQ-VAE**.
- **Parallel decoding.** Since SimHash code positions are unordered and do not have residual dependencies, FLASH predicts all semantic ID positions simultaneously instead of autoregressively.
- **Semantic grounding.** An explicit alignment loss compensates for semantic information lost during discrete hashing by preserving information from the original LLM embedding space.



## Architecture

<p align="center">
  <img src="asset/model.png" alt="FLASH framework" width="100%">
</p>

1. **SimHash Tokenization** — Each item's LLM text embedding is hashed into `m` codebook tokens via locality-sensitive hashing.
2. **Token Aggregation** — Codebook embeddings are concatenated and projected to a single item vector, optionally aligned with the LLM embedding (cosine similarity loss).
3. **GPT-2 Backbone** — Causal self-attention over the item sequence.
4. **Parallel Prediction Heads** — `m` independent ResBlock heads, each predicting one codebook position.

## Quick Start

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model=Flash --category=Beauty
```

Available categories: `Beauty`, `Sports_and_Outdoors`, `Toys_and_Games`, `CDs_and_Vinyl`

Datasets are downloaded automatically on first run. The OpenAI text embeddings can be downloaded from: https://drive.google.com/file/d/1HLy4GVfIKWVlhajXmq9_hI8oWggeU7EL/view?usp=sharing

## Key Hyperparameters

| Parameter            | Default | Description                                       |
| -------------------- | ------- | ------------------------------------------------- |
| `n_codebook`       | 32      | Number of codebook positions (semantic ID length) |
| `codebook_size`    | 256     | Entries per codebook                              |
| `n_embd`           | 448     | Hidden dimension                                  |
| `n_layer`          | 2       | Transformer layers                                |
| `align_weight`     | 0.2     | LLM alignment loss weight (0 = off)               |
| `codebook_dropout` | 0.2 | position-level dropout ratio                      |
| `temperature`      | 0.07    | Initial cosine logit temperature                  |
| `lr`               | 0.001   | Learning rate (dataset-dependent)                 |

All hyperparameters can be set via CLI (`--key=value`) or config files:

- `genrec/default.yaml` — global defaults
- `genrec/datasets/AmazonReviews2014/config.yaml` — dataset config
- `genrec/models/Flash/config.yaml` — model config



### Beauty

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=Flash \
    --category=Beauty \
    --lr=0.01 \
    --n_codebook=32 \
    --align_weight=0.1
```


### Toys and Games

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=Flash \
    --category=Toys_and_Games \
    --lr=0.003 \
    --n_codebook=64 \
    --align_weight=0.2
```

### Sports and Outdoors

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=Flash \
    --category=Sports_and_Outdoors \
    --lr=0.003 \
    --n_codebook=64 \
    --n_embd=896 \
    --align_weight=0.2
```

### CDs and Vinyl

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model=Flash \
    --category=CDs_and_Vinyl \
    --lr=0.001 \
    --n_codebook=64 \
    --codebook_size=512 \
    --align_weight=0.1
```

## Project Structure

```
genrec/
├── models/Flash/
│   ├── model.py          # Flash model (GPT-2 + parallel heads)
│   ├── tokenizer.py      # SimHash tokenizer
│   └── config.yaml       # Default hyperparameters
├── datasets/             # Dataset loaders
├── pipeline.py           # Training pipeline
├── trainer.py            # Training loop + evaluation
├── evaluator.py          # Metrics (Recall, NDCG)
└── utils.py              # Config, logging, utilities
```

## Citing this work

Please cite our paper if you find this work or code helpful.

```bibtex
@inproceedings{liu2026rethinking,
  title={Rethinking Semantic ID Construction for Generative Recommendation: SimHash with Parallel Decoding and Semantic Alignment},
  author={Yuqing Liu and Huiyuan Chen and Yibo Wang and Wooseong Yang and Philip S. Yu},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  url={https://arxiv.org/abs/2610.07402}
}
```

## License

Flash is CC-BY-NC 4.0 licensed.
