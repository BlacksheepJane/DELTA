# DELTA: Decoupling Latent Heterogeneity in Asymmetric Low-Rank Compression

## Abstract

DELTA is an asymmetric low-rank compression method that jointly optimizes rank
truncation, factor-specific bit-widths, and quantization directions under a
fixed memory budget. It preserves more effective singular directions through
low-bit factor representations and uses activation-aware reconstruction error
to select layerwise configurations, substantially improving accuracy at high
compression ratios without additional training.

## Installation

```bash
conda create -n delta python=3.9 -y
conda activate delta
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

For other CUDA versions, install the corresponding PyTorch wheel. The modified
[SINQ](https://github.com/huawei-csl/SINQ) runtime required by DELTA is included
in `delta/_vendor/sinq`; do not install another SINQ package.

```bash
python -m unittest discover -s tests -v
bash -n scripts/*.sh
```

## Experimental Environment

| Component | Configuration |
| --- | --- |
| GPU | 8 × NVIDIA H20-3e|
| OS | Ubuntu 22.04.3 LTS, Linux 5.15.0-91 |
| Python | 3.9 |
| PyTorch / CUDA | 2.8.0+cu128 / 12.8 |

## Compression

```bash
bash scripts/compress_llama7b.sh \
  --model-path /path/to/huggyllama/llama-7b \
  --ratio 0.2 \
  --output-dir outputs/llama7b-r020 \
  --profiling-path cache/llama7b-wikitext2-256-seed3.pt \
  --devices 0,1,2,3
```

An existing profiling file is reused; otherwise it is generated and saved at
`--profiling-path`. GPU IDs passed to `--devices` are logical CUDA indices.

## Perplexity Evaluation

```bash
bash scripts/eval_ppl.sh \
  outputs/llama7b-r020 \
  results/llama7b-r020/ppl \
  --device cuda:0 \
  --batch-size 4
```

WikiText-2 and C4 are evaluated by default. Use `--datasets wikitext2` to
evaluate WikiText-2 only. Results are written to `ppl.json` and `ppl.csv`.

## Downstream Evaluation

```bash
bash scripts/eval_downstream.sh \
  outputs/llama7b-r020 \
  results/llama7b-r020/downstream \
  0,1,2,3 \
  --batch-size 1
```

The 0-shot task set contains OpenBookQA, WinoGrande, HellaSwag, ARC-Easy, PIQA,
and MathQA. Add `--limit 10` for a smoke test.

## Acknowledgements

We thank the authors and maintainers of
[SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM),
[SINQ](https://github.com/huawei-csl/SINQ), and
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for their valuable
open-source contributions.

## License

DELTA is released under the Apache License 2.0. See `NOTICE` and
`delta/_vendor/sinq/LICENSE` for third-party attribution.

## Citation

If you find DELTA useful or relevant to your research, please kindly cite our paper:

```bibtex
@inproceedings{zhan2026delta,
  title={{DELTA}: Decoupling Latent Heterogeneity in Asymmetric Low-Rank Compression},
  author={Zhan, Jialin and Liu, Fangxin and Wang, Junjie and Yang, Ning and Jiang, Li and Guan, Haibing},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year={2026}
}
