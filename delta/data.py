"""Dataset loading for DELTA calibration and evaluation."""

import random
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader


C4_VALIDATION_URL = (
    "https://huggingface.co/datasets/allenai/c4/resolve/"
    "f998d2cd8b92435980789e3ecb2f89b4c68bfe1e/"
    "en/c4-validation.00000-of-00008.json.gz"
)


class TokenChunkDataset(torch.utils.data.Dataset):
    def __init__(self, chunks: torch.Tensor) -> None:
        self.chunks = chunks

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.chunks[index]

    def __len__(self) -> int:
        return int(self.chunks.shape[0])


def _load_local_split(path: str, split: str) -> Dataset:
    local_path = Path(path).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError("Dataset path does not exist: {}".format(local_path))
    if local_path.is_dir():
        arrow_path = local_path / "wikitext-{}.arrow".format(split)
        if arrow_path.exists():
            return Dataset.from_file(str(arrow_path))
        loaded = load_from_disk(str(local_path))
        if isinstance(loaded, DatasetDict):
            return loaded[split]
        return loaded
    return load_dataset("json", data_files=str(local_path), split="train")


def load_wikitext2(
    split: str,
    cache_dir: Optional[str] = None,
    data_path: Optional[str] = None,
) -> Dataset:
    if data_path:
        return _load_local_split(data_path, split)
    return load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split=split,
        cache_dir=cache_dir,
    )


def load_c4_validation(
    cache_dir: Optional[str] = None,
    data_path: Optional[str] = None,
) -> Dataset:
    source = str(Path(data_path).expanduser().resolve()) if data_path else C4_VALIDATION_URL
    if data_path and not Path(source).exists():
        raise FileNotFoundError("C4 file does not exist: {}".format(source))
    return load_dataset(
        "json",
        data_files={"validation": source},
        split="validation",
        cache_dir=cache_dir,
    )


def get_calibration_data(
    tokenizer,
    nsamples: int = 256,
    seqlen: int = 2048,
    seed: int = 3,
    cache_dir: Optional[str] = None,
    wikitext_path: Optional[str] = None,
) -> List[dict]:
    if nsamples <= 0 or seqlen <= 0:
        raise ValueError("nsamples and seqlen must be positive")
    dataset = load_wikitext2("train", cache_dir, wikitext_path)
    encoded = tokenizer(
        "\n\n".join(dataset["text"]), return_tensors="pt", verbose=False
    ).input_ids
    if encoded.shape[1] <= seqlen:
        raise ValueError("WikiText-2 training split is shorter than seqlen")

    generator = random.Random(seed)
    batches = []
    for _ in range(nsamples):
        start = generator.randint(0, encoded.shape[1] - seqlen - 1)
        input_ids = encoded[:, start : start + seqlen].clone()
        batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        )
    return batches


def _token_chunks(
    texts,
    tokenizer,
    seqlen: int,
    max_samples: Optional[int],
) -> torch.Tensor:
    token_ids = tokenizer(
        "\n\n".join(texts), return_tensors="pt", verbose=False
    ).input_ids[0]
    count = int(token_ids.numel() // seqlen)
    if max_samples is not None:
        count = min(count, int(max_samples))
    if count <= 0:
        raise ValueError("Dataset does not contain a complete token sequence")
    return token_ids[: count * seqlen].reshape(count, seqlen)


def get_ppl_dataloader(
    name: str,
    tokenizer,
    seqlen: int = 2048,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    wikitext_path: Optional[str] = None,
    c4_path: Optional[str] = None,
) -> DataLoader:
    normalized = name.strip().lower()
    if normalized in ("wikitext2", "wikitext-2"):
        dataset = load_wikitext2("test", cache_dir, wikitext_path)
        texts = dataset["text"]
    elif normalized == "c4":
        dataset = load_c4_validation(cache_dir, c4_path)
        count = min(2000, len(dataset))
        texts = dataset.select(range(count))["text"]
    else:
        raise ValueError("Unsupported PPL dataset: {}".format(name))
    chunks = _token_chunks(texts, tokenizer, seqlen, max_samples)
    return DataLoader(TokenChunkDataset(chunks), batch_size=batch_size, shuffle=False)
