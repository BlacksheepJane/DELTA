import tempfile
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from delta.checkpoint import load_delta_checkpoint, save_delta_checkpoint
from delta.modeling import DeltaLinear, set_submodule
from delta.quantization import SINQLinearFactor, _quantize_candidate


def _tiny_tokenizer():
    backend = Tokenizer(
        WordLevel(
            vocab={"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    )


def _factor(weight, params):
    candidate = _quantize_candidate(
        weight,
        4,
        params,
        device="cpu",
        metadata_dtype=torch.float16,
    )
    return SINQLinearFactor(
        candidate["qweight"],
        candidate["meta"],
        params,
        nbits=4,
        rank=64,
    )


class CheckpointTest(unittest.TestCase):
    def test_safetensors_roundtrip(self):
        torch.manual_seed(11)
        config = LlamaConfig(
            vocab_size=5,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
        model = LlamaForCausalLM(config).half().eval()
        module_path = "model.layers.0.self_attn.q_proj"
        original = model.get_submodule(module_path)
        u_weight = torch.eye(64)
        v_weight = original.weight.detach().float().cpu()
        replacement = DeltaLinear(
            _factor(u_weight, {"optimize": False, "axis": 1}),
            _factor(v_weight, {"optimize": False, "axis": 0}),
            in_features=64,
            out_features=64,
        )
        set_submodule(model, module_path, replacement)
        input_ids = torch.tensor([[1, 3, 4, 2]])
        with torch.no_grad():
            expected = model(input_ids=input_ids).logits

        with tempfile.TemporaryDirectory() as directory:
            save_delta_checkpoint(
                model,
                _tiny_tokenizer(),
                directory,
                choices={"format": "test", "layers": {}},
                compression_config={"method": "test"},
                max_shard_size="10MB",
            )
            loaded_model, loaded_tokenizer = load_delta_checkpoint(directory)
            loaded_model.eval()
            with torch.no_grad():
                actual = loaded_model(input_ids=input_ids).logits
            self.assertEqual(loaded_tokenizer.bos_token_id, 1)
            self.assertIsInstance(loaded_model.get_submodule(module_path), DeltaLinear)
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
