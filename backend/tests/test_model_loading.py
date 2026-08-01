"""Checkpoint reuse must preserve exact weights and independent adapter state."""
import unittest
from unittest.mock import patch

from app.services import inference


class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        torch = inference.torch
        self.registry = inference.ModelRegistry()
        self.config = dict(d_model=16, n_heads=2, d_ff=32, n_layers=2, max_seq_len=32)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            self.original = inference.MusicTransformer(
                vocab_size=self.registry._trained_vocab_size(), dropout=0,
                pad_id=self.registry.tokenizer.pad_id, **self.config).eval()
        self.state = self.original.state_dict()
        self.checkpoint = dict(model_state_dict=self.state, config=self.config, optimizer='not retained')

    def test_loading_skips_random_initialization_and_preserves_logits(self):
        torch = inference.torch
        rng = torch.get_rng_state().clone()
        loaded = self.registry._build_base_model(self.checkpoint, self.config, torch.device('cpu')).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(any(p.is_meta for p in loaded.parameters()))
        # Compare logits on the trained vocabulary and all new genre rows.
        trained_vocab = self.registry._trained_vocab_size()
        self.registry._resize_embedding_inplace(self.original, trained_vocab,
                                               self.registry.tokenizer.vocab_size, torch.device('cpu'))
        ids = torch.tensor([[1, 4, 7, 10]])
        with torch.inference_mode():
            expected = self.original(ids)
            actual = loaded(ids)
        torch.testing.assert_close(actual[:, :, :expected.shape[-1]], expected, rtol=0, atol=0)
        none_id = self.registry.tokenizer.encode_genre('none')
        for index in range(trained_vocab, self.registry.tokenizer.vocab_size):
            self.assertTrue(torch.equal(loaded.token_emb.weight[index], self.state['token_emb.weight'][none_id]))

    def test_f1_storage_is_shared_but_overlaid_embeddings_and_adapter_modules_are_independent(self):
        from peft import LoraConfig, TaskType, get_peft_model
        torch = inference.torch
        device = torch.device('cpu')
        with patch.object(torch, 'load', return_value=self.checkpoint) as load:
            first = self.registry._read_checkpoint('ft_f1', device)
            second = self.registry._read_checkpoint('ft_f1', device)
        load.assert_called_once()
        self.assertIs(first, second)
        self.assertNotIn('optimizer', first)
        plain = self.registry._build_base_model(first, self.config, device).eval()
        adapter_base = self.registry._build_base_model(second, self.config, device)
        self.assertIsNot(plain.layers[0], adapter_base.layers[0])
        self.assertEqual(plain.layers[0].attn.w_q.weight.data_ptr(), adapter_base.layers[0].attn.w_q.weight.data_ptr())
        self.assertNotEqual(plain.token_emb.weight.data_ptr(), adapter_base.token_emb.weight.data_ptr())
        ids = torch.tensor([[1, 4, 7, 10]])
        with torch.inference_mode():
            before = plain(ids).clone()
        wrapped = get_peft_model(adapter_base, LoraConfig(r=2, lora_alpha=4,
            target_modules=['w_q', 'w_v'], task_type=TaskType.FEATURE_EXTRACTION)).eval()
        with torch.no_grad():
            for name, parameter in wrapped.named_parameters():
                if 'lora_B' in name:
                    parameter.fill_(.2)
            adapter_base.token_emb.weight[4].add_(.5)
        with torch.inference_mode():
            torch.testing.assert_close(plain(ids), before, rtol=0, atol=0)
            self.assertFalse(torch.equal(wrapped.get_base_model()(ids), before))


if __name__ == '__main__':
    unittest.main()
