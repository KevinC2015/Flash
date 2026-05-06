import os
import json
import math
import tempfile

import numpy as np

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class EmbeddingSimHash:
    """
    SimHash on LLM text embeddings for semantic, locality-sensitive IDs.

    Args:
        codebook_size:     entries per codebook (default 256)
        n_codebook:        number of codebook positions (default 16)
        emb_dim:           dimension of the text embeddings
        seed:              random seed for reproducibility
    """

    def __init__(self, codebook_size=256, n_codebook=16,
                 emb_dim=3072, seed=42):
        self.codebook_size = codebook_size
        self.n_codebook = n_codebook
        self.emb_dim = emb_dim
        self.n_hashes_per_code = int(math.ceil(math.log2(max(codebook_size, 2))))

        # Random hyperplanes: (n_codebook, n_hashes_per_code, emb_dim)
        rng = np.random.default_rng(seed)
        self.hyperplanes = rng.standard_normal(
            (n_codebook, self.n_hashes_per_code, emb_dim)
        ).astype(np.float32)

        # Powers of two for bit -> integer packing
        self.bit_weights = (1 << np.arange(self.n_hashes_per_code)).astype(np.int64)

    def encode_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode a batch of text embeddings to codebook indices.

        Args:
            embeddings: (N, emb_dim) float array
        """
        # (N, D) @ (D, n_codebook * n_hashes_per_code)
        H = self.hyperplanes.reshape(-1, self.emb_dim).T
        proj = embeddings @ H
        proj = proj.reshape(-1, self.n_codebook, self.n_hashes_per_code)

        # Sign bits -> integer per codebook position
        bits = (proj > 0).astype(np.int64)
        ints = (bits * self.bit_weights).sum(-1)
        codes = ints % self.codebook_size
        return codes


class FlashTokenizer(AbstractTokenizer):
    """
    Tokenizer for Flash using SimHash semantic IDs.
    """

    def __init__(self, config: dict, dataset: AbstractDataset):
        super(FlashTokenizer, self).__init__(config, dataset)
        self.item2id = dataset.item2id
        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']
        self.item2tokens = self._init_tokenizer(dataset)
        self.eos_token = self.n_digit * self.codebook_size + 1
        self.ignored_label = -100

    @property
    def n_digit(self):
        return self.config['n_codebook']

    @property
    def codebook_size(self):
        return self.config['codebook_size']

    @property
    def max_token_seq_len(self) -> int:
        return self.config['max_item_seq_len']

    @property
    def vocab_size(self) -> int:
        return self.eos_token + 1

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        """Convert raw semantic IDs (0-based per codebook) to global token IDs."""
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.n_digit):
                tokens[digit] += self.codebook_size * digit + 1
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

    def _init_tokenizer(self, dataset: AbstractDataset):
        """Initialize tokenizer with SimHash semantic IDs (cached to disk)."""
        n_codebook = self.config['n_codebook']
        codebook_size = self.config['codebook_size']
        seed = self.config['rand_seed']

        sem_ids_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'simhash_n{n_codebook}_c{codebook_size}_s{seed}.sem_ids'
        )

        if not os.path.exists(sem_ids_path):
            item2sem_ids = self._generate_simhash_ids(dataset, n_codebook, codebook_size, seed)

            self.log(f'[TOKENIZER] Saving to {sem_ids_path}...')

            tmp_dir = os.path.dirname(sem_ids_path)
            fd, tmp_path = tempfile.mkstemp(
                prefix='.sem_ids_', suffix='.tmp', dir=tmp_dir
            )
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(item2sem_ids, f)
                os.replace(tmp_path, sem_ids_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
        else:
            self.log(f'[TOKENIZER] Loading SimHash semantic IDs from {sem_ids_path}...')

        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)
        return item2tokens

    def _generate_simhash_ids(self, dataset, n_codebook, codebook_size, seed):
        """
        Generate semantic IDs using SimHash on LLM text embeddings.

        Requires {sent_emb_model}.sent_emb in the dataset cache.
        """
        self.log(f'[TOKENIZER] Generating SimHash semantic IDs from text embeddings...')

        sent_emb_model = self.config.get('sent_emb_model', 'text-embedding-3-large')
        sent_emb_dim = self.config.get('sent_emb_dim', 3072)
        emb_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(sent_emb_model)}.sent_emb'
        )
        assert os.path.exists(emb_path), \
            f'Text embedding file not found: {emb_path}. SimHash requires text embeddings.'

        raw_emb = np.fromfile(emb_path, dtype=np.float32).reshape(-1, sent_emb_dim)
        assert raw_emb.shape[0] == dataset.n_items - 1, \
            f'Expected {dataset.n_items - 1} embeddings, got {raw_emb.shape[0]}'

        hasher = EmbeddingSimHash(
            codebook_size=codebook_size,
            n_codebook=n_codebook,
            emb_dim=sent_emb_dim,
            seed=seed,
        )

        # Batch encode all items (item IDs 1..n_items-1 map to rows 0..n_items-2)
        all_codes = hasher.encode_batch(raw_emb)  # (n_items-1, n_codebook)

        item2sem_ids = {}
        collisions = 0
        seen_tuples = set()
        for i in range(1, dataset.n_items):
            item = self.id2item[i]
            codes = all_codes[i - 1].tolist()
            code_key = tuple(codes)
            if code_key in seen_tuples:
                collisions += 1
            seen_tuples.add(code_key)
            item2sem_ids[item] = codes

        self.log(f'[TOKENIZER] Generated {len(item2sem_ids)} SimHash IDs, {collisions} collisions')
        self.log(f'[TOKENIZER] Hashes/code: {hasher.n_hashes_per_code}, emb_dim: {sent_emb_dim}')
        return item2sem_ids

    def _tokenize_first_n_items(self, item_seq: list) -> tuple:
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        labels = [self.item2id[item] for item in item_seq[1:]]
        labels.extend([self.ignored_label] * pad_lens)
        return input_ids, attention_mask, labels, seq_lens

    def _tokenize_later_items(self, item_seq: list, pad_labels: bool = True) -> tuple:
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        labels = [self.ignored_label] * seq_lens
        labels[-1] = self.item2id[item_seq[-1]]
        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        if pad_labels:
            labels.extend([self.ignored_label] * pad_lens)
        return input_ids, attention_mask, labels, seq_lens

    def tokenize_function(self, example: dict, split: str) -> dict:
        max_item_seq_len = self.config['max_item_seq_len']
        item_seq = example['item_seq'][0]
        if split == 'train':
            n_return_examples = max(len(item_seq) - max_item_seq_len, 1)
            input_ids, attention_mask, labels, seq_lens = self._tokenize_first_n_items(
                item_seq=item_seq[:min(len(item_seq), max_item_seq_len + 1)]
            )
            all_input_ids, all_attention_mask, all_labels, all_seq_lens = \
                [input_ids], [attention_mask], [labels], [seq_lens]
            for i in range(1, n_return_examples):
                cur_item_seq = item_seq[i:i + max_item_seq_len + 1]
                input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(cur_item_seq)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)
            return {
                'input_ids': all_input_ids, 'attention_mask': all_attention_mask,
                'labels': all_labels, 'seq_lens': all_seq_lens,
            }
        else:
            input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                item_seq=item_seq[-(max_item_seq_len + 1):], pad_labels=False
            )
            return {
                'input_ids': [input_ids], 'attention_mask': [attention_mask],
                'labels': [labels[-1:]], 'seq_lens': [seq_lens]
            }

    def tokenize(self, datasets: dict) -> dict:
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=True, batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: '
            )
        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')
        return tokenized_datasets
