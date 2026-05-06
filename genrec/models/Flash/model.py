import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class ResBlock(nn.Module):
    """Residual Block: x + SiLU(Linear(x))"""

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class Flash(AbstractModel):
    """
    Flash: Unordered Semantic ID + LLM Alignment for Sequential Recommendation.

    Architecture:
        1. item_id → SimHash on LLM text emb → token set {c1, ..., c_m}
        2. wte(tokens) → (optional codebook dropout) → concat → Linear
           → input_emb (n_embd)  ── aligned with LLM text embedding (cos sim)
        3. GPT-2 Transformer backbone (causal self-attention)
        4. n_codebook parallel ResBlock heads (one per token position)
        5. Per-head cosine logits with learnable per-head temperatures
        6. Fixed-position CE loss (averaged per-head cross-entropy)

    """

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(Flash, self).__init__(config, dataset, tokenizer)

        self.register_buffer('item_id2tokens', self._map_item_tokens(), persistent=False)

        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=tokenizer.max_token_seq_len,
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_inner=config['n_inner'],
            activation_function=config['activation_function'],
            resid_pdrop=config['resid_pdrop'],
            embd_pdrop=config['embd_pdrop'],
            attn_pdrop=config['attn_pdrop'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            initializer_range=config['initializer_range'],
            eos_token_id=tokenizer.eos_token,
        )

        self.gpt2 = GPT2Model(gpt2config)

        # Parallel prediction heads (one per codebook position)
        self.n_pred_head = self.tokenizer.n_digit
        pred_head_list = []
        for i in range(self.n_pred_head):
            pred_head_list.append(ResBlock(self.config['n_embd']))
        self.pred_heads = nn.Sequential(*pred_head_list)

        # Aggregation: concat m codebook embeddings → Linear → item vector
        self.token_aggregator = nn.Linear(self.n_pred_head * config['n_embd'], config['n_embd'])

        init_temp = float(self.config['temperature'])
        self.per_head_temperature = bool(self.config.get('per_head_temperature', True))
        if self.per_head_temperature:
            self.log_temperature = nn.Parameter(
                torch.full((self.n_pred_head,), float(np.log(init_temp)))
            )
        else:
            self.log_temperature = nn.Parameter(
                torch.tensor(float(np.log(init_temp)))
            )
        self.codebook_dropout = self.config.get('codebook_dropout', 0.0)

        # Cross-entropy loss module (stateless; instantiate once).
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

        # --- LLM Text Embedding Alignment ---
        self.align_weight = self.config.get('align_weight', 0.0)
        self.codebook_l2_weight = self.config.get('codebook_l2_weight', 0.0)

        if self.align_weight > 0:
            text_emb_dim = config.get('sent_emb_dim', 3072)
            self._load_text_embeddings(dataset, config, text_emb_dim)
            # Project text embedding down to the item embedding space.
            self.text_projector = nn.Linear(text_emb_dim, config['n_embd'])

        self.generate_w_decoding_graph = False  

    def _load_text_embeddings(self, dataset, config, text_emb_dim):
        sent_emb_model = config.get('sent_emb_model', 'text-embedding-3-large')
        sent_emb_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(sent_emb_model)}.sent_emb'
        )
        assert os.path.exists(sent_emb_path), \
            f'Text embedding file not found: {sent_emb_path}.'
        raw_emb = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, text_emb_dim)
        assert raw_emb.shape[0] == dataset.n_items - 1
        pad_vec = np.zeros((1, text_emb_dim), dtype=np.float32)
        full_emb = np.concatenate([pad_vec, raw_emb], axis=0)
        self.register_buffer('text_embeddings', torch.from_numpy(full_emb), persistent=False)

    def _map_item_tokens(self) -> torch.Tensor:
        item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict, return_loss=True):
        # Encode: token set → codebook dropout → concat → Linear → item embedding
        input_tokens = self.item_id2tokens[batch['input_ids']]       # (B, L, n_digit)
        token_embs = self.gpt2.wte(input_tokens)                     # (B, L, n_digit, n_embd)
        B, L, m, d = token_embs.shape


        if self.training and self.codebook_dropout > 0.0:
            # (B, L, n_digit) mask — each token position independently dropped
            keep_mask = (torch.rand(B, L, m, device=token_embs.device)
                         > self.codebook_dropout).float()

            min_keep = max(m // 2, 1)
            too_few = keep_mask.sum(dim=-1) < min_keep  # (B, L)
            if too_few.any():
                flat_mask = keep_mask.view(-1, m)
                flat_too_few = too_few.view(-1)
                n_fix = int(flat_too_few.sum().item())
                # Random per-row permutations, take first min_keep indices
                rand_keys = torch.rand(n_fix, m, device=token_embs.device)
                fix_indices = rand_keys.argsort(dim=-1)[:, :min_keep]
                new_rows = torch.zeros(n_fix, m, device=token_embs.device)
                new_rows.scatter_(1, fix_indices, 1.0)
                flat_mask[flat_too_few] = new_rows

            scale = m / keep_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            token_embs = token_embs * keep_mask.unsqueeze(-1) * scale.unsqueeze(-1)

        # Aggregate codebook embeddings: concat → Linear → (B, L, n_embd)
        input_embs = self.token_aggregator(token_embs.view(B, L, m * d))

        # GPT-2 forward
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch['attention_mask'],
        )

        # Parallel prediction heads
        final_states = [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2)
                        for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)  # (B, L, n_heads, n_embd)
        outputs.final_states = final_states

        if return_loss:
            assert 'labels' in batch

            # --- Fixed-position CE Recommendation Loss ---
            rec_loss = self._fixed_position_ce_loss(final_states, batch['labels'])
            total_loss = rec_loss

            # --- Alignment Loss: align input_embs (pre-GPT-2) with LLM text embeddings ---
            if self.align_weight > 0.0:
                align_input = self._compute_alignment_loss(
                    batch['input_ids'], batch['attention_mask'],
                    input_embs, self.text_projector,
                )
                total_loss = total_loss + self.align_weight * align_input

            # --- L2 Regularization on codebook embeddings ---
            if self.codebook_l2_weight > 0.0:
                # Penalize large codebook embedding norms (exclude PAD and EOS)
                codebook_embs = self.gpt2.wte.weight[1:-1]  # (n_codebook * codebook_size, n_embd)
                l2_loss = codebook_embs.pow(2).sum() / codebook_embs.shape[0]
                total_loss = total_loss + self.codebook_l2_weight * l2_loss

            outputs.loss = total_loss
        return outputs

    def _fixed_position_ce_loss(self, final_states, labels):
        """
        Fixed-position cross-entropy loss.

        head_i predicts token_i using cosine similarity / temperature.
        Tokens are treated in their Coco Hash canonical order.

        For each head i:
            logits_i = normalize(h_i) @ normalize(wte_i).T / τ
            loss_i = CrossEntropy(logits_i, target_code_i)
        rec_loss = mean(loss_0, ..., loss_{m-1})
        """
        label_mask = labels.view(-1) != -100
        if label_mask.sum() == 0:
            return torch.tensor(0.0, device=labels.device)

        n_heads = self.n_pred_head
        n_embd = self.config['n_embd']
        codebook_size = self.config['codebook_size']

        # Selected states: (N_valid, n_heads, n_embd)
        selected_states = final_states.view(-1, n_heads, n_embd)[label_mask]
        selected_states = F.normalize(selected_states, dim=-1)
        selected_states = torch.chunk(selected_states, n_heads, dim=1)

        # Token embeddings (exclude pad=0 and eos=last)
        token_emb = self.gpt2.wte.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, n_heads, dim=0)

        # Per-head temperatures (positive via exp of learnable log-temperatures)
        temps = self.log_temperature.exp()
        if not self.per_head_temperature:
            temps = temps.expand(n_heads)

        # Per-head logits: cosine / temperature_i
        token_logits = [
            torch.matmul(selected_states[i].squeeze(dim=1), token_embs[i].T) / temps[i]
            for i in range(n_heads)
        ]

        # Target tokens for each head
        token_labels = self.item_id2tokens[labels.view(-1)[label_mask]]

        # Per-head CE loss
        losses = [
            self.ce_loss(token_logits[i], token_labels[:, i] - i * codebook_size - 1)
            for i in range(n_heads)
        ]

        rec_loss = torch.mean(torch.stack(losses))
        return rec_loss

    def _compute_alignment_loss(self, input_ids, attention_mask, hidden, projector):
        """
        Semantic regularization: align a hidden representation (per input item)
        with the LLM text embedding of the SAME item (negative cosine similarity).

        The LLM text embedding (frozen) is projected down from text_emb_dim
        to n_embd via a trainable Linear, then cosine-aligned with `hidden`:

            L = -mean( cos( hidden, projector(text_embs) ) )

        Args:
            input_ids: (B, L) item IDs in the sequence
            attention_mask: (B, L)
            hidden: (B, L, n_embd) — the representation to align (input_embs)
            projector: nn.Linear module from text_emb_dim to n_embd
        """
        # Mask: only align at non-padding positions
        valid_mask = attention_mask.float()  # (B, L)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=input_ids.device)

        # Look up LLM text embeddings for the same input items
        text_embs = self.text_embeddings[input_ids]          # (B, L, text_emb_dim)

        # Project text embedding down to item embedding space
        text_proj = projector(text_embs)                      # (B, L, n_embd)

        # Negative cosine similarity loss (lower = better alignment)
        h_norm = F.normalize(hidden, dim=-1)
        t_norm = F.normalize(text_proj, dim=-1)
        cos_sim = (h_norm * t_norm).sum(dim=-1)   # (B, L)
        align_loss = -(cos_sim * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

        return align_loss

    # ===== Inference: Cached Logit Scoring =====

    def generate(self, batch, n_return_sequences=1):
        """
        Generate top-k predictions using cached logit calculation.

        Efficient approach:
          1. Cache per-codebook log-softmax: p(j) = log_softmax(E_j · g_j(s) / τ) ∈ R^M
             This is O(M × d × m) and independent of N_items.
          2. For each candidate item with semantic ID (c1,...,cm):
             score = Σ_j log p(j)_{c_j}
             This is O(N × m) via a single gather operation.
          3. Return top-k items by score.

        Total: O(M×d×m + N×m) instead of naive O(N×m×d).
        """
        outputs = self.forward(batch, return_loss=False)

        # Get last position states: (B, 1, n_heads, n_embd)
        states = outputs.final_states.gather(
            dim=1,
            index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(
                -1, 1, self.n_pred_head, self.config['n_embd'])
        )
        states = F.normalize(states, dim=-1)

        n_heads = self.n_pred_head
        codebook_size = self.config['codebook_size']
        token_emb = self.gpt2.wte.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)

        # Step 1: Cache per-codebook log-probs (efficient logit cache)
        # For each head j: p(j) = log_softmax(E_j · g_j(s) / τ_j) ∈ R^(B, M)
        temps = self.log_temperature.exp()
        if not self.per_head_temperature:
            temps = temps.expand(n_heads)
        token_embs = torch.chunk(token_emb, n_heads, dim=0)
        logits = [
            torch.matmul(states[:, 0, i, :], token_embs[i].T) / temps[i]
            for i in range(n_heads)
        ]
        logits = [F.log_softmax(l, dim=-1) for l in logits]
        # Concatenate: (B, n_heads * codebook_size)
        token_logits = torch.cat(logits, dim=-1)

        # Step 2: Score all items via gather
        # item_id2tokens[1:, :] has shape (N_items-1, n_heads) with global token IDs
        # Subtract 1 to get 0-based index into token_logits
        item_logits = torch.gather(
            input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items - 1, -1),
            dim=-1,
            index=(self.item_id2tokens[1:, :] - 1).unsqueeze(0).expand(
                token_logits.shape[0], -1, -1)
        ).sum(dim=-1)  # (B, N_items-1) — sum of log-probs across codebook digits

        # Step 3: Top-k
        preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1  # +1 for 1-based item IDs
        return preds.unsqueeze(-1)  # (B, n_return_sequences, 1)
