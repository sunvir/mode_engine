# MoDE Engine  
# Copyright (c) 2026 Sunvir S. Gujral  
# All rights reserved.

# This project is licensed under the MoDE Engine Source‑Available License (MESAL) v1.2.  
# Non‑commercial use only. Commercial use requires a separate license.

# Attribution is required for all uses, modifications, and redistributions.

# For commercial licensing inquiries, contact the Author.
"""MoDEInferencer — batched inference with streaming and non-streaming modes.

This module is pure ML: no web framework, no async runtime.  Callers are
responsible for wrapping the synchronous streaming generator in whatever
async/thread executor they require.

gate_indices shape
------------------
``gate_indices`` passed to :meth:`MoDEInferencer.generate` must have shape
``(num_layers, batch_size)`` where *num_layers* is the total number of
decoder layers in the model (including unexpanded ones).

``ExpandedDecoderLayer`` at layer *i* indexes ``gate_indices[i]`` to obtain
a ``(batch_size,)`` tensor of expert assignments — one per sequence.
Unexpanded layers never read ``gate_indices``, so their row values are
ignored; using ``-1`` for those rows is conventional.

To use the **same expert for all layers**, repeat the row::

    per_seq = torch.tensor([2, 0, -1])           # (batch_size,)
    gate_indices = per_seq.unsqueeze(0).expand(num_layers, -1)

To assign **different experts per layer**::

    gate_indices = torch.tensor([
        [0, 1, -1],   # layer 0: seq-0→expert-0, seq-1→expert-1, seq-2→passthrough
        [2, 0,  1],   # layer 1: seq-0→expert-2, seq-1→expert-0, seq-2→expert-1
        ...
    ])  # shape (num_layers, batch_size)

Usage (non-streaming)::

    inferencer = MoDEInferencer(model, tokenizer, device)
    num_layers = len(plugin.get_layers(model))
    # Same expert (2) for all layers, except seq 2 which is passthrough
    gate_indices = torch.tensor([2, 0, -1]).unsqueeze(0).expand(num_layers, -1)
    result = inferencer.generate(
        prompts=["Hello", "World", "Test"],
        gate_indices=gate_indices,
    )
    print(result.texts)

Usage (streaming)::

    for token_batch in inferencer.generate(
        prompts=["Hello", "World"],
        gate_indices=torch.tensor([[1, 0]] * num_layers),
        stream=True,
    ):
        # token_batch[i] is the decoded token for sequence i, or None if finished
        print(token_batch)
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Generator
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GenerationConfig:
    """Knobs for the token-sampling loop.

    Attributes:
        max_new_tokens:      Maximum tokens to generate per sequence.
        temperature:         Softmax temperature.  1.0 = no change.
        top_p:               Nucleus-sampling probability mass.  1.0 = disabled.
        top_k:               Keep only the top-k logits.  0 = disabled.
        do_sample:           If False, use greedy decoding (argmax).
        eos_token_id:        Stop when this token is sampled.  Falls back to
                             ``tokenizer.eos_token_id`` if None.
        pad_token_id:        Token used to fill finished sequences in the
                             output tensor.  Falls back to
                             ``tokenizer.pad_token_id`` if None.
        repetition_penalty:  Penalise tokens that already appear in the
                             prompt/generated text.  1.0 = disabled.
    """

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = False
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    repetition_penalty: float = 1.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class InferenceResult:
    """Completed generation result for an entire batch.

    Attributes:
        texts:                   Decoded output strings, one per prompt.
        token_ids:               Raw generated token ids, one list per prompt.
        prompt_token_counts:     Number of prompt tokens per sequence.
        generated_token_counts:  Number of generated tokens per sequence.
    """

    texts: list[str]
    token_ids: list[list[int]]
    prompt_token_counts: list[int]
    generated_token_counts: list[int]


# ---------------------------------------------------------------------------
# Inferencer
# ---------------------------------------------------------------------------

class MoDEInferencer:
    """Batched token-generation wrapper for a MoDE-expanded causal LM.

    The caller owns the model and tokenizer (obtained from
    ``MoDEEngine.load_model``).  This class holds only references and
    introduces no model-level state.

    Args:
        model:      A causal LM (possibly with MoDE expansion blocks).
                    Must already be in ``eval()`` mode.
        tokenizer:  The corresponding HuggingFace tokenizer.
        device:     The device the model lives on.
    """

    def __init__(self, model: nn.Module, tokenizer: Any, device: torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: list[str],
        gate_indices: torch.Tensor,
        config: GenerationConfig | None = None,
        stream: bool = False,
    ) -> InferenceResult | Generator[list[str | None], None, None]:
        """Generate text for a batch of prompts.

        Args:
            prompts:       One prompt string per sequence.
            gate_indices:  Shape ``(num_layers, batch_size)``, dtype ``torch.long``.
                           ``gate_indices[i]`` is a ``(batch_size,)`` tensor
                           selecting the expert for each sequence at layer *i*.
                           Use ``-1`` for passthrough (base model only).
                           Unexpanded layers never read their row, so ``-1``
                           is conventional for those positions.
            config:        Sampling configuration.  Defaults to greedy,
                           256 max new tokens.
            stream:        If True, return a synchronous generator that
                           yields one ``list[str | None]`` per decode step.
                           Each element is the decoded new token for that
                           sequence, or ``None`` once the sequence has
                           finished.  If False, block until all sequences
                           are done and return an ``InferenceResult``.

        Returns:
            ``InferenceResult`` when ``stream=False``.
            ``Generator[list[str | None], None, None]`` when ``stream=True``.
        """
        config = config or GenerationConfig()
        config.eos_token_id = config.eos_token_id if config.eos_token_id is not None else self.tokenizer.eos_token_id
        config.pad_token_id = config.pad_token_id if config.pad_token_id is not None else self.tokenizer.pad_token_id

        batch_size = len(prompts)
        gate_indices = self._validate_gate_indices(gate_indices, batch_size)
        input_ids, attention_mask = self._tokenize_batch(prompts)

        if stream:
            return self._generate_streaming(input_ids, attention_mask, gate_indices, config)
        return self._generate_non_streaming(input_ids, attention_mask, gate_indices, config)

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _tokenize_batch(
        self, prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise *prompts* with left-padding.

        Left-padding is required for decoder-only models: all sequences must
        share the same last-real-token position so that the first generated
        token's attention over the prompt is correct.

        Returns:
            ``(input_ids, attention_mask)`` — both on ``self.device``,
            shape ``(batch_size, max_seq_len)``.
        """
        original_padding_side = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = "left"
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

        return (
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_gate_indices(
        self, gate_indices: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Validate and normalise *gate_indices*.

        Expected shape: ``(num_layers, batch_size)``.

        Raises:
            ValueError: if ndim != 2 or the batch dimension does not match.
        """
        if gate_indices.ndim != 2:
            raise ValueError(
                f"gate_indices must be 2-D with shape (num_layers, batch_size), "
                f"got shape {tuple(gate_indices.shape)}.  "
                f"To use the same expert for all layers: "
                f"gate_indices.unsqueeze(0).expand(num_layers, -1)"
            )
        if gate_indices.size(1) != batch_size:
            raise ValueError(
                f"gate_indices.size(1)={gate_indices.size(1)} must equal "
                f"batch_size={batch_size}"
            )
        if gate_indices.dtype not in (torch.int32, torch.int64):
            logger.debug(
                "gate_indices dtype %s — coercing to torch.long", gate_indices.dtype
            )
        return gate_indices.to(dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Single forward step
    # ------------------------------------------------------------------

    def _forward_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gate_indices: torch.Tensor,
        past_key_values: Any | None,
    ) -> tuple[torch.Tensor, Any]:
        """Run one forward pass and return ``(logits_last_pos, past_key_values)``.

        Args:
            input_ids:       ``(batch, seq_len)`` — prompt on first call,
                             ``(batch, 1)`` on subsequent decode steps.
            attention_mask:  ``(batch, total_seq_len_so_far)``
            gate_indices:    ``(batch,)`` — constant across all decode steps.
            past_key_values: KV cache from the previous step, or ``None``.

        Returns:
            Tuple of logits at the last position ``(batch, vocab_size)``
            and the updated KV cache.
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                gate_indices=gate_indices,
            )
        return outputs.logits[:, -1, :], outputs.past_key_values

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        config: GenerationConfig,
        past_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Sample the next token for every sequence in the batch.

        Pipeline (applied in order):
        1. Repetition penalty
        2. Temperature scaling
        3. Top-k filtering
        4. Top-p (nucleus) filtering
        5. Greedy argmax  or  categorical sampling

        Args:
            logits:         ``(batch, vocab_size)`` — raw logits at last pos.
            config:         Sampling configuration.
            past_token_ids: ``(batch, current_seq_len)`` — for repetition
                            penalty; the full sequence seen so far.

        Returns:
            ``(batch,)`` integer tensor of sampled token ids.
        """
        logits = logits.float()  # work in fp32 for numerical stability

        # 1. Repetition penalty
        if config.repetition_penalty != 1.0:
            for i in range(logits.size(0)):
                unique_tokens = past_token_ids[i].unique()
                # Divide positive logits, multiply negative logits
                token_logits = logits[i, unique_tokens]
                penalised = torch.where(
                    token_logits > 0,
                    token_logits / config.repetition_penalty,
                    token_logits * config.repetition_penalty,
                )
                logits[i, unique_tokens] = penalised

        # 2. Temperature
        if config.temperature != 1.0:
            logits = logits / config.temperature

        # 3. Top-k
        if config.top_k > 0:
            top_k = min(config.top_k, logits.size(-1))
            kth_values = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
            logits = logits.masked_fill(logits < kth_values, float("-inf"))

        # 4. Top-p (nucleus)
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens beyond the nucleus (shift right so the first token is kept)
            remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > config.top_p
            sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
            logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)

        # 5. Decode
        if config.do_sample:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        return torch.argmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Non-streaming generation
    # ------------------------------------------------------------------

    def _generate_non_streaming(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gate_indices: torch.Tensor,
        config: GenerationConfig,
    ) -> InferenceResult:
        batch_size = input_ids.size(0)
        prompt_lengths = attention_mask.sum(dim=-1).tolist()

        generated_ids: list[list[int]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        current_ids = input_ids
        current_mask = attention_mask
        past_key_values = None

        for step in range(config.max_new_tokens):
            logits, past_key_values = self._forward_step(
                current_ids, current_mask, gate_indices, past_key_values
            )

            # Build full sequence seen so far for repetition penalty
            if step == 0:
                full_ids = current_ids
            else:
                full_ids = torch.cat([full_ids, current_ids], dim=-1)

            next_tokens = self._sample_next_token(logits, config, full_ids)

            for i in range(batch_size):
                if not finished[i]:
                    token = next_tokens[i].item()
                    generated_ids[i].append(token)
                    if token == config.eos_token_id:
                        finished[i] = True

            if finished.all():
                break

            # Next decode step: single new token, extended mask
            current_ids = next_tokens.unsqueeze(-1)
            current_mask = torch.cat(
                [current_mask, torch.ones(batch_size, 1, dtype=current_mask.dtype, device=self.device)],
                dim=-1,
            )

        # Decode — strip EOS from output text
        texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]

        return InferenceResult(
            texts=texts,
            token_ids=generated_ids,
            prompt_token_counts=[int(p) for p in prompt_lengths],
            generated_token_counts=[len(ids) for ids in generated_ids],
        )

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    def _generate_streaming(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gate_indices: torch.Tensor,
        config: GenerationConfig,
    ) -> Generator[list[str | None], None, None]:
        """Yield one decoded token per sequence per step.

        Each yielded value is a ``list`` of length ``batch_size`` where
        element *i* is the decoded string for the new token at that step,
        or ``None`` if sequence *i* has already finished.

        Note: single tokens that form multi-byte UTF-8 characters may decode
        to empty strings until the full character is accumulated.  This
        matches HuggingFace's streaming behaviour; character-boundary
        handling is the caller's responsibility.
        """
        batch_size = input_ids.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        current_ids = input_ids
        current_mask = attention_mask
        past_key_values = None
        full_ids = None

        for step in range(config.max_new_tokens):
            logits, past_key_values = self._forward_step(
                current_ids, current_mask, gate_indices, past_key_values
            )

            if step == 0:
                full_ids = current_ids
            else:
                full_ids = torch.cat([full_ids, current_ids], dim=-1)

            next_tokens = self._sample_next_token(logits, config, full_ids)

            token_texts: list[str | None] = []
            for i in range(batch_size):
                if finished[i]:
                    token_texts.append(None)
                else:
                    token = next_tokens[i].item()
                    if token == config.eos_token_id:
                        finished[i] = True
                        token_texts.append(None)
                    else:
                        token_texts.append(
                            self.tokenizer.decode([token], skip_special_tokens=True)
                        )

            yield token_texts

            if finished.all():
                return

            current_ids = next_tokens.unsqueeze(-1)
            current_mask = torch.cat(
                [current_mask, torch.ones(batch_size, 1, dtype=current_mask.dtype, device=self.device)],
                dim=-1,
            )
