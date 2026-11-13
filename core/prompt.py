"""Text encoding: Qwen3 chat-template prompt embeds (BluePixel semantics).

Extracted from vins_infer.py `_get_qwen3_prompt_embeds` / `embedding_prompt` /
`_prepare_text_ids`. Kept 1:1 with the reference implementation so tokenization,
padding, hidden-state layer selection and text position ids match exactly.
"""
from __future__ import annotations

import torch

BLUEPIXEL_HIDDEN_LAYERS = (9, 18, 27)
BLUEPIXEL_MAX_SEQUENCE_LENGTH = 512


def _prepare_text_ids(x: torch.Tensor, t_coord: torch.Tensor | None = None) -> torch.Tensor:
    """(B, L, D) -> (B, L, 4) text position ids: (t, h=0, w=0, seq)."""
    B, L, _ = x.shape
    out_ids = []
    for i in range(B):
        t = torch.arange(1) if t_coord is None else t_coord[i]
        if t.dim() == 0:
            t = t.unsqueeze(0)
        h = torch.arange(1)
        w = torch.arange(1)
        seq_dim = torch.arange(L)
        coords = torch.cartesian_prod(t, h, w, seq_dim)
        out_ids.append(coords)
    return torch.stack(out_ids)


def _get_qwen3_prompt_embeds(
    text_encoder,
    tokenizer,
    prompt: str | list[str],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    max_sequence_length: int = BLUEPIXEL_MAX_SEQUENCE_LENGTH,
    hidden_states_layers: tuple[int, ...] = BLUEPIXEL_HIDDEN_LAYERS,
) -> torch.Tensor:
    """Qwen3 hidden states from layers (9,18,27) concatenated per position.

    Output: (B, L, 3 * hidden_size) = (B, 512, 12288) for Qwen3-8B-like enc.
    """
    if device is None:
        device = next(text_encoder.parameters()).device
    dtype = text_encoder.dtype if dtype is None else dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt

    texts = []
    for single_prompt in prompt:
        messages = [{"role": "user", "content": single_prompt}]
        texts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        ))

    inputs = tokenizer(
        texts, return_tensors="pt", padding="max_length",
        truncation=True, max_length=max_sequence_length,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output = text_encoder(
        input_ids=input_ids, attention_mask=attention_mask,
        output_hidden_states=True, use_cache=False,
    )
    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
    out = out.to(dtype=dtype, device=device)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
    return prompt_embeds


def encode_prompt_bluepixel(
    text_encoder,
    tokenizer,
    prompt: str | list[str],
    device,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = BLUEPIXEL_MAX_SEQUENCE_LENGTH,
    hidden_states_layers: tuple[int, ...] = BLUEPIXEL_HIDDEN_LAYERS,
    dtype: torch.dtype | None = None,
):
    """Full text conditioning: (prompt_embeds, text_ids)."""
    if prompt is None:
        prompt = ""
    prompt = [prompt] if isinstance(prompt, str) else prompt

    prompt_embeds = _get_qwen3_prompt_embeds(
        text_encoder=text_encoder, tokenizer=tokenizer, prompt=prompt,
        device=device, max_sequence_length=max_sequence_length,
        hidden_states_layers=hidden_states_layers, dtype=dtype,
    )
    batch_size, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    text_ids = _prepare_text_ids(prompt_embeds).to(device)
    return prompt_embeds, text_ids
