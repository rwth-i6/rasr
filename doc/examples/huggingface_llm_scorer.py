"""
`LlmScorer` for `llm-timesync-beam-search` backed by a HuggingFace causal LM, e.g. Qwen, with a key/value cache
per history.

Parameters in the `llm` selection of the search algorithm:
  model           HuggingFace model name or local path (required)
  device          torch device, default "cuda" if available else "cpu"
  prompt          text every history starts with, default empty
  max-batch-size  maximum number of requests per forward pass, default 64
  case-variants   comma-separated casings of each word to score, out of original, lower, capitalized and upper,
                  default "original,lower,capitalized,upper"
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import librasr


@dataclass
class _State:
    keys: list  # Per layer: [heads, time, dim] for all tokens of the history but the last one
    values: list
    last_token: int


class HuggingFaceLlmScorer(librasr.LlmScorer):
    CASINGS = {
        "original": lambda word: word,
        "lower": str.lower,
        "capitalized": lambda word: word[:1].upper() + word[1:].lower(),
        "upper": str.upper,
    }

    def __init__(self, config):
        super().__init__(config)
        if config["model"] is None:
            raise ValueError("No LLM configured: set `model` in the `llm` selection of the search algorithm")
        self.device = torch.device(config["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_batch_size = int(config["max-batch-size"] or 64)
        casings = config["case-variants"] or "original,lower,capitalized,upper"
        self.casings = [self.CASINGS[name.strip()] for name in casings.split(",")]

        self.tokenizer = AutoTokenizer.from_pretrained(config["model"])
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(config["model"], dtype=dtype).to(self.device).eval()

        # Every history needs a first token. Models without BOS token, such as Qwen, use their EOS token instead.
        start = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
        self.initial = [start] + self.tokenizer(config["prompt"] or "", add_special_tokens=False).input_ids
        self.states = {}

    def reset(self):
        self.states.clear()

    def initial_tokens(self):
        return self.initial

    def sentence_end_tokens(self):
        return [self.tokenizer.eos_token_id]

    def spelling_variants(self, words):
        return [list(dict.fromkeys(casing(word) for casing in self.casings)) for word in words]

    def tokenize(self, texts):
        return self.tokenizer(list(texts), add_special_tokens=False).input_ids

    def cleanup(self, active_histories):
        active = set(active_histories)
        self.states = {history: state for history, state in self.states.items() if history in active}

    @torch.inference_mode()
    def score(self, requests):
        costs = []
        for start in range(0, len(requests), self.max_batch_size):
            costs.extend(self._score_batch(requests[start : start + self.max_batch_size]))
        return costs

    def _score_batch(self, requests):
        # Each row feeds the last token of its history (or the whole history if it has no state) and all request
        # tokens but the last one, so that its outputs predict exactly the request tokens
        states = [self.states.get(request.history) for request in requests]
        inputs = [
            ([state.last_token] if state else list(request.prefix)) + list(request.tokens[:-1])
            for state, request in zip(states, requests)
        ]
        past_lengths = [state.keys[0].shape[1] if state else 0 for state in states]
        cache, past_mask = self._batched_cache(states, past_lengths)

        input_ids, input_mask = self._pad([torch.tensor(ids) for ids in inputs])
        max_past = past_mask.shape[1]
        position_ids = torch.tensor(past_lengths).unsqueeze(1) + torch.arange(input_ids.shape[1]).unsqueeze(0)
        logits = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=torch.cat([past_mask, input_mask], dim=1).to(self.device),
            position_ids=position_ids.to(self.device),
            past_key_values=cache,
            use_cache=True,
        ).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        costs = []
        for row, (request, ids, past_length) in enumerate(zip(requests, inputs, past_lengths)):
            tokens = torch.tensor(list(request.tokens), device=self.device)
            positions = torch.arange(len(ids) - len(tokens), len(ids), device=self.device)
            costs.append((-log_probs[row, positions, tokens]).tolist())

            # The new history holds everything fed so far, followed by the last request token
            span = slice(max_past - past_length, max_past + len(ids))
            self.states[request.token_histories[-1]] = _State(
                keys=[layer.keys[row, :, span].clone() for layer in cache.layers],
                values=[layer.values[row, :, span].clone() for layer in cache.layers],
                last_token=request.tokens[-1],
            )
        return costs

    def _batched_cache(self, states, past_lengths):
        """Left-padded cache of the given states and the matching attention mask."""
        max_past = max(past_lengths)
        past_mask = torch.zeros(len(states), max_past, dtype=torch.long)
        cache = DynamicCache()
        if max_past == 0:
            return cache, past_mask
        reference = next(state for state in states if state)
        for layer in range(len(reference.keys)):
            keys, values = [], []
            for state, length in zip(states, past_lengths):
                pad = max_past - length
                key = state.keys[layer] if state else reference.keys[layer][:, :0]
                value = state.values[layer] if state else reference.values[layer][:, :0]
                keys.append(torch.nn.functional.pad(key, (0, 0, pad, 0)))
                values.append(torch.nn.functional.pad(value, (0, 0, pad, 0)))
            cache.update(torch.stack(keys), torch.stack(values), layer)
        for row, length in enumerate(past_lengths):
            past_mask[row, max_past - length :] = 1
        return cache, past_mask

    def _pad(self, sequences):
        """Right-padded batch of 1-d tensors and the matching attention mask."""
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        batch = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_id)
        mask = torch.nn.utils.rnn.pad_sequence([torch.ones_like(s) for s in sequences], batch_first=True)
        return batch, mask
