"""Stage 5: evaluation of the final finetuned model.

No FUTO-published Finnish test set exists, so this is a proxy eval:

  1. Top-1 / top-4 next-token accuracy on held-out Finnish text (the corpus
     holdout sentences + OPUS fi subtitles), measured over real sentences.
  2. Autocorrect smoke test: 100 synthetic '<XBU><typo><XBC>' prompts; the
     model's greedy continuation is checked against the intended word.
     This is the behaviour the keyboard actually relies on.

Results land in workdir/eval/results.json; the stage is idempotent (re-running
re-measures, it is cheap and side-effect free).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

import config_fi


def _topk_accuracy(model, tok, sentences: list[str], k: int = 4) -> tuple[float, float, int]:
    """Top-1 and top-k next-token accuracy over the given sentences."""
    device = next(model.parameters()).device
    model.eval()
    hit1 = hitk = total = 0
    with torch.no_grad():
        for s in sentences:
            ids = tok.encode(s + " ", add_special_tokens=False).ids
            if len(ids) < 4:
                continue
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            logits = model(input_ids=x).logits[0]  # one row per input position
            targets = torch.tensor(ids[1:], dtype=torch.long, device=device)
            topk = logits.topk(k, dim=-1).indices
            hit1 += int((logits.argmax(-1) == targets).sum())
            # Per-row membership: does the true next token land in this
            # position's own top-k?
            hitk += int((topk == targets.unsqueeze(1)).any(1).sum())
            total += len(ids) - 1
    if total == 0:
        return 0.0, 0.0, 0
    return hit1 / total, hitk / total, total


def _autocorrect_rate(model, tok, pairs: list[tuple[str, str]],
                      n: int, seed: int) -> tuple[float, int]:
    """Greedy '<XBU><typo><XBC>' -> does the top continuation contain the
    intended word?"""
    device = next(model.parameters()).device
    model.eval()
    rng = random.Random(seed)
    sample = rng.sample(pairs, min(n, len(pairs)))
    correct = 0
    for typo, word in sample:
        prompt_ids = (tok.encode(config_fi.XBU, add_special_tokens=False).ids
                      + [tok.encode(ch, add_special_tokens=False).ids[0]
                         for ch in config_fi.typed_to_keypress_chars(typo)]
                      + tok.encode(config_fi.XBC, add_special_tokens=False).ids)
        with torch.no_grad():
            out = model.generate(torch.tensor([prompt_ids], device=device),
                                 max_new_tokens=8, do_sample=False)
        cont = tok.decode(out[0].tolist()[len(prompt_ids):])
        if word in cont:
            correct += 1
    return correct / len(sample), len(sample)


def _holdout_sentences(workdir: Path, n: int, seed: int) -> list[str]:
    lines = (workdir / "corpus" / "holdout_sentences.txt").read_text(
        encoding="utf-8").splitlines()
    rng = random.Random(seed)
    rng.shuffle(lines)
    out = []
    for l in lines:
        l = l.strip(" .")
        if 8 <= len(l) <= 300:
            out.append(l)
        if len(out) >= n:
            break
    return out


def _typo_pairs(workdir: Path, n: int, seed: int) -> list[tuple[str, str]]:
    """(typo, intended_word) pairs from the most frequent Finnish words."""
    from lib_train import _typo_variants  # reuse the synthetic typo generator
    pairs = json.loads((workdir / "corpus" / "wordfreq.json").read_text())
    words = [w for w, _f in pairs if 3 <= len(w) <= 12 and w.isalpha()]
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    for w in rng.sample(words, min(len(words), n * 3)):
        for t in _typo_variants(w, rng, 1):
            out.append((t, w))
            if len(out) >= n:
                return out
    return out


def run_eval(workdir: Path, p: dict) -> None:
    from transformers import LlamaForCausalLM
    from lib_train import _load_tokenizer

    final_dir = workdir / "finetune" / "stage_c" / "final"
    if not (final_dir / "model.safetensors").exists():
        raise SystemExit("final checkpoint missing: run the 'finetune' stage first")
    spm = workdir / "tokenizer" / "spm_fi.model"
    if not spm.exists():
        raise SystemExit("tokenizer missing: run the 'tokenizer' stage first")

    tok = _load_tokenizer(workdir)
    model = LlamaForCausalLM.from_pretrained(str(final_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    n_acc = p.get("eval_acc_sentences", 2000)
    n_ac = p.get("eval_autocorrect", 100)

    t0 = time.time()
    sents = _holdout_sentences(workdir, n_acc, seed=20261031)
    top1, top4, npos = _topk_accuracy(model, tok, sents, k=4)
    pairs = _typo_pairs(workdir, n_ac, seed=20261032)
    ac_rate, ac_n = _autocorrect_rate(model, tok, pairs, n_ac, seed=20261033)
    elapsed = time.time() - t0

    out = workdir / "eval" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(final_dir),
        "top1_next_token": round(top1, 4),
        "top4_next_token": round(top4, 4),
        "next_token_positions": npos,
        "autocorrect_rate": round(ac_rate, 4),
        "autocorrect_samples": ac_n,
        "seconds": round(elapsed, 1),
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"[eval] top-1={top1:.3f} top-4={top4:.3f} "
          f"({npos:,} positions, {len(sents):,} sentences)")
    print(f"[eval] autocorrect {ac_rate:.3f} over {ac_n} synthetic typos")
    print(f"[eval] -> {out} ({elapsed:.0f}s)")
