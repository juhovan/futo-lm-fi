"""Stages 3-4: pretrain the 36M base model and run the 3-stage autocorrect
finetune, exactly as prescribed for the FUTO keyboard LM.

Pretrain: custom bf16 loop over packed corpus shards (no HF Trainer: fewer
surfaces, resumable checkpoints, same hyperparameters as the verified
reference recipe).

Finetune stages (see the futo-portuguese GUIDE.md phase 4):
  a) isolated  <XBU><CHAR_*...><XBC>word<XEC>  triples, loss masked to the
     correction span
  b) real sentences with ~25% of words replaced by typo triples, full labels
  c) casual-register sentences (subtitles) with a lower typo rate, short polish

Every stage loads from the previous one's final checkpoint and writes an HF
save_pretrained dir, so the pipeline is fully resumable.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

import config_fi

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------- tokenization --

class _EncodeResult:
    __slots__ = ("ids",)

    def __init__(self, ids: list[int]):
        self.ids = ids


class _SpmTokenizer:
    """SentencePiece model with the .encode/.decode surface the pipeline uses.

    `tokenizers.Tokenizer.from_file` only parses its own JSON; the trained
    `spm_fi.model` is a binary SPM model, so we encode directly with the SPM
    C++ encoder (sentencepiece bindings). That is the same encoder the FUTO
    app uses at inference on the SPM bytes embedded in the GGUF, so training
    packs, finetune data, eval and app inference all share one tokenization
    semantics (inverted space, atomic user symbols).
    """

    def __init__(self, model_file: Path):
        import sentencepiece
        self._sp = sentencepiece.SentencePieceProcessor()
        self._sp.Load(str(model_file))

    def encode(self, text: str, add_special_tokens: bool = False):
        if add_special_tokens:
            raise ValueError("add_special_tokens=True is not supported")
        return _EncodeResult(self._sp.EncodeAsIds(text, add_bos=False,
                                                   add_eos=False))

    def decode(self, ids: list[int]) -> str:
        return self._sp.Decode(list(ids))


# ------------------------------------------------------------------ packing --

def _build_packs(workdir: Path, seq_len: int, prefix: str, force: bool) -> list[Path]:
    """Tokenize the corpus shards into int32 packed sequences (one-off)."""

    packs_dir = workdir / "pretrain" / prefix
    packs_dir.mkdir(parents=True, exist_ok=True)
    manifest = packs_dir / "manifest.json"
    if manifest.exists() and not force:
        return sorted(packs_dir.glob("pack_*.npy"))

    spm_file = workdir / "tokenizer" / "spm_fi.model"
    if not spm_file.exists():
        raise SystemExit("tokenizer missing: run the 'tokenizer' stage first")
    tok = _SpmTokenizer(spm_file)

    shards = sorted((workdir / "corpus" / "shards").glob("shard_*.txt"))
    if not shards:
        raise SystemExit("corpus missing: run the 'corpus' stage first")

    rng = random.Random(20261031)
    order = list(range(len(shards)))
    rng.shuffle(order)

    packs: list[Path] = []
    buf: list[int] = []
    pack_idx = 0
    for i in order:
        text = shards[i].read_text(encoding="utf-8")
        buf.extend(tok.encode(text, add_special_tokens=False).ids)
        while len(buf) >= seq_len:
            out = packs_dir / f"pack_{pack_idx:06d}.npy"
            np.save(out, np.asarray(buf[:seq_len], dtype=np.int32))
            del buf[:seq_len]
            packs.append(out)
            pack_idx += 1
    if buf:
        out = packs_dir / f"pack_{pack_idx:06d}.npy"
        np.save(out, np.asarray(buf + [1] * (seq_len - len(buf)), dtype=np.int32))
        packs.append(out)

    manifest.write_text(json.dumps({"seq_len": seq_len, "packs": len(packs)}))
    print(f"[{prefix}] packed {len(packs):,} x {seq_len} = {len(packs) * seq_len:,} tokens")
    return packs


class _PackedDataset(torch.utils.data.IterableDataset):
    """Cycles the packed sequences forever (3-epoch passes to avoid pure
    repetition on the tail of each pass)."""

    def __init__(self, packs: list[Path]):
        self.packs = packs

    def __iter__(self):
        idx = 0
        while True:
            for _ in range(3):
                for p in self.packs[idx:]:
                    # packs are 1-D (seq_len,); add the batch dim here so
                    # torch.cat(micro, dim=0) yields (micro, seq_len).
                    yield torch.from_numpy(np.load(p)).long().unsqueeze(0)
                idx = 0


# ------------------------------------------------------------------ pretrain --

def run_pretrain(workdir: Path, p: dict) -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    out_dir = workdir / "pretrain" / "final"
    if out_dir.exists() and (out_dir / "model.safetensors").exists():
        print(f"[pretrain] {out_dir} exists; skipping (FLUFI_FORCE=1 to retrain)")
        return

    rp = config_fi.PRETRAIN
    steps = p["pretrain_steps"]
    seq_len = rp["seq_len"]
    force = os.environ.get("FLUFI_FORCE") == "1"
    packs = _build_packs(workdir, seq_len, "packs", force=force)

    model = LlamaForCausalLM(LlamaConfig(**config_fi.LLAMA_CONFIG))
    model.to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=rp["learning_rate"],
                              weight_decay=rp["weight_decay"], betas=(0.9, 0.95))

    def lr_lambda(s: int) -> float:
        if s < rp["warmup_steps"]:
            return max(0.01, s / max(1, rp["warmup_steps"]))
        done = (s - rp["warmup_steps"]) / max(1, steps - rp["warmup_steps"])
        return max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, done))))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    start_step = 0
    ckpt = workdir / "pretrain" / "ckpt"
    if (ckpt / "state.pt").exists() and not force:
        state = torch.load(ckpt / "state.pt", map_location=DEVICE)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optimizer"])
        start_step = state["step"]
        print(f"[pretrain] resuming from step {start_step}")

    it = iter(_PackedDataset(packs))
    grad_acc = rp["grad_accum"]
    micro = rp["micro_batch"]
    t0 = time.time()

    def save(step: int, final: bool) -> None:
        d = out_dir if final else ckpt
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optim.state_dict(), "step": step},
                   d / "state.pt")
        if final:
            model.save_pretrained(str(d))

    model.train()
    loss_sum = 0.0
    for step in range(start_step, steps):
        batch_x = torch.cat([next(it) for _ in range(micro)], dim=0).to(DEVICE)
        inp = batch_x[:, :-1].contiguous()
        tgt = batch_x[:, 1:].contiguous()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            loss = model(input_ids=inp, labels=tgt).loss
            (loss / grad_acc).backward()
        loss_sum += loss.item()
        if (step + 1) % grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            if (step + 1) % 50 == 0:
                elapsed = time.time() - t0
                per_step = elapsed / (step + 1 - start_step)
                eta = per_step * (steps - step - 1)
                print(f"[pretrain] step {step + 1}/{steps} loss={loss_sum / grad_acc:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {per_step:.2f}s/step "
                      f"eta={eta / 60:.1f}m", flush=True)
                loss_sum = 0.0
            if (step + 1) % rp["save_steps"] == 0:
                save(step + 1, final=False)
                print(f"[pretrain] checkpoint -> {ckpt}", flush=True)

    save(steps, final=True)
    print(f"[pretrain] done; final HF checkpoint -> {out_dir}")


# ------------------------------------------------------------- finetune data --

def _typo_variants(word: str, rng: random.Random, n: int) -> list[str]:
    """Synthetic Finnish typos: layout-adjacent substitution, transposition,
    doubling, omission, and diacritic drop (the dominant Finnish class)."""
    variants: list[str] = []
    w = word

    def _sub_key(ch: str) -> str:
        adj = config_fi.KEY_ADJACENCY.get(ch.upper(), "")
        return rng.choice(adj).lower() if adj else ch

    if len(w) >= 3:
        i = rng.randrange(len(w))
        variants.append(w[:i] + _sub_key(w[i]) + w[i + 1:])
        i = rng.randrange(len(w) - 1)
        variants.append(w[:i] + w[i + 1] + w[i] + w[i + 2:])
    if len(w) >= 2:
        i = rng.randrange(len(w))
        variants.append(w[:i] + w[i] + w[i] + w[i + 1:])
        i = rng.randrange(len(w))
        variants.append(w[:i] + w[i + 1:])
    bare = "".join(config_fi.DIACRITIC_BASE.get(ch, ch) for ch in w.lower())
    if bare != w.lower():
        variants.append(bare)
    rng.shuffle(variants)
    return [v for v in dict.fromkeys(variants) if v and v != w][:n]


def _load_tokenizer(workdir: Path):
    spm_file = workdir / "tokenizer" / "spm_fi.model"
    if not spm_file.exists():
        raise SystemExit("tokenizer missing: run the 'tokenizer' stage first")
    return _SpmTokenizer(spm_file)


def _build_stage_a(workdir: Path, n: int, variants_per_word: int, seq_len: int) -> Path:
    """Isolated correction triples packed into seq_len tiles with a parallel
    correction-span mask (loss is computed only over the correction span)."""
    out = workdir / "finetune" / "data_a.npz"
    if out.exists():
        return out

    tok = _load_tokenizer(workdir)
    pairs = json.loads((workdir / "corpus" / "wordfreq.json").read_text())
    words: list[str] = []
    freqs: list[int] = []
    for w, f in pairs:
        if 2 <= len(w) <= 16 and w.isalpha():
            words.append(w)
            freqs.append(f)
    if not words:
        raise SystemExit("wordfreq.json has no usable words")

    weights = np.log1p(np.asarray(freqs, dtype=np.float64))
    weights /= weights.sum()
    np_rng = np.random.default_rng(20261032)
    rng = random.Random(20261032)

    xbu = tok.encode(config_fi.XBU, add_special_tokens=False).ids
    xbc = tok.encode(config_fi.XBC, add_special_tokens=False).ids
    xec = tok.encode(config_fi.XEC, add_special_tokens=False).ids

    flat_ids: list[int] = []
    flat_mask: list[bool] = []
    seen: set[str] = set()
    made = 0
    while made < n:
        w = words[np_rng.choice(len(words), p=weights)]
        if w in seen:
            # Word pool exhausted: no new words left to source fresh triples.
            if len(seen) >= len(words):
                print(f"[finetune] stage-a: word pool exhausted at {made:,} "
                      f"triples (requested {n:,}); stopping")
                break
            continue
        seen.add(w)
        for typo in _typo_variants(w, rng, variants_per_word):
            char_ids: list[int] = []
            for ch in config_fi.typed_to_keypress_chars(typo):
                char_ids += tok.encode(ch, add_special_tokens=False).ids
            ids = xbu + char_ids + xbc + tok.encode(w + " ", add_special_tokens=False).ids + xec
            span_start = len(xbu + char_ids + xbc)
            for j, tid in enumerate(ids):
                flat_ids.append(tid)
                flat_mask.append(j >= span_start)
            made += 1
            if made >= n:
                break

    # Tile into seq_len chunks, padded with BOS.
    npacks = (len(flat_ids) + seq_len - 1) // seq_len
    ids = np.zeros((npacks, seq_len), dtype=np.int16)
    mask = np.zeros((npacks, seq_len), dtype=bool)
    for i in range(len(flat_ids)):
        pk, off = divmod(i, seq_len)
        ids[pk, off] = flat_ids[i]
        mask[pk, off] = flat_mask[i]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=ids, mask=mask)
    print(f"[finetune] stage-a data: {made:,} triples -> {npacks:,} x {seq_len} tiles")
    return out


def _stage_b_c_sentences(workdir: Path, sentences_n: int, casual: bool) -> list[str]:
    """Real sentences: corpus holdout/shards, or OPUS fi subtitles (casual)."""
    if not casual:
        rng = random.Random(20261033)
        lines = (workdir / "corpus" / "holdout_sentences.txt").read_text(encoding="utf-8").splitlines()
        if len(lines) < sentences_n:
            extra = []
            for shard in sorted((workdir / "corpus" / "shards").glob("shard_*.txt"))[:5]:
                for line in shard.open(encoding="utf-8"):
                    extra.extend(line.split(". "))
            lines += [l for l in extra if 8 <= len(l) <= 300]
        rng.shuffle(lines)
        return [l.strip(" .") for l in lines if 8 <= len(l) <= 300][:sentences_n]

    from datasets import load_dataset

    src = config_fi.CORPUS_SOURCES[1]
    ds = load_dataset(src["dataset"], name=src.get("config"), split=src["split"], streaming=True)
    out = []
    for row in ds:
        t = row.get("translation")
        # datasets 3.x pair-config rows: translation={en, fi}; pick the configured side.
        if isinstance(t, dict):
            t = t.get(src.get("subcolumn", "fi"))
        t = (t or "").strip()
        if 4 <= len(t) <= 200:
            out.append(t)
            if len(out) >= sentences_n:
                break
    return out


def _build_stage_b(workdir: Path, spec: dict, casual: bool, stage_name: str) -> Path:
    """Pack sentences with word-level typo substitution into seq_len int16
    packs (full labels; no masking)."""
    out = workdir / "finetune" / f"data_{stage_name}.npz"
    if out.exists():
        return out
    tok = _load_tokenizer(workdir)
    sents = _stage_b_c_sentences(workdir, spec["sentences"], casual)
    rng = random.Random(20261034 if casual else 20261035)

    xbu = tok.encode(config_fi.XBU, add_special_tokens=False).ids
    xbc = tok.encode(config_fi.XBC, add_special_tokens=False).ids
    xec = tok.encode(config_fi.XEC, add_special_tokens=False).ids

    buf: list[int] = []
    packs: list[np.ndarray] = []

    def flush() -> None:
        nonlocal buf
        while len(buf) >= spec["seq_len"]:
            packs.append(np.asarray(buf[:spec["seq_len"]], dtype=np.int16))
            buf = buf[spec["seq_len"]:]

    for sent in sents:
        words = [w for w in sent.split() if w.strip(".,;:!?")]
        if not words:
            continue
        ids: list[int] = []
        for w in words:
            base = w.strip(".,;:!?'\"")
            if rng.random() < spec["typo_rate"] and len(base) >= 2:
                typo = _typo_variants(base, rng, 1)[0] or base
                char_ids: list[int] = []
                for ch in config_fi.typed_to_keypress_chars(typo):
                    char_ids += tok.encode(ch, add_special_tokens=False).ids
                ids += xbu + char_ids + xbc + tok.encode(base + " ", add_special_tokens=False).ids + xec
                continue
            ids += tok.encode(w + " ", add_special_tokens=False).ids
        buf.extend(ids)
        flush()

    arr = (np.stack(packs) if packs
           else np.zeros((1, spec["seq_len"]), dtype=np.int16))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, packs=arr)
    print(f"[finetune] stage-{stage_name} data: {len(packs):,} x {spec['seq_len']} packs "
          f"({len(sents):,} sentences)")
    return out


# ------------------------------------------------------------- finetune loop --

def _batches(packs: np.ndarray, batch: int):
    """Yield (batch, seq_len) int64 tiles from a (N, seq_len) array, cycling
    forever (like _PackedDataset) so long finetune schedules never exhaust."""
    n = len(packs)
    while True:
        for i in range(0, n, batch):
            idx = np.array([j % n for j in range(i, i + batch)])
            yield torch.from_numpy(packs[idx].astype(np.int32)).long()


def _train_stage(model, loader, steps: int, learning_rate: float,
                 stage: str, out_dir: Path) -> None:
    """One finetune stage. `loader` yields (x, mask_or_None) tuples where
    x is (batch, seq_len) int64. When mask is given, the loss is restricted
    to the correction span of each triple (stage a)."""
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    it = iter(loader)
    t0 = time.time()
    for step in range(steps):
        x, m = next(it)
        inp = x[:, :-1].contiguous().to(DEVICE)
        tgt = x[:, 1:].contiguous().to(DEVICE)
        logits = model(input_ids=inp).logits.float().view(-1, model.config.vocab_size)
        if m is not None:
            mm = m.to(DEVICE)
            # label k targets input position k+1 (tgt[k] = x[k+1]); active where
            # that position is a correction-span token (word pieces + <XEC>).
            sel = mm[:, 1:].bool()
            labels = torch.where(sel, tgt, -100)
            loss = torch.nn.functional.cross_entropy(logits, labels.reshape(-1),
                                                      ignore_index=-100)
        else:
            loss = torch.nn.functional.cross_entropy(logits, tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        if (step + 1) % 50 == 0:
            print(f"[finetune:{stage}] step {step + 1}/{steps} loss={loss.item():.4f} "
                  f"{(time.time() - t0) / (step + 1):.2f}s/step", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    print(f"[finetune:{stage}] final -> {out_dir}")


def _load_from(d: Path):
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(str(d), torch_dtype=torch.float32)
    model.to(DEVICE)
    return model


def run_finetune(workdir: Path, p: dict) -> None:
    fs = config_fi.FINETUNE
    final_dir = workdir / "finetune" / "stage_c" / "final"
    if final_dir.exists() and (final_dir / "model.safetensors").exists():
        print(f"[finetune] {final_dir} exists; skipping (FLUFI_FORCE=1 to retrain)")
        return

    # ---- stage a: isolated triples, masked loss ------------------------------
    a_dir = workdir / "finetune" / "stage_a" / "final"
    if a_dir.exists() and (a_dir / "model.safetensors").exists():
        model = _load_from(a_dir)
    else:
        model = _load_from(workdir / "pretrain" / "final")
        fa = fs["a"]
        data = np.load(_build_stage_a(workdir, p["ft_a_steps"] * fa["batch"],
                                      fa["typo_variants_per_word"], fa["seq_len"]))
        ids, mask = data["ids"], data["mask"]
        batch = fa["batch"]

        def loader_a():
            n = len(ids)
            while True:
                for i in range(0, n, batch):
                    idx = np.array([j % n for j in range(i, i + batch)])
                    yield torch.from_numpy(ids[idx].astype(np.int32)).long(), \
                        torch.from_numpy(mask[idx])
        _train_stage(model, loader_a(), p["ft_a_steps"], fa["learning_rate"], "a", a_dir)

    # ---- stage b: in-context full text ----------------------------------------
    b_dir = workdir / "finetune" / "stage_b" / "final"
    if b_dir.exists() and (b_dir / "model.safetensors").exists():
        model = _load_from(b_dir)
    else:
        model = _load_from(a_dir)
        fb = fs["b"]
        packs = np.load(_build_stage_b(workdir, fb, casual=False, stage_name="b"))["packs"]

        def loader_b():
            for x in _batches(packs, fb["batch"]):
                yield x, None
        _train_stage(model, loader_b(), p["ft_b_steps"], fb["learning_rate"], "b", b_dir)

    # ---- stage c: casual-register polish ---------------------------------------
    fc = fs["c"]
    model = _load_from(b_dir)
    packs = np.load(_build_stage_b(workdir, fc, casual=True, stage_name="c"))["packs"]

    def loader_c():
        for x in _batches(packs, fc["batch"]):
            yield x, None
    _train_stage(model, loader_c(), p["ft_c_steps"], fc["learning_rate"], "c", final_dir)
