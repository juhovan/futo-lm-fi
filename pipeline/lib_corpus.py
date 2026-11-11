"""Stage 1: assemble the Finnish corpus.

Streams the sources from config_fi.CORPUS_SOURCES, filters + dedupes, writes
~256 MiB text shards, builds a word-frequency map (used by the finetune typo
generator) and a holdout sentence set (used by the next-word eval).

Idempotent: an existing stats.json with >=80% of the token budget is
accepted; set FLUFI_FORCE=1 to rebuild.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import config_fi

SHARD_BYTES = 256 * 1024 * 1024
HOLDOUT_SENTENCES = 5_000
WORDFREQ_CAP = 300_000

# Heuristic Finnish-ness: ratio of characters from the Finnish alphabet.
FI_ALPHA = set(
    "abcdefghijklmnopqrstuvwxyz0123456789 \n\t.,;:!?'"
    "“”‘’-–—&%()[]{}<>/\\@#$*=+_^~"
    "äöåÄÖÅ"
)
FI_MIN_ALPHA_RATIO = 0.85

# A few high-frequency Finnish stopword fragments used as a second filter for
# sources that may be mixed-locale (catches web crawls full of other langs).
FI_STOPWORD_HINTS = [" ja ", " on ", " ei ", " se ", " että ", " kun ", " mikä ", " tämä "]


def _is_finnish(text: str) -> bool:
    n = len(text)
    if n < 40:
        return False
    ratio = sum(ch in FI_ALPHA for ch in text) / n
    if ratio < FI_MIN_ALPHA_RATIO:
        return False
    hint_hits = sum(h in text.lower() for h in FI_STOPWORD_HINTS)
    return hint_hits >= 1 or ratio > 0.97


class _Stream:
    """One HF source, streamed lazily with automatic restart on exhaustion."""

    def __init__(self, src: dict):
        self.src = src
        self.ds = None
        self.ds_iter = None

    def _ensure(self):
        if self.ds is None:
            from datasets import load_dataset

            self.ds = load_dataset(
                self.src["dataset"],
                name=self.src.get("config"),
                split=self.src["split"],
                streaming=True,
            )
            # datasets 3.x: an IterableDataset is not itself an iterator.
            self.ds_iter = iter(self.ds)
        assert self.ds is not None

    def next_doc(self) -> str | None:
        """Next accepted text, or None when this source is exhausted."""
        self._ensure()
        for _ in range(10_000):  # bounded search window per call
            try:
                row = next(self.ds_iter)
            except StopIteration:
                return None
            val = row.get(self.src["column"])
            sub = self.src.get("subcolumn")
            if sub and isinstance(val, dict):  # datasets 3.x pair-config row
                val = val.get(sub)
            text = (val or "").strip()
            if not text:
                continue
            if not (self.src["min_chars"] <= len(text) <= self.src["max_chars"]):
                continue
            if self.src.get("filter_finnish") and not _is_finnish(text):
                continue
            return text
        return None


def run(workdir: Path, p: dict) -> None:
    corpus_dir = workdir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = corpus_dir / "shards"
    shards_dir.mkdir(exist_ok=True)

    budget_tokens = p["token_budget"]
    budget_chars = int(budget_tokens * config_fi.CHARS_PER_TOKEN)

    # ---- resume support ----------------------------------------------------
    stats_path = corpus_dir / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if stats.get("est_tokens", 0) >= 0.8 * budget_tokens:
            print(f"[corpus] existing corpus has {stats['est_tokens']:,} tokens "
                  f"(>=80% of budget {budget_tokens:,}); skipping (FLUFI_FORCE=1 to rebuild)")
            return

    streams: dict[str, _Stream] = {s["name"]: _Stream(s) for s in config_fi.CORPUS_SOURCES}
    alive = set(streams)
    rng = random.Random(20261030)
    weights = {name: s.src["weight"] for name, s in streams.items()}
    names = list(streams)

    seen_hashes: set[str] = set()
    holdout: list[str] = []
    word_freq: Counter = Counter()
    token_re = re.compile(r"[a-zA-ZäöåÄÖÅ']+")

    total_chars = 0
    doc_count = 0
    per_source_docs: dict[str, int] = {}
    shard_idx = 0
    shard_out = None
    shard_size = 0
    done = False

    while not done:
        # Weighted source pick; drop dead sources.
        while True:
            pick = rng.choices(names, weights=[weights[n] for n in names], k=1)[0]
            if pick in alive:
                break
            alive.discard(pick)
            if not alive:
                done = True
                break
        if done:
            break

        s = streams[pick]
        text = s.next_doc()
        if text is None:
            alive.discard(pick)
            continue

        h = hashlib.sha1(text[:200].encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # Holdout sampling before sharding: a random spread of short complete
        # sentences for next-word evaluation.
        if (len(holdout) < HOLDOUT_SENTENCES
                and 8 <= len(text) <= 300
                and "\n" not in text
                and rng.random() < 0.15):
            holdout.append(text)

        # Word frequency (whitespace words, lowercased, alpha only).
        for tok in token_re.findall(text.lower()):
            if 2 <= len(tok) <= 24:
                word_freq[tok] += 1

        # Shard writing (256 MiB text files).
        line = text.replace("\n", " ") + "\n"
        if shard_out is None:
            shard_idx += 1
            shard_out = (shards_dir / f"shard_{shard_idx:05d}.txt").open("w", encoding="utf-8")
        while len(line) > SHARD_BYTES - shard_size and shard_size > 0:
            shard_out.write(line[:SHARD_BYTES - shard_size])
            line = line[SHARD_BYTES - shard_size:]
            shard_out.close()
            shard_idx += 1
            shard_out = (shards_dir / f"shard_{shard_idx:05d}.txt").open("w", encoding="utf-8")
            shard_size = 0
        shard_out.write(line)
        shard_size += len(line.encode("utf-8"))
        total_chars += len(text)
        doc_count += 1
        per_source_docs[pick] = per_source_docs.get(pick, 0) + 1

        if total_chars >= budget_chars:
            done = True
            print(f"[corpus] token budget reached after {doc_count:,} docs; stopping")

    if shard_out:
        shard_out.close()

    # ---- artifacts ----------------------------------------------------------
    with open(corpus_dir / "holdout_sentences.txt", "w", encoding="utf-8") as f:
        for s in holdout:
            f.write(s + "\n")

    (corpus_dir / "wordfreq.json").write_text(json.dumps(word_freq.most_common(WORDFREQ_CAP),
                                                          ensure_ascii=False))

    est_tokens = total_chars // config_fi.CHARS_PER_TOKEN
    stats = {
        "est_tokens": est_tokens,
        "total_chars": total_chars,
        "docs": doc_count,
        "per_source": per_source_docs,
        "shards": shard_idx,
        "holdout_sentences": len(holdout),
        "wordfreq_entries": min(len(word_freq), WORDFREQ_CAP),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[corpus] {stats}")
