"""Stage 2: train the Finnish SentencePiece BPE tokenizer.

- Pulls the reference English FUTO model from HF, extracts its embedded
  SentencePiece, and keeps the filler/structural/CHAR/emoji slots byte-for-byte
  in layout (see config_fi.build_user_symbols).
- Trains a 15008-vocab BPE with treat_whitespace_as_suffix=True ("inverted
  space", required by the FUTO loader) on a RAM-bounded sample of the corpus.
- Validates the hard constraints from the reference layout and the FUTO C++
  source (26 contiguous <CHAR_*> ids, all structural tokens resolvable).

Idempotent: skips when spm_fi.model + slot_map.json exist.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import config_fi


def _download_reference_spm(workdir: Path) -> Path:
    ref_dir = workdir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    out = ref_dir / "spm_ref.model"
    if out.exists():
        return out

    from huggingface_hub import hf_hub_download

    gguf_path = hf_hub_download(
        config_fi.REFERENCE_HF_REPO,
        config_fi.REFERENCE_GGUF_FILENAME,
        local_dir=str(ref_dir),
    )
    print(f"[tokenizer] downloaded reference {gguf_path}")

    # Extract the embedded SentencePiece bytes (keyboardlm.ext_tokenizer_data).
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    field = reader.fields["keyboardlm.ext_tokenizer_data"]
    spm_bytes = b"".join(bytes(field.parts[i]) for i in field.data)
    out.write_bytes(spm_bytes)
    print(f"[tokenizer] extracted reference SentencePiece ({len(spm_bytes):,} bytes)")
    return out


def _sample_shards(workdir: Path, sample_bytes: int) -> Path:
    sample_path = workdir / "tokenizer" / "spm_sample.txt"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if sample_path.exists() and sample_path.stat().st_size >= sample_bytes // 2:
        return sample_path

    shards = sorted((workdir / "corpus" / "shards").glob("shard_*.txt"))
    if not shards:
        raise SystemExit("corpus stage missing: run the 'corpus' stage first")
    rng = random.Random(7)
    chosen = [shards[0]] + [s for s in shards[1:] if rng.random() < 0.25]
    rng.shuffle(chosen)

    taken = 0
    with open(sample_path, "w", encoding="utf-8") as f:
        for shard in chosen:
            if taken >= sample_bytes:
                break
            # Random seek into the shard, copy ~16 MiB chunks.
            size = shard.stat().st_size
            for _ in range(64):
                if taken >= sample_bytes:
                    break
                off = rng.randrange(0, max(1, size - 16 * 1024 * 1024))
                with open(shard, "rb") as s:
                    s.seek(off)
                    chunk = s.read(16 * 1024 * 1024)
                f.write(chunk.decode("utf-8", errors="ignore"))
                taken += len(chunk)
    print(f"[tokenizer] sampled {taken:,} bytes of SPM training data -> {sample_path}")
    return sample_path


def run(workdir: Path, p: dict) -> None:
    import sentencepiece as spm

    tok_dir = workdir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    out_model = tok_dir / "spm_fi.model"
    out_json = tok_dir / "slot_map.json"

    if out_model.exists() and out_json.exists():
        print(f"[tokenizer] {out_model} exists; skipping (FLUFI_FORCE=1 to retrain)")
        return

    ref_spm = _download_reference_spm(workdir)
    ref_proc = spm.SentencePieceProcessor()
    ref_proc.Load(str(ref_spm))
    ref_pieces = [ref_proc.IdToPiece(i) for i in range(config_fi.VOCAB_SIZE)]
    user_symbols = config_fi.build_user_symbols(ref_pieces)
    print(f"[tokenizer] {len(user_symbols)} user symbols "
          f"(24 filler + 146 fi + 8 structural + 26 CHAR + 56 fi + 40 emoji)")

    sample_bytes = int(config_fi.SPM_SAMPLE_GB * 1024 ** 3)
    sample = _sample_shards(workdir, sample_bytes)

    spm.SentencePieceTrainer.Train(
        input=str(sample),
        input_format="text",
        model_prefix=str(tok_dir / "spm_fi"),
        vocab_size=config_fi.VOCAB_SIZE,
        model_type="bpe",
        treat_whitespace_as_suffix=True,   # the "inverted_space" feature
        character_coverage=0.9995,
        user_defined_symbols=user_symbols,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        byte_fallback=True,
        input_sentence_size=1_000_000,
        shuffle_input_sentence=True,
    )

    # ---- hard validations (mirror the FUTO C++ loader asserts) -------------
    proc = spm.SentencePieceProcessor()
    proc.Load(str(out_model))
    size = proc.GetPieceSize()
    assert size == config_fi.VOCAB_SIZE, f"piece size {size} != {config_fi.VOCAB_SIZE}"

    char_ids = [proc.PieceToId(config_fi.CHAR_TOKENS[i]) for i in range(26)]
    assert char_ids == list(range(char_ids[0], char_ids[0] + 26)), \
        f"<CHAR_A>..<CHAR_Z> not contiguous/sequential: {char_ids}"
    for tok in [config_fi.XBU, config_fi.XBC, config_fi.XEC, "<XC0>"]:
        tid = proc.PieceToId(tok)
        assert tid > 3, f"structural token {tok} missing (id {tid})"

    slot_map = {
        "vocab_size": size,
        "char_a_id": char_ids[0],
        "char_z_id": char_ids[25],
        "xbu": proc.PieceToId(config_fi.XBU),
        "xbc": proc.PieceToId(config_fi.XBC),
        "xec": proc.PieceToId(config_fi.XEC),
        "user_symbols": user_symbols,
    }
    out_json.write_text(json.dumps(slot_map, indent=2))
    print(f"[tokenizer] validations passed; slot map: {slot_map | {'user_symbols': '200+ items elided'}}")

    # Also keep the model under its Llama-convention name for the export stage.
    (tok_dir / "spm.model").write_bytes(out_model.read_bytes())
    print(f"[tokenizer] wrote {out_model} + {out_json} + spm.model")
