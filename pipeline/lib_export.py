"""Stage 6: HF checkpoint -> FUTO-importable GGUF.

Mirrors the verified futo-portuguese recipe (pt_05/pt_06/06b), inlined:

  1. Stage the HF checkpoint + spm_fi.model into a Llama-convention dir
     (tokenizer.model + tokenizer_config.json + special_tokens_map.json).
  2. llama.cpp convert_hf_to_gguf.py (CPU) -> vanilla Q8_0 GGUF (~37MB).
  3. Patch the keyboardlm.* fields the FUTO app validates:
       keyboardlm.languages            'fi'
       keyboardlm.finetuning_count     0
       keyboardlm.history              '<date>: Model created (fi v1)'
       keyboardlm.features             'base_v1 inverted_space xbu_char_autocorrect_v1'
       keyboardlm.ext_tokenizer_type   'sentencepiece'
       keyboardlm.ext_tokenizer_data   raw spm bytes (UINT8 array)
  4. Downgrade to GGUF v2 and strip fields the FUTO-bundled llama.cpp does
     not understand (it only parses GGUF v2).

A llama-cli smoke test proves the shipped file actually loads and generates.
Output: workdir/gguf/fi_futo.gguf (+ .vanilla.gguf kept for debugging).
"""
from __future__ import annotations

import datetime
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import config_fi
import gguf as g

LLAMA_CPP = Path("/opt/llama.cpp")

SPECIAL_TOKENS_MAP = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
}
TOKENIZER_CONFIG = {
    "tokenizer_class": "LlamaTokenizer",
    "model_max_length": 2048,
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "add_bos_token": True,
    "add_eos_token": False,
    "clean_up_tokenization_spaces": False,
    "legacy": False,
}

# FUTO keyboard features. char_embed_mixing_v1 gates the app's typed-token
# mixing: when advertised, the app copies token_embd.weight out of the GGUF
# into a float table (dequantizing q8_0 -> f32) and uses it in
# DecodePromptAndMixes. The tensor is part of every LLaMA GGUF, so
# advertising it costs nothing; without it the app's unguarded mix loop
# dereferences an empty table and crashes.
FUTO_FEATURES = "base_v1 inverted_space xbu_char_autocorrect_v1 char_embed_mixing_v1"

# Fields the FUTO-bundled (older) llama.cpp does NOT understand. Strip them
# in the v2 downgrade step.
STRIP_V2 = {
    "general.size_label",
    "general.type",
    "llama.attention.key_length",
    "llama.attention.value_length",
    "llama.vocab_size",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.pre",
}

WRITER_MANAGED = {
    "GGUF.version",
    "GGUF.tensor_count",
    "GGUF.kv_count",
    "general.architecture",
}


# ------------------------------------------------------------- field utils --

def _value(field) -> object:
    """Decode a GGUFReader field to a Python value."""
    t = field.types[0]
    if t == g.GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8")
    if len(field.types) == 1:
        p = field.parts[field.data[0]]
        return p.tolist()[0] if hasattr(p, "tolist") else p
    inner = field.types[1]
    if inner == g.GGUFValueType.STRING:
        return [bytes(field.parts[i]).decode("utf-8") for i in field.data]
    if inner == g.GGUFValueType.UINT8:
        return bytes(int(field.parts[i].tolist()[0] if hasattr(field.parts[i], "tolist")
                         else field.parts[i]) for i in field.data)
    return [field.parts[i].tolist()[0] if hasattr(field.parts[i], "tolist") else field.parts[i]
            for i in field.data]


def _copy_field(writer, name: str, field) -> None:
    """Copy a reader field to a writer, preserving its type/sub-type."""
    t = field.types[0]
    val = _value(field)
    if t == g.GGUFValueType.STRING:
        writer.add_string(name, val)
    elif t == g.GGUFValueType.UINT32:
        writer.add_uint32(name, val)
    elif t == g.GGUFValueType.INT32:
        writer.add_int32(name, val)
    elif t == g.GGUFValueType.UINT64:
        writer.add_uint64(name, val)
    elif t == g.GGUFValueType.INT64:
        writer.add_int64(name, val)
    elif t == g.GGUFValueType.FLOAT32:
        writer.add_float32(name, val)
    elif t == g.GGUFValueType.FLOAT64:
        writer.add_float64(name, val)
    elif t == g.GGUFValueType.BOOL:
        writer.add_bool(name, val)
    elif t in (g.GGUFValueType.UINT8, g.GGUFValueType.INT8,
               g.GGUFValueType.UINT16, g.GGUFValueType.INT16):
        if len(field.types) == 1:
            getattr(writer, f"add_{t.name.lower()}")(name, val)
        else:
            writer.add_array(name, list(val))
    elif t == g.GGUFValueType.ARRAY:
        inner = field.types[1]
        if inner == g.GGUFValueType.UINT8:
            # Force the sub-type; add_array would infer INT32 from ints.
            writer.add_key_value(name, val, g.GGUFValueType.ARRAY,
                                 sub_type=g.GGUFValueType.UINT8)
        else:
            writer.add_array(name, val)
    else:
        raise ValueError(f"Unhandled GGUF type {t} for field {name}")


def _patch_metadata(in_path: Path, out_path: Path, spm: Path,
                    languages: str, name: str, history: str) -> None:
    """Copy all fields + tensors, set the keyboardlm.* FUTO fields."""
    reader = g.GGUFReader(str(in_path))
    arch = _value(reader.fields["general.architecture"])
    writer = g.GGUFWriter(str(out_path), arch)
    for fname, field in reader.fields.items():
        if fname in WRITER_MANAGED:
            continue
        _copy_field(writer, fname, field)
    writer.add_string("general.name", name)
    writer.add_string("keyboardlm.languages", languages)
    writer.add_uint32("keyboardlm.finetuning_count", 0)
    writer.add_string("keyboardlm.history", history)
    writer.add_string("keyboardlm.features", FUTO_FEATURES)
    writer.add_string("keyboardlm.ext_tokenizer_type", "sentencepiece")
    writer.add_key_value("keyboardlm.ext_tokenizer_data", spm.read_bytes(),
                         g.GGUFValueType.ARRAY, sub_type=g.GGUFValueType.UINT8)
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _downgrade_v2(in_path: Path, out_path: Path) -> None:
    """Re-emit as GGUF v2, stripping fields the app's llama.cpp can't parse,
    then patch the version byte 3 -> 2."""
    reader = g.GGUFReader(str(in_path))
    arch = _value(reader.fields["general.architecture"])
    writer = g.GGUFWriter(str(out_path), arch)
    for fname, field in reader.fields.items():
        if fname in WRITER_MANAGED or fname in STRIP_V2:
            continue
        _copy_field(writer, fname, field)
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    with open(out_path, "r+b") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file (magic={magic!r})")
        f.seek(4)
        ver = struct.unpack("<I", f.read(4))[0]
        if ver == 3:
            f.seek(4)
            f.write(struct.pack("<I", 2))
            print(f"  GGUF version 3 -> 2")
        else:
            print(f"  GGUF version is v{ver}; leaving as-is")


# ------------------------------------------------------------------ stages --

def _stage_checkpoint(checkpoint: Path, spm: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in checkpoint.iterdir():
        if f.name in {"tokenizer.model", "tokenizer.json",
                     "tokenizer_config.json", "special_tokens_map.json"}:
            continue
        if f.is_file():
            shutil.copy2(f, dest / f.name)
            copied += 1
    shutil.copy2(spm, dest / "tokenizer.model")
    (dest / "tokenizer_config.json").write_text(json.dumps(TOKENIZER_CONFIG, indent=2))
    (dest / "special_tokens_map.json").write_text(json.dumps(SPECIAL_TOKENS_MAP, indent=2))
    print(f"[export] staged {copied} checkpoint files + {spm.name} -> {dest}")


def _convert_hf_to_gguf(staged: Path, out_vanilla: Path) -> None:
    conv = LLAMA_CPP / "convert_hf_to_gguf.py"
    if not conv.exists():
        raise SystemExit(f"{conv} not found; llama.cpp missing from the image")
    cmd = [sys.executable, str(conv), str(staged), "--outfile", str(out_vanilla),
           "--outtype", "q8_0"]
    print(f"[export] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[export] vanilla GGUF: {out_vanilla.stat().st_size:,} bytes")


def _smoke_test(gguf_path: Path) -> None:
    """Prove the shipped file loads and generates in the bundled llama-cli."""
    cli = LLAMA_CPP / "build" / "bin" / "llama-cli"
    if not cli.exists():
        print("[export] llama-cli not found; skipping smoke test")
        return
    for prompt in config_fi.SMOKE_PROMPTS:
        cmd = [str(cli), "-m", str(gguf_path), "-p", prompt, "-n", "6", "--temp", "0"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print(f"[export] smoke {prompt!r}: rc={r.returncode}")
        print((r.stdout or r.stderr).strip()[-1200:])
        if r.returncode != 0:
            raise SystemExit(f"llama-cli smoke test failed for {prompt!r} (rc={r.returncode})")


def run_export(workdir: Path, p: dict) -> None:
    import os
    final_dir = workdir / "finetune" / "stage_c" / "final"
    if not (final_dir / "model.safetensors").exists():
        raise SystemExit("final checkpoint missing: run the 'finetune' stage first")
    spm = workdir / "tokenizer" / "spm_fi.model"
    if not spm.exists():
        raise SystemExit("tokenizer missing: run the 'tokenizer' stage first")

    out = workdir / "gguf" / "fi_futo.gguf"
    if out.exists() and not os.environ.get("FLUFI_FORCE"):
        print(f"[export] {out} exists; skipping (FLUFI_FORCE=1 to rebuild)")
        return
    vanilla = out.parent / "fi_futo.vanilla.gguf"

    _stage_checkpoint(final_dir, spm, workdir / "gguf" / "_staged")
    _convert_hf_to_gguf(workdir / "gguf" / "_staged", vanilla)
    history = f"{datetime.date.today().isoformat()}: Model created (fi v1)"
    _patch_metadata(vanilla, out, spm, languages="fi",
                    name="Finnish v1", history=history)
    v2 = out.parent / "fi_futo_v3.gguf"
    _downgrade_v2(out, v2)
    print(f"[export] final: {v2} ({v2.stat().st_size:,} bytes)")
    _smoke_test(v2)
    print(f"DONE: {v2} — import via FUTO Keyboard's Languages & Models.")
