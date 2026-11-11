"""FUTO Keyboard Finnish LM pipeline.

One-off training job; stages are idempotent and resumable, so the whole thing
can be re-run later (or resumed mid-way) via:

    docker compose run --rm futo-lm-fi <stage>

Stages:
    corpus     download + filter + dedupe + shard the Finnish corpus, build
               the word-frequency map and a next-word holdout set
    tokenizer  train the 15008-vocab inverted-space SentencePiece BPE with the
               300 user-defined symbols pinned to the reference slot layout
    pretrain   train the 36M Llama base model on the packed corpus
    finetune   3-stage autocorrect finetune (isolated triples, in-context,
               casual-register polish)
    eval       next-word top-1/top-4 + Finnish autocorrect suite, on the
               pretrain checkpoint ("floor") and the final model
    export     HF -> GGUF v2 with the FUTO keyboardlm.* metadata, Q6_K output
               weight, reference diff and a llama-cli smoke test
    all        run every stage in order
    idle       container no-op (the service default command)

Environment:
    FLUFI_MODE         mini | full            (default: mini)
    FLUFI_TOKEN_BUDGET int, overrides mode budget
    FLUFI_FORCE        1 re-runs stages even when their markers exist
    FLUFI_WORK         work dir               (default: /work)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config_fi  # noqa: E402


def work() -> Path:
    p = Path(os.environ.get("FLUFI_WORK", "/work"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def log(msg: str) -> None:
    print(f"[futo-lm-fi] {msg}", flush=True)


def marker(stage: str, workdir: Path) -> Path:
    return workdir / stage / "DONE"


def done(stage: str, workdir: Path, force: bool) -> bool:
    if force:
        return False
    m = marker(stage, workdir)
    return m.exists()


def mark_done(stage: str, workdir: Path, extra: dict | None = None) -> None:
    m = marker(stage, workdir)
    m.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if extra:
        payload.update(extra)
    m.write_text(json.dumps(payload, indent=2))
    log(f"stage '{stage}' marked done -> {m}")


def mode_params() -> dict:
    mode = os.environ.get("FLUFI_MODE", "mini")
    if mode not in config_fi.MODES:
        raise SystemExit(f"FLUFI_MODE={mode!r} unknown; pick one of {list(config_fi.MODES)}")
    p = dict(config_fi.MODES[mode])
    p["mode"] = mode
    budget = os.environ.get("FLUFI_TOKEN_BUDGET", "").strip()
    if budget:
        p["token_budget"] = int(budget)
        log(f"FLUFI_TOKEN_BUDGET overrides the {mode} budget to {p['token_budget']:,}")
    return p


STAGES = ["corpus", "tokenizer", "pretrain", "finetune", "eval", "export"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=STAGES + ["all", "idle"])
    args = ap.parse_args()

    if args.stage == "idle":
        log("idle (container no-op; run a stage with: docker compose run --rm futo-lm-fi <stage>)")
        while True:
            time.sleep(3600)

    workdir = work()
    force = os.environ.get("FLUFI_FORCE", "") == "1"
    p = mode_params()
    log(f"stage={args.stage} mode={p['mode']} token_budget={p['token_budget']:,} force={force}")

    import lib_corpus
    import lib_tokenizer
    import lib_train
    import lib_eval
    import lib_export

    runners = {
        "corpus": lambda: lib_corpus.run(workdir, p),
        "tokenizer": lambda: lib_tokenizer.run(workdir, p),
        "pretrain": lambda: lib_train.run_pretrain(workdir, p),
        "finetune": lambda: lib_train.run_finetune(workdir, p),
        "eval": lambda: lib_eval.run_eval(workdir, p),
        "export": lambda: lib_export.run_export(workdir, p),
    }

    stages = STAGES if args.stage == "all" else [args.stage]
    for stage in stages:
        if done(stage, workdir, force):
            log(f"stage '{stage}' already done ({marker(stage, workdir)}); set FLUFI_FORCE=1 to re-run")
            continue
        t0 = time.time()
        log(f"=== stage: {stage} ===")
        runners[stage]()
        mark_done(stage, workdir)
        log(f"=== stage: {stage} finished in {time.time() - t0:.0f}s ===")


if __name__ == "__main__":
    main()
