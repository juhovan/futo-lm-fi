# futo-lm-fi

Finnish language-model pipeline for the [FUTO Keyboard](https://github.com/futo-org/android-keyboard):
corpus build -> SentencePiece tokenizer -> 36M-param Llama pretrain -> correction
finetuning (3 stages) -> eval -> GGUF export, importable by the keyboard's
Languages & Models screen.

The recipe ports the verified
[futo-portuguese](https://github.com/danmaxis/futo-portuguese) pipeline to
Finnish (data, word lists, key adjacency, diacritics).

## Layout

- `pipeline/` — the training pipeline: `python3 pipeline/pipeline.py <stage>`
  where stage is `corpus`, `tokenizer`, `pretrain`, `finetune`, `eval`,
  `export` or `all`; `config_fi.py` holds the model and data config.
- `Dockerfile` — one-off training job container: pinned PyTorch training stack
  plus CPU llama.cpp for the GGUF conversion and the final `llama-cli` smoke
  test. The pipeline is baked in at `/pipeline`; all artifacts go to `/work`.
- `.github/workflows/build.yml` — builds the image on every push and publishes
  it to `ghcr.io/juhovan/futo-lm-fi` as `sha-<commit>` (plus `latest`).

## Running stages

The container is idle by default. Run a stage inside it:

```
docker compose run --rm futo-lm-fi corpus      # download + filter Finnish text
docker compose run --rm futo-lm-fi tokenizer   # train the SPM BPE
docker compose run --rm futo-lm-fi pretrain    # hours on one GPU
docker compose run --rm futo-lm-fi finetune    # 3-stage autocorrect finetune
docker compose run --rm futo-lm-fi eval        # next-word top-4 + typo smoke
docker compose run --rm futo-lm-fi export      # GGUF + metadata + smoke test
docker compose run --rm futo-lm-fi all        # everything in order
```

Every stage writes a DONE marker in the work volume, so a resumed `all`
skips finished stages; `FLUFI_FORCE=1` re-runs a stage. The finished model
lands at `gguf/fi_futo_v3.gguf` under the work volume.

## Data sources

Public, no credentials: `Finnish-NLP/oscar_2301_fi_cleaned` (primary corpus),
the Finnish side of `Helsinki-NLP/opus-100`, and
`wikimedia/wikipedia` 20231101.fi. A 5k-sentence holdout is reserved before
training for the eval stage.
