# futo-lm-fi

Finnish language-model pipeline for the [FUTO Keyboard](https://github.com/futo-org/android-keyboard):
corpus build -> SentencePiece tokenizer -> 36M-param Llama pretrain -> correction
finetuning (3 stages) -> eval -> GGUF export, importable by the keyboard's
Languages & Models screen.

## Layout

- `pipeline/` — the training pipeline (`pipeline.py <stage>`: `corpus`,
  `tokenizer`, `pretrain`, `finetune`, `eval`, `export`, `all`),
  `config_fi.py` (model + data config).
- `Dockerfile` — one-off training job container (PyTorch + pinned training
  stack + CPU llama.cpp for GGUF conversion and the smoke test). The pipeline
  is baked in at `/pipeline`; all artifacts go to `/work` (host-mounted
  `/docker/data/futo-lm-fi`).
- `.github/workflows/build.yml` — builds the image on every push to `master`
  and publishes it to `ghcr.io/juhovan/futo-lm-fi` with a `sha-<commit>` tag
  (plus `latest` on master).

## Deploying a pipeline change

1. Push to `master`. The workflow builds and pushes the image.
2. Get the image digest from the run (docker/build-push-action log, or
   `docker manifest inspect ghcr.io/juhovan/futo-lm-fi:sha-<commit>`).
3. Update the pin in the infra repo's
   `ansible/host_vars/koti-docker-prod-llm.24-7.fi.yml`
   (`futo_lm_fi.image: ghcr.io/juhovan/futo-lm-fi:sha-<commit>@sha256:<digest>`).
4. Deploy with `--tags futo-lm-fi`; the host pulls the image and recreates the
   idle container.

## Running stages

The host runs the idle container and one-off stages with
`docker compose -f /docker/compose/futo-lm-fi/docker-compose.yml run --rm futo-lm-fi <stage>`.
Every stage writes a DONE marker under `/docker/data/futo-lm-fi`, so
`run futo-lm-fi all` resumes where a previous run stopped
(`FLUFI_FORCE=1` re-runs a stage).
