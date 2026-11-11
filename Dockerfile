# FUTO Keyboard Finnish LM pipeline (one-off training job container).
#
# Base image is the same verified PyTorch image the futo-portuguese pipeline
# ran end-to-end on: torch 2.6.0 + CUDA 12.4 + cudnn9, preinstalled. The
# runtime variant (not -devel) is enough: no CUDA extensions are compiled in
# the image (llama.cpp is built CPU-only, GGML_CUDA=OFF), and the runtime
# variant is several GB smaller — a -devel base fills a free GitHub runner's
# disk during the build.
#
# The job itself needs no GPU for corpus/tokenizer/export/eval; the pretrain and
# finetune stages use the reserved NVIDIA device (see the host's compose override).
#
# The pipeline code is baked into the image at /pipeline; a code change means a
# new image (built by .github/workflows/build.yml, pushed to ghcr.io/juhovan/futo-lm-fi).

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/work/.cache/huggingface

# Build + fetch tooling (pinned).
RUN pip install --no-cache-dir cmake==3.30.0 requests==2.32.3

# Training / packaging stack (pinned; versions verified to exist and interoperate
# as of 2026-10).
RUN pip install --no-cache-dir \
    "transformers==4.55.1" \
    "datasets==3.6.0" \
    "tokenizers==0.21.0" \
    "sentencepiece==0.2.0" \
    "gguf==0.19.0" \
    "huggingface_hub==0.34.6"

# llama.cpp: CPU build. Used for (a) HF -> GGUF conversion, (b) requantizing
# output.weight to Q6_K to match the reference English model, (c) the final
# llama-cli smoke test that proves the shipped .gguf actually loads and runs.
ARG LLAMA_CPP_TAG=v0.8.0
# The runtime pytorch base image ships no C++ toolchain (the -devel variant
# did), so install git + build-essential in this layer before the clone.
# (No system cmake either; the pip cmake above provides it.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --recurse-submodules --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && git fetch --depth 1 origin tag "${LLAMA_CPP_TAG}" 2>/dev/null \
       || git fetch --depth 1 origin "${LLAMA_CPP_TAG}" \
    && git checkout -q "${LLAMA_CPP_TAG}" \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build -j"$(nproc)" \
    && rm -rf /opt/llama.cpp/.git

# Pipeline code (versioned in this repo; the entrypoint runs it directly).
COPY pipeline/ /pipeline/

# All artifacts go under /work (a host-mounted data volume).
WORKDIR /work

ENTRYPOINT ["python", "/pipeline/pipeline.py"]
CMD ["idle"]
