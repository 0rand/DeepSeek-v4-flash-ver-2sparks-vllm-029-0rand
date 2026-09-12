#!/bin/bash
# Fix DSV4 mm-prefix span derivation in the V2 Model Runner.
#
# Root cause (INVESTIGATION-vision-grounding-bias.md §12):
#   With the V2 model runner (default on CUDA), DefaultModelState.prepare_attn
#   derives mm_prefix bidirectional ranges via compute_mm_prefix_ranges ->
#   PlaceholderRange.extract_embeds_range(), which returns only the IMAGE-embed
#   runs (per-row-pair runs of the N-layout), NOT the full sentinel block
#   [pad + IMAGE_START ... IMAGE_END]. The DSL-specific span derivation
#   (mm_prefix_span_leading_pad_modulus=4, mm_prefix_clamp_sliding_window=True)
#   exists only in the V1 runner (gpu_model_runner.py).
#
# Fix: port that logic into attn_utils.compute_mm_prefix_ranges and thread the
# config flags through DefaultModelState.prepare_attn. Python-only, no recompile.
#
# Idempotent: safe to apply to an already-patched tree (patch reversed with -R,
# errors tolerated).
set -euo pipefail

VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
PATCH_FILE="$(cd "$(dirname "$0")" && pwd)/dsv4-mm-prefix-span.patch"

# paths inside the patch are relative to vllm/; patch from parent of vllm/
cd "$(dirname "$VLLM_DIR")"

if patch -p1 --dry-run -N -s < "$PATCH_FILE" 2>/dev/null; then
    echo "[fix-dsv4-mm-prefix-span] applying patch to $VLLM_DIR"
    patch -p1 -N -s < "$PATCH_FILE"
elif patch -p1 --dry-run -R -s < "$PATCH_FILE" 2>/dev/null; then
    echo "[fix-dsv4-mm-prefix-span] already applied - skipping"
else
    echo "[fix-dsv4-mm-prefix-span] ERROR: cannot apply and not already applied" >&2
    exit 1
fi

echo "[fix-dsv4-mm-prefix-span] done"
