"""
export_dinov2_migraphx.py
=========================
Exports DINOv2 from HuggingFace to a MIGraphX-compiled .mxr file for AMD MI250X.

Pipeline
--------
  HuggingFace (PyTorch) ──► ONNX ──► MIGraphX .mxr (FP16, GPU-compiled)

The .mxr is compiled specifically for the GPU present at export time.
Do NOT copy .mxr files across different GPU families — recompile on each target.

Requirements (must be run inside ROCm environment)
--------------------------------------------------
  pip install torch torchvision transformers optimum[exporters] onnx onnxsim
  # migraphx is pre-installed in /opt/rocm; ensure PYTHONPATH includes it:
  #   export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH

Usage
-----
  # Export dinov2-base (default)
  python export_dinov2_migraphx.py

  # Export a different variant
  python export_dinov2_migraphx.py --model facebook/dinov2-large

  # Export with a specific fixed batch size (better kernel fusion)
  python export_dinov2_migraphx.py --batch-size 4

  # Skip ONNX simplification (faster, but less optimized)
  python export_dinov2_migraphx.py --no-simplify

  # Keep FP32 (default is FP16)
  python export_dinov2_migraphx.py --fp32
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoImageProcessor

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Thin wrapper that returns only last_hidden_state
#    (ONNX export needs a single well-defined output tensor graph)
# ─────────────────────────────────────────────────────────────────────────────
class DINOv2Wrapper(torch.nn.Module):
    """
    Wraps HuggingFace DINOv2 and exposes two clean outputs:
      - global_embedding : CLS token  [B, hidden_dim]
      - patch_embeddings : patch tokens [B, num_patches, hidden_dim]

    Having explicit outputs makes the ONNX graph unambiguous and lets
    MIGraphX fuse the entire forward pass including the final slice ops.
    """

    def __init__(self, hf_model: torch.nn.Module):
        super().__init__()
        self.model = hf_model

    def forward(self, pixel_values: torch.Tensor):
        out = self.model(pixel_values=pixel_values)
        hidden = out.last_hidden_state           # [B, 1+N, D]
        global_emb = hidden[:, 0, :]             # [B, D]
        patch_embs = hidden[:, 1:, :]            # [B, N, D]
        return global_emb, patch_embs


# ─────────────────────────────────────────────────────────────────────────────
# 2. Export to ONNX
# ─────────────────────────────────────────────────────────────────────────────
def export_to_onnx(
    model_id: str,
    onnx_path: Path,
    batch_size: int,
    image_size: int,
    use_fp16: bool,
    simplify: bool,
) -> dict:
    """
    Loads DINOv2 from HuggingFace, traces it with torch.onnx.export,
    and writes an ONNX file.

    Returns a dict of input/output shape info needed by MIGraphX.
    """
    log.info(f"Loading {model_id} ...")
    dtype = torch.float16 if use_fp16 else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hf_model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    wrapped  = DINOv2Wrapper(hf_model).to(device).eval()

    hidden_dim   = hf_model.config.hidden_size
    patch_size   = hf_model.config.patch_size        # 14 for all dinov2 variants
    num_patches  = (image_size // patch_size) ** 2   # 256 for 224px

    log.info(
        f"  hidden_dim={hidden_dim}  patch_size={patch_size}  "
        f"num_patches={num_patches}  dtype={dtype}"
    )

    # Dummy input that matches inference-time shape
    dummy = torch.zeros(batch_size, 3, image_size, image_size, dtype=dtype, device=device)

    input_names  = ["pixel_values"]
    output_names = ["global_embedding", "patch_embeddings"]

    # Dynamic axes let MIGraphX see which dims can vary.
    # We fix batch and spatial dims here for best kernel fusion.
    # If you need dynamic batch, add batch to dynamic_axes below.
    dynamic_axes = {}   # fully static — best performance on MI250X

    log.info(f"Exporting ONNX → {onnx_path} ...")
    t0 = time.perf_counter()

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (dummy,),
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,           # opset 17 covers all ViT ops cleanly
            do_constant_folding=True,   # fold BN stats, positional embeddings, etc.
            export_params=True,
            verbose=False,
        )

    log.info(f"ONNX export done in {time.perf_counter()-t0:.1f}s")

    # ── Optional: ONNX Simplifier (removes redundant ops, helps MIGraphX) ──
    if simplify:
        try:
            import onnx
            import onnxsim

            log.info("Running onnxsim simplification ...")
            model_onnx = onnx.load(str(onnx_path))
            model_simplified, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_simplified, str(onnx_path))
                log.info("  onnxsim simplification succeeded ✓")
            else:
                log.warning("  onnxsim check failed — keeping original ONNX")
        except ImportError:
            log.warning("onnxsim not installed — skipping simplification (pip install onnxsim)")

    # ── Basic ONNX validation ──
    try:
        import onnx
        onnx.checker.check_model(str(onnx_path))
        log.info("ONNX model validation passed ✓")
    except Exception as e:
        log.warning(f"ONNX validation warning: {e}")

    onnx_size_mb = onnx_path.stat().st_size / 1e6
    log.info(f"ONNX file size: {onnx_size_mb:.1f} MB")

    return {
        "input_name":    input_names[0],
        "output_names":  output_names,
        "batch_size":    batch_size,
        "image_size":    image_size,
        "hidden_dim":    hidden_dim,
        "num_patches":   num_patches,
        "dtype_str":     "fp16" if use_fp16 else "fp32",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Compile ONNX → MIGraphX .mxr
# ─────────────────────────────────────────────────────────────────────────────
def compile_to_mxr(
    onnx_path: Path,
    mxr_path: Path,
    shape_info: dict,
    use_fp16: bool,
) -> None:
    """
    Parses the ONNX file with MIGraphX, compiles it for the GPU target,
    and saves the compiled .mxr binary.

    MIGraphX compilation applies:
      - Operator fusion (attention blocks, layer norms, GELU)
      - Dead code elimination
      - Memory layout optimization for HBM2e bandwidth on MI250X
      - FP16 Tensor Core kernel selection (if use_fp16=True)
    """
    try:
        import migraphx
    except ImportError:
        log.error(
            "migraphx Python module not found.\n"
            "Make sure you are inside the ROCm environment and run:\n"
            "  export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH"
        )
        sys.exit(1)

    B  = shape_info["batch_size"]
    IS = shape_info["image_size"]

    # Tell MIGraphX the exact static input shape.
    # Providing explicit shapes here (rather than relying on ONNX defaults)
    # enables full static-shape compilation — the most aggressive fusion path.
    input_shapes = {
        shape_info["input_name"]: [B, 3, IS, IS]
    }

    log.info(f"Parsing ONNX with MIGraphX ...")
    log.info(f"  Fixed input shape: {input_shapes}")
    t0 = time.perf_counter()

    program = migraphx.parse_onnx(
        str(onnx_path),
        map_input_dims=input_shapes,    # fix all dims — enables maximum fusion
        default_dim_value=1,            # fallback for any remaining dynamic dims
    )

    log.info(f"  Parse done in {time.perf_counter()-t0:.1f}s")

    # ── Compile for GPU ──
    # migraphx.get_target("gpu") targets the GPU present at compile time.
    # On MI250X this compiles for gfx90a (CDNA2).
    log.info("Compiling for GPU target (this takes 2–10 min for DINOv2) ...")
    t0 = time.perf_counter()

    compile_options = migraphx.compile_options()

    if use_fp16:
        # quantize_fp16() converts eligible FP32 ops to FP16 during compilation.
        # This is AMD's recommended path for MI250X (CDNA2 has native FP16 matrix units).
        log.info("  Applying FP16 quantization ...")
        migraphx.quantize_fp16(program)

    program.compile(
        migraphx.get_target("gpu"),
        offload_copy=True,    # auto-manage CPU↔GPU transfers
        fast_math=True,       # allow reassociation for better vectorization
    )

    elapsed = time.perf_counter() - t0
    log.info(f"  Compilation done in {elapsed:.1f}s")

    # ── Save compiled program ──
    log.info(f"Saving .mxr → {mxr_path} ...")
    migraphx.save(program, str(mxr_path))

    mxr_size_mb = mxr_path.stat().st_size / 1e6
    log.info(f".mxr file size: {mxr_size_mb:.1f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Verify the compiled .mxr runs correctly
# ─────────────────────────────────────────────────────────────────────────────
def verify_mxr(mxr_path: Path, shape_info: dict) -> None:
    """
    Loads the compiled .mxr back from disk, runs a dummy forward pass,
    and prints output shapes to confirm the export is correct.
    """
    try:
        import migraphx
    except ImportError:
        log.warning("Cannot verify — migraphx not importable")
        return

    log.info(f"Verifying .mxr by loading and running a dummy forward pass ...")
    program = migraphx.load(str(mxr_path))

    B  = shape_info["batch_size"]
    IS = shape_info["image_size"]
    dtype_np = np.float16 if shape_info["dtype_str"] == "fp16" else np.float32

    dummy_input = np.random.rand(B, 3, IS, IS).astype(dtype_np)

    t0 = time.perf_counter()
    results = program.run({shape_info["input_name"]: migraphx.argument(dummy_input)})
    elapsed_ms = (time.perf_counter() - t0) * 1000

    global_emb = np.array(results[0])
    patch_embs = np.array(results[1])

    log.info("Verification results:")
    log.info(f"  global_embedding shape : {global_emb.shape}   (expected [{B}, {shape_info['hidden_dim']}])")
    log.info(f"  patch_embeddings shape : {patch_embs.shape}  (expected [{B}, {shape_info['num_patches']}, {shape_info['hidden_dim']}])")
    log.info(f"  Inference time         : {elapsed_ms:.1f} ms")

    expected_global = (B, shape_info["hidden_dim"])
    expected_patch  = (B, shape_info["num_patches"], shape_info["hidden_dim"])

    assert global_emb.shape == expected_global, f"Shape mismatch: {global_emb.shape} != {expected_global}"
    assert patch_embs.shape == expected_patch,  f"Shape mismatch: {patch_embs.shape} != {expected_patch}"

    log.info("Shape assertions passed ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Write Triton config.pbtxt for the MIGraphX backend
# ─────────────────────────────────────────────────────────────────────────────
def write_triton_config(output_dir: Path, model_name: str, shape_info: dict) -> None:
    """
    Writes a ready-to-use Triton config.pbtxt that loads the .mxr file
    via Triton's MIGraphX backend (backend: "migraphx").
    """
    B   = shape_info["batch_size"]
    D   = shape_info["hidden_dim"]
    N   = shape_info["num_patches"]
    IS  = shape_info["image_size"]
    fp  = "TYPE_FP16" if shape_info["dtype_str"] == "fp16" else "TYPE_FP32"

    config = f"""name: "{model_name}"
backend: "migraphx"
max_batch_size: 0

# ── Input ──────────────────────────────────────────────────────────
input [
  {{
    name:      "pixel_values"
    data_type: {fp}
    dims:      [ {B}, 3, {IS}, {IS} ]
  }}
]

# ── Outputs ────────────────────────────────────────────────────────
output [
  {{
    name:      "global_embedding"
    data_type: {fp}
    dims:      [ {B}, {D} ]
  }},
  {{
    name:      "patch_embeddings"
    data_type: {fp}
    dims:      [ {B}, {N}, {D} ]
  }}
]

# ── Instance: 1 GPU, device 0 ──────────────────────────────────────
instance_group [
  {{
    count: 1
    kind:  KIND_GPU
    gpus:  [ 0 ]
  }}
]
"""

    triton_model_dir = output_dir / "triton_repo" / model_name
    triton_model_dir.mkdir(parents=True, exist_ok=True)
    config_path = triton_model_dir / "config.pbtxt"
    config_path.write_text(config)
    log.info(f"Triton config written → {config_path}")

    # Also create the version directory symlink location note
    version_dir = triton_model_dir / "1"
    version_dir.mkdir(exist_ok=True)
    note = version_dir / "README.txt"
    note.write_text(
        f"Place the compiled {model_name}.mxr file in this directory.\n"
        f"Expected filename: model.mxr\n"
    )
    log.info(
        f"\nTriton model repo layout:\n"
        f"  {triton_model_dir}/\n"
        f"  ├── config.pbtxt\n"
        f"  └── 1/\n"
        f"      └── model.mxr   ← copy your .mxr here\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Export DINOv2 → ONNX → MIGraphX .mxr")
    p.add_argument(
        "--model", default="facebook/dinov2-base",
        help="HuggingFace model ID (default: facebook/dinov2-base)"
    )
    p.add_argument(
        "--output-dir", default="./dinov2_export",
        help="Directory to write ONNX and .mxr files (default: ./dinov2_export)"
    )
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Fixed batch size to compile for (default: 1). "
             "Larger batch = better throughput but less flexible."
    )
    p.add_argument(
        "--image-size", type=int, default=224,
        help="Input image size in pixels (default: 224)"
    )
    p.add_argument(
        "--fp32", action="store_true",
        help="Export in FP32 instead of FP16 (FP16 is faster on MI250X)"
    )
    p.add_argument(
        "--no-simplify", action="store_true",
        help="Skip onnxsim simplification step"
    )
    p.add_argument(
        "--onnx-only", action="store_true",
        help="Stop after ONNX export (skip MIGraphX compilation)"
    )
    p.add_argument(
        "--skip-verify", action="store_true",
        help="Skip the verification forward pass after compilation"
    )
    return p.parse_args()


def main():
    args = parse_args()

    use_fp16    = not args.fp32
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive a clean short name from the model ID, e.g. "dinov2-base"
    model_short = args.model.split("/")[-1]
    dtype_tag   = "fp16" if use_fp16 else "fp32"
    model_name  = f"{model_short}_{dtype_tag}_b{args.batch_size}"

    onnx_path   = output_dir / f"{model_name}.onnx"
    mxr_path    = output_dir / f"{model_name}.mxr"

    log.info("=" * 60)
    log.info("DINOv2 → ONNX → MIGraphX Export")
    log.info("=" * 60)
    log.info(f"  Model       : {args.model}")
    log.info(f"  Batch size  : {args.batch_size}")
    log.info(f"  Image size  : {args.image_size}x{args.image_size}")
    log.info(f"  Precision   : {'FP16' if use_fp16 else 'FP32'}")
    log.info(f"  Output dir  : {output_dir}")
    log.info("=" * 60)

    # ── Step 1: HuggingFace → ONNX ──
    shape_info = export_to_onnx(
        model_id   = args.model,
        onnx_path  = onnx_path,
        batch_size = args.batch_size,
        image_size = args.image_size,
        use_fp16   = use_fp16,
        simplify   = not args.no_simplify,
    )

    if args.onnx_only:
        log.info("--onnx-only set, stopping after ONNX export.")
        log.info(f"ONNX file: {onnx_path}")
        return

    # ── Step 2: ONNX → MIGraphX .mxr ──
    compile_to_mxr(
        onnx_path  = onnx_path,
        mxr_path   = mxr_path,
        shape_info = shape_info,
        use_fp16   = use_fp16,
    )

    # ── Step 3: Verify ──
    if not args.skip_verify:
        verify_mxr(mxr_path, shape_info)

    # ── Step 4: Write Triton config ──
    write_triton_config(output_dir, model_name, shape_info)

    log.info("=" * 60)
    log.info("Export complete!")
    log.info(f"  ONNX : {onnx_path}")
    log.info(f"  MXR  : {mxr_path}")
    log.info("=" * 60)
    log.info(
        "\nNext steps:\n"
        f"  1. Copy {mxr_path.name} to your Triton model repo under 1/model.mxr\n"
        f"  2. Copy config.pbtxt to the model directory\n"
        f"  3. Start Triton with --model-repository=<path>\n"
        f"\nNOTE: This .mxr is compiled for the GPU currently in this machine.\n"
        f"      Recompile on a different GPU architecture."
    )


if __name__ == "__main__":
    main()
