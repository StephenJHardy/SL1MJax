"""Isolated forward/adjoint DFT benchmarks with GPU memory safety guards.

Each case runs in a fresh subprocess so compilation caches and allocator peaks
do not leak between cases. The parent process records JSON and CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_CASES = "4096x1024,16384x4096"


def _parse_cases(value: str) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    for item in value.split(","):
        visibility_text, separator, pixel_text = item.strip().lower().partition("x")
        if not separator:
            raise argparse.ArgumentTypeError(
                f"case {item!r} must have VISIBILITIESxPIXELS form"
            )
        visibility_count = int(visibility_text)
        pixel_count = int(pixel_text)
        if visibility_count < 1 or pixel_count < 1:
            raise argparse.ArgumentTypeError("benchmark dimensions must be positive")
        cases.append((visibility_count, pixel_count))
    if not cases:
        raise argparse.ArgumentTypeError("at least one benchmark case is required")
    return cases


def _gpu_memory_bytes(gpu_index: int) -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    values = [
        int(line.strip()) * 1024**2
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    return values[gpu_index] if gpu_index < len(values) else None


def _block_until_ready(value: Any) -> None:
    import jax

    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _memory_stats(device: Any) -> dict[str, int]:
    stats = device.memory_stats() or {}
    return {
        str(key): int(value)
        for key, value in stats.items()
        if isinstance(value, int | float)
    }


def _run_child(case: dict[str, Any]) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from sl1mjax.direct_operator import DirectDFTConfig, direct_scalar_visibility
    from sl1mjax.rime import _pixel_basis_kernel
    from sl1mjax.sky import pixel_basis_from_name

    precision = str(case["precision"])
    real_dtype = jnp.float32 if precision == "float32" else jnp.float64
    complex_dtype = jnp.complex64 if precision == "float32" else jnp.complex128
    visibility_count = int(case["visibility_count"])
    pixel_count = int(case["pixel_count"])
    rng = np.random.default_rng(int(case["seed"]))
    side = math.ceil(math.sqrt(pixel_count))
    axis = np.linspace(-2e-3, 2e-3, side, dtype=np.float64)
    l_numpy, m_numpy = np.meshgrid(-axis, axis)
    l = jnp.asarray(l_numpy.ravel()[:pixel_count], dtype=real_dtype)
    m = jnp.asarray(m_numpy.ravel()[:pixel_count], dtype=real_dtype)
    uvw = jnp.asarray(
        rng.normal(size=(visibility_count, 3)) * np.asarray([2e4, 2e4, 5e3]),
        dtype=real_dtype,
    )
    intensity = jnp.asarray(
        rng.uniform(0.0, 1.0, size=pixel_count) / pixel_count,
        dtype=real_dtype,
    )
    basis = pixel_basis_from_name(
        str(case["pixel_model"]),
        gaussian_sigma_pixels=float(case["gaussian_sigma_pixels"]),
    )
    pixel_size_rad = float(case["pixel_size_rad"])
    mode = str(case["mode"])

    if mode == "explicit":
        config = DirectDFTConfig(
            visibility_chunk_size=int(case["visibility_tile_size"]),
            pixel_chunk_size=int(case["pixel_tile_size"]),
            max_response_bytes=int(case["max_response_bytes"]),
            precision=precision,  # type: ignore[arg-type]
        )

        def apply(values: Any) -> Any:
            return direct_scalar_visibility(
                values,
                l,
                m,
                uvw,
                pixel_basis=basis,
                pixel_size_rad=pixel_size_rad,
                config=config,
            )

    else:
        native_chunk_size = int(case["native_visibility_chunk_size"])

        def apply(values: Any) -> Any:
            pieces = []
            for start in range(0, visibility_count, native_chunk_size):
                response = _pixel_basis_kernel(
                    uvw[start : start + native_chunk_size],
                    l,
                    m,
                    basis,
                    pixel_size_rad,
                    include_projection=False,
                )
                pieces.append(response @ values)
            return jnp.concatenate(pieces)

    operation = str(case["operation"])
    if operation == "forward":
        compiled = jax.jit(apply)
    else:

        def objective(values: Any) -> Any:
            prediction = apply(values)
            return jnp.mean(
                jnp.square(prediction.real) + jnp.square(prediction.imag)
            )

        compiled = jax.jit(jax.value_and_grad(objective))

    device = jax.devices()[0]
    memory_before = _memory_stats(device)
    started = time.perf_counter()
    first = compiled(intensity)
    _block_until_ready(first)
    compile_and_first_s = time.perf_counter() - started
    for _ in range(int(case["warmups"])):
        warmed = compiled(intensity)
        _block_until_ready(warmed)
    samples_s: list[float] = []
    result: Any = first
    for _ in range(int(case["repeats"])):
        started = time.perf_counter()
        result = compiled(intensity)
        _block_until_ready(result)
        samples_s.append(time.perf_counter() - started)
    memory_after = _memory_stats(device)
    if operation == "forward":
        checksum = float(jnp.sum(jnp.abs(result)))
    else:
        loss, gradient = result
        checksum = float(loss + jnp.sum(jnp.abs(gradient)))
    median_s = statistics.median(samples_s)
    return {
        **case,
        "status": "ok",
        "device": str(device),
        "jax_backend": jax.default_backend(),
        "compile_and_first_s": compile_and_first_s,
        "samples_s": samples_s,
        "median_s": median_s,
        "min_s": min(samples_s),
        "max_s": max(samples_s),
        "visibility_pixel_products_per_s": (
            visibility_count * pixel_count / median_s
        ),
        "checksum": checksum,
        "output_dtype": str(jnp.dtype(complex_dtype)),
        "memory_before": memory_before,
        "memory_after": memory_after,
    }


def _child_entry(payload: str) -> int:
    case = json.loads(payload)
    try:
        result = _run_child(case)
    except Exception as error:
        result = {
            **case,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


def _csv_row(result: dict[str, Any]) -> dict[str, Any]:
    memory = result.get("memory_after", {})
    return {
        "status": result["status"],
        "mode": result["mode"],
        "operation": result["operation"],
        "precision": result["precision"],
        "pixel_model": result["pixel_model"],
        "visibility_count": result["visibility_count"],
        "pixel_count": result["pixel_count"],
        "response_gib": result["response_bytes"] / 1024**3,
        "guarded_native_gib": result["guarded_native_bytes"] / 1024**3,
        "explicit_tile_mib": result["explicit_tile_bytes"] / 1024**2,
        "compile_and_first_s": result.get("compile_and_first_s"),
        "median_s": result.get("median_s"),
        "products_per_s": result.get("visibility_pixel_products_per_s"),
        "peak_device_gib": memory.get("peak_bytes_in_use", 0) / 1024**3,
        "device": result.get("device"),
        "error": result.get("error") or result.get("skip_reason"),
    }


def _write_results(
    results: list[dict[str, Any]], output_json: Path, output_csv: Path
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = [_csv_row(result) for result in results]
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=_parse_cases, default=_parse_cases(DEFAULT_CASES))
    parser.add_argument(
        "--modes",
        default="autodiff,explicit",
        help="comma-separated subset of autodiff,explicit",
    )
    parser.add_argument(
        "--operation",
        choices=("forward", "forward-backward"),
        default="forward-backward",
    )
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--pixel-model",
        choices=(
            "delta",
            "gaussian-paraxial",
            "gaussian-wide-field",
            "compound-paraxial",
            "compound-wide-field",
        ),
        default="delta",
    )
    parser.add_argument("--gaussian-sigma-pixels", type=float, default=0.5)
    parser.add_argument("--pixel-size-rad", type=float, default=1e-5)
    parser.add_argument("--visibility-tile-size", type=int, default=256)
    parser.add_argument("--pixel-tile-size", type=int, default=1024)
    parser.add_argument("--native-visibility-chunk-size", type=int, default=256)
    parser.add_argument("--max-response-mib", type=float, default=512.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--platform", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--device-memory-gib", type=float)
    parser.add_argument("--safe-memory-fraction", type=float, default=0.8)
    parser.add_argument("--native-memory-factor", type=float, default=2.0)
    parser.add_argument("--allow-unsafe-native", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/direct_operator_benchmark.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/direct_operator_benchmark.csv"),
    )
    parser.add_argument("--_child-case", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments._child_case is not None:
        return _child_entry(arguments._child_case)
    modes = tuple(item.strip() for item in arguments.modes.split(",") if item.strip())
    if not modes or set(modes) - {"autodiff", "explicit"}:
        parser.error("--modes must contain autodiff and/or explicit")
    if not 0 < arguments.safe_memory_fraction <= 1:
        parser.error("--safe-memory-fraction must lie in (0, 1]")
    if arguments.native_memory_factor < 1:
        parser.error("--native-memory-factor must be at least one")
    if arguments.warmups < 0 or arguments.repeats < 1:
        parser.error("warmups must be nonnegative and repeats positive")
    detected_memory = _gpu_memory_bytes(arguments.gpu_index)
    device_memory_bytes = (
        int(arguments.device_memory_gib * 1024**3)
        if arguments.device_memory_gib is not None
        else detected_memory
    )
    if arguments.platform == "gpu" and device_memory_bytes is None:
        parser.error(
            "GPU memory was not detected; pass --device-memory-gib explicitly"
        )
    real_bytes = 4 if arguments.precision == "float32" else 8
    complex_bytes = 2 * real_bytes
    configured_explicit_tile_bytes = (
        arguments.visibility_tile_size
        * arguments.pixel_tile_size
        * complex_bytes
    )
    max_response_bytes = int(arguments.max_response_mib * 1024**2)
    if configured_explicit_tile_bytes > max_response_bytes:
        parser.error("explicit tile exceeds --max-response-mib")

    results: list[dict[str, Any]] = []
    for visibility_count, pixel_count in arguments.cases:
        response_bytes = visibility_count * pixel_count * complex_bytes
        explicit_tile_bytes = (
            min(visibility_count, arguments.visibility_tile_size)
            * min(pixel_count, arguments.pixel_tile_size)
            * complex_bytes
        )
        guarded_native_bytes = int(
            response_bytes * arguments.native_memory_factor
        )
        for mode in modes:
            case: dict[str, Any] = {
                "mode": mode,
                "operation": arguments.operation,
                "precision": arguments.precision,
                "pixel_model": arguments.pixel_model,
                "gaussian_sigma_pixels": arguments.gaussian_sigma_pixels,
                "pixel_size_rad": arguments.pixel_size_rad,
                "visibility_count": visibility_count,
                "pixel_count": pixel_count,
                "visibility_tile_size": arguments.visibility_tile_size,
                "pixel_tile_size": arguments.pixel_tile_size,
                "native_visibility_chunk_size": (
                    arguments.native_visibility_chunk_size
                ),
                "max_response_bytes": max_response_bytes,
                "response_bytes": response_bytes,
                "guarded_native_bytes": guarded_native_bytes,
                "explicit_tile_bytes": explicit_tile_bytes,
                "device_memory_bytes": device_memory_bytes,
                "warmups": arguments.warmups,
                "repeats": arguments.repeats,
                "seed": arguments.seed,
            }
            unsafe_native = (
                mode == "autodiff"
                and device_memory_bytes is not None
                and guarded_native_bytes
                > device_memory_bytes * arguments.safe_memory_fraction
            )
            if unsafe_native and not arguments.allow_unsafe_native:
                result = {
                    **case,
                    "status": "skipped",
                    "skip_reason": "native memory safety guard",
                }
            else:
                environment = os.environ.copy()
                environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
                environment["JAX_PLATFORM_NAME"] = arguments.platform
                if arguments.platform == "gpu":
                    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.gpu_index)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--_child-case",
                    json.dumps(case, separators=(",", ":")),
                ]
                try:
                    completed = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=arguments.timeout_s,
                        env=environment,
                    )
                    output_lines = [
                        line for line in completed.stdout.splitlines() if line.strip()
                    ]
                    result = json.loads(output_lines[-1])
                    if completed.returncode != 0 and result["status"] == "ok":
                        result["status"] = "error"
                        result["error"] = completed.stderr[-2000:]
                except subprocess.TimeoutExpired:
                    result = {
                        **case,
                        "status": "timeout",
                        "error": f"exceeded {arguments.timeout_s} seconds",
                    }
                except (IndexError, json.JSONDecodeError) as error:
                    result = {
                        **case,
                        "status": "error",
                        "error": f"child output could not be parsed: {error}",
                    }
            results.append(result)
            row = _csv_row(result)
            print(
                f"{mode:8s} V={visibility_count:7d} P={pixel_count:7d} "
                f"{result['status']:7s} median={row['median_s']} "
                f"peak_GiB={row['peak_device_gib']:.3f}"
            )
            _write_results(
                results, arguments.output_json, arguments.output_csv
            )
    return (
        0
        if all(result["status"] in {"ok", "skipped"} for result in results)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
