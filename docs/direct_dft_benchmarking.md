# Direct DFT GPU benchmarking

`scripts/benchmark_direct_operator.py` compares the native autodiff reference
with the dual-chunked explicit-adjoint operator. Every mode and problem size
runs in a fresh subprocess. This isolates JIT caches, contains an out-of-memory
failure, and gives each case an independent allocator peak.

The benchmark reports:

- compilation plus first execution time;
- median steady-state forward or forward/backward time;
- visibility-pixel products per second;
- JAX peak device allocation when the backend exposes it;
- the full complex response-matrix size;
- a conservative native-autodiff working-memory estimate;
- the exact explicit response-tile size.

JAX GPU preallocation is disabled in child processes so allocator statistics
and failure boundaries are meaningful. The native safety guard skips a case
when its estimated working set exceeds 80% of device memory. The estimate is
the response size multiplied by `--native-memory-factor`, which defaults to
two. This is intentionally conservative: kernel intermediates and compiler
choices mean the full native peak is not exactly one response matrix.

## RTX 3080 Ti sequence

Run a small smoke sweep first:

```bash
uv run scripts/benchmark_direct_operator.py \
  --platform gpu \
  --precision float32 \
  --cases 4096x1024,16384x4096 \
  --repeats 5 \
  --output-json outputs/direct_operator_3080ti_smoke.json \
  --output-csv outputs/direct_operator_3080ti_smoke.csv
```

Then map the memory crossover:

```bash
uv run scripts/benchmark_direct_operator.py \
  --platform gpu \
  --precision float32 \
  --cases 8192x4096,16384x8192,32768x8192,32768x16384,65536x16384 \
  --visibility-tile-size 256 \
  --pixel-tile-size 1024 \
  --native-memory-factor 2 \
  --safe-memory-fraction 0.8 \
  --repeats 5 \
  --output-json outputs/direct_operator_3080ti_crossover.json \
  --output-csv outputs/direct_operator_3080ti_crossover.csv
```

The script normally detects memory with `nvidia-smi`. If that is unavailable,
pass `--device-memory-gib 12`. Do not use `--allow-unsafe-native` for the first
run; it deliberately permits cases likely to exhaust device memory.

To separate compile time from kernel throughput, use `median_s`, not
`compile_and_first_s`. The relevant crossover is the first shape where the
native mode fails or is skipped while the explicit mode completes. Before
selecting production tiles, also compare speed among several explicit tile
shapes:

```bash
for tiles in 128x512 256x1024 512x1024 512x2048; do
  visibility_tile=${tiles%x*}
  pixel_tile=${tiles#*x}
  uv run scripts/benchmark_direct_operator.py \
    --platform gpu \
    --precision float32 \
    --modes explicit \
    --cases 32768x16384 \
    --visibility-tile-size "$visibility_tile" \
    --pixel-tile-size "$pixel_tile" \
    --repeats 5 \
    --output-json "outputs/direct_operator_3080ti_tiles_${tiles}.json" \
    --output-csv "outputs/direct_operator_3080ti_tiles_${tiles}.csv"
done
```

Start with the delta model. After selecting tile sizes, repeat representative
cases with `--pixel-model gaussian-wide-field` and
`--pixel-model compound-wide-field`; those kernels perform more arithmetic and
may have different optimal tiles.
