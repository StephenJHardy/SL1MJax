# Exact direct DFT operator

SL1MJax has two mathematically equivalent direct sky-to-visibility paths:

- `predict_stokes_i`: the original JAX-autodifferentiated reference.
- `predict_stokes_i_explicit`: a dual-chunked forward pass with a custom VJP
  that recomputes the same response tiles in the backward pass.

The explicit path never constructs the full visibility-by-pixel matrix. For
visibility tile size \(C_v\) and pixel tile size \(C_p\), its response storage
is \(O(C_v C_p)\), while its arithmetic remains the exact \(O(VP)\) direct
measurement equation. The backward pass stores only the coordinates and
recomputes response tiles; it does not retain all forward responses.

`DirectDFTConfig` controls both tile dimensions, places an explicit byte limit
on one complex response tile, and selects float32/complex64 or
float64/complex128 arithmetic. Float64 remains the application default;
float32 is available for GPU benchmarking.

## Validation

`tests/test_direct_operator.py` checks:

- non-divisible visibility and pixel tile boundaries;
- forward parity with the original operator;
- custom-VJP gradient parity with native JAX autodiff;
- the real transpose identity;
- use from JIT-compiled Optax inference;
- response-tile memory-budget enforcement;
- float32 forward and VJP agreement with the float64 reference.

Forward and backward parity are parameterized over all current models:
delta, Gaussian paraxial, Gaussian wide-field, compound paraxial, and
compound wide-field.

The explicit path is opt-in until GPU benchmarks establish good defaults:

```python
InferenceConfig(
    operator_mode="explicit",
    direct_dft=DirectDFTConfig(
        visibility_chunk_size=256,
        pixel_chunk_size=1024,
    ),
)
```

The CLI equivalents are `--operator-mode explicit`,
`--visibility-tile-size`, and `--pixel-tile-size`.

## H100 benchmark boundary

The GPU harness and staged 3080 Ti commands are documented in
`docs/direct_dft_benchmarking.md`. Sweep both tile dimensions before changing
the kernel. Million-by-million runs should only be attempted after a smaller
geometric sweep demonstrates stable memory and throughput; exact \(O(VP)\)
compute remains intentionally unchanged.
