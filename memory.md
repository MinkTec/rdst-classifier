# Inference Benchmark Progress

## 2026-06-25

- Initial `cargo bench --bench rdst_bench` compiled successfully, but every existing benchmark skipped because external `dart_rdst_classifier` fixtures/assets are not present in this workspace.
- Added an in-repo deterministic synthetic benchmark model/data path to `benches/rdst_bench.rs` so inference performance can be measured locally without external files.
- Baseline command: `cargo bench --bench rdst_bench -- --noplot --sample-size 10 --measurement-time 2 --warm-up-time 1 synthetic`.
- Baseline synthetic results: batch predict `10.071 ms`, batch predict_proba `10.653 ms`, single predict `1.1473 ms`, single predict_proba `1.0655 ms`.
- Kept optimization: scale feature matrices in place to avoid allocating a second scaled feature matrix.
- Kept optimization: cache transform shapelet groups on `RdstClassifier` load and reuse them for each inference call. This was the largest win, especially for single-window calls.
- Rejected experiment: direct input scoring without subsequence buffers. It improved one batch case slightly but regressed single-window latency to about `0.80 ms`, likely because the existing contiguous buffers help vectorized distance accumulation.
- Kept optimization: bypass Rayon for `n_samples == 1` to avoid parallel scheduling overhead on single-window inference.
- Kept optimization: normalize the contiguous subsequence buffer in place after non-normalized shapelets are scored, avoiding a separate normalized buffer allocation.
- Rejected experiment: manual indexed L1 distance loop. It regressed single-window benches versus the iterator/zip accumulation.
- Kept optimization: specialize `get_all_subsequences` for `dilation == 1` with `copy_from_slice`; it improved or held steady in the synthetic benches.
- Rejected experiment: early-abandon L1 distance accumulation. Branch overhead dominated and all synthetic benches regressed.
- Final command: `cargo bench --bench rdst_bench -- --noplot --sample-size 20 --measurement-time 3 --warm-up-time 1 --discard-baseline synthetic`.
- Final synthetic results: batch predict `4.1471 ms`, batch predict_proba `3.8240 ms`, single predict `490.45 us`, single predict_proba `498.24 us`.
- Final speedups versus baseline medians: batch predict `2.43x`, batch predict_proba `2.79x`, single predict `2.34x`, single predict_proba `2.14x`.
- Verification: `cargo test` passed (`60` tests; doc examples ignored as before). External Dart fixture/production benches still skip when those assets are absent.

## 2026-06-25, second pass

- Research notes: stable Rust SIMD options for this crate are mainly `std::arch` target intrinsics with runtime CPU detection; portable `std::simd` is still less attractive for stable/library use. Bigger wins looked more likely from fusing inference stages and removing allocations than from byte tricks alone.
- Current-state re-baseline was noisy, so the prior stable final numbers stayed the comparison baseline: batch predict `4.1471 ms`, batch predict_proba `3.8240 ms`, single predict `490.45 us`, single predict_proba `498.24 us`.
- Kept optimization: fused StandardScaler and Ridge scoring into the transform path. `RdstClassifier` now precomputes scaler-adjusted ridge coefficients/intercepts and computes class scores directly while shapelet features are produced, avoiding the full feature matrix allocation, scale pass, and final dot-product pass in `predict`/`predict_proba`.
- Kept optimization: runtime-dispatched AVX2 L1 distance kernel using `std::arch` for contiguous subsequence-vs-shapelet distance. Existing scalar/auto-vectorized path remains the fallback for non-AVX2 CPUs.
- Rejected experiment: replacing the AVX lane-store horizontal sum with an intrinsic horizontal reduction; it regressed synthetic timings.
- Rejected experiment: special-casing binary score updates; it was not consistently faster than the simple row loop.
- Kept optimization: reuse subsequence buffers across shapelet groups per sample instead of allocating a new `Vec<f64>` for every group. This was the largest second-pass batch win.
- Kept optimization: reuse sliding mean/std and scratch buffers across normalized groups per sample.
- Kept optimization: AVX2 in-place normalization for channel slices; it was neutral-to-positive and preserves scalar fallback.
- Rejected deployment experiment: `RUSTFLAGS='-C target-cpu=native'` did not improve synthetic timings on this machine and made single-window timings worse, so no repo config was added.
- Final verification: `cargo fmt --check`, `git diff --check`, and `cargo test` passed.
- Final portable-build batch command: `cargo bench --bench rdst_bench -- --noplot --sample-size 20 --measurement-time 3 --warm-up-time 1 --discard-baseline synthetic`.
- Final portable-build batch results: batch predict `1.6154 ms`, batch predict_proba `1.5764 ms`.
- Final portable-build single reruns: single predict `197.48 us`, single predict_proba `215.19 us`.
- Second-pass speedups versus the first-pass final medians: batch predict `2.57x`, batch predict_proba `2.43x`, single predict `2.48x`, single predict_proba `2.32x`.
- Total speedups versus the original synthetic baseline medians: batch predict `6.24x`, batch predict_proba `6.76x`, single predict `5.81x`, single predict_proba `4.95x`.

## 2026-06-25, post-push exploration

- Pushed commit `42cd484` (`Optimize RDST inference hot path`) to `origin/main`.
- GPU research: `wgpu` is the practical Rust-native cross-platform compute route for Vulkan/Metal/D3D12/WebGPU-style backends, but WGSL/wgpu compute is effectively `f32`/`f16` oriented for portable shaders. This RDST engine is currently `f64` and regression-tested against aeon/Dart outputs, so a GPU backend would either need an approximate `f32` mode or backend-specific nonportable shader extensions. GPU dispatch and upload overhead also makes the current single-window target unlikely to beat AVX2 CPU execution.
- NPU research: CoreML/ANE, NNAPI, and similar NPUs are designed around fixed neural-network graphs and low-precision tensor ops (`f16`, `i8`, sometimes `f32`). RDST's custom dilated shapelet search, argmin, occurrence counts, and `f64` arithmetic do not map cleanly without a major approximation/reformulation.
- Recommendation: preserve the exact CPU path as the default library ABI. If GPU is pursued later, make it an optional approximate/batch-only backend, likely `f32`, and benchmark only for large batches or very large production models where transfer/dispatch overhead can be amortized.
- Follow-up experiment: AVX2 normalization is not always worthwhile for small shapelet lengths. Raising the SIMD normalization cutoff from length `>= 4` to `>= 8` improved the synthetic run: batch predict `1.4844 ms`, batch predict_proba `1.5047 ms`, single predict `176.11 us`, single predict_proba `172.73 us`.
- Rejected follow-up: disabling AVX2 normalization for synthetic lengths via cutoff `>= 16` was slightly worse (`1.5478 ms`, `1.5547 ms`, `183.01 us`, `180.55 us`), so the cutoff remains `>= 8`.
