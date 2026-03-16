//! Shared horizontal reduction helpers and 4-accumulator loop macros for SIMD.
//!
//! Provides canonical `hsum_avx256` and `hsum_avx512` functions so that
//! every AVX2/AVX-512 kernel reduces accumulators through a single code path.
//! Also provides [`simd_4acc_dot_loop!`] and [`simd_4acc_l2_loop!`] macros
//! that encode the 4-accumulator ILP unrolling pattern used across all ISAs.

#![allow(clippy::incompatible_msrv)] // SIMD intrinsics gated behind target_feature + runtime detection

// =============================================================================
// Horizontal sum helpers
// =============================================================================

/// Horizontal sum of 8 packed f32 values in an AVX2 `__m256` register.
///
/// Reduces `[a, b, c, d, e, f, g, h]` → `a + b + c + d + e + f + g + h`.
///
/// # Safety
///
/// Caller must ensure CPU supports AVX2 (enforced by `#[target_feature]`
/// and runtime detection via `simd_level()`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn hsum_avx256(v: std::arch::x86_64::__m256) -> f32 {
    // SAFETY: All intrinsics require AVX2 which is guaranteed by #[target_feature].
    // No pointer arithmetic or memory access — operates purely on register values.
    use std::arch::x86_64::{
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32,
        _mm_movehdup_ps, _mm_movehl_ps,
    };
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

/// Horizontal sum of 16 packed f32 values in an AVX-512 `__m512` register.
///
/// Wraps the native `_mm512_reduce_add_ps` intrinsic for API symmetry
/// with [`hsum_avx256`].
///
/// # Safety
///
/// Caller must ensure CPU supports AVX-512F (enforced by `#[target_feature]`
/// and runtime detection via `simd_level()`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
#[allow(dead_code)] // Available for AVX-512 kernels; currently they use _mm512_reduce_add_ps directly
pub(crate) unsafe fn hsum_avx512(v: std::arch::x86_64::__m512) -> f32 {
    // SAFETY: `_mm512_reduce_add_ps` requires AVX-512F, guaranteed by #[target_feature].
    // No pointer arithmetic — operates purely on register values.
    std::arch::x86_64::_mm512_reduce_add_ps(v)
}

// =============================================================================
// 4-accumulator reduction helpers
// =============================================================================

/// Combines 4 AVX2 accumulators into one via binary-tree addition.
///
/// # Safety
///
/// Caller must ensure CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)] // Available for custom 4-acc kernels outside the macro
pub(crate) unsafe fn reduce_4acc_avx256(
    a0: std::arch::x86_64::__m256,
    a1: std::arch::x86_64::__m256,
    a2: std::arch::x86_64::__m256,
    a3: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    // SAFETY: `_mm256_add_ps` requires AVX2, guaranteed by #[target_feature].
    // No pointer arithmetic — operates purely on register values.
    use std::arch::x86_64::_mm256_add_ps;
    let sum01 = _mm256_add_ps(a0, a1);
    let sum23 = _mm256_add_ps(a2, a3);
    _mm256_add_ps(sum01, sum23)
}

/// Combines 4 AVX-512 accumulators into one via binary-tree addition.
///
/// # Safety
///
/// Caller must ensure CPU supports AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
#[allow(dead_code)] // Available for custom 4-acc kernels outside the macro
pub(crate) unsafe fn reduce_4acc_avx512(
    a0: std::arch::x86_64::__m512,
    a1: std::arch::x86_64::__m512,
    a2: std::arch::x86_64::__m512,
    a3: std::arch::x86_64::__m512,
) -> std::arch::x86_64::__m512 {
    // SAFETY: `_mm512_add_ps` requires AVX-512F, guaranteed by #[target_feature].
    // No pointer arithmetic — operates purely on register values.
    use std::arch::x86_64::_mm512_add_ps;
    let sum01 = _mm512_add_ps(a0, a1);
    let sum23 = _mm512_add_ps(a2, a3);
    _mm512_add_ps(sum01, sum23)
}

// =============================================================================
// 4-accumulator loop macros
// =============================================================================

/// 4-accumulator unrolled SIMD loop for dot product (ILP optimization).
///
/// Processes `4 × lane` elements per iteration using 4 independent
/// accumulators to hide FMA latency. Works across AVX2, AVX-512, and NEON
/// by accepting ISA-specific intrinsics as parameters.
///
/// Returns `(combined_accumulator, updated_a_ptr, updated_b_ptr)`.
///
/// # Arguments
///
/// - `$a_ptr`, `$b_ptr` — Starting pointers for the two input vectors
/// - `$end` — End pointer for the main loop (aligned to `4 × lane`)
/// - `$zero` — Zero-init expression (e.g., `_mm256_setzero_ps()`)
/// - `$load` — SIMD load intrinsic (e.g., `_mm256_loadu_ps`)
/// - `$fmadd` — FMA intrinsic with signature `fmadd(a, b, acc) → a*b + acc`
/// - `$add` — SIMD add intrinsic (e.g., `_mm256_add_ps`)
/// - `$lane` — Number of f32 elements per SIMD register (4/8/16)
///
/// # Safety
///
/// Must be invoked inside an `unsafe` context where the specified
/// SIMD intrinsics are valid for the current CPU.
#[macro_export]
macro_rules! simd_4acc_dot_loop {
    ($a_ptr:expr, $b_ptr:expr, $end:expr,
     $zero:expr, $load:ident, $fmadd:ident, $add:ident, $lane:expr) => {{
        let mut acc0 = $zero;
        let mut acc1 = $zero;
        let mut acc2 = $zero;
        let mut acc3 = $zero;
        let mut a_p = $a_ptr;
        let mut b_p = $b_ptr;

        while a_p < $end {
            let va0 = $load(a_p);
            let vb0 = $load(b_p);
            acc0 = $fmadd(va0, vb0, acc0);

            let va1 = $load(a_p.add($lane));
            let vb1 = $load(b_p.add($lane));
            acc1 = $fmadd(va1, vb1, acc1);

            let va2 = $load(a_p.add(2 * $lane));
            let vb2 = $load(b_p.add(2 * $lane));
            acc2 = $fmadd(va2, vb2, acc2);

            let va3 = $load(a_p.add(3 * $lane));
            let vb3 = $load(b_p.add(3 * $lane));
            acc3 = $fmadd(va3, vb3, acc3);

            a_p = a_p.add(4 * $lane);
            b_p = b_p.add(4 * $lane);
        }

        let sum01 = $add(acc0, acc1);
        let sum23 = $add(acc2, acc3);
        ($add(sum01, sum23), a_p, b_p)
    }};
}

/// 4-accumulator unrolled SIMD loop for squared L2 distance.
///
/// Same structure as [`simd_4acc_dot_loop!`] but computes `sum((a-b)²)`
/// instead of `sum(a·b)`. Requires an additional `$sub` intrinsic.
///
/// # Safety
///
/// Same requirements as [`simd_4acc_dot_loop!`].
#[macro_export]
macro_rules! simd_4acc_l2_loop {
    ($a_ptr:expr, $b_ptr:expr, $end:expr,
     $zero:expr, $load:ident, $sub:ident, $fmadd:ident, $add:ident, $lane:expr) => {{
        let mut acc0 = $zero;
        let mut acc1 = $zero;
        let mut acc2 = $zero;
        let mut acc3 = $zero;
        let mut a_p = $a_ptr;
        let mut b_p = $b_ptr;

        while a_p < $end {
            let va0 = $load(a_p);
            let vb0 = $load(b_p);
            let diff0 = $sub(va0, vb0);
            acc0 = $fmadd(diff0, diff0, acc0);

            let va1 = $load(a_p.add($lane));
            let vb1 = $load(b_p.add($lane));
            let diff1 = $sub(va1, vb1);
            acc1 = $fmadd(diff1, diff1, acc1);

            let va2 = $load(a_p.add(2 * $lane));
            let vb2 = $load(b_p.add(2 * $lane));
            let diff2 = $sub(va2, vb2);
            acc2 = $fmadd(diff2, diff2, acc2);

            let va3 = $load(a_p.add(3 * $lane));
            let vb3 = $load(b_p.add(3 * $lane));
            let diff3 = $sub(va3, vb3);
            acc3 = $fmadd(diff3, diff3, acc3);

            a_p = a_p.add(4 * $lane);
            b_p = b_p.add(4 * $lane);
        }

        let sum01 = $add(acc0, acc1);
        let sum23 = $add(acc2, acc3);
        ($add(sum01, sum23), a_p, b_p)
    }};
}

// Re-export macros for crate-internal use
#[allow(unused_imports)]
pub(crate) use simd_4acc_dot_loop;
#[allow(unused_imports)]
pub(crate) use simd_4acc_l2_loop;
