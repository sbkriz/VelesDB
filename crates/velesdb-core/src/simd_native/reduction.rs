//! Shared horizontal reduction helpers for SIMD accumulators.
//!
//! Provides canonical `hsum_avx256` and `hsum_avx512` functions so that
//! every AVX2/AVX-512 kernel reduces accumulators through a single code path.

#![allow(clippy::incompatible_msrv)] // SIMD intrinsics gated behind target_feature + runtime detection

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
