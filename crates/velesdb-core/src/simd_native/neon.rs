//! ARM NEON kernel implementations for aarch64.
//!
//! Contains hand-tuned NEON SIMD kernels for dot product and squared L2 distance
//! with 1-acc and 4-acc variants for different vector sizes.
//!
//! NEON is always available on aarch64, so no runtime detection is needed.

// SAFETY: Numeric casts in this file are intentional and safe:
// - All casts are from well-bounded values (vector dimensions, loop indices)
// - All casts are validated by extensive SIMD tests (simd_native_tests.rs)
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::similar_names)]

// =============================================================================
// Dot Product
// =============================================================================

/// ARM NEON dot product with 4 accumulators for ILP optimization (EPIC-052/US-009).
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();

    if len >= 64 {
        return dot_product_neon_4acc(a, b);
    }

    let simd_len = len / 4;
    // SAFETY: `vdupq_n_f32` is a non-faulting register initialisation on aarch64.
    // - Condition 1: NEON is always present on aarch64; no runtime detection needed.
    // - Condition 2: Immediate value 0.0 is a valid f32 constant accepted by the instruction.
    // Reason: Initialise the SIMD accumulator register to zero before the reduction loop.
    let mut sum = unsafe { vdupq_n_f32(0.0) };

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..simd_len {
        let offset = i * 4;
        // SAFETY: `vld1q_f32` loads 4 f32 values from an unaligned address on aarch64.
        // - Condition 1: `offset + 4 <= simd_len * 4 <= len`, so both pointers stay within slice bounds.
        // - Condition 2: `vld1q_f32` is documented to support unaligned loads on ARM64.
        // Reason: Core NEON computation for dot product accumulation per 4-element block.
        unsafe {
            let va = vld1q_f32(a_ptr.add(offset));
            let vb = vld1q_f32(b_ptr.add(offset));
            sum = vfmaq_f32(sum, va, vb);
        }
    }

    // SAFETY: `vaddvq_f32` reduces a 128-bit register to a scalar f32 on aarch64.
    // - Condition 1: NEON is always present on aarch64; intrinsic is always available.
    // - Condition 2: `sum` is a valid float32x4_t value set by `vdupq_n_f32`/`vfmaq_f32`.
    // Reason: Horizontal reduction of the SIMD accumulator to a scalar dot-product result.
    let mut result = unsafe { vaddvq_f32(sum) };

    let base = simd_len * 4;
    for i in base..len {
        result += a[i] * b[i];
    }

    result
}

/// NEON FMA wrapper with x86-compatible argument order.
///
/// NEON `vfmaq_f32(acc, a, b)` = acc + a*b, but [`simd_4acc_dot_loop!`] expects
/// `fmadd(a, b, acc)` = a*b + acc. This wrapper reorders the arguments.
///
/// SAFETY: `vfmaq_f32` is a non-faulting register operation on aarch64.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_fma_compat(
    a: std::arch::aarch64::float32x4_t,
    b: std::arch::aarch64::float32x4_t,
    acc: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    std::arch::aarch64::vfmaq_f32(acc, a, b)
}

/// ARM NEON dot product with 4 accumulators for large vectors.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_product_neon_4acc(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    // SAFETY: `add` on a raw pointer derived from a valid slice.
    // - Condition 1: `len / 16 * 16 <= len`, so `end_main` is within or at the end of the slice.
    // - Condition 2: `add(len)` yields the one-past-the-end pointer, which is valid for comparison.
    // Reason: Establish loop bounds for the 16-element-wide main body and scalar tail.
    let end_main = unsafe { a.as_ptr().add(len / 16 * 16) };
    let end_ptr = unsafe { a.as_ptr().add(len) };

    // SAFETY: 4-accumulator ILP loop using NEON intrinsics. All pointer bounds
    // guaranteed by `end_main`. `neon_fma_compat` reorders args to match macro convention.
    let (combined, mut a_ptr, mut b_ptr) = unsafe {
        crate::simd_4acc_dot_loop!(
            a.as_ptr(), b.as_ptr(), end_main,
            vdupq_n_f32(0.0), vld1q_f32, neon_fma_compat, vaddq_f32, 4
        )
    };

    // SAFETY: `vaddvq_f32` reduces a 128-bit register to a scalar f32 on aarch64.
    // Reason: Horizontal reduction of the combined SIMD accumulator to a scalar result.
    let mut result = unsafe { vaddvq_f32(combined) };

    while a_ptr < end_ptr {
        // SAFETY: Raw pointer dereference for scalar tail processing.
        // - Condition 1: Loop condition `a_ptr < end_ptr` guarantees both pointers are within slice bounds.
        // - Condition 2: `b_ptr` advances in step with `a_ptr` so it remains within the `b` slice.
        // Reason: Handle the remaining 0-15 elements that the 16-wide SIMD loop did not cover.
        unsafe {
            result += *a_ptr * *b_ptr;
            a_ptr = a_ptr.add(1);
            b_ptr = b_ptr.add(1);
        }
    }

    result
}

// =============================================================================
// Cosine Similarity
// =============================================================================

/// ARM NEON cosine similarity — fused single-pass kernel.
///
/// Computes `dot(a,b)`, `norm(a)^2`, and `norm(b)^2` simultaneously in one
/// pass over the data, using 3 independent NEON accumulators. For vectors
/// with >= 64 elements, delegates to [`cosine_fused_neon_4acc`] which uses
/// 12 accumulators (3 products x 4-way ILP).
///
/// This replaces the prior 3-pass approach (`dot_product_neon` called 3x).
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn cosine_neon(a: &[f32], b: &[f32]) -> f32 {
    if a.len() >= 64 {
        // SAFETY: `cosine_fused_neon_4acc` requires NEON (guaranteed on aarch64)
        // and len >= 64 (checked above).
        // Reason: Delegate to the 4-accumulator ILP variant for large vectors.
        return unsafe { cosine_fused_neon_4acc(a, b) };
    }
    // SAFETY: `cosine_fused_neon_1acc` requires NEON (guaranteed on aarch64).
    // Reason: Single-accumulator variant for small/medium vectors.
    unsafe { cosine_fused_neon_1acc(a, b) }
}

/// Single-accumulator fused cosine for vectors with < 64 elements.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn cosine_fused_neon_1acc(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let simd_len = len / 4;

    // SAFETY: `vdupq_n_f32` is a non-faulting register initialisation on aarch64.
    // - Condition 1: NEON is always present on aarch64.
    // - Condition 2: Immediate value 0.0 is valid for the instruction.
    // Reason: Initialise three SIMD accumulators (dot, norm_a, norm_b).
    let mut dot_acc = vdupq_n_f32(0.0);
    let mut na_acc = vdupq_n_f32(0.0);
    let mut nb_acc = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..simd_len {
        let offset = i * 4;
        // SAFETY: `vld1q_f32`/`vfmaq_f32` are non-faulting NEON operations.
        // - Condition 1: `offset + 4 <= simd_len * 4 <= len`, pointers within bounds.
        // - Condition 2: `vld1q_f32` supports unaligned loads on ARM64.
        // Reason: Single-pass accumulation of dot, norm_a_sq, norm_b_sq.
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        dot_acc = vfmaq_f32(dot_acc, va, vb);
        na_acc = vfmaq_f32(na_acc, va, va);
        nb_acc = vfmaq_f32(nb_acc, vb, vb);
    }

    // SAFETY: `vaddvq_f32` reduces a 128-bit register to scalar on aarch64.
    // - Condition 1: All accumulators are valid float32x4_t values.
    // Reason: Horizontal reduction of the three accumulators.
    let mut dot = vaddvq_f32(dot_acc);
    let mut norm_a_sq = vaddvq_f32(na_acc);
    let mut norm_b_sq = vaddvq_f32(nb_acc);

    let base = simd_len * 4;
    for i in base..len {
        let x = a[i];
        let y = b[i];
        dot += x * y;
        norm_a_sq += x * x;
        norm_b_sq += y * y;
    }

    finalize_cosine(dot, norm_a_sq, norm_b_sq)
}

/// Four-accumulator fused cosine for vectors with >= 64 elements.
///
/// Uses 12 NEON registers (3 products x 4-way ILP) and processes 16
/// elements per iteration, following the pattern from `cosine_fused_avx2_2acc`.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn cosine_fused_neon_4acc(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let main_end = len / 16 * 16;
    // SAFETY: `add` on a raw pointer derived from a valid slice.
    // - Condition 1: `main_end <= len`, so pointer stays within the allocation.
    // - Condition 2: `add(len)` yields one-past-end, valid for comparison.
    // Reason: Establish loop bounds for 16-wide main body and scalar tail.
    let end_main = a.as_ptr().add(main_end);
    let end_ptr = a.as_ptr().add(len);

    let (dot, norm_a_sq, norm_b_sq) = cosine_fused_neon_main_loop(a.as_ptr(), b.as_ptr(), end_main);

    let (dot, norm_a_sq, norm_b_sq) = cosine_fused_neon_scalar_tail(
        end_main,
        b.as_ptr().add(main_end),
        end_ptr,
        dot,
        norm_a_sq,
        norm_b_sq,
    );

    finalize_cosine(dot, norm_a_sq, norm_b_sq)
}

/// Reduces 4 NEON f32x4 accumulators to a single scalar sum.
///
/// SAFETY: All inputs must be valid `float32x4_t` values.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn reduce_4acc_neon(
    a0: std::arch::aarch64::float32x4_t,
    a1: std::arch::aarch64::float32x4_t,
    a2: std::arch::aarch64::float32x4_t,
    a3: std::arch::aarch64::float32x4_t,
) -> f32 {
    use std::arch::aarch64::*;
    // SAFETY: `vaddq_f32`/`vaddvq_f32` are non-faulting register operations.
    // - Condition 1: All accumulators hold valid float32x4_t values.
    // Reason: Reduce 4 accumulators to scalar result.
    let ab01 = vaddq_f32(a0, a1);
    let ab23 = vaddq_f32(a2, a3);
    vaddvq_f32(vaddq_f32(ab01, ab23))
}

/// Main 16-wide SIMD loop for fused cosine (4-acc ILP).
///
/// Returns `(dot, norm_a_sq, norm_b_sq)` accumulated over full 16-element blocks.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn cosine_fused_neon_main_loop(
    mut a_ptr: *const f32,
    mut b_ptr: *const f32,
    end_main: *const f32,
) -> (f32, f32, f32) {
    use std::arch::aarch64::*;

    // SAFETY: `vdupq_n_f32` is a non-faulting register initialisation on aarch64.
    // - Condition 1: NEON is always present on aarch64.
    // - Condition 2: Immediate 0.0 is valid.
    // Reason: Initialise 12 accumulators (3 products x 4-way ILP).
    let (mut d0, mut d1, mut d2, mut d3) =
        (vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let (mut na0, mut na1, mut na2, mut na3) =
        (vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let (mut nb0, mut nb1, mut nb2, mut nb3) =
        (vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));

    while a_ptr < end_main {
        // SAFETY: Loop condition guarantees 16 elements remain before `end_main`.
        // - Condition 1: `vld1q_f32` supports unaligned loads on ARM64.
        // Reason: 16-wide single-pass accumulation with 4-way ILP.
        let va0 = vld1q_f32(a_ptr);
        let vb0 = vld1q_f32(b_ptr);
        d0 = vfmaq_f32(d0, va0, vb0);
        na0 = vfmaq_f32(na0, va0, va0);
        nb0 = vfmaq_f32(nb0, vb0, vb0);

        let va1 = vld1q_f32(a_ptr.add(4));
        let vb1 = vld1q_f32(b_ptr.add(4));
        d1 = vfmaq_f32(d1, va1, vb1);
        na1 = vfmaq_f32(na1, va1, va1);
        nb1 = vfmaq_f32(nb1, vb1, vb1);

        let va2 = vld1q_f32(a_ptr.add(8));
        let vb2 = vld1q_f32(b_ptr.add(8));
        d2 = vfmaq_f32(d2, va2, vb2);
        na2 = vfmaq_f32(na2, va2, va2);
        nb2 = vfmaq_f32(nb2, vb2, vb2);

        let va3 = vld1q_f32(a_ptr.add(12));
        let vb3 = vld1q_f32(b_ptr.add(12));
        d3 = vfmaq_f32(d3, va3, vb3);
        na3 = vfmaq_f32(na3, va3, va3);
        nb3 = vfmaq_f32(nb3, vb3, vb3);

        a_ptr = a_ptr.add(16);
        b_ptr = b_ptr.add(16);
    }

    (
        reduce_4acc_neon(d0, d1, d2, d3),
        reduce_4acc_neon(na0, na1, na2, na3),
        reduce_4acc_neon(nb0, nb1, nb2, nb3),
    )
}

/// Scalar tail for fused cosine — handles the remaining 0..15 elements.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn cosine_fused_neon_scalar_tail(
    mut a_ptr: *const f32,
    mut b_ptr: *const f32,
    end_ptr: *const f32,
    mut dot: f32,
    mut norm_a_sq: f32,
    mut norm_b_sq: f32,
) -> (f32, f32, f32) {
    while a_ptr < end_ptr {
        // SAFETY: Loop condition guarantees both pointers are within slice bounds.
        // - Condition 1: `b_ptr` advances in step with `a_ptr`.
        // Reason: Handle remaining elements the 16-wide loop did not cover.
        let x = *a_ptr;
        let y = *b_ptr;
        dot += x * y;
        norm_a_sq += x * x;
        norm_b_sq += y * y;
        a_ptr = a_ptr.add(1);
        b_ptr = b_ptr.add(1);
    }
    (dot, norm_a_sq, norm_b_sq)
}

/// Finalize cosine from dot product and squared norms.
#[cfg(target_arch = "aarch64")]
#[inline]
fn finalize_cosine(dot: f32, norm_a_sq: f32, norm_b_sq: f32) -> f32 {
    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

// =============================================================================
// Squared L2 Distance
// =============================================================================

/// ARM NEON squared L2 distance.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn squared_l2_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let simd_len = len / 4;

    // SAFETY: `vdupq_n_f32` is a non-faulting register initialisation on aarch64.
    // - Condition 1: NEON is always present on aarch64; no runtime detection needed.
    // - Condition 2: Immediate value 0.0 is a valid f32 constant accepted by the instruction.
    // Reason: Initialise the SIMD accumulator register to zero before the squared-diff loop.
    let mut sum = unsafe { vdupq_n_f32(0.0) };

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..simd_len {
        let offset = i * 4;
        // SAFETY: `vld1q_f32`/`vsubq_f32`/`vfmaq_f32` are non-faulting NEON operations.
        // - Condition 1: `offset + 4 <= simd_len * 4 <= len`, so both pointers stay within slice bounds.
        // - Condition 2: `vld1q_f32` is documented to support unaligned loads on ARM64.
        // Reason: Compute squared element-wise differences for the L2 distance accumulator.
        unsafe {
            let va = vld1q_f32(a_ptr.add(offset));
            let vb = vld1q_f32(b_ptr.add(offset));
            let diff = vsubq_f32(va, vb);
            sum = vfmaq_f32(sum, diff, diff);
        }
    }

    // SAFETY: `vaddvq_f32` reduces a 128-bit register to a scalar f32 on aarch64.
    // - Condition 1: NEON is always present on aarch64; intrinsic is always available.
    // - Condition 2: `sum` is a valid float32x4_t value set by `vdupq_n_f32`/`vfmaq_f32`.
    // Reason: Horizontal reduction of the squared-difference accumulator to a scalar result.
    let mut result = unsafe { vaddvq_f32(sum) };

    let base = simd_len * 4;
    for i in base..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result
}
