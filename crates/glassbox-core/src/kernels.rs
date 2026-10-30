//! Dot products and matrix multiplication.
//!
//! Every SIMD kernel has a scalar twin that defines the expected result; the
//! tests below check one against the other on awkward shapes. Kernel choice
//! happens once per matmul call: AVX2+FMA when the CPU has it (runtime
//! detection), `simd128` on wasm when compiled with it, scalar otherwise.

use crate::quant::{d, BlockQ4_0, BlockQ8_0, QK};
use std::sync::atomic::{AtomicBool, Ordering};

static SIMD: AtomicBool = AtomicBool::new(true);

/// Turn SIMD kernels off or on globally. Used by the kernel-progression benchmark.
pub fn set_simd(on: bool) {
    SIMD.store(on, Ordering::Relaxed);
}

pub fn simd_enabled() -> bool {
    SIMD.load(Ordering::Relaxed) && simd_available()
}

pub fn simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    return is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return true;
    #[allow(unreachable_code)]
    false
}

// ---------- scalar reference ----------

pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn dot_q8_scalar(a: &[BlockQ8_0], b: &[BlockQ8_0]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let s: i32 = x.qs.iter().zip(&y.qs).map(|(&p, &q)| p as i32 * q as i32).sum();
            s as f32 * d(x.d) * d(y.d)
        })
        .sum()
}

pub fn dot_q4_scalar(a: &[BlockQ4_0], b: &[BlockQ8_0]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let mut s = 0i32;
            for j in 0..QK / 2 {
                s += ((x.qs[j] & 0xf) as i32 - 8) * y.qs[j] as i32;
                s += ((x.qs[j] >> 4) as i32 - 8) * y.qs[j + QK / 2] as i32;
            }
            s as f32 * d(x.d) * d(y.d)
        })
        .sum()
}

// ---------- AVX2 + FMA ----------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    /// # Safety
    /// The CPU must support AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= n {
            for (k, acc) in acc.iter_mut().enumerate() {
                let off = i + 8 * k;
                *acc = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(off)), _mm256_loadu_ps(pb.add(off)), *acc);
            }
            i += 32;
        }
        while i + 8 <= n {
            acc[0] = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc[0]);
            i += 8;
        }
        let v = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        let mut s = hsum(v);
        while i < n {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    /// Signed i8 x i8 products of 32 lanes, summed to 8 i32 lanes. `maddubs`
    /// wants unsigned x signed, so move the sign of `x` onto `y` first.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn mul_sum_i8(x: __m256i, y: __m256i) -> __m256 {
        let ax = _mm256_sign_epi8(x, x);
        let sy = _mm256_sign_epi8(y, x);
        let dot = _mm256_maddubs_epi16(ax, sy);
        _mm256_cvtepi32_ps(_mm256_madd_epi16(dot, _mm256_set1_epi16(1)))
    }

    /// # Safety
    /// The CPU must support AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q8(a: &[BlockQ8_0], b: &[BlockQ8_0]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        for (x, y) in a.iter().zip(b) {
            let qx = _mm256_loadu_si256(x.qs.as_ptr() as *const __m256i);
            let qy = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            let scale = _mm256_set1_ps(d(x.d) * d(y.d));
            acc = _mm256_fmadd_ps(scale, mul_sum_i8(qx, qy), acc);
        }
        hsum(acc)
    }

    /// # Safety
    /// The CPU must support AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q4(a: &[BlockQ4_0], b: &[BlockQ8_0]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let mask = _mm256_set1_epi8(0xf);
        let eight = _mm256_set1_epi8(8);
        for (x, y) in a.iter().zip(b) {
            let bytes = _mm_loadu_si128(x.qs.as_ptr() as *const __m128i);
            // Low nibbles are weights 0..16, high nibbles 16..32.
            let both = _mm256_set_m128i(_mm_srli_epi16(bytes, 4), bytes);
            let qx = _mm256_sub_epi8(_mm256_and_si256(both, mask), eight);
            let qy = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            let scale = _mm256_set1_ps(d(x.d) * d(y.d));
            acc = _mm256_fmadd_ps(scale, mul_sum_i8(qx, qy), acc);
        }
        hsum(acc)
    }
}

// ---------- wasm simd128 ----------

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use super::*;
    use std::arch::wasm32::*;

    fn hsum(v: v128) -> f32 {
        f32x4_extract_lane::<0>(v)
            + f32x4_extract_lane::<1>(v)
            + f32x4_extract_lane::<2>(v)
            + f32x4_extract_lane::<3>(v)
    }

    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let mut acc = [f32x4_splat(0.0); 4];
        let mut i = 0;
        while i + 16 <= n {
            for (k, acc) in acc.iter_mut().enumerate() {
                let off = i + 4 * k;
                // SAFETY: off + 4 <= n, so both 16-byte loads are in bounds.
                let (x, y) = unsafe {
                    (
                        v128_load(a.as_ptr().add(off) as *const v128),
                        v128_load(b.as_ptr().add(off) as *const v128),
                    )
                };
                *acc = f32x4_add(*acc, f32x4_mul(x, y));
            }
            i += 16;
        }
        let mut s = hsum(f32x4_add(f32x4_add(acc[0], acc[1]), f32x4_add(acc[2], acc[3])));
        while i < n {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    fn dot_i8x32(x: &[i8; 32], y: &[i8; 32]) -> i32 {
        let mut acc = i32x4_splat(0);
        for h in 0..2 {
            // SAFETY: each array holds 32 bytes; h * 16 + 16 <= 32.
            let (a, b) = unsafe {
                (
                    v128_load(x.as_ptr().add(h * 16) as *const v128),
                    v128_load(y.as_ptr().add(h * 16) as *const v128),
                )
            };
            acc = i32x4_add(
                acc,
                i32x4_dot_i16x8(i16x8_extend_low_i8x16(a), i16x8_extend_low_i8x16(b)),
            );
            acc = i32x4_add(
                acc,
                i32x4_dot_i16x8(i16x8_extend_high_i8x16(a), i16x8_extend_high_i8x16(b)),
            );
        }
        i32x4_extract_lane::<0>(acc)
            + i32x4_extract_lane::<1>(acc)
            + i32x4_extract_lane::<2>(acc)
            + i32x4_extract_lane::<3>(acc)
    }

    pub fn dot_q8(a: &[BlockQ8_0], b: &[BlockQ8_0]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| dot_i8x32(&x.qs, &y.qs) as f32 * d(x.d) * d(y.d))
            .sum()
    }

    pub fn dot_q4(a: &[BlockQ4_0], b: &[BlockQ8_0]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                // SAFETY: qs is 16 bytes.
                let bytes = unsafe { v128_load(x.qs.as_ptr() as *const v128) };
                let mask = u8x16_splat(0xf);
                let lo = i8x16_sub(v128_and(bytes, mask), i8x16_splat(8));
                let hi = i8x16_sub(u8x16_shr(bytes, 4), i8x16_splat(8));
                let mut q = [0i8; 32];
                // SAFETY: q is 32 bytes, written as two 16-byte halves.
                unsafe {
                    v128_store(q.as_mut_ptr() as *mut v128, lo);
                    v128_store(q.as_mut_ptr().add(16) as *mut v128, hi);
                }
                dot_i8x32(&q, &y.qs) as f32 * d(x.d) * d(y.d)
            })
            .sum()
    }
}

pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    if simd_enabled() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: simd_enabled() checked AVX2 and FMA at runtime.
        return unsafe { avx2::dot_f32(a, b) };
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return wasm::dot_f32(a, b);
    }
    dot_f32_scalar(a, b)
}

pub fn dot_q8(a: &[BlockQ8_0], b: &[BlockQ8_0]) -> f32 {
    if simd_enabled() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: simd_enabled() checked AVX2 and FMA at runtime.
        return unsafe { avx2::dot_q8(a, b) };
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return wasm::dot_q8(a, b);
    }
    dot_q8_scalar(a, b)
}

pub fn dot_q4(a: &[BlockQ4_0], b: &[BlockQ8_0]) -> f32 {
    if simd_enabled() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: simd_enabled() checked AVX2 and FMA at runtime.
        return unsafe { avx2::dot_q4(a, b) };
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return wasm::dot_q4(a, b);
    }
    dot_q4_scalar(a, b)
}

// ---------- matmul ----------

#[derive(Clone)]
pub enum Weights {
    F32(Vec<f32>),
    Q8_0(Vec<BlockQ8_0>),
    Q4_0(Vec<BlockQ4_0>),
}

/// A row-major `[rows, cols]` weight matrix, as stored by `nn.Linear` (out, in).
#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub w: Weights,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    Q8_0,
    Q4_0,
}

impl Matrix {
    pub fn f32(rows: usize, cols: usize, w: Vec<f32>) -> Self {
        assert_eq!(w.len(), rows * cols);
        Matrix {
            rows,
            cols,
            w: Weights::F32(w),
        }
    }

    pub fn dtype(&self) -> DType {
        match self.w {
            Weights::F32(_) => DType::F32,
            Weights::Q8_0(_) => DType::Q8_0,
            Weights::Q4_0(_) => DType::Q4_0,
        }
    }

    pub fn bytes(&self) -> usize {
        match &self.w {
            Weights::F32(w) => w.len() * 4,
            Weights::Q8_0(w) => std::mem::size_of_val(w.as_slice()),
            Weights::Q4_0(w) => std::mem::size_of_val(w.as_slice()),
        }
    }

    /// Quantize an f32 matrix. Rows whose length isn't a multiple of 32 stay f32.
    pub fn quantize(&self, to: DType) -> Matrix {
        let Weights::F32(w) = &self.w else {
            return self.clone();
        };
        if !self.cols.is_multiple_of(QK) {
            return self.clone();
        }
        let w = match to {
            DType::F32 => Weights::F32(w.clone()),
            DType::Q8_0 => Weights::Q8_0(crate::quant::quantize_q8_0(w)),
            DType::Q4_0 => Weights::Q4_0(crate::quant::quantize_q4_0(w)),
        };
        Matrix {
            rows: self.rows,
            cols: self.cols,
            w,
        }
    }

    pub fn row(&self, r: usize, out: &mut [f32]) {
        let c = self.cols;
        match &self.w {
            Weights::F32(w) => out.copy_from_slice(&w[r * c..(r + 1) * c]),
            Weights::Q8_0(w) => crate::quant::dequantize_q8_0(&w[r * c / QK..(r + 1) * c / QK], out),
            Weights::Q4_0(w) => crate::quant::dequantize_q4_0(&w[r * c / QK..(r + 1) * c / QK], out),
        }
    }
}

struct SyncPtr(*mut f32);
// SAFETY: the matmul tasks write disjoint (row, token) cells through this pointer.
unsafe impl Sync for SyncPtr {}

pub(crate) fn par_for(n: usize, f: impl Fn(usize) + Sync + Send) {
    #[cfg(feature = "threads")]
    {
        use rayon::prelude::*;
        (0..n).into_par_iter().for_each(f);
    }
    #[cfg(not(feature = "threads"))]
    (0..n).for_each(f);
}

const ROWS_PER_TASK: usize = 16;
const TOKEN_TILE: usize = 8;

/// `out[t, r] = dot(x[t, :], w[r, :])` for `n` tokens. `out` is `[n, rows]`.
pub fn matmul(out: &mut [f32], x: &[f32], n: usize, m: &Matrix) {
    let (rows, cols) = (m.rows, m.cols);
    assert_eq!(x.len(), n * cols);
    assert_eq!(out.len(), n * rows);
    // Quantized weights multiply against activations quantized to Q8_0, so the
    // inner loop is integer multiply-adds.
    let xq: Vec<BlockQ8_0> = match m.w {
        Weights::F32(_) => Vec::new(),
        _ => crate::quant::quantize_q8_0(x),
    };
    let bpr = cols / QK;
    let ptr = SyncPtr(out.as_mut_ptr());
    let tasks = rows.div_ceil(ROWS_PER_TASK);
    par_for(tasks, |task| {
        let ptr = &ptr;
        let r0 = task * ROWS_PER_TASK;
        let r1 = (r0 + ROWS_PER_TASK).min(rows);
        // Tiling: a tile of tokens is reused across the rows of this task, so
        // each weight row is read from memory once per tile, not once per token.
        for t0 in (0..n).step_by(TOKEN_TILE) {
            let t1 = (t0 + TOKEN_TILE).min(n);
            for r in r0..r1 {
                for t in t0..t1 {
                    let v = match &m.w {
                        Weights::F32(w) => dot_f32(&w[r * cols..(r + 1) * cols], &x[t * cols..(t + 1) * cols]),
                        Weights::Q8_0(w) => dot_q8(&w[r * bpr..(r + 1) * bpr], &xq[t * bpr..(t + 1) * bpr]),
                        Weights::Q4_0(w) => dot_q4(&w[r * bpr..(r + 1) * bpr], &xq[t * bpr..(t + 1) * bpr]),
                    };
                    // SAFETY: t < n and r < rows, so the index is inside `out`, and
                    // each (t, r) cell is written by exactly one task.
                    unsafe { *ptr.0.add(t * rows + r) = v };
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{quantize_q4_0, quantize_q8_0};

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.f()).collect()
        }
    }

    #[test]
    fn simd_dots_match_scalar() {
        let mut rng = Rng(0x9e3779b97f4a7c15);
        for n in [1, 7, 8, 31, 33, 63, 100, 576, 1000, 1537] {
            let (a, b) = (rng.vec(n), rng.vec(n));
            let s = dot_f32_scalar(&a, &b);
            let v = dot_f32(&a, &b);
            assert!((s - v).abs() <= 1e-4 * (1.0 + s.abs()), "n={n} {s} {v}");
        }
        for blocks in [1, 2, 3, 5, 18, 48] {
            let (a, b) = (rng.vec(blocks * QK), rng.vec(blocks * QK));
            let (qa, qb, q4) = (quantize_q8_0(&a), quantize_q8_0(&b), quantize_q4_0(&a));
            let (s, v) = (dot_q8_scalar(&qa, &qb), dot_q8(&qa, &qb));
            assert!((s - v).abs() <= 1e-4 * (1.0 + s.abs()), "q8 {blocks} {s} {v}");
            let (s, v) = (dot_q4_scalar(&q4, &qb), dot_q4(&q4, &qb));
            assert!((s - v).abs() <= 1e-4 * (1.0 + s.abs()), "q4 {blocks} {s} {v}");
        }
    }

    #[test]
    fn matmul_matches_naive() {
        let mut rng = Rng(42);
        for (rows, cols, n) in [(1, 32, 1), (17, 64, 3), (33, 96, 9), (50, 160, 17)] {
            let w = rng.vec(rows * cols);
            let x = rng.vec(n * cols);
            for dt in [DType::F32, DType::Q8_0, DType::Q4_0] {
                let m = Matrix::f32(rows, cols, w.clone()).quantize(dt);
                let mut out = vec![0.0; n * rows];
                matmul(&mut out, &x, n, &m);
                let mut deq = vec![0.0; cols];
                for r in 0..rows {
                    m.row(r, &mut deq);
                    for t in 0..n {
                        let want = dot_f32_scalar(&deq, &x[t * cols..(t + 1) * cols]);
                        // Q8/Q4 also quantize activations, so allow their rounding.
                        let tol = if dt == DType::F32 {
                            1e-4
                        } else {
                            0.05 * (cols as f32).sqrt() / 4.0
                        };
                        assert!((out[t * rows + r] - want).abs() < tol, "{dt:?} {rows}x{cols} n={n}");
                    }
                }
            }
        }
    }
}
