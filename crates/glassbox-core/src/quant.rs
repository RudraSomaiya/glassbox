//! GGML-compatible Q8_0 and Q4_0 blocks: 32 weights sharing one f16 scale.

use half::f16;

pub const QK: usize = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub d: u16,
    pub qs: [i8; QK],
}

/// Low nibble of `qs[j]` holds weight `j`, high nibble holds weight `j + 16`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub d: u16,
    pub qs: [u8; QK / 2],
}

pub fn d(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

pub fn quantize_q8_0(x: &[f32]) -> Vec<BlockQ8_0> {
    assert_eq!(x.len() % QK, 0);
    x.chunks_exact(QK)
        .map(|c| {
            let amax = c.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let mut qs = [0i8; QK];
            for (q, v) in qs.iter_mut().zip(c) {
                *q = (v * id).round() as i8;
            }
            BlockQ8_0 {
                d: f16::from_f32(d).to_bits(),
                qs,
            }
        })
        .collect()
}

/// Same rounding as ggml's reference `quantize_row_q4_0`, so a GGUF written
/// by llama.cpp and one quantized here agree bit for bit.
pub fn quantize_q4_0(x: &[f32]) -> Vec<BlockQ4_0> {
    assert_eq!(x.len() % QK, 0);
    x.chunks_exact(QK)
        .map(|c| {
            let max = c.iter().fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
            let d = max / -8.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let mut qs = [0u8; QK / 2];
            for j in 0..QK / 2 {
                let q0 = ((c[j] * id + 8.5) as i32).min(15) as u8;
                let q1 = ((c[j + QK / 2] * id + 8.5) as i32).min(15) as u8;
                qs[j] = q0 | (q1 << 4);
            }
            BlockQ4_0 {
                d: f16::from_f32(d).to_bits(),
                qs,
            }
        })
        .collect()
}

pub fn dequantize_q8_0(b: &[BlockQ8_0], out: &mut [f32]) {
    for (blk, o) in b.iter().zip(out.chunks_exact_mut(QK)) {
        let d = d(blk.d);
        for (v, &q) in o.iter_mut().zip(&blk.qs) {
            *v = q as f32 * d;
        }
    }
}

pub fn dequantize_q4_0(b: &[BlockQ4_0], out: &mut [f32]) {
    for (blk, o) in b.iter().zip(out.chunks_exact_mut(QK)) {
        let d = d(blk.d);
        for j in 0..QK / 2 {
            o[j] = ((blk.qs[j] & 0xf) as i32 - 8) as f32 * d;
            o[j + QK / 2] = ((blk.qs[j] >> 4) as i32 - 8) as f32 * d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_error_is_bounded() {
        let x: Vec<f32> = (0..256).map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0).collect();
        let mut y = vec![0.0; x.len()];
        dequantize_q8_0(&quantize_q8_0(&x), &mut y);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 0.03, "{a} {b}");
        }
        dequantize_q4_0(&quantize_q4_0(&x), &mut y);
        for (a, b) in x.iter().zip(&y) {
            // Q4_0 is asymmetric: the side opposite the max can clip by one step.
            assert!((a - b).abs() < 1.0, "{a} {b}");
        }
    }
}
