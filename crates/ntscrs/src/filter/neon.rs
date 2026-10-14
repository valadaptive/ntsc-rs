use core::arch::aarch64::*;

use super::TransferFunction;

/// Transpose four rows into four columns, or the reverse.
#[inline(always)]
unsafe fn transpose(v: [float32x4_t; 4]) -> [float32x4_t; 4] {
    unsafe {
        let a = vtrn1q_f32(v[0], v[1]);
        let b = vtrn2q_f32(v[0], v[1]);
        let c = vtrn1q_f32(v[2], v[3]);
        let d = vtrn2q_f32(v[2], v[3]);
        [
            vcombine_f32(vget_low_f32(a), vget_low_f32(c)),
            vcombine_f32(vget_low_f32(b), vget_low_f32(d)),
            vcombine_f32(vget_high_f32(a), vget_high_f32(c)),
            vcombine_f32(vget_high_f32(b), vget_high_f32(d)),
        ]
    }
}

#[inline(always)]
unsafe fn step(
    sample: float32x4_t,
    num: &[float32x4_t; 4],
    den: &[float32x4_t; 3],
    z: &mut [float32x4_t; 4],
) -> float32x4_t {
    unsafe {
        let out = vfmaq_f32(z[0], num[0], sample);
        // Preserve the existing coefficient-SIMD path's two fused operations and their order.
        z[0] = vfmaq_f32(vfmaq_f32(z[1], num[1], sample), den[0], out);
        z[1] = vfmaq_f32(vfmaq_f32(z[2], num[2], sample), den[1], out);
        z[2] = vfmaq_f32(vfmaq_f32(z[3], num[3], sample), den[2], out);
        out
    }
}

impl TransferFunction {
    /// Run four independent scanlines in the four NEON lanes. Transposing 4x4 tiles avoids
    /// gathering and scattering one sample at a time in the main loop. The caller provides a
    /// multiple of four rows; each row is a disjoint, valid mutable slice.
    pub(super) unsafe fn filter_rows_neon<const ROWS: usize>(
        &self,
        signal: &mut [&mut [f32]; ROWS],
        initial: [f32; ROWS],
        delay: usize,
    ) {
        unsafe {
            assert!(ROWS >= 4 && ROWS.is_multiple_of(4));
            let width = signal[0].len();
            for row in signal.iter() {
                assert_eq!(row.len(), width);
            }
            if width == 0 {
                return;
            }
            let zero = vdupq_n_f32(0.0);
            let num = core::array::from_fn(|i| vdupq_n_f32(self.coeffs[i]));
            let den = core::array::from_fn(|i| vdupq_n_f32(-self.coeffs[4 + i]));
            // ROWS is a small compile-time constant. Only the first ROWS/4 entries are used.
            let mut state = [[zero; 4]; ROWS];
            for group in 0..ROWS / 4 {
                let mut initial_rows = [[0.0; 4]; 4];
                for lane in 0..4 {
                    self.initial_condition_into(
                        initial[group * 4 + lane],
                        &mut initial_rows[lane][..self.len()],
                    );
                }
                state[group] = transpose(initial_rows.map(|r| vld1q_f32(r.as_ptr())));
            }
            let read_column = |group: usize, x: usize, rows: &[&mut [f32]; ROWS]| {
                let values = core::array::from_fn::<_, 4, _>(|lane| rows[group * 4 + lane][x]);
                vld1q_f32(values.as_ptr())
            };
            for x in 0..delay {
                for group in 0..ROWS / 4 {
                    step(
                        read_column(group, x.min(width - 1), signal),
                        &num,
                        &den,
                        &mut state[group],
                    );
                }
            }
            let mut x = delay;
            while x + 4 <= width {
                let mut tiles = [[zero; 4]; ROWS];
                for group in 0..ROWS / 4 {
                    tiles[group] = transpose(core::array::from_fn(|lane| {
                        vld1q_f32(signal[group * 4 + lane].as_ptr().add(x))
                    }));
                }
                for column in 0..4 {
                    for group in 0..ROWS / 4 {
                        tiles[group][column] =
                            step(tiles[group][column], &num, &den, &mut state[group]);
                    }
                }
                for group in 0..ROWS / 4 {
                    let rows = transpose(tiles[group]);
                    for lane in 0..4 {
                        vst1q_f32(
                            signal[group * 4 + lane].as_mut_ptr().add(x - delay),
                            rows[lane],
                        );
                    }
                }
                x += 4;
            }
            for x in x..width + delay {
                for group in 0..ROWS / 4 {
                    let out = step(
                        read_column(group, x.min(width - 1), signal),
                        &num,
                        &den,
                        &mut state[group],
                    );
                    let mut values = [0.0; 4];
                    vst1q_f32(values.as_mut_ptr(), out);
                    for lane in 0..4 {
                        signal[group * 4 + lane][x - delay] = values[lane];
                    }
                }
            }
        }
    }
}
