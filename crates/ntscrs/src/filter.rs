use std::num::NonZero;

use fearless_simd::{Level, dispatch, f32x4, prelude::*};

/// Multiplies two polynomials (lowest coefficients first).
/// Note that this function does not trim trailing zero coefficients--see below for that.
/// This is how filters are multiplied--a transfer function is a fraction of polynomials.
fn polynomial_multiply(a: &[f32], b: &[f32], dst: &mut [f32]) -> usize {
    let degree = a.len() + b.len() - 1;

    for ai in 0..a.len() {
        for bi in 0..b.len() {
            dst[ai + bi] += a[ai] * b[bi];
        }
    }

    degree
}

/// Helper function to trim trailing zeros. Takes a slice and returns a sub-slice of it with no trailing zeros.
fn trim_zeros(mut input: &[f32]) -> &[f32] {
    while input.last().is_some_and(|last| *last == 0.0) {
        input = &input[..input.len() - 1]
    }
    input
}

/// Rational transfer function for an IIR filter in the z-transform domain.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TransferFunction {
    /// Coefficients for the numerator and denominator. The first 4 elements hold the numerator, and the last 3 hold the
    /// denominator (without the implicit 1.0 of the first element).
    coeffs: [f32; 7],
    /// Number of actual coefficients. The leading 1 in the denominator *is* included here.
    len: NonZero<u32>,
}

fn filter_signal_simd<const ROWS: usize>(
    level: Level,
    tf: &TransferFunction,
    signal: &mut [&mut [f32]; ROWS],
    initial: [f32; ROWS],
    delay: usize,
) -> bool {
    if level.is_fallback() {
        return false;
    }
    dispatch!(level, simd =>
        tf.filter_signal_in_place_fixed_size_simdx4::<_, ROWS>(
            simd, signal, initial, delay,
        )
    );
    true
}

impl TransferFunction {
    #[inline(always)]
    pub fn new(num: &[f32], den: &[f32]) -> Self {
        assert!(num.len() <= 4 && den.len() <= 3, "Filter is too large.");
        let len = den.len() + 1;
        assert!(
            num.len() <= len,
            "Numerator length exceeds denominator length."
        );

        let mut coeffs = [0f32; 7];
        let (dst_num, dst_den) = coeffs.split_at_mut(4);
        dst_num[..num.len()].copy_from_slice(num);
        dst_den[..den.len()].copy_from_slice(den);
        let len = NonZero::new(len as u32).unwrap();

        TransferFunction { coeffs, len }
    }

    pub fn with_scale(&self, scale: f32) -> Self {
        let (num, den) = self.coeffs.split_at(4);
        let mut new_num = [0f32; 4];

        new_num[0] = scale * num[0] + (1.0 - scale);
        for i in 1..self.len() {
            new_num[i] = scale * num[i] + (1.0 - scale) * den[i - 1];
        }

        TransferFunction::new(&new_num[..self.len()], &den[..self.len() - 1])
    }

    /// Chain this filter into itself `n` times. This multiplies the filter's coefficients (see polynomial_multiply
    /// above), which produces the same result as running a signal through the filter three times but is faster.
    pub fn cascade_self(&self, n: usize) -> Self {
        let mut filt = self.clone();
        for _ in 1..n {
            filt = &filt * self;
        }
        filt
    }

    /// Whether we should use SIMD and hence process rows in chunks.
    pub fn should_use_simd(&self, level: Level) -> bool {
        !level.is_fallback() && (2..=4).contains(&self.len.get())
    }

    #[inline(always)]
    fn num_den(&self) -> (&[f32], &[f32]) {
        let len = self.len();
        (&self.coeffs[..len], &self.coeffs[4..4 + len - 1])
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len.get() as usize
    }

    /// Return initial conditions for the filter that results in a given steady-state value (e.g. "start" the filter as
    /// if every previous sample was the given value).
    fn initial_condition_into(&self, value: f32, dst: &mut [f32]) {
        // Adapted from scipy
        // https://github.com/scipy/scipy/blob/da82ac849a4ccade2d954a0998067e6aa706dd70/scipy/signal/_signaltools.py#L3609-L3742

        let filter_len = self.len.get() as usize;
        assert_eq!(dst.len(), filter_len);

        // The last element here will always be 0--in the loop below, we intentionally do not initialize the last
        // element of zi.
        dst.fill(0.0);

        if value.abs() == 0.0 {
            return;
        }

        let (num, den) = self.num_den();

        let mut b_sum = 0.0;
        for i in 1..filter_len {
            let num_i = num.get(i).unwrap_or(&0.0);
            let den_i = den.get(i - 1).unwrap_or(&0.0);
            b_sum += num_i - den_i * num[0];
        }

        dst[0] = b_sum / (den.iter().sum::<f32>() + 1.0);
        let mut a_sum = 1.0;
        let mut c_sum = 0.0;
        for i in 1..filter_len - 1 {
            let num_i = num.get(i).unwrap_or(&0.0);
            let den_i = den.get(i - 1).unwrap_or(&0.0);
            a_sum += den_i;
            c_sum += num_i - den_i * num[0];
            dst[i] = (a_sum * dst[0] - c_sum) * value;
        }
        dst[0] *= value;
    }

    #[inline(always)]
    fn filter_sample(
        filter_len: usize,
        num: &[f32],
        den: &[f32],
        z: &mut [f32],
        sample: f32,
    ) -> f32 {
        // This function gets auto-vectorized by the compiler. The following attempts to optimize it have backfired:
        // - Special-casing a function for when there's only one nonzero coefficient in the numerator: slower.
        // - Using a smallvec to store all the coefficients: slower.
        let filt_sample = z[0] + (num[0] * sample);
        for i in 0..filter_len - 1 {
            z[i] = z[i + 1] + (num[i + 1] * sample) - (den[i] * filt_sample);
        }
        filt_sample
    }

    /// Scalar implementation of a linear filter.
    /// Adapted from https://github.com/scipy/scipy/blob/da82ac849a4ccade2d954a0998067e6aa706dd70/scipy/signal/_lfilter.c.in#L543
    #[inline(always)]
    fn filter_signal_in_place_impl<const ROWS: usize>(
        signal: &mut [&mut [f32]; ROWS],
        num: &[f32],
        den: &[f32],
        z: [&mut [f32]; ROWS],
        delay: usize,
    ) {
        let filter_len = num.len();
        for row_idx in 0..ROWS {
            let signal = &mut signal[row_idx];
            for i in 0..(signal.len() + delay) {
                // Either the loop bound extending past items.len() or the min() call seems to prevent the optimizer from
                // determining that we're in-bounds here. Since i.min(items.len() - 1) never exceeds items.len() - 1 by
                // definition, this is safe.
                let sample = unsafe { signal.get_unchecked(i.min(signal.len() - 1)) };
                let filt_sample = Self::filter_sample(filter_len, num, den, z[row_idx], *sample);
                if i >= delay {
                    signal[i - delay] = filt_sample;
                }
            }
        }
    }

    /// Specialized version of filter_signal_in_place for a fixed number of filter coefficients. About 15% faster than
    /// the dynamic-size version. All we need to do is specify the size as a const generic param--the compiler
    /// specializes the rest.
    #[inline(never)]
    fn filter_signal_in_place_fixed_size<const SIZE: usize, const ROWS: usize>(
        &self,
        signal: &mut [&mut [f32]; ROWS],
        initial: [f32; ROWS],
        delay: usize,
    ) {
        let mut z_rows = [[0f32; SIZE]; ROWS];
        z_rows.iter_mut().zip(initial).for_each(|(z, initial)| {
            self.initial_condition_into(initial, z);
        });
        let z_rows_ref = z_rows.each_mut().map(|z| z.as_mut_slice());

        let (num, den) = self.num_den();
        Self::filter_signal_in_place_impl::<ROWS>(signal, num, den, z_rows_ref, delay);
    }

    /// SIMD implementation of a linear filter. This isn't row-parallel--the architecture is a bit more complex.
    /// The filter coefficients are stored in a single SIMD register, meaning this works for filters with up to 4
    /// coefficients. This allows for easy updating of the filter state--we can do it for all the coefficients at once.
    /// However, this adds a long data dependency--sure, we're doing all these SIMD operations on the coefficients, but
    /// we have to wait for them all to make it through the pipeline before we can repeat it on the next pixel, because
    /// IIR filters are based on a feedback loop and therefore inherently serial. This is where looping over multiple
    /// rows at a time comes in--after we finish processing a sample on one row, we start in on the next. This scheme,
    /// in my testing, is faster than the naive SIMD translation of just doing the scalar approach (one coefficient at
    /// a time) on multiple rows.
    ///
    /// Note that this is `inline(always)` so that the dispatch functions below will actually compile this with the
    /// correct architecture-specific SIMD features enabled.
    #[inline(always)]
    fn filter_signal_in_place_fixed_size_simdx4<S: Simd, const ROWS: usize>(
        &self,
        simd: S,
        signal: &mut [&mut [f32]; ROWS],
        initial: [f32; ROWS],
        delay: usize,
    ) {
        // Ensure the chunks are actually of equal length
        let width = signal[0].len();
        for i in 1..ROWS {
            assert_eq!(signal[i].len(), width);
        }

        let len = self.len();
        let mut z = initial.map(|initial| {
            let mut dest = [0.0; 4];
            self.initial_condition_into(initial, &mut dest[..len]);
            f32x4::simd_from(dest, simd)
        });

        let mut num = [0.0f32; 4];
        let mut den = [0.0f32; 4];
        let (my_num, my_den) = self.num_den();
        num[0..self.len()].copy_from_slice(my_num);
        den[0..self.len() - 1].copy_from_slice(my_den);

        let num = f32x4::simd_from(num, simd);
        let den = -f32x4::simd_from(den, simd);

        for i in 0..(width + delay) {
            for j in 0..ROWS {
                // While the compiler cannot elide this bounds check in the scalar version either, here it is probably
                // also because it doesn't know that each row of the signal has the same length.
                let sample =
                    unsafe { f32x4::splat(simd, *signal[j].get_unchecked(i.min(width - 1))) };
                let filt_sample = num.mul_add(sample, z[j]);

                if i >= delay {
                    unsafe {
                        *signal[j].get_unchecked_mut(i - delay) = filt_sample[0];
                    }
                }

                // Add the sample * the numerator, subtract the filtered sample * the denominator, and shift it all over
                z[j] = den.mul_add(
                    // Filtered sample * the denominator, pre-negated and shifted
                    f32x4::splat(simd, filt_sample[0]),
                    // Sample * the numerator, which we are now shifting over
                    filt_sample.slide::<1>(0.0),
                );
            }
        }
    }

    /// Filter a signal in-place, modifying the given slice.
    /// # Type parameters
    /// - `ROWS` - The number of rows to process at once. This makes SIMD faster by removing loop-carried dependencies:
    ///   once one pixel of each row is done, we can move onto the next row instead of waiting to finish calculating the
    ///   next filter state.
    /// # Arguments
    /// - `signal` - The slice containing the signal to be filtered.
    /// - `initial` - The initial steady-state value of the filter.
    /// - `scale` - Scale the filter output by this amount. For example, a scale of -1 turns a lowpass filter into a
    ///   highpass filter.
    /// - `delay` - Offset the filter output backwards (to the left) by this amount.
    pub fn filter_signal_in_place<const ROWS: usize>(
        &self,
        level: Level,
        signal: &mut [&mut [f32]; ROWS],
        initial: [f32; ROWS],
        delay: usize,
    ) {
        if self.should_use_simd(level) && filter_signal_simd(level, self, signal, initial, delay) {
            return;
        }

        match self.len() {
            // Specialize fixed-size implementations for filter sizes 1-4
            1 => self.filter_signal_in_place_fixed_size::<1, ROWS>(signal, initial, delay),
            2 => self.filter_signal_in_place_fixed_size::<2, ROWS>(signal, initial, delay),
            3 => self.filter_signal_in_place_fixed_size::<3, ROWS>(signal, initial, delay),
            4 => self.filter_signal_in_place_fixed_size::<4, ROWS>(signal, initial, delay),
            _ => unreachable!("Filters with an order > 4 are not supported"),
        }
    }
}

impl std::ops::Mul<&TransferFunction> for &TransferFunction {
    type Output = TransferFunction;

    fn mul(self, rhs: &TransferFunction) -> Self::Output {
        let (lhs_num, lhs_den) = self.num_den();
        let (rhs_num, rhs_den) = rhs.num_den();
        // Add back the implicit leading 1 coefficient for denominator multiplication
        let mut lhs_den_full = [0f32; 4];
        lhs_den_full[0] = 1.0;
        lhs_den_full[1..1 + lhs_den.len()].copy_from_slice(lhs_den);
        let mut rhs_den_full = [0f32; 4];
        rhs_den_full[0] = 1.0;
        rhs_den_full[1..1 + rhs_den.len()].copy_from_slice(rhs_den);

        let mut num = [0f32; 7];
        let mut den = [0f32; 7];
        polynomial_multiply(lhs_num, rhs_num, &mut num);
        polynomial_multiply(
            &lhs_den_full[..self.len()],
            &rhs_den_full[..rhs.len()],
            &mut den,
        );
        let num = trim_zeros(&num);
        // Remove the leading 1 from the multiplied denominator
        let den = &trim_zeros(&den)[1..];

        TransferFunction::new(num, den)
    }
}
