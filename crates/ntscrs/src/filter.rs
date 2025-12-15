use fearless_simd::{Level, dispatch, f32x4, prelude::*};

/// Multiplies two polynomials (lowest coefficients first).
/// Note that this function does not trim trailing zero coefficients--see below for that.
/// This is how filters are multiplied--a transfer function is a fraction of polynomials.
fn polynomial_multiply(a: &[f32], b: &[f32]) -> Vec<f32> {
    let degree = a.len() + b.len() - 1;

    let mut out = vec![0f32; degree];

    for ai in 0..a.len() {
        for bi in 0..b.len() {
            out[ai + bi] += a[ai] * b[bi];
        }
    }

    out
}

/// Helper function to trim trailing zeros. Takes a slice and returns a sub-slice of it with no trailing zeros.
fn trim_zeros(input: &[f32]) -> &[f32] {
    let mut end = input.len() - 1;
    while input[end].abs() == 0.0 {
        end -= 1;
    }
    &input[0..=end]
}

/// Rational transfer function for an IIR filter in the z-transform domain.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TransferFunction {
    /// Coefficients for the numerator polynomial. Padded with trailing zeros to match the number of coefficients
    /// in the denominator.
    num: Vec<f32>,
    /// Coefficients for the denominator polynomial.
    den: Vec<f32>,
}

fn filter_signal_simd<const ROWS: usize>(
    level: Level,
    tf: &TransferFunction,
    signal: &mut [&mut [f32]; ROWS],
    initial: [f32; ROWS],
    scale: f32,
    delay: usize,
) -> bool {
    if level.is_fallback() {
        return false;
    }
    dispatch!(level, simd =>
        tf.filter_signal_in_place_fixed_size_simdx4::<_, ROWS>(
            simd, signal, initial, scale, delay,
        )
    );
    true
}

impl TransferFunction {
    pub fn new<I, J>(num: I, den: J) -> Self
    where
        I: IntoIterator<Item = f32>,
        J: IntoIterator<Item = f32>,
    {
        let mut num = num.into_iter().collect::<Vec<f32>>();
        let den = den.into_iter().collect::<Vec<f32>>();
        if num.len() > den.len() {
            panic!("Numerator length exceeds denominator length.");
        }

        // Zero-pad the numerator from the right
        num.resize(den.len(), 0.0);

        TransferFunction { num, den }
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
        !level.is_fallback() && (2..=4).contains(&self.den.len())
    }

    /// Return initial conditions for the filter that results in a given steady-state value (e.g. "start" the filter as
    /// if every previous sample was the given value).
    fn initial_condition_into(&self, value: f32, dst: &mut [f32]) {
        // Adapted from scipy
        // https://github.com/scipy/scipy/blob/da82ac849a4ccade2d954a0998067e6aa706dd70/scipy/signal/_signaltools.py#L3609-L3742

        let filter_len = self.num.len();
        assert_eq!(dst.len(), filter_len);
        assert_eq!(self.den.len(), filter_len);
        if value.abs() == 0.0 {
            dst.fill(0.0);
            return;
        }
        // The last element here will always be 0--in the loop below, we intentionally do not initialize the last
        // element of zi.
        dst[filter_len - 1] = 0.0;

        let first_nonzero_coeff = self
            .den
            .iter()
            .find_map(|coeff| {
                if coeff.abs() != 0.0 {
                    Some(*coeff)
                } else {
                    None
                }
            })
            .expect("There must be at least one nonzero coefficient in the denominator.");

        let norm_num = self
            .num
            .iter()
            .map(|item| *item / first_nonzero_coeff)
            .collect::<Vec<f32>>();
        let norm_den = self
            .den
            .iter()
            .map(|item| *item / first_nonzero_coeff)
            .collect::<Vec<f32>>();

        let mut b_sum = 0.0;
        for i in 1..filter_len {
            let num_i = norm_num.get(i).unwrap_or(&0.0);
            let den_i = norm_den.get(i).unwrap_or(&0.0);
            b_sum += num_i - den_i * norm_num[0];
        }

        dst[0] = b_sum / norm_den.iter().sum::<f32>();
        let mut a_sum = 1.0;
        let mut c_sum = 0.0;
        for i in 1..filter_len - 1 {
            let num_i = norm_num.get(i).unwrap_or(&0.0);
            let den_i = norm_den.get(i).unwrap_or(&0.0);
            a_sum += den_i;
            c_sum += num_i - den_i * norm_num[0];
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
        scale: f32,
    ) -> f32 {
        // This function gets auto-vectorized by the compiler. The following attempts to optimize it have backfired:
        // - Special-casing a function for when there's only one nonzero coefficient in the numerator: slower.
        // - Using a smallvec to store all the coefficients: slower.
        let filt_sample = z[0] + (num[0] * sample);
        for i in 0..filter_len - 1 {
            z[i] = z[i + 1] + (num[i + 1] * sample) - (den[i + 1] * filt_sample);
        }
        (filt_sample - sample) * scale + sample
    }

    /// Scalar implementation of a linear filter.
    /// Adapted from https://github.com/scipy/scipy/blob/da82ac849a4ccade2d954a0998067e6aa706dd70/scipy/signal/_lfilter.c.in#L543
    #[inline(always)]
    fn filter_signal_in_place_impl<const ROWS: usize>(
        signal: &mut [&mut [f32]; ROWS],
        num: &[f32],
        den: &[f32],
        z: [&mut [f32]; ROWS],
        scale: f32,
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
                let filt_sample =
                    Self::filter_sample(filter_len, num, den, z[row_idx], *sample, scale);
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
        scale: f32,
        delay: usize,
    ) {
        let mut z_rows = [[0f32; SIZE]; ROWS];
        z_rows.iter_mut().zip(initial).for_each(|(z, initial)| {
            self.initial_condition_into(initial, z);
        });
        let z_rows_ref = z_rows.each_mut().map(|z| z.as_mut_slice());

        Self::filter_signal_in_place_impl::<ROWS>(
            signal, &self.num, &self.den, z_rows_ref, scale, delay,
        );
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
        scale: f32,
        delay: usize,
    ) {
        // Ensure the chunks are actually of equal length
        let width = signal[0].len();
        for i in 1..ROWS {
            assert_eq!(signal[i].len(), width);
        }

        let mut z = initial.map(|initial| {
            let mut dest = [0.0; 4];
            self.initial_condition_into(initial, &mut dest[..self.num.len()]);
            f32x4::simd_from(dest, simd)
        });

        let mut num = [0.0f32; 4];
        let mut den = [0.0f32; 4];
        num[0..self.num.len()].copy_from_slice(&self.num);
        den[0..self.den.len()].copy_from_slice(&self.den);

        let num = f32x4::simd_from(num, simd);
        let den = -f32x4::simd_from(den, simd).slide::<1>(0.0);

        let is_unit_scale = scale == 1.0;

        for i in 0..(width + delay) {
            for j in 0..ROWS {
                // While the compiler cannot elide this bounds check in the scalar version either, here it is probably
                // also because it doesn't know that each row of the signal has the same length.
                let sample =
                    unsafe { f32x4::splat(simd, *signal[j].get_unchecked(i.min(width - 1))) };
                let filt_sample = num.mul_add(sample, z[j]);

                if i >= delay {
                    // If the filter scale is 1.0, we can skip scaling the sample. Either this branch is easily
                    // predicted or hoisted by the compiler, and this is ~4.5% faster.
                    let final_samp = if is_unit_scale {
                        filt_sample
                    } else {
                        let samp_diff = filt_sample - sample;
                        samp_diff.mul_add(scale, sample)
                    };
                    unsafe {
                        *signal[j].get_unchecked_mut(i - delay) = final_samp[0];
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
        scale: f32,
        delay: usize,
    ) {
        let filter_len = usize::max(self.num.len(), self.den.len());

        if self.should_use_simd(level)
            && filter_signal_simd(level, self, signal, initial, scale, delay)
        {
            return;
        }

        match filter_len {
            // Specialize fixed-size implementations for filter sizes 1-4
            1 => self.filter_signal_in_place_fixed_size::<1, ROWS>(signal, initial, scale, delay),
            2 => self.filter_signal_in_place_fixed_size::<2, ROWS>(signal, initial, scale, delay),
            3 => self.filter_signal_in_place_fixed_size::<3, ROWS>(signal, initial, scale, delay),
            4 => self.filter_signal_in_place_fixed_size::<4, ROWS>(signal, initial, scale, delay),
            _ => {
                // Fall back to the general-length implementation
                let mut z: [Vec<f32>; ROWS] = initial
                    .into_iter()
                    .map(|init| {
                        let mut dst = vec![0.0; filter_len];
                        self.initial_condition_into(init, &mut dst);
                        dst
                    })
                    .collect::<Vec<Vec<_>>>()
                    .try_into()
                    .unwrap();
                let z: [&mut [f32]; ROWS] = z
                    .iter_mut()
                    .map(|z| z.as_mut_slice())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                Self::filter_signal_in_place_impl::<ROWS>(
                    signal, &self.num, &self.den, z, scale, delay,
                )
            }
        }
    }
}

impl std::ops::Mul<&TransferFunction> for &TransferFunction {
    type Output = TransferFunction;

    fn mul(self, rhs: &TransferFunction) -> Self::Output {
        TransferFunction::new(
            polynomial_multiply(trim_zeros(&self.num), trim_zeros(&rhs.num)),
            polynomial_multiply(trim_zeros(&self.den), trim_zeros(&rhs.den)),
        )
    }
}
