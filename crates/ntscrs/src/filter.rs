use std::mem::MaybeUninit;

use crate::f32x4::{get_supported_simd_type, F32x4, SupportedSimdType};

/// Multiplies two polynomials (lowest coefficients first).
/// Note that this function does not trim trailing zero coefficients--see below for that.
/// This is how filters are multiplied--a transfer function is a fraction of polynomials.
pub fn polynomial_multiply(a: &[f32], b: &[f32]) -> Vec<f32> {
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

// TODO: this is a copy of Rust's each_mut, which was stabilized in 1.77.
// We should probably just use that function.
fn each_mut<T, const N: usize>(arr: &mut [T; N]) -> [&mut T; N] {
    // Unlike in `map`, we don't need a guard here, as dropping a reference
    // is a noop.
    let mut out = unsafe { MaybeUninit::<[MaybeUninit<&mut T>; N]>::uninit().assume_init() };
    for (src, dst) in arr.iter_mut().zip(&mut out) {
        dst.write(src);
    }

    // SAFETY: All elements of `dst` are properly initialized and
    // `MaybeUninit<T>` has the same layout as `T`, so this cast is valid.
    unsafe { (&mut out as *mut _ as *mut [&mut T; N]).read() }
}

/// Rational transfer function for an IIR filter in the z-transform domain.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TransferFunction {
    /// Coefficients for the numerator polynomial. Padded with trailing zeros to match the number of coefficients
    /// in the denominator.
    pub num: Vec<f32>,
    /// Coefficients for the denominator polynomial.
    pub den: Vec<f32>,
}

impl TransferFunction {
    pub fn new<I, J>(num: I, den: J) -> Self
    where
        I: IntoIterator<Item = f32>,
        J: IntoIterator<Item = f32>,
    {
        let mut num = num.into_iter().collect::<Vec<f32>>();
        let mut den = den.into_iter().collect::<Vec<f32>>();
        if num.len() > den.len() {
            panic!("Numerator length exceeds denominator length.");
        }

        // Resize to 4 so we can use the SIMD implementation. This is faster than the scalar implementation, even if
        // we only use 2 coefficients.
        if den.len() == 2 || den.len() == 3 {
            den.resize(4, 0.0);
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

    /// Whether processing the rows in chunks or one-by-one is faster.
    pub fn should_use_row_chunks(&self) -> bool {
        // Row chunks are ~25% slower than processing each row one-by-one with the scalar implementation.
        // We should only process rows in chunks with the SIMD implementation.
        self.den.len() == 4 && get_supported_simd_type() != SupportedSimdType::None
    }

    /// Return initial conditions for the filter that results in a given steady-state value (e.g. "start" the filter as
    /// if every previous sample was the given value).
    fn initial_condition(&self, value: f32) -> Vec<f32> {
        // Adapted from scipy
        // https://github.com/scipy/scipy/blob/da82ac849a4ccade2d954a0998067e6aa706dd70/scipy/signal/_signaltools.py#L3609-L3742

        let filter_len = usize::max(self.num.len(), self.den.len());
        // The last element here will always be 0--in the loop below, we intentionally do not initialize the last
        // element of zi.
        let mut zi = vec![0f32; filter_len];
        if value.abs() == 0.0 {
            return zi;
        }

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

        zi[0] = b_sum / norm_den.iter().sum::<f32>();
        let mut a_sum = 1.0;
        let mut c_sum = 0.0;
        for i in 1..filter_len - 1 {
            let num_i = norm_num.get(i).unwrap_or(&0.0);
            let den_i = norm_den.get(i).unwrap_or(&0.0);
            a_sum += den_i;
            c_sum += num_i - den_i * norm_num[0];
            zi[i] = (a_sum * zi[0] - c_sum) * value;
        }
        zi[0] *= value;

        zi
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

    /// Filter a signal, reading from one slice and writing into another.
    /// # Arguments
    /// - `src` - The slice containing the signal to be filtered.
    /// - `dst` - The slice to place the filtered signal into. This must be the same size as `src`.
    /// - `initial` - The initial steady-state value of the filter.
    /// - `scale` - Scale the effect of the filter on the output by this amount.
    /// - `delay` - Offset the filter output backwards (to the left) by this amount.
    pub fn filter_signal_into(
        &self,
        src: &[f32],
        dst: &mut [f32],
        initial: f32,
        scale: f32,
        delay: usize,
    ) {
        if dst.len() != src.len() {
            panic!(
                "Source slice is {} samples but destination is {} samples",
                src.len(),
                src.len()
            );
        }

        let filter_len = usize::max(self.num.len(), self.den.len());
        let mut z = self.initial_condition(initial);

        for i in 0..(src.len() + delay) {
            // Either the loop bound extending past items.len() or the min() call seems to prevent the optimizer from
            // determining that we're in-bounds here. Since i.min(items.len() - 1) never exceeds items.len() - 1 by
            // definition, this is safe.
            let sample = unsafe { src.get_unchecked(i.min(src.len() - 1)) };
            let filt_sample =
                Self::filter_sample(filter_len, &self.num, &self.den, &mut z, *sample, scale);
            if i >= delay {
                dst[i - delay] = filt_sample;
            }
        }
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
            *z = self.initial_condition(initial).try_into().unwrap();
        });
        let z_rows_ref = each_mut::<[f32; SIZE], ROWS>(&mut z_rows).map(|z| z.as_mut_slice());

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
    unsafe fn filter_signal_in_place_fixed_size_simdx4<S: F32x4, const ROWS: usize>(
        &self,
        signal: &mut [&mut [f32]; ROWS],
        initial: [f32; ROWS],
        scale: f32,
        delay: usize,
    ) {
        let mut z_rows = [[0f32; 4]; ROWS];
        z_rows.iter_mut().zip(initial).for_each(|(z, initial)| {
            *z = self.initial_condition(initial).try_into().unwrap();
        });
        let mut z = z_rows;

        let mut num: [f32; 4] = [0f32; 4];
        num.copy_from_slice(&self.num);

        let mut den: [f32; 4] = [0f32; 4];
        den.copy_from_slice(&self.den);

        let num = S::load(&num);
        let den = S::load(&den);
        let scale_b = S::set1(scale);
        let width = signal[0].len();

        for i in 0..(width + delay) {
            for j in 0..ROWS {
                let mut zmm = S::load4(&z[j]);
                // Either the loop bound extending past items.len() or the min() call seems to prevent the optimizer from
                // determining that we're in-bounds here. Since i.min(items.len() - 1) never exceeds items.len() - 1 by
                // definition, this is safe.
                let sample = S::load1(signal[j].get_unchecked(i.min(width - 1)));
                let filt_sample = num.mul_add(sample, zmm).swizzle(0, 0, 0, 0);

                // Add the sample * the numerator, subtract the filtered sample * the denominator
                zmm = num.mul_add(sample, zmm);
                zmm = den.neg_mul_add(filt_sample, zmm);

                // Shift it all over
                zmm = zmm.swizzle(1, 2, 3, 0);

                // Zero out the last element
                zmm = zmm.insert::<3>(0.0);

                zmm.store(&mut z[j]);

                if i >= delay {
                    let samp_diff = filt_sample - sample;
                    let final_samp = samp_diff.mul_add(scale_b, sample);
                    final_samp.store1(signal[j].get_unchecked_mut(i - delay));
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    /// Process the signal using SSE4.1 intrinsics.
    unsafe fn filter_signal_dispatch_sse41<const ROWS: usize>(
        &self,
        signal: &mut [&mut [f32]; ROWS],
        initial: [f32; ROWS],
        scale: f32,
        delay: usize,
    ) {
        use crate::f32x4::x86_64::SseF32x4;
        unsafe {
            self.filter_signal_in_place_fixed_size_simdx4::<SseF32x4, ROWS>(
                signal, initial, scale, delay,
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    /// Process the signal using AVX2 intrinsics.
    unsafe fn filter_signal_dispatch_avx2<const ROWS: usize>(
        &self,
        signal: &mut [&mut [f32]; ROWS],
        initial: [f32; ROWS],
        scale: f32,
        delay: usize,
    ) {
        use crate::f32x4::x86_64::AvxF32x4;
        unsafe {
            self.filter_signal_in_place_fixed_size_simdx4::<AvxF32x4, ROWS>(
                signal, initial, scale, delay,
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    /// Process the signal using NEON intrinsics.
    unsafe fn filter_signal_dispatch_neon<const ROWS: usize>(
        &self,
        signal: &mut [&mut [f32]; ROWS],
        initial: [f32; ROWS],
        scale: f32,
        delay: usize,
    ) {
        use crate::f32x4::aarch64::ArmF32x4;
        unsafe {
            self.filter_signal_in_place_fixed_size_simdx4::<ArmF32x4, ROWS>(
                signal, initial, scale, delay,
            );
        }
    }

    /// Filter a signal in-place, modifying the given slice.
    /// # Type parameters
    /// - `ROWS` - The number of rows to process at once. This makes SIMD faster by removing loop-carried dependencies:
    /// once one pixel of each row is done, we can move onto the next row instead of waiting to finish calculating the
    /// next filter state.
    /// # Arguments
    /// - `signal` - The slice containing the signal to be filtered.
    /// - `initial` - The initial steady-state value of the filter.
    /// - `scale` - Scale the filter output by this amount. For example, a scale of -1 turns a lowpass filter into a
    ///   highpass filter.
    /// - `delay` - Offset the filter output backwards (to the left) by this amount.
    pub fn filter_signal_in_place<const ROWS: usize>(
        &self,
        signal: &mut [&mut [f32]; ROWS],
        initial: [f32; ROWS],
        scale: f32,
        delay: usize,
    ) {
        let filter_len = usize::max(self.num.len(), self.den.len());

        if filter_len == 4 {
            #[cfg(target_arch = "x86_64")]
            match get_supported_simd_type() {
                SupportedSimdType::Sse41 => {
                    unsafe {
                        self.filter_signal_dispatch_sse41::<ROWS>(signal, initial, scale, delay);
                    }
                    return;
                }
                SupportedSimdType::Avx2 => {
                    unsafe {
                        self.filter_signal_dispatch_avx2::<ROWS>(signal, initial, scale, delay);
                    }
                    return;
                }
                _ => {}
            }

            #[cfg(target_arch = "aarch64")]
            if get_supported_simd_type() == SupportedSimdType::Neon {
                unsafe {
                    self.filter_signal_dispatch_neon::<ROWS>(signal, initial, scale, delay);
                }
                return;
            }

            self.filter_signal_in_place_fixed_size::<4, ROWS>(signal, initial, scale, delay);
        } else {
            match filter_len {
                // Specialize fixed-size implementations for filter sizes 1-8
                1 => {
                    self.filter_signal_in_place_fixed_size::<1, ROWS>(signal, initial, scale, delay)
                }
                2 => {
                    self.filter_signal_in_place_fixed_size::<2, ROWS>(signal, initial, scale, delay)
                }
                3 => {
                    self.filter_signal_in_place_fixed_size::<3, ROWS>(signal, initial, scale, delay)
                }
                // 4 is covered in the branch above
                5 => {
                    self.filter_signal_in_place_fixed_size::<5, ROWS>(signal, initial, scale, delay)
                }
                6 => {
                    self.filter_signal_in_place_fixed_size::<6, ROWS>(signal, initial, scale, delay)
                }
                7 => {
                    self.filter_signal_in_place_fixed_size::<7, ROWS>(signal, initial, scale, delay)
                }
                8 => {
                    self.filter_signal_in_place_fixed_size::<8, ROWS>(signal, initial, scale, delay)
                }
                _ => {
                    // Fall back to the general-length implementation
                    let mut z: [Vec<f32>; ROWS] = initial
                        .into_iter()
                        .map(|init| self.initial_condition(init))
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
                    );
                }
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
