/// Multiplies two polynomials (lowest coefficients first).
/// Note that this function does not trim trailing zero coefficients--see below for that.
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

/// Rational transfer function for an IIR filter in the z-transform domain.
#[derive(Debug)]
pub struct TransferFunction {
    /// Coefficients for the numerator polynomial. Padded with trailing zeros to match the number of coefficients
    /// in the denominator.
    pub num: Vec<f32>,
    /// Coefficients for the denominator polynomial.
    pub den: Vec<f32>,
    _private: (),
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

        TransferFunction {
            num,
            den,
            _private: (),
        }
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

    #[inline(always)]
    fn filter_signal_in_place_impl(
        &self,
        signal: &mut [f32],
        num: &[f32],
        den: &[f32],
        z: &mut [f32],
        scale: f32,
        delay: usize,
    ) {
        let filter_len = num.len();
        for i in 0..(signal.len() + delay) {
            // Either the loop bound extending past items.len() or the min() call seems to prevent the optimizer from
            // determining that we're in-bounds here. Since i.min(items.len() - 1) never exceeds items.len() - 1 by
            // definition, this is safe.
            let sample = unsafe { signal.get_unchecked(i.min(signal.len() - 1)) };
            let filt_sample = Self::filter_sample(filter_len, num, den, z, *sample, scale);
            if i >= delay {
                signal[i - delay] = filt_sample;
            }
        }
    }

    /// Specialized version of filter_signal_in_place for fixed-size arrays. About 15% faster than the dynamic-size
    /// version. All we need to do is specify the size as a const generic param--the compiler specializes the rest.
    fn filter_signal_in_place_fixed_size<const SIZE: usize>(
        &self,
        signal: &mut [f32],
        initial: f32,
        scale: f32,
        delay: usize,
    ) {
        let z = self.initial_condition(initial);

        let mut num_fixed: [f32; SIZE] = [0f32; SIZE];
        num_fixed.copy_from_slice(&self.num);

        let mut den_fixed: [f32; SIZE] = [0f32; SIZE];
        den_fixed.copy_from_slice(&self.den);

        let mut z_fixed: [f32; SIZE] = [0f32; SIZE];
        z_fixed.copy_from_slice(&z);

        self.filter_signal_in_place_impl(
            signal,
            &num_fixed,
            &den_fixed,
            &mut z_fixed,
            scale,
            delay,
        );
    }

    /// Filter a signal in-place, modifying the given slice.
    /// # Arguments
    /// - `signal` - The slice containing the signal to be filtered.
    /// - `initial` - The initial steady-state value of the filter.
    /// - `scale` - Scale the filter output by this amount. For example, a scale of -1 turns a lowpass filter into a
    ///   highpass filter.
    /// - `delay` - Offset the filter output backwards (to the left) by this amount.
    pub fn filter_signal_in_place(
        &self,
        signal: &mut [f32],
        initial: f32,
        scale: f32,
        delay: usize,
    ) {
        let filter_len = usize::max(self.num.len(), self.den.len());

        match filter_len {
            // Specialize fixed-size implementations for filter sizes 1-8
            1 => self.filter_signal_in_place_fixed_size::<1>(signal, initial, scale, delay),
            2 => self.filter_signal_in_place_fixed_size::<2>(signal, initial, scale, delay),
            3 => self.filter_signal_in_place_fixed_size::<3>(signal, initial, scale, delay),
            4 => self.filter_signal_in_place_fixed_size::<4>(signal, initial, scale, delay),
            5 => self.filter_signal_in_place_fixed_size::<5>(signal, initial, scale, delay),
            6 => self.filter_signal_in_place_fixed_size::<6>(signal, initial, scale, delay),
            7 => self.filter_signal_in_place_fixed_size::<7>(signal, initial, scale, delay),
            8 => self.filter_signal_in_place_fixed_size::<8>(signal, initial, scale, delay),
            _ => {
                let mut z = self.initial_condition(initial);
                self.filter_signal_in_place_impl(
                    signal, &self.num, &self.den, &mut z, scale, delay,
                );
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
