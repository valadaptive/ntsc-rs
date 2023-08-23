// Multiplies two polynomials (lowest coefficients first).
pub fn polynomial_multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
    let degree = a.len() + b.len() - 1;

    let mut out = vec![0f64; degree];

    for ai in 0..a.len() {
        for bi in 0..b.len() {
            out[ai + bi] += a[ai] * b[bi];
        }
    }

    out
}

#[derive(Debug)]
pub struct TransferFunction {
    pub num: Vec<f64>,
    pub den: Vec<f64>,
    _private: (),
}

impl TransferFunction {
    pub fn new<I, J>(num: I, den: J) -> Self
    where
        I: IntoIterator<Item = f64>,
        J: IntoIterator<Item = f64>,
    {
        let mut num = num.into_iter().collect::<Vec<f64>>();
        let den = den.into_iter().collect::<Vec<f64>>();
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

    fn initial_condition(&self, value: f64) -> Vec<f64> {
        // Adapted from scipy
        // https://github.com/scipy/scipy/blob/da82ac849a4ccade2d954a0998067e6aa706dd70/scipy/signal/_signaltools.py#L3609-L3742

        let filter_len = usize::max(self.num.len(), self.den.len());
        let mut zi = vec![0f64; filter_len - 1];
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
            .collect::<Vec<f64>>();
        let norm_den = self
            .den
            .iter()
            .map(|item| *item / first_nonzero_coeff)
            .collect::<Vec<f64>>();

        let mut b_sum = 0.0;
        for i in 1..filter_len {
            let num_i = norm_num.get(i).unwrap_or(&0.0);
            let den_i = norm_den.get(i).unwrap_or(&0.0);
            b_sum += num_i - den_i * norm_num[0];
        }

        zi[0] = b_sum / norm_den.iter().sum::<f64>();
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
    fn filter_sample(filter_len: usize, num: &Vec<f64>, den: &Vec<f64>, z: &mut Vec<f64>, sample: f64, scale: f64) -> f64 {
            // Either the loop bound extending past items.len() or the min() call seems to prevent the optimizer from
            // determining that we're in-bounds here. Since i.min(items.len() - 1) never exceeds items.len() - 1 by
            // definition, this is safe.
            let filt_sample = z[0] + (num[0] * sample);
            for i in 0..filter_len - 2 {
                z[i] = z[i + 1] + (num[i + 1] * sample) - (den[i + 1] * filt_sample);
            }
            if filter_len > 1 {
                z[filter_len - 2] = (num[filter_len - 1] * sample)
                    - (den[filter_len - 1] * filt_sample);
            }
            (filt_sample - sample) * scale + sample
    }

    pub fn filter_signal<'a, I>(&self, items: I) -> Vec<f64>
    where
        I: IntoIterator<Item = &'a f64>,
    {
        let mut yout = items.into_iter().cloned().collect::<Vec<f64>>();
        self.filter_signal_in_place(&mut yout, 0.0, 1.0, 0);
        yout
    }

    pub fn filter_signal_into(
        &self,
        src: &[f64],
        dst: &mut [f64],
        initial: f64,
        scale: f64,
        delay: usize,
    ) {
        if dst.len() < src.len() {
            panic!("Destination array not big enough");
        }

        let filter_len = usize::max(self.num.len(), self.den.len());
        let mut z = self.initial_condition(initial);

        for i in 0..(src.len() + delay) {
            // Either the loop bound extending past items.len() or the min() call seems to prevent the optimizer from
            // determining that we're in-bounds here. Since i.min(items.len() - 1) never exceeds items.len() - 1 by
            // definition, this is safe.
            let sample = unsafe { src.get_unchecked(i.min(src.len() - 1)) };
            let filt_sample = Self::filter_sample(filter_len, &self.num, &self.den, &mut z, *sample, scale);
            if i >= delay {
                dst[i - delay] = filt_sample;
            }
        }
    }

    pub fn filter_signal_in_place(
        &self,
        items: &mut [f64],
        initial: f64,
        scale: f64,
        delay: usize,
    ) {
        let filter_len = usize::max(self.num.len(), self.den.len());
        let mut z = self.initial_condition(initial);

        for i in 0..(items.len() + delay) {
            // Either the loop bound extending past items.len() or the min() call seems to prevent the optimizer from
            // determining that we're in-bounds here. Since i.min(items.len() - 1) never exceeds items.len() - 1 by
            // definition, this is safe.
            let sample = unsafe { items.get_unchecked(i.min(items.len() - 1)) };
            let filt_sample = Self::filter_sample(filter_len, &self.num, &self.den, &mut z, *sample, scale);
            if i >= delay {
                items[i - delay] = filt_sample;
            }
        }
    }
}

fn trim_zeros(input: &[f64]) -> &[f64] {
    let mut end = input.len() - 1;
    while input[end].abs() == 0.0 {
        end -= 1;
    }
    &input[0..=end]
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
