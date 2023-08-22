// Multiplies two polynomials (lowest coefficients first).
pub fn polynomial_multiply(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
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
    _private: ()
}

impl TransferFunction {
    pub fn new<I, J>(num: I, den: J) -> Self
    where
        I: IntoIterator<Item = f64>,
        J: IntoIterator<Item = f64>,
    {
        let num = num.into_iter().collect::<Vec<f64>>();
        let den = den.into_iter().collect::<Vec<f64>>();
        if num.len() > den.len() {
            panic!("Numerator length exceeds denominator length.");
        }

        TransferFunction {
            num,
            den,
            _private: ()
        }
    }

    pub fn steady_state_condition(&self, value: f64) -> Vec<f64> {
        // Adapted from scipy
        // https://github.com/scipy/scipy/blob/da82ac849a4ccade2d954a0998067e6aa706dd70/scipy/signal/_signaltools.py#L3609-L3742

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

        let norm_num = self.num.iter().map(|item| {
            *item / first_nonzero_coeff
        }).collect::<Vec<f64>>();
        let norm_den = self.den.iter().map(|item| {
            *item / first_nonzero_coeff
        }).collect::<Vec<f64>>();

        let filter_len = usize::max(self.num.len(), self.den.len());

        let mut b_sum = 0.0;
        for i in 1..filter_len {
            let num_i = norm_num.get(i).unwrap_or(&0.0);
            let den_i = norm_den.get(i).unwrap_or(&0.0);
            b_sum += num_i - den_i * norm_num[0];
        }

        let mut zi = vec![0f64; filter_len - 1];
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

        zi
    }

    pub fn filter_signal<'a, I>(&self, items: I) -> Vec<f64>
    where
        I: IntoIterator<Item = &'a f64>,
    {
        let mut yout = items.into_iter().cloned().collect::<Vec<f64>>();
        self.filter_signal_in_place(&mut yout, 0.0, 1.0, 0);
        yout
    }

    pub fn filter_signal_in_place(&self, items: &mut [f64], initial: f64, scale: f64, delay: usize) {
        // Zero-pad the numerator from the right
        let num_padded = {
            let mut num = self.num.clone();
            num.resize(self.den.len(), 0.0);
            num
        };
        let filter_len = usize::max(self.num.len(), self.den.len());

        let mut z = if initial.abs() == 0.0 {
            vec![0f64; filter_len - 1]
        } else {
            self.steady_state_condition(initial)
        };

        for i in 0..items.len() {
            let sample = items[i];
            let filt_sample = z[0] + (num_padded[0] * sample);
            for i in 0..filter_len - 2 {
                z[i] = z[i + 1] + (num_padded[i + 1] * sample) - (self.den[i + 1] * filt_sample);
            }
            z[filter_len - 2] =
                (num_padded[filter_len - 1] * sample) - (self.den[filter_len - 1] * filt_sample);
            if i >= delay {
                items[i - delay] = (filt_sample - sample) * scale + sample;
            }
        }
    }
}

impl std::ops::Mul<&TransferFunction> for &TransferFunction {
    type Output = TransferFunction;

    fn mul(self, rhs: &TransferFunction) -> Self::Output {
        TransferFunction {
            num: polynomial_multiply(&self.num, &rhs.num),
            den: polynomial_multiply(&self.den, &rhs.den),
            _private: ()
        }
    }
}
