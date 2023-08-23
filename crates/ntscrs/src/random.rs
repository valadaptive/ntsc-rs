use rand::distributions::Distribution;

pub struct Geometric {
    lambda: f64,
}

impl Geometric {
    pub fn new(p: f64) -> Self {
        if p <= 0.0 || p > 1.0 {
            panic!("Invalid probability");
        }

        Geometric {
            lambda: (1.0 - p).ln(),
        }
    }
}

impl Distribution<usize> for Geometric {
    // We can simulate a geometric distribution by taking the floor of an exponential distribution
    // https://en.wikipedia.org/wiki/Geometric_distribution#Related_distributions
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> usize {
        (rng.gen::<f64>().ln() / self.lambda) as usize
    }
}

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

pub trait FromSeeder {
    fn from_seeder(input: u64) -> Self;
}

impl FromSeeder for u64 {
    #[inline(always)]
    fn from_seeder(input: u64) -> Self {
        input
    }
}

impl FromSeeder for i32 {
    #[inline(always)]
    fn from_seeder(input: u64) -> Self {
        (input & (u32::MAX as u64)) as i32
    }
}

impl FromSeeder for f64 {
    #[inline(always)]
    fn from_seeder(input: u64) -> Self {
        f64::from_bits((input | 0x3FF0000000000000) & 0x3FFFFFFFFFFFFFFF) - 1.0
    }
}

impl FromSeeder for f32 {
    #[inline(always)]
    fn from_seeder(input: u64) -> Self {
        f32::from_bits(((input & 0xFFFFFFFF) as u32 >> 9) >> 9 | 0x3F800000) - 1.0
    }
}

#[derive(Clone, Copy)]
pub struct Seeder {
    state: u64,
}

impl Seeder {
    pub fn new<T: Mix>(seed: T) -> Self {
        Seeder {
            state: splitmix64(seed.mix()),
        }
    }

    pub fn mix<T: Mix>(mut self, input: T) -> Self {
        self.state = splitmix64(self.state) ^ input.mix();
        self
    }

    pub fn finalize<T: FromSeeder>(self) -> T {
        T::from_seeder(self.state)
    }
}

pub trait Mix {
    fn mix(&self) -> u64;
}

macro_rules! impl_mix_for {
    ($ty: tt) => {
        impl Mix for $ty {
            #[inline(always)]
            fn mix(&self) -> u64 {
                splitmix64(*self as u64)
            }
        }
    };
}

macro_rules! impl_mix_for_float {
    ($ty: tt) => {
        impl Mix for $ty {
            #[inline(always)]
            fn mix(&self) -> u64 {
                splitmix64(self.to_bits() as u64)
            }
        }
    };
}

impl_mix_for!(i8);
impl_mix_for!(u8);
impl_mix_for!(i16);
impl_mix_for!(u16);
impl_mix_for!(i32);
impl_mix_for!(u32);
impl_mix_for!(i64);
impl_mix_for!(u64);
impl_mix_for!(isize);
impl_mix_for!(usize);

impl_mix_for_float!(f32);
impl_mix_for_float!(f64);
