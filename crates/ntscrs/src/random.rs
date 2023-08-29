use std::hash::{Hasher, Hash};
use rand::distributions::Distribution;
use siphasher::sip::SipHasher;

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
        f32::from_bits(((input & 0xFFFFFFFF) as u32 >> 9) | 0x3F800000) - 1.0
    }
}

/// RNG seed generator which allows you to mix in as much of your own entropy as you want before generating the final
/// seed.
#[derive(Clone)]
pub struct Seeder {
    state: SipHasher,
}

impl Seeder {
    pub fn new<T: Hash>(seed: T) -> Self {
        let mut hasher = SipHasher::new_with_keys(0, 0);
        seed.hash(&mut hasher);
        Seeder {
            state: hasher,
        }
    }

    pub fn mix<T: Hash>(mut self, input: T) -> Self {
        input.hash(&mut self.state);
        self
    }

    pub fn finalize<T: FromSeeder>(self) -> T {
        T::from_seeder(self.state.finish())
    }
}
