use std::hash::Hasher;

use rand::distributions::Distribution;
use siphasher::sip::SipHasher;

pub fn key_seed(seed: u64, key: u64, frame_num: usize) -> u64 {
    let mut hasher = SipHasher::new_with_keys(seed, key);
    hasher.write_u64(frame_num as u64);
    hasher.finish()
}

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

pub fn seek_rng(seed: u64, index: u64) -> u64 {
    splitmix64(splitmix64(seed)) ^ splitmix64(index)
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

impl FromSeeder for f64 {
    #[inline(always)]
    fn from_seeder(input: u64) -> Self {
        f64::from_bits((input | 0x3FF0000000000000u64) & 0x3FFFFFFFFFFFFFFFu64) - 1.0
    }
}

pub struct Seeder {
    state: u64,
}

// TODO: replace key_seed fully
impl Seeder {
    pub fn new(seed: u64) -> Self {
        Seeder {
            state: splitmix64(seed),
        }
    }

    pub fn mix_u64(mut self, input: u64) -> Self {
        self.state = splitmix64(self.state) ^ splitmix64(input);
        self
    }

    pub fn finalize<T: FromSeeder>(self) -> T {
        T::from_seeder(self.state)
    }
}
