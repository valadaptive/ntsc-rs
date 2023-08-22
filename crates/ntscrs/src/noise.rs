use crate::random::Seeder;

const JITTER_SEED: u64 = 1;
const VALUE_SEED: u64 = 2;

/// Sample jittered value noise at a given time. Useful for varying things over time.
/// `t` is the time to sample at. The value noise has a frequency of 1 noise value per 1.0t.
/// `jitter` is the amount to move around the points which are interpolated to make value noise.
/// This is useful for avoiding periodic artifacts.
/// `seed` is the random seed.
pub fn sample_noise(t: f64, jitter: f64, seed: u64) -> f64 {
    let left_coord = t as u64;
    let cellspace_coord = t.fract();

    let mut left_jitter = (Seeder::new(left_coord)
        .mix_u64(seed)
        .mix_u64(JITTER_SEED)
        .finalize::<f64>()
        - 0.5)
        * jitter;
    let mut right_jitter = (Seeder::new(left_coord.wrapping_add(1))
        .mix_u64(seed)
        .mix_u64(JITTER_SEED)
        .finalize::<f64>()
        - 0.5)
        * jitter;

    let (dist_offset, rand_coord) = if cellspace_coord < left_jitter {
        right_jitter = left_jitter;
        left_jitter = (Seeder::new(left_coord.wrapping_sub(1))
            .mix_u64(seed)
            .mix_u64(JITTER_SEED)
            .finalize::<f64>()
            - 0.5)
            * jitter;
        (-1.0, left_coord.wrapping_sub(1))
    } else if cellspace_coord > right_jitter + 1.0 {
        left_jitter = right_jitter;
        right_jitter = (Seeder::new(left_coord.wrapping_add(2))
            .mix_u64(seed)
            .mix_u64(JITTER_SEED)
            .finalize::<f64>()
            - 0.5)
            * jitter;
        (1.0, left_coord.wrapping_add(1))
    } else {
        (0.0, left_coord)
    };
    let mut dist =
        (cellspace_coord - (left_jitter + dist_offset)) / (right_jitter + 1.0 - left_jitter);
    let left_rand: f64 = Seeder::new(rand_coord)
        .mix_u64(seed)
        .mix_u64(VALUE_SEED)
        .finalize();
    let right_rand: f64 = Seeder::new(rand_coord.wrapping_add(1))
        .mix_u64(seed)
        .mix_u64(VALUE_SEED)
        .finalize();

    // Smoothstep
    dist = dist * dist * (3.0 - 2.0 * dist);

    (left_rand * (1.0 - dist)) + (right_rand * dist)
}
