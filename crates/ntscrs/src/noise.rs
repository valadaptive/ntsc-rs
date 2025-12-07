use clatter::{Simplex1d, Simplex2d};
use fearless_simd::{Level, Simd, SimdBase as _, SimdFloat, dispatch};

pub trait Noise<BaseNoise: Sampleable> {
    fn generate<S: Simd>(
        &self,
        point: <BaseNoise::Coords as ScalarCoords>::SimdCoords<S>,
    ) -> S::f32s;
}

pub struct Fbm {
    pub seed: i32,
    pub octaves: usize,
    pub gain: f32,
    pub lacunarity: f32,
    pub frequency: f32,
}

impl<BaseNoise: Sampleable> Noise<BaseNoise> for Fbm {
    #[inline(always)]
    fn generate<S: Simd>(
        &self,
        mut point: <BaseNoise::Coords as ScalarCoords>::SimdCoords<S>,
    ) -> <S as Simd>::f32s {
        point = BaseNoise::Coords::map(point, |x| x * self.frequency);
        let mut result = BaseNoise::sample(self.seed, point);

        let mut amplitude = self.gain;
        for _ in 1..self.octaves {
            point = BaseNoise::Coords::map(point, |x| x * self.lacunarity);
            result = BaseNoise::sample(self.seed, point).madd(amplitude, result);
            amplitude *= self.gain;
        }
        result
    }
}

pub struct Simplex {
    pub seed: i32,
    pub frequency: f32,
}
impl<BaseNoise: Sampleable> Noise<BaseNoise> for Simplex {
    #[inline(always)]
    fn generate<S: Simd>(
        &self,
        point: <BaseNoise::Coords as ScalarCoords>::SimdCoords<S>,
    ) -> <S as Simd>::f32s {
        BaseNoise::sample(
            self.seed,
            BaseNoise::Coords::map(point, |x| x * self.frequency),
        )
    }
}

#[inline(always)]
pub fn sample_noise_1d_inner<S: Simd, B: Sampleable<Coords = [f32; 1]>, N: Noise<B>>(
    simd: S,
    noise: &N,
    start_coords: [f32; 1],
    dimensions: [usize; 1],
    destination: &mut [f32],
) {
    assert_eq!(destination.len(), dimensions[0]);

    let mut chunks = destination.chunks_exact_mut(S::f32s::N);
    let mut coords = start_coords.splat_x(simd);
    for chunk in chunks.by_ref() {
        let noise = noise.generate::<S>(coords);
        chunk.copy_from_slice(noise.as_slice());
        coords[0] += S::f32s::N as f32;
    }
    let remainder = chunks.into_remainder();
    if !remainder.is_empty() {
        let noise = noise.generate::<S>(coords);
        remainder.copy_from_slice(&noise.as_slice()[..remainder.len()]);
    }
}

#[inline(always)]
pub fn add_noise_1d_inner<S: Simd, B: Sampleable<Coords = [f32; 1]>, N: Noise<B>>(
    simd: S,
    noise: &N,
    intensity: f32,
    start_coords: [f32; 1],
    dimensions: [usize; 1],
    destination: &mut [f32],
) {
    assert_eq!(destination.len(), dimensions[0]);

    let mut chunks = destination.chunks_exact_mut(S::f32s::N);
    let mut coords = start_coords.splat_x(simd);
    for chunk in chunks.by_ref() {
        let noise = noise.generate::<S>(coords);
        let chunk_f32s = S::f32s::from_slice(simd, chunk);
        chunk.copy_from_slice((chunk_f32s + (noise * intensity)).as_slice());
        coords[0] += S::f32s::N as f32;
    }
    let remainder = chunks.into_remainder();
    if !remainder.is_empty() {
        let noise = noise.generate::<S>(coords);
        let mut chunk_f32s = S::f32s::splat(simd, 0.0);
        for (i, &value) in remainder.iter().enumerate() {
            chunk_f32s[i] = value;
        }
        remainder
            .copy_from_slice(&(chunk_f32s + (noise * intensity)).as_slice()[..remainder.len()]);
    }
}

#[inline(always)]
pub fn sample_noise_2d_inner<S: Simd, B: Sampleable<Coords = [f32; 2]>, N: Noise<B>>(
    simd: S,
    noise: &N,
    mut start_coords: [f32; 2],
    dimensions: [usize; 2],
    destination: &mut [f32],
) {
    assert_eq!(destination.len(), dimensions[0] * dimensions[1]);

    for row in destination.chunks_exact_mut(dimensions[0]) {
        let mut chunks = row.chunks_exact_mut(S::f32s::N);
        let mut coords = start_coords.splat_x(simd);
        for chunk in chunks.by_ref() {
            let noise = noise.generate::<S>(coords);
            chunk.copy_from_slice(noise.as_slice());
            coords[0] += S::f32s::N as f32;
        }
        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            let noise = noise.generate::<S>(coords);
            remainder.copy_from_slice(&noise.as_slice()[..remainder.len()]);
        }
        start_coords[1] += 1.0;
    }
}

pub fn sample_noise_1d<B: Sampleable<Coords = [f32; 1]>, N: Noise<B>>(
    noise: &N,
    start_coords: [f32; 1],
    dimensions: [usize; 1],
    destination: &mut [f32],
) {
    let level = Level::new();
    dispatch!(level, simd => sample_noise_1d_inner(simd, noise, start_coords, dimensions, destination))
}

pub fn add_noise_1d<B: Sampleable<Coords = [f32; 1]>, N: Noise<B>>(
    noise: &N,
    intensity: f32,
    start_coords: [f32; 1],
    dimensions: [usize; 1],
    destination: &mut [f32],
) {
    let level = Level::new();
    dispatch!(level, simd => add_noise_1d_inner(simd, noise, intensity, start_coords, dimensions, destination))
}

pub fn sample_noise_2d<B: Sampleable<Coords = [f32; 2]>, N: Noise<B>>(
    noise: &N,
    start_coords: [f32; 2],
    dimensions: [usize; 2],
    destination: &mut [f32],
) {
    let level = Level::new();
    dispatch!(level, simd => sample_noise_2d_inner(simd, noise, start_coords, dimensions, destination))
}

pub trait ScalarCoords: Copy {
    type SimdCoords<S: Simd>: Copy;
    type Dimensions;
    fn splat_x<S: Simd>(self, simd: S) -> Self::SimdCoords<S>;
    fn map<S: Simd>(
        coords: Self::SimdCoords<S>,
        f: impl Fn(S::f32s) -> S::f32s,
    ) -> Self::SimdCoords<S>;
}

impl<const N: usize> ScalarCoords for [f32; N] {
    type SimdCoords<S: Simd> = [S::f32s; N];
    type Dimensions = [usize; N];

    #[inline(always)]
    fn splat_x<S: Simd>(self, simd: S) -> Self::SimdCoords<S> {
        let mut coords = self.map(|c| S::f32s::splat(simd, c));
        for i in 0..S::f32s::N {
            coords[0][i] += i as f32;
        }
        coords
    }

    #[inline(always)]
    fn map<S: Simd>(
        coords: Self::SimdCoords<S>,
        f: impl Fn(<S as Simd>::f32s) -> <S as Simd>::f32s,
    ) -> Self::SimdCoords<S> {
        coords.map(f)
    }
}

pub trait Sampleable {
    type Coords: ScalarCoords;
    fn sample<S: Simd>(seed: i32, coords: <Self::Coords as ScalarCoords>::SimdCoords<S>)
    -> S::f32s;
}

impl Sampleable for Simplex1d {
    type Coords = [f32; 1];

    #[inline(always)]
    fn sample<S: Simd>(
        seed: i32,
        coords: <Self::Coords as ScalarCoords>::SimdCoords<S>,
    ) -> <S as Simd>::f32s {
        Simplex1d::with_seed(seed).sample::<S>(coords).value
    }
}

impl Sampleable for Simplex2d {
    type Coords = [f32; 2];

    #[inline(always)]
    fn sample<S: Simd>(
        seed: i32,
        coords: <Self::Coords as ScalarCoords>::SimdCoords<S>,
    ) -> <S as Simd>::f32s {
        Simplex2d::with_seed(seed).sample::<S>(coords).value
    }
}
