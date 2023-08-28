use std::collections::VecDeque;

use core::f32::consts::PI;
use glam::{Mat3, Vec3};
use image::RgbImage;
use rand::{Rng, RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use simdnoise::NoiseBuilder;

use crate::{
    filter::TransferFunction,
    random::{Geometric, Seeder},
    shift::{shift_row, shift_row_to, BoundaryHandling},
};

pub use crate::settings::*;

const YIQ_MATRIX: Mat3 = Mat3 {
    x_axis: Vec3 {
        x: 0.299,
        y: 0.5959,
        z: 0.2115,
    },
    y_axis: Vec3 {
        x: 0.587,
        y: -0.2746,
        z: -0.5227,
    },
    z_axis: Vec3 {
        x: 0.114,
        y: -0.3213,
        z: 0.3112,
    },
};

const RGB_MATRIX: Mat3 = Mat3 {
    x_axis: Vec3 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    },
    y_axis: Vec3 {
        x: 0.956,
        y: -0.272,
        z: -1.106,
    },
    z_axis: Vec3 {
        x: 0.619,
        y: -0.647,
        z: 1.703,
    },
};

// 315/88 Mhz rate * 4
// TODO: why do we multiply by 4? composite-video-simulator does this for every filter and ntscqt defines NTSC_RATE the
// same way as we do here.
const NTSC_RATE: f32 = (315000000.00 / 88.0) * 4.0;

/// Create a simple constant-k lowpass filter with the given frequency cutoff, which can then be used to filter a signal.
pub fn make_lowpass(cutoff: f32, rate: f32) -> TransferFunction {
    let time_interval = 1.0 / rate;
    let tau = (cutoff * 2.0 * PI).recip();
    let alpha = time_interval / (tau + time_interval);

    TransferFunction::new(vec![alpha], vec![1.0, -(1.0 - alpha)])
}

/// Simulate three constant-k lowpass filters in a row by multiplying the coefficients. The original code
/// (composite-video-simulator and ntscqt) applies a lowpass filter 3 times in a row, but it's more efficient to
/// multiply the coefficients and just apply the filter once, which is mathematically equivalent.
pub fn make_lowpass_triple(cutoff: f32, rate: f32) -> TransferFunction {
    let time_interval = 1.0 / rate;
    let tau = (cutoff * 2.0 * PI).recip();
    let alpha = time_interval / (tau + time_interval);

    let tf = TransferFunction::new(vec![alpha], vec![1.0, -(1.0 - alpha)]);

    &(&tf * &tf) * &tf
}

/// Create an IIR notch filter.
pub fn make_notch_filter(freq: f32, quality: f32) -> TransferFunction {
    // Adapted from scipy and simplified
    // https://github.com/scipy/scipy/blob/686422c4f0a71be1b4258309590fd3e9de102e18/scipy/signal/_filter_design.py#L5099-L5171
    if freq > 1.0 || freq < 0.0 {
        panic!("Frequency outside valid range");
    }

    let bandwidth = (freq / quality) * PI;
    let freq = freq * PI;

    let beta = (bandwidth * 0.5).tan();

    let gain = (1.0 + beta).recip();

    let num = vec![gain, -2.0 * freq.cos() * gain, gain];
    let den = vec![1.0, -2.0 * freq.cos() * gain, 2.0 * gain - 1.0];

    TransferFunction::new(num, den)
}

/// Filter initial condition.
enum InitialCondition {
    /// Convenience value--just use 0.
    Zero,
    /// Set the initial filter condition to a constant.
    Constant(f32),
    /// Set the initial filter condition to that of the first sample to be filtered.
    FirstSample,
}

/// Apply a given IIR filter to each row of one color plane.
/// # Arguments
/// - `plane` - The entire data (width * height) for the plane to be filtered.
/// - `width` - The width of each row.
/// - `filter` - The filter to apply to the plane.
/// - `initial` - The initial steady-state value of the filter.
/// - `scale` - Scale the effect of the filter on the signal by this amount.
/// - `delay` - Offset the filter output backwards (to the left) by this amount.
fn filter_plane(
    plane: &mut [f32],
    width: usize,
    filter: &TransferFunction,
    initial: InitialCondition,
    scale: f32,
    delay: usize,
) {
    plane.chunks_mut(width).for_each(|field| {
        let initial = match initial {
            InitialCondition::Zero => 0.0,
            InitialCondition::Constant(c) => c,
            InitialCondition::FirstSample => field[0],
        };
        filter.filter_signal_in_place(field, initial, scale, delay);
    });
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum YiqField {
    Upper,
    Lower,
    Both,
}

/// Borrowed YIQ data in a planar format.
/// Each plane is densely packed with regards to rows--if we skip fields, we just leave them out of these planes, which
/// squashes them vertically.
pub struct YiqView<'a> {
    pub y: &'a mut [f32],
    pub i: &'a mut [f32],
    pub q: &'a mut [f32],
    /// This refers to the number of rendered rows. For instance, if the input frame is 480 pixels high but we're only
    /// doing the effect on even-numbered fields, then resolution.1 will be 240.
    pub resolution: (usize, usize),
    /// The source field that this data is for.
    pub field: YiqField,
}

/// Owned YIQ data.
pub struct YiqOwned {
    /// Densely-packed planar YUV data. The Y plane comes first in memory, then I, then Q.
    data: Box<[f32]>,
    /// This refers to the number of rendered rows. For instance, if the input frame is 480 pixels high but we're only
    /// doing the effect on even-numbered fields, then resolution.1 will be 240.
    resolution: (usize, usize),
    /// The source field that this data is for.
    field: YiqField,
}

impl YiqOwned {
    pub fn from_image(image: &RgbImage, field: YiqField) -> YiqOwned {
        let width = image.width() as usize;
        let height = match field {
            YiqField::Upper | YiqField::Lower => (image.height() + 1) / 2,
            YiqField::Both => image.height(),
        } as usize;

        // We write into the destination array differently depending on whether we're using the upper field, lower
        // field, or both. row_lshift determines whether we left-shift the source row index (doubling it). When we use
        // only one of the fields, the source row index needs to be double the destination row index so we take every
        // other row. When we use both fields, we just use the source row index as-is.
        // The row_offset determines whether we skip the first row (when using the lower field).
        let (row_lshift, row_offset): (usize, usize) = match field {
            YiqField::Upper => (1, 0),
            YiqField::Lower => (1, 1),
            YiqField::Both => (0, 0),
        };

        let num_pixels = width * height;

        let mut data = vec![0f32; num_pixels * 3];
        let (y, iq) = data.split_at_mut(num_pixels);
        let (i, q) = iq.split_at_mut(num_pixels);

        let src_data = image.as_raw();

        y.chunks_mut(width)
            .zip(i.chunks_mut(width).zip(q.chunks_mut(width)))
            .enumerate()
            .for_each(|(row_idx, (y, (i, q)))| {
                let src_row_idx = (row_idx << row_lshift) + row_offset;
                let src_offset = src_row_idx * width;
                for pixel_idx in 0..width {
                    let yiq_pixel = YIQ_MATRIX
                        * Vec3::new(
                            (src_data[((pixel_idx + src_offset) * 3) + 0] as f32) / 255.0,
                            (src_data[((pixel_idx + src_offset) * 3) + 1] as f32) / 255.0,
                            (src_data[((pixel_idx + src_offset) * 3) + 2] as f32) / 255.0,
                        );
                    y[pixel_idx] = yiq_pixel[0];
                    i[pixel_idx] = yiq_pixel[1];
                    q[pixel_idx] = yiq_pixel[2];
                }
            });

        YiqOwned {
            data: data.into_boxed_slice(),
            resolution: (width, height),
            field,
        }
    }
}

impl<'a> From<&'a mut YiqOwned> for YiqView<'a> {
    fn from(value: &'a mut YiqOwned) -> Self {
        let num_pixels = value.resolution.0 * value.resolution.1;
        let (y, iq) = value.data.split_at_mut(num_pixels);
        let (i, q) = iq.split_at_mut(num_pixels);
        YiqView { y, i, q, resolution: value.resolution, field: value.field }
    }
}

impl From<&YiqView<'_>> for RgbImage {
    fn from(image: &YiqView) -> Self {
        let width = image.resolution.0;
        let output_height = image.resolution.1 * if image.field == YiqField::Both { 1 } else { 2 };
        let num_pixels = width * output_height;
        let mut dst = vec![0u8; num_pixels * 3];

        // If the row index modulo 2 equals this number, that row was not rendered in the source data and we need to
        // interpolate between the rows above and beneath it.
        let skip_field: usize = match image.field {
            YiqField::Upper => 1,
            YiqField::Lower => 0,
            // The row index modulo 2 never reaches 2, meaning we don't skip any rows
            YiqField::Both => 2,
        };

        let row_rshift = match image.field {
            YiqField::Both => 0,
            YiqField::Upper | YiqField::Lower => 1,
        };

        dst.chunks_mut(width * 3)
            .enumerate()
            .for_each(|(row_idx, dst_row)| {
                // Inner fields with lines above and below them. Interpolate between those fields
                if (row_idx & 1) == skip_field && row_idx != 0 && row_idx != output_height - 1 {
                    for (pix_idx, pixel) in dst_row.chunks_mut(3).enumerate() {
                        let src_idx_lower = ((row_idx - 1) >> 1) * width + pix_idx;
                        let src_idx_upper = ((row_idx + 1) >> 1) * width + pix_idx;

                        let interp_pixel = Vec3::new(
                            (image.y[src_idx_lower] + image.y[src_idx_upper]) * 0.5,
                            (image.i[src_idx_lower] + image.i[src_idx_upper]) * 0.5,
                            (image.q[src_idx_lower] + image.q[src_idx_upper]) * 0.5,
                        );

                        let rgb = RGB_MATRIX * interp_pixel;
                        pixel[0] = (rgb[0] * 255.0).clamp(0.0, 255.0) as u8;
                        pixel[1] = (rgb[1] * 255.0).clamp(0.0, 255.0) as u8;
                        pixel[2] = (rgb[2] * 255.0).clamp(0.0, 255.0) as u8;
                    }
                } else {
                    // Copy the field directly
                    for (pix_idx, pixel) in dst_row.chunks_mut(3).enumerate() {
                        let src_idx = (row_idx >> row_rshift) * width + pix_idx;
                        let rgb = RGB_MATRIX
                            * Vec3::new(image.y[src_idx], image.i[src_idx], image.q[src_idx]);
                        pixel[0] = (rgb[0] * 255.0).clamp(0.0, 255.0) as u8;
                        pixel[1] = (rgb[1] * 255.0).clamp(0.0, 255.0) as u8;
                        pixel[2] = (rgb[2] * 255.0).clamp(0.0, 255.0) as u8;
                    }
                }
            });

        RgbImage::from_raw(width as u32, output_height as u32, dst).unwrap()
    }
}

/// Apply a lowpass filter to the input chroma, emulating broadcast NTSC's bandwidth cutoffs.
/// (Well, almost--Wikipedia (https://en.wikipedia.org/wiki/YIQ) puts the Q bandwidth at 0.4 MHz, not 0.6. Although
/// that statement seems unsourced and I can't find any info on it...
fn composite_chroma_lowpass(frame: &mut YiqView) {
    let i_filter = make_lowpass_triple(1300000.0, NTSC_RATE);
    let q_filter = make_lowpass_triple(600000.0, NTSC_RATE);

    let width = frame.resolution.0;

    filter_plane(
        &mut frame.i,
        width,
        &i_filter,
        InitialCondition::Zero,
        1.0,
        2,
    );
    filter_plane(
        &mut frame.q,
        width,
        &q_filter,
        InitialCondition::Zero,
        1.0,
        4,
    );
}

/// Apply a less intense lowpass filter to the input chroma.
fn composite_chroma_lowpass_lite(frame: &mut YiqView) {
    let filter = make_lowpass_triple(2600000.0, NTSC_RATE);

    let width = frame.resolution.0;

    filter_plane(&mut frame.i, width, &filter, InitialCondition::Zero, 1.0, 1);
    filter_plane(&mut frame.q, width, &filter, InitialCondition::Zero, 1.0, 1);
}

/// Calculate the chroma subcarrier phase for a given row/field
fn chroma_phase_offset(
    scanline_phase_shift: PhaseShift,
    offset: i32,
    frame_num: usize,
    line_num: usize,
) -> usize {
    (match scanline_phase_shift {
        PhaseShift::Degrees90 | PhaseShift::Degrees270 => {
            (frame_num as i32 + offset + ((line_num as i32) >> 1)) & 3
        }
        PhaseShift::Degrees180 => (((frame_num + line_num) & 2) as i32 + offset) & 3,
        PhaseShift::Degrees0 => 0,
    } & 3) as usize
}

const I_MULT: [f32; 4] = [1.0, 0.0, -1.0, 0.0];
const Q_MULT: [f32; 4] = [0.0, 1.0, 0.0, -1.0];

fn chroma_into_luma_line(
    y: &mut [f32],
    i: &mut [f32],
    q: &mut [f32],
    xi: usize,
    subcarrier_amplitude: f32,
) {
    y.into_iter()
        .zip(i.into_iter().zip(q.into_iter()))
        .enumerate()
        .for_each(|(index, (y, (i, q)))| {
            let phase = (index + (xi & 3)) & 3;
            *y += (*i * I_MULT[phase] + *q * Q_MULT[phase]) * subcarrier_amplitude / 50.0;
            // *i = 0.0;
            // *q = 0.0;
        });
}

/// Modulate the chrominance signal (I and Q planes) into the Y (luminance) plane.
fn chroma_into_luma(
    phase_shift: PhaseShift,
    phase_offset: i32,
    yiq: &mut YiqView,
    subcarrier_amplitude: f32,
    frame_num: usize,
) {
    let width = yiq.resolution.0;

    let y_lines = yiq.y.chunks_mut(width);
    let i_lines = yiq.i.chunks_mut(width);
    let q_lines = yiq.q.chunks_mut(width);

    y_lines
        .zip(i_lines.zip(q_lines))
        .enumerate()
        .for_each(|(index, (y, (i, q)))| {
            let xi = chroma_phase_offset(phase_shift, phase_offset, frame_num, index * 2);

            chroma_into_luma_line(y, i, q, xi, subcarrier_amplitude);
        });
}

fn luma_into_chroma_line(
    y: &mut [f32],
    i: &mut [f32],
    q: &mut [f32],
    xi: usize,
    subcarrier_amplitude: f32,
) {
    let mut delay = VecDeque::<f32>::with_capacity(4);
    delay.push_back(16.0 / 255.0);
    delay.push_back(16.0 / 255.0);
    delay.push_back(y[0]);
    delay.push_back(y[1]);
    let mut sum: f32 = delay.iter().sum();
    let width = y.len();

    for index in 0..width {
        // Box-blur the signal to get the luminance.
        // TODO: add an option to use a real lowpass filter or notch filter here?
        let c = y[usize::min(index + 2, width - 1)];
        sum -= delay.pop_front().unwrap();
        delay.push_back(c);
        sum += c;
        y[index] = sum * 0.25;
        let mut chroma = c - y[index];

        chroma = (chroma * 50.0) / subcarrier_amplitude;

        let offset = (index + (xi & 3)) & 3;

        let i_modulated = -(chroma * I_MULT[offset]);
        let q_modulated = -(chroma * Q_MULT[offset]);

        // TODO: ntscQT seems to mess this up, giving chroma a "jagged" look reminiscent of the "dot crawl" artifact.
        // Is that worth trying to replicate, or should it just be left like this?
        if index < width - 1 {
            i[index + 1] = i_modulated * 0.5;
            q[index + 1] = q_modulated * 0.5;
        }
        i[index] += i_modulated;
        q[index] += q_modulated;
        if index > 0 {
            i[index - 1] += i_modulated * 0.5;
            q[index - 1] += q_modulated * 0.5;
        }
    }
}

/// Demodulate the chroma signal from the Y (luma) plane back into the I and Q planes.
fn luma_into_chroma(
    phase_shift: PhaseShift,
    phase_offset: i32,
    yiq: &mut YiqView,
    subcarrier_amplitude: f32,
    frame_num: usize,
) {
    let width = yiq.resolution.0;

    let y_lines = yiq.y.chunks_mut(width);
    let i_lines = yiq.i.chunks_mut(width);
    let q_lines = yiq.q.chunks_mut(width);

    y_lines
        .zip(i_lines.zip(q_lines))
        .enumerate()
        .for_each(|(index, (y, (i, q)))| {
            let xi = chroma_phase_offset(phase_shift, phase_offset, frame_num, index * 2);

            luma_into_chroma_line(y, i, q, xi, subcarrier_amplitude);
        });
}

/// We use a seeded RNG to generate random noise deterministically, but we don't want every pass which uses noise to use
/// the *same* noise. Each pass gets its own random seed which is mixed into the RNG.
mod noise_seeds {
    pub const VIDEO_COMPOSITE: u64 = 0;
    pub const VIDEO_CHROMA: u64 = 1;
    pub const HEAD_SWITCHING: u64 = 2;
    pub const TRACKING_NOISE: u64 = 3;
    pub const VIDEO_CHROMA_PHASE: u64 = 4;
    pub const EDGE_WAVE: u64 = 5;
    pub const SNOW: u64 = 6;
    pub const CHROMA_LOSS: u64 = 7;
}

/// Helper function to apply gradient noise to a single row of a single plane.
fn video_noise_line(row: &mut [f32], seeder: Seeder, index: usize, frequency: f32, intensity: f32) {
    let width = row.len();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seeder.mix(index as u64).finalize());
    let noise_seed = rng.next_u32();
    let offset = rng.gen::<f32>() * width as f32;

    let noise = NoiseBuilder::gradient_1d_offset(offset, width)
        .with_seed(noise_seed as i32)
        .with_freq(frequency)
        .generate()
        .0;

    row.iter_mut().enumerate().for_each(|(x, pixel)| {
        *pixel += noise[x] as f32 * 0.25 * intensity;
    });
}

/// Add noise to an NTSC-encoded signal.
fn composite_noise(
    yiq: &mut YiqView,
    seed: u64,
    frequency: f32,
    intensity: f32,
    frame_num: usize,
) {
    let width = yiq.resolution.0;
    let seeder = Seeder::new(seed)
        .mix(noise_seeds::VIDEO_COMPOSITE)
        .mix(frame_num);

    yiq.y
        .chunks_mut(width)
        .enumerate()
        .for_each(|(index, row)| {
            video_noise_line(row, seeder, index, frequency, intensity);
        });
}

/// Add noise to the chrominance (I and Q) planes of a de-modulated signal.
fn chroma_noise(yiq: &mut YiqView, seed: u64, frequency: f32, intensity: f32, frame_num: usize) {
    let width = yiq.resolution.0;
    let seeder = Seeder::new(seed)
        .mix(noise_seeds::VIDEO_CHROMA)
        .mix(frame_num);

    yiq.i
        .chunks_mut(width)
        .zip(yiq.q.chunks_mut(width))
        .enumerate()
        .for_each(|(index, (i, q))| {
            video_noise_line(i, seeder, index, frequency, intensity);
            video_noise_line(q, seeder, index, frequency, intensity);
        });
}

/// Add per-scanline chroma phase error.
fn chroma_phase_noise(yiq: &mut YiqView, seed: u64, intensity: f32, frame_num: usize) {
    let width = yiq.resolution.0;
    let seeder = Seeder::new(seed)
        .mix(noise_seeds::VIDEO_CHROMA_PHASE)
        .mix(frame_num);

    yiq.i
        .chunks_mut(width)
        .zip(yiq.q.chunks_mut(width))
        .enumerate()
        .for_each(|(index, (i, q))| {
            // Phase shift angle in radians. Mapped so that an intensity of 1.0 is a phase shift ranging from a full
            // rotation to the left - a full rotation to the right.
            let phase_shift = (seeder.mix(index).finalize::<f32>() - 0.5) * PI * 4.0 * intensity;
            let (sin_angle, cos_angle) = phase_shift.sin_cos();

            for (i, q) in i.iter_mut().zip(q.iter_mut()) {
                // Treat (i, q) as a 2D vector and rotate it by the phase shift amount.
                let rotated_i = (*i * cos_angle) - (*q * sin_angle);
                let rotated_q = (*i * sin_angle) + (*q * cos_angle);

                *i = rotated_i;
                *q = rotated_q;
            }
        });
}

/// Emulate VHS head-switching at the bottom of the image.
fn head_switching(
    yiq: &mut YiqView,
    num_rows: usize,
    offset: usize,
    shift: f32,
    seed: u64,
    frame_num: usize,
) {
    let (width, height) = yiq.resolution;
    if offset > num_rows {
        return;
    }
    let num_affected_rows = num_rows - offset;

    let start_row = height - num_affected_rows;
    let affected_rows = &mut yiq.y[start_row * width..];

    let seeder = Seeder::new(seed)
        .mix(noise_seeds::HEAD_SWITCHING)
        .mix(frame_num);

    affected_rows
        .chunks_mut(width)
        .enumerate()
        .for_each(|(index, row)| {
            let index = num_affected_rows - index;
            let row_shift = shift * ((index + offset) as f32 / num_rows as f32).powf(1.5);
            shift_row(
                row,
                row_shift + (seeder.mix(index).finalize::<f32>() - 0.5),
                BoundaryHandling::Constant(0.0),
            );
        });
}

/// Helper function for generating "snow".
fn row_speckles<R: Rng>(row: &mut [f32], rng: &mut R, intensity: f32) {
    if intensity <= 0.0 {
        return;
    }
    // Turn each pixel into "snow" with probability snow_intensity * intensity_scale
    // We can simulate the distance between each "snow" pixel with a geometric distribution which avoids having to
    // loop over every pixel
    let dist = Geometric::new(intensity);
    let mut pixel_idx = 0usize;
    loop {
        pixel_idx += rng.sample(&dist);
        if pixel_idx >= row.len() {
            break;
        }

        let transient_len: f32 = rng.gen_range(8.0..=64.0);

        for i in pixel_idx..(pixel_idx + transient_len.ceil() as usize).min(row.len()) {
            let x = (i - pixel_idx) as f32;
            // Quadratic decay to 0
            row[i] += (1.0 - (x / transient_len)).powi(2) * 2.0 * rng.gen::<f32>();
        }

        // Make sure we advance the pixel index each time. Our geometric distribution gives us the time between
        // successive events, which can be 0 for very high probabilities.
        pixel_idx += 1;
    }
}

/// Emulate VHS tracking error/noise.
/// TODO: this is inaccurate and doesn't let you control the position of the noise.
/// I need to revamp this at some point.
fn tracking_noise(
    yiq: &mut YiqView,
    seed: u64,
    num_rows: usize,
    wave_intensity: f32,
    snow_intensity: f32,
    noise_intensity: f32,
    frame_num: usize,
) {
    let (width, height) = yiq.resolution;

    let seeder = Seeder::new(seed)
        .mix(noise_seeds::TRACKING_NOISE)
        .mix(frame_num);
    let noise_seed = seeder.finalize::<i32>();
    let offset = seeder.mix(1).finalize::<f32>() * yiq.resolution.1 as f32;
    let shift_noise = NoiseBuilder::gradient_1d_offset(offset, num_rows)
        .with_seed(noise_seed)
        .with_freq(0.5)
        .generate()
        .0;

    let start_row = height - num_rows;
    let affected_rows = &mut yiq.y[start_row * width..];

    affected_rows
        .chunks_mut(width)
        .enumerate()
        .for_each(|(index, row)| {
            // This iterates from the top down. Increase the intensity as we approach the bottom of the picture.
            let intensity_scale = index as f32 / num_rows as f32;
            shift_row(
                row,
                shift_noise[index] * intensity_scale * wave_intensity * 0.25,
                BoundaryHandling::Constant(0.0),
            );

            video_noise_line(
                row,
                seeder,
                index,
                0.25,
                intensity_scale.powi(2) * noise_intensity * 4.0,
            );

            // Turn each pixel into "snow" with probability snow_intensity * intensity_scale
            // We can simulate the distance between each "snow" pixel with a geometric distribution which avoids having to
            // loop over every pixel
            row_speckles(
                row,
                &mut Xoshiro256PlusPlus::seed_from_u64(seeder.mix(index).finalize()),
                snow_intensity * intensity_scale.powi(2),
            );
        });
}

/// Add random bits of "snow" to an NTSC-encoded signal.
fn snow(yiq: &mut YiqView, seed: u64, intensity: f32, frame_num: usize) {
    let seeder = Seeder::new(seed).mix(noise_seeds::SNOW).mix(frame_num);

    yiq.y
        .chunks_mut(yiq.resolution.0)
        .enumerate()
        .for_each(|(index, row)| {
            // Turn each pixel into "snow" with probability snow_intensity * intensity_scale
            // We can simulate the distance between each "snow" pixel with a geometric distribution which avoids having to
            // loop over every pixel
            row_speckles(
                row,
                &mut Xoshiro256PlusPlus::seed_from_u64(seeder.mix(index).finalize()),
                intensity,
            );
        });
}

/// Offset the chrominance (I and Q) planes horizontally and/or vertically.
/// Note how the horizontal shift is a float (the signal is continuous), but the vertical shift is an int (each scanline
/// is discrete).
fn chroma_delay(yiq: &mut YiqView, offset: (f32, isize)) {
    let copy_or_shift = |src: &mut [f32], dst: &mut [f32]| {
        if offset.0.abs() == 0.0 {
            dst.copy_from_slice(src);
        } else {
            shift_row_to(src, dst, offset.0, BoundaryHandling::Constant(0.0));
        }
    };

    let (width, height) = yiq.resolution;

    if offset.1 == 0 {
        // Only a horizontal shift is necessary. We can do this in-place easily.
        yiq.i
            .chunks_mut(width)
            .zip(yiq.q.chunks_mut(width))
            .for_each(|(i, q)| {
                shift_row(i, offset.0, BoundaryHandling::Constant(0.0));
                shift_row(q, offset.0, BoundaryHandling::Constant(0.0));
            });
    } else if offset.1 > 0 {
        // Some finagling is required to shift vertically. This branch shifts the chroma planes downwards.
        let offset = offset.1 as usize;
        // Starting from the bottom, copy (or write a horizontally-shifted copy of) each row downwards.
        for dst_row_idx in (0..height).rev() {
            let (i_src_part, i_dst_part) = yiq.i.split_at_mut(dst_row_idx * width);
            let (q_src_part, q_dst_part) = yiq.q.split_at_mut(dst_row_idx * width);

            let dst_row_range = 0..width;
            let dst_i = &mut i_dst_part[dst_row_range.clone()];
            let dst_q = &mut q_dst_part[dst_row_range.clone()];

            if dst_row_idx < offset {
                dst_i.fill(0.0);
                dst_q.fill(0.0);
            } else {
                let src_row_idx = dst_row_idx - offset;
                let src_row_range = src_row_idx * width..(src_row_idx + 1) * width;
                let src_i = &mut i_src_part[src_row_range.clone()];
                let src_q = &mut q_src_part[src_row_range.clone()];

                copy_or_shift(src_i, dst_i);
                copy_or_shift(src_q, dst_q);
            }
        }
    } else {
        let offset = (-offset.1) as usize;
        // Starting from the top, copy (or write a horizontally-shifted copy of) each row upwards.
        for dst_row_idx in 0..height {
            let (i_dst_part, i_src_part) = yiq.i.split_at_mut((dst_row_idx + 1) * width);
            let (q_dst_part, q_src_part) = yiq.q.split_at_mut((dst_row_idx + 1) * width);

            let dst_row_range = dst_row_idx * width..(dst_row_idx + 1) * width;
            let dst_i = &mut i_dst_part[dst_row_range.clone()];
            let dst_q = &mut q_dst_part[dst_row_range.clone()];

            if dst_row_idx >= height - offset {
                dst_i.fill(0.0);
                dst_q.fill(0.0);
            } else {
                let src_row_range = (offset - 1) * width..offset * width;
                let src_i = &mut i_src_part[src_row_range.clone()];
                let src_q = &mut q_src_part[src_row_range.clone()];

                copy_or_shift(src_i, dst_i);
                copy_or_shift(src_q, dst_q);
            }
        }
    }
}

/// Emulate VHS waviness / horizontal shift noise.
fn vhs_edge_wave(yiq: &mut YiqView, seed: u64, intensity: f32, speed: f32, frame_num: usize) {
    let width = yiq.resolution.0;

    let seeder = Seeder::new(seed).mix(noise_seeds::EDGE_WAVE);
    let noise_seed: i32 = seeder.finalize();
    let offset = seeder.mix(1).finalize::<f32>() * yiq.resolution.1 as f32;
    let noise = NoiseBuilder::gradient_2d_offset(offset, width, frame_num as f32 * speed, 1)
        .with_seed(noise_seed)
        .with_freq(0.05)
        .generate()
        .0;

    for plane in [&mut yiq.y, &mut yiq.i, &mut yiq.q] {
        plane
            .chunks_mut(width)
            .enumerate()
            .for_each(|(index, row)| {
                let shift = (noise[index] / 0.022) * intensity * 0.5;
                shift_row(row, shift, BoundaryHandling::Extend);
            })
    }
}

/// Drop out the chrominance signal from random lines.
fn chroma_loss(yiq: &mut YiqView, intensity: f32, seed: u64, frame_num: usize) {
    let (width, height) = yiq.resolution;

    let seeder = Seeder::new(seed)
        .mix(noise_seeds::CHROMA_LOSS)
        .mix(frame_num);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seeder.finalize());
    // We blank out each row with a probability of `intensity` (0 to 1). Instead of going over each row and checking
    // whether to blank out the chroma, use a geometric distribution to simulate that process and tell us which rows
    // to blank.
    let dist = Geometric::new(intensity);

    let mut row_idx = 0usize;
    loop {
        row_idx += rng.sample(&dist);
        if row_idx >= height {
            break;
        }

        let row_range = row_idx * width..(row_idx + 1) * width;
        yiq.i[row_range.clone()].fill(0.0);
        yiq.q[row_range.clone()].fill(0.0);
        row_idx += 1;
    }
}

/// Vertically blend each chroma scanline with the one above it, as VHS does.
fn chroma_vert_blend(yiq: &mut YiqView) {
    let width = yiq.resolution.0;
    let mut delay_i = vec![0f32; width];
    let mut delay_q = vec![0f32; width];

    yiq.i
        .chunks_mut(width)
        .zip(yiq.q.chunks_mut(width))
        .for_each(|(i_row, q_row)| {
            // TODO: check if this is faster than interleaved (I think the cache locality is better this way)
            i_row.iter_mut().enumerate().for_each(|(index, i)| {
                let c_i = *i;
                *i = (delay_i[index] + c_i) * 0.5;
                delay_i[index] = c_i;
            });
            q_row.iter_mut().enumerate().for_each(|(index, q)| {
                let c_q = *q;
                *q = (delay_q[index] + c_q) * 0.5;
                delay_q[index] = c_q;
            });
        });
}

impl NtscEffect {
    pub fn apply_effect_to_yiq(&self, yiq: &mut YiqView, frame_num: usize, seed: u64) {
        let (width, _) = yiq.resolution;

        match self.chroma_lowpass_in {
            ChromaLowpass::Full => {
                composite_chroma_lowpass(yiq);
            }
            ChromaLowpass::Light => {
                composite_chroma_lowpass_lite(yiq);
            }
            ChromaLowpass::None => {}
        };

        chroma_into_luma(
            self.video_scanline_phase_shift,
            self.video_scanline_phase_shift_offset,
            yiq,
            50.0,
            frame_num,
        );

        if self.composite_preemphasis > 0.0 {
            let preemphasis_filter = make_lowpass(315000000.0 / 88.0 / 2.0, NTSC_RATE);
            filter_plane(
                &mut yiq.y,
                width,
                &preemphasis_filter,
                InitialCondition::Zero,
                -self.composite_preemphasis,
                0,
            );
        }

        if self.composite_noise_intensity > 0.0 {
            composite_noise(yiq, seed, 0.25, self.composite_noise_intensity, frame_num);
        }

        if self.snow_intensity > 0.0 {
            snow(yiq, seed, self.snow_intensity * 0.01, frame_num);
        }

        if let Some(HeadSwitchingSettings {
            height,
            offset,
            horiz_shift,
        }) = self.head_switching
        {
            head_switching(
                yiq,
                height as usize,
                offset as usize,
                horiz_shift,
                seed,
                frame_num,
            );
        }

        if let Some(TrackingNoiseSettings {
            height,
            wave_intensity,
            snow_intensity,
            noise_intensity,
        }) = self.tracking_noise
        {
            tracking_noise(
                yiq,
                seed,
                height as usize,
                wave_intensity,
                snow_intensity,
                noise_intensity,
                frame_num,
            );
        }

        luma_into_chroma(
            self.video_scanline_phase_shift,
            self.video_scanline_phase_shift_offset,
            yiq,
            50.0,
            frame_num,
        );

        if let Some(ringing) = &self.ringing {
            let notch_filter = make_notch_filter(ringing.frequency, ringing.power);
            filter_plane(
                &mut yiq.y,
                width,
                &notch_filter,
                InitialCondition::FirstSample,
                ringing.intensity,
                1,
            );
        }

        if self.chroma_noise_intensity > 0.0 {
            chroma_noise(yiq, seed, 0.05, self.chroma_noise_intensity, frame_num);
        }

        if self.chroma_phase_noise_intensity > 0.0 {
            chroma_phase_noise(yiq, seed, self.chroma_phase_noise_intensity, frame_num);
        }

        if self.chroma_delay.0 != 0.0 || self.chroma_delay.1 != 0 {
            chroma_delay(yiq, (self.chroma_delay.0, self.chroma_delay.1 as isize));
        }

        if let Some(vhs_settings) = &self.vhs_settings {
            if vhs_settings.edge_wave > 0.0 {
                vhs_edge_wave(
                    yiq,
                    seed,
                    vhs_settings.edge_wave,
                    vhs_settings.edge_wave_speed,
                    frame_num,
                );
            }

            if let Some(tape_speed) = &vhs_settings.tape_speed {
                let VHSTapeParams {
                    luma_cut,
                    chroma_cut,
                    chroma_delay,
                } = tape_speed.filter_params();

                // TODO: add an option to control whether there should be a line on the left from the filter starting
                // at 0. it's present in both the original C++ code and Python port but probably not an actual VHS
                // TODO: use a better filter! this effect's output looks way more smear-y than real VHS
                let luma_filter = make_lowpass_triple(luma_cut, NTSC_RATE);
                let chroma_filter = make_lowpass_triple(chroma_cut, NTSC_RATE);
                filter_plane(
                    &mut yiq.y,
                    width,
                    &luma_filter,
                    InitialCondition::Zero,
                    1.0,
                    0,
                );
                filter_plane(
                    &mut yiq.i,
                    width,
                    &chroma_filter,
                    InitialCondition::Zero,
                    1.0,
                    chroma_delay,
                );
                filter_plane(
                    &mut yiq.q,
                    width,
                    &chroma_filter,
                    InitialCondition::Zero,
                    1.0,
                    chroma_delay,
                );
                let luma_filter_single = make_lowpass(luma_cut, NTSC_RATE);
                filter_plane(
                    &mut yiq.y,
                    width,
                    &luma_filter_single,
                    InitialCondition::Zero,
                    -1.6,
                    0,
                );
            }

            if vhs_settings.chroma_loss > 0.0 {
                chroma_loss(yiq, vhs_settings.chroma_loss, seed, frame_num);
            }

            if vhs_settings.chroma_vert_blend {
                chroma_vert_blend(yiq);
            }

            if vhs_settings.sharpen > 0.0 {
                if let Some(tape_speed) = &vhs_settings.tape_speed {
                    let VHSTapeParams { luma_cut, .. } = tape_speed.filter_params();
                    let luma_sharpen_filter = make_lowpass_triple(luma_cut * 4.0, NTSC_RATE);
                    // The composite-video-simulator code sharpens the chroma plane, but ntscqt and this effect do not.
                    // I'm not sure if I'm implementing it wrong, but chroma sharpening looks awful.
                    // let chroma_sharpen_filter = make_lowpass_triple(chroma_cut * 4.0, 0.0, NTSC_RATE);
                    filter_plane(
                        &mut yiq.y,
                        width,
                        &luma_sharpen_filter,
                        InitialCondition::Zero,
                        -vhs_settings.sharpen * 2.0,
                        0,
                    );
                    // filter_plane_scaled(&mut yiq.i, width, &chroma_sharpen_filter, -vhs_settings.sharpen * 0.85);
                    // filter_plane_scaled(&mut yiq.q, width, &chroma_sharpen_filter, -vhs_settings.sharpen * 0.85);
                }
            }
        }

        match self.chroma_lowpass_out {
            ChromaLowpass::Full => {
                composite_chroma_lowpass(yiq);
            }
            ChromaLowpass::Light => {
                composite_chroma_lowpass_lite(yiq);
            }
            ChromaLowpass::None => {}
        };
    }

    pub fn apply_effect(&self, input_frame: &RgbImage, frame_num: usize, seed: u64) -> RgbImage {
        let field = match self.use_field {
            UseField::Alternating => if frame_num & 1 == 0 {
                YiqField::Upper
            } else {
                YiqField::Lower
            },
            UseField::Upper => YiqField::Upper,
            UseField::Lower => YiqField::Lower,
            UseField::Both => YiqField::Both,
        };
        let mut yiq = YiqOwned::from_image(input_frame, field);
        let mut view = YiqView::from(&mut yiq);
        self.apply_effect_to_yiq(&mut view, frame_num, seed);
        RgbImage::from(&view)
    }
}
