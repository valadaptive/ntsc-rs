use std::collections::VecDeque;

use core::f32::consts::PI;
use image::RgbImage;
use macros::FullSettings;
use glam::{Vec3, Mat3};
use rand::{rngs::SmallRng, Rng, RngCore, SeedableRng};
use simdnoise::NoiseBuilder;

use crate::{
    filter::TransferFunction,
    random::{Geometric, Seeder},
    shift::{shift_row, BoundaryHandling, shift_row_to},
};

const YIQ_MATRIX: Mat3 = Mat3 {
    x_axis: Vec3 {x: 0.299, y: 0.5959, z: 0.2115},
    y_axis: Vec3 {x: 0.587, y: -0.2746, z: -0.5227},
    z_axis: Vec3 {x: 0.114, y: -0.3213, z: 0.3112},
};

const RGB_MATRIX: Mat3 = Mat3 {
    x_axis: Vec3 {x: 1.0, y: 1.0, z: 1.0},
    y_axis: Vec3 {x: 0.956, y: -0.272, z: -1.106},
    z_axis: Vec3 {x: 0.619, y: -0.647, z: 1.703},
};


const NTSC_RATE: f32 = (315000000.00 / 88.0) * 4.0; // 315/88 Mhz rate * 4

struct YiqPlanar {
    y: Vec<f32>,
    i: Vec<f32>,
    q: Vec<f32>,
    resolution: (usize, usize),
    field: YiqField,
}

/// Create a lowpass filter with the given parameters, which can then be used to filter a signal.
pub fn make_lowpass(cutoff: f32, rate: f32) -> TransferFunction {
    let time_interval = 1.0 / rate;
    let tau = (cutoff * 2.0 * PI).recip();
    let alpha = time_interval / (tau + time_interval);

    TransferFunction::new(vec![alpha], vec![1.0, -(1.0 - alpha)])
}

/// Create a lowpass filter with the given parameters, which can then be used to filter a signal.
/// This is equivalent to applying the same lowpass filter 3 times.
pub fn make_lowpass_triple(cutoff: f32, rate: f32) -> TransferFunction {
    let time_interval = 1.0 / rate;
    let tau = (cutoff * 2.0 * PI).recip();
    let alpha = time_interval / (tau + time_interval);

    let tf = TransferFunction::new(vec![alpha], vec![1.0, -(1.0 - alpha)]);

    &(&tf * &tf) * &tf
}

pub fn make_notch_filter(freq: f32, quality: f32) -> TransferFunction {
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

/// Apply a given filter to one color plane.
fn filter_plane(
    plane: &mut Vec<f32>,
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

#[derive(PartialEq, Eq)]
pub enum YiqField {
    Upper,
    Lower,
    Both,
}

impl YiqPlanar {
    pub fn from_image(image: &RgbImage, field: YiqField) -> Self {
        let width = image.width() as usize;
        let height = match field {
            YiqField::Upper | YiqField::Lower => (image.height() + 1) / 2,
            YiqField::Both => image.height(),
        } as usize;

        let (row_lshift, row_offset): (usize, usize) = match field {
            YiqField::Upper => (1, 0),
            YiqField::Lower => (1, 1),
            YiqField::Both => (0, 0),
        };

        let num_pixels = width * height;
        let mut y = vec![0f32; num_pixels];
        let mut i = vec![0f32; num_pixels];
        let mut q = vec![0f32; num_pixels];

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

        YiqPlanar {
            y,
            i,
            q,
            resolution: (width, height),
            field,
        }
    }
}

impl From<&YiqPlanar> for RgbImage {
    fn from(image: &YiqPlanar) -> Self {
        let width = image.resolution.0;
        let output_height = image.resolution.1 * if image.field == YiqField::Both { 1 } else { 2 };
        let num_pixels = width * output_height;
        let mut dst = vec![0u8; num_pixels * 3];

        // If the row index modulo 2 equals this number, skip that row.
        let skip_field: usize = match image.field {
            YiqField::Upper => 1,
            YiqField::Lower => 0,
            // The row index modulo 2 never reaches 2, meaning we don't skip any fields
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

fn composite_chroma_lowpass(frame: &mut YiqPlanar) {
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

fn composite_chroma_lowpass_lite(frame: &mut YiqPlanar) {
    let filter = make_lowpass_triple(2600000.0, NTSC_RATE);

    let width = frame.resolution.0;

    filter_plane(&mut frame.i, width, &filter, InitialCondition::Zero, 1.0, 1);
    filter_plane(&mut frame.q, width, &filter, InitialCondition::Zero, 1.0, 1);
}

const I_MULT: [f32; 4] = [1.0, 0.0, -1.0, 0.0];
const Q_MULT: [f32; 4] = [0.0, 1.0, 0.0, -1.0];

fn chroma_luma_line_offset(
    scanline_phase_shift: PhaseShift,
    offset: i32,
    field_num: usize,
    line_num: usize,
) -> usize {
    (match scanline_phase_shift {
        PhaseShift::Degrees90 | PhaseShift::Degrees270 => {
            (field_num as i32 + offset + ((line_num as i32) >> 1)) & 3
        }
        PhaseShift::Degrees180 => (((field_num + line_num) & 2) as i32 + offset) & 3,
        PhaseShift::Degrees0 => 0,
    } & 3) as usize
}

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
            let offset = (index + (xi & 3)) & 3;
            *y += (*i * I_MULT[offset] + *q * Q_MULT[offset]) * subcarrier_amplitude / 50.0;
            // *i = 0.0;
            // *q = 0.0;
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
        // box-blur the signal to get the luminance
        // TODO: add an option to use a real lowpass filter here?
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

fn video_noise_line(row: &mut [f32], seeder: Seeder, index: usize, frequency: f32, intensity: f32) {
    let width = row.len();
    let mut rng = SmallRng::seed_from_u64(seeder.mix(index as u64).finalize());
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

fn composite_noise(
    yiq: &mut YiqPlanar,
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

fn chroma_noise(yiq: &mut YiqPlanar, seed: u64, frequency: f32, intensity: f32, frame_num: usize) {
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

fn chroma_phase_noise(yiq: &mut YiqPlanar, seed: u64, intensity: f32, frame_num: usize) {
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

fn head_switching(
    yiq: &mut YiqPlanar,
    num_rows: usize,
    offset: usize,
    shift: f32,
    seed: u64,
    frame_num: usize,
) {
    let (width, height) = yiq.resolution;
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

        let transient_len: f32 = rng.gen_range(8.0..64.0);

        for i in
            pixel_idx..(pixel_idx + transient_len.ceil() as usize).min(row.len())
        {
            let x = (i - pixel_idx) as f32;
            // Quadratic decay to 0
            row[i] += (1.0 - (x / transient_len)).powi(2) * 2.0 * rng.gen::<f32>();
        }

        // Make sure we advance the pixel index each time. Our geometric distribution gives us the time between
        // successive events, which can be 0 for very high probabilities.
        pixel_idx += 1;
    }
}

fn tracking_noise(
    yiq: &mut YiqPlanar,
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

            video_noise_line(row, seeder, index, 0.25, intensity_scale.powi(2) * noise_intensity * 4.0);

            // Turn each pixel into "snow" with probability snow_intensity * intensity_scale
            // We can simulate the distance between each "snow" pixel with a geometric distribution which avoids having to
            // loop over every pixel
            row_speckles(
                row,
                &mut SmallRng::seed_from_u64(seeder.mix(index).finalize()),
                snow_intensity * intensity_scale.powi(2),
            );
        });
}

fn snow(yiq: &mut YiqPlanar, seed: u64, intensity: f32, frame_num: usize) {
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
                &mut SmallRng::seed_from_u64(seeder.mix(index).finalize()),
                intensity,
            );
        });
}

fn chroma_delay(yiq: &mut YiqPlanar, offset: (f32, isize)) {
    let copy_or_shift = |src: &mut [f32], dst: &mut [f32]| {
        if offset.0.abs() == 0.0 {
            dst.copy_from_slice(src);
        } else {
            shift_row_to(src, dst, offset.0, BoundaryHandling::Constant(0.0));
        }
    };

    let (width, height) = yiq.resolution;

    if offset.1 == 0 {
        yiq.i.chunks_mut(width).zip(yiq.q.chunks_mut(width)).for_each(|(i, q)| {
            shift_row(i, offset.0, BoundaryHandling::Constant(0.0));
            shift_row(q, offset.0, BoundaryHandling::Constant(0.0));
        });
    } else if offset.1 > 0 {
        let offset = offset.1 as usize;
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

fn vhs_edge_wave(yiq: &mut YiqPlanar, seed: u64, intensity: f32, speed: f32, frame_num: usize) {
    let width = yiq.resolution.0;

    let seeder = Seeder::new(seed).mix(noise_seeds::EDGE_WAVE);
    let noise_seed: i32 = seeder.finalize();
    let offset = seeder.mix(1).finalize::<f32>() * yiq.resolution.1 as f32;
    let noise = NoiseBuilder::gradient_2d_offset(
        offset,
        width,
        frame_num as f32 * speed,
        1,
    )
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

fn chroma_loss(
    yiq: &mut YiqPlanar,
    intensity: f32,
    seed: u64,
    frame_num: usize
) {
    let (width, height) = yiq.resolution;

    let seeder = Seeder::new(seed).mix(noise_seeds::CHROMA_LOSS).mix(frame_num);

    let mut rng = SmallRng::seed_from_u64(seeder.finalize());
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

fn chroma_vert_blend(yiq: &mut YiqPlanar) {
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PhaseShift {
    Degrees0,
    Degrees90,
    Degrees180,
    Degrees270,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VHSTapeSpeed {
    SP,
    LP,
    EP,
}

struct VHSTapeParams {
    luma_cut: f32,
    chroma_cut: f32,
    chroma_delay: usize,
}

impl VHSTapeSpeed {
    fn filter_params(&self) -> VHSTapeParams {
        match self {
            VHSTapeSpeed::SP => VHSTapeParams {
                luma_cut: 2400000.0,
                chroma_cut: 320000.0,
                chroma_delay: 4,
            },
            VHSTapeSpeed::LP => VHSTapeParams {
                luma_cut: 1900000.0,
                chroma_cut: 300000.0,
                chroma_delay: 5,
            },
            VHSTapeSpeed::EP => VHSTapeParams {
                luma_cut: 1400000.0,
                chroma_cut: 280000.0,
                chroma_delay: 6,
            },
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct VHSSettings {
    pub tape_speed: Option<VHSTapeSpeed>,
    pub chroma_vert_blend: bool,
    pub chroma_loss: f32,
    pub sharpen: f32,
    pub edge_wave: f32,
    pub edge_wave_speed: f32,
}

impl Default for VHSSettings {
    fn default() -> Self {
        Self {
            tape_speed: Some(VHSTapeSpeed::LP),
            chroma_vert_blend: true,
            chroma_loss: 0.0,
            sharpen: 1.0,
            edge_wave: 1.0,
            edge_wave_speed: 4.0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ChromaLowpass {
    None,
    Light,
    Full,
}

#[derive(Clone, PartialEq)]
pub struct HeadSwitchingSettings {
    pub height: usize,
    pub offset: usize,
    pub horiz_shift: f32,
}

impl Default for HeadSwitchingSettings {
    fn default() -> Self {
        Self {
            height: 8,
            offset: 3,
            horiz_shift: 72.0,
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct HeadSwitchingNoiseSettings {
    pub height: usize,
    pub wave_intensity: f32,
    pub snow_intensity: f32,
    pub noise_intensity: f32,
}

impl Default for HeadSwitchingNoiseSettings {
    fn default() -> Self {
        Self {
            height: 24,
            wave_intensity: 5.0,
            snow_intensity: 0.005,
            noise_intensity: 0.005,
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct RingingSettings {
    pub frequency: f32,
    pub power: f32,
    pub intensity: f32,
}

impl Default for RingingSettings {
    fn default() -> Self {
        Self {
            frequency: 0.45,
            power: 4.0,
            intensity: 4.0,
        }
    }
}

pub struct SettingsBlock<T> {
    pub enabled: bool,
    pub settings: T,
}

impl<T: Default + Clone> From<&Option<T>> for SettingsBlock<T> {
    fn from(opt: &Option<T>) -> Self {
        Self {
            enabled: opt.is_some(),
            settings: match opt {
                Some(v) => v.clone(),
                None => T::default(),
            },
        }
    }
}

impl<T: Default> From<Option<T>> for SettingsBlock<T> {
    fn from(opt: Option<T>) -> Self {
        Self {
            enabled: opt.is_some(),
            settings: opt.unwrap_or_else(T::default),
        }
    }
}

impl<T> From<SettingsBlock<T>> for Option<T> {
    fn from(value: SettingsBlock<T>) -> Self {
        if value.enabled {
            Some(value.settings)
        } else {
            None
        }
    }
}

impl<T: Clone> From<&SettingsBlock<T>> for Option<T> {
    fn from(value: &SettingsBlock<T>) -> Self {
        if value.enabled {
            Some(value.settings.clone())
        } else {
            None
        }
    }
}

impl<T: Default> Default for SettingsBlock<T> {
    fn default() -> Self {
        Self {
            enabled: true,
            settings: T::default(),
        }
    }
}

#[derive(FullSettings)]
pub struct NtscEffect {
    pub chroma_lowpass_in: ChromaLowpass,
    pub composite_preemphasis: f32,
    pub video_scanline_phase_shift: PhaseShift,
    pub video_scanline_phase_shift_offset: i32,
    #[settings_block]
    pub head_switching: Option<HeadSwitchingSettings>,
    #[settings_block]
    pub tracking_noise: Option<HeadSwitchingNoiseSettings>,
    pub composite_noise_intensity: f32,
    #[settings_block]
    pub ringing: Option<RingingSettings>,
    pub chroma_noise_intensity: f32,
    pub snow_intensity: f32,
    pub chroma_phase_noise_intensity: f32,
    pub chroma_delay: (f32, isize),
    #[settings_block]
    pub vhs_settings: Option<VHSSettings>,
    pub chroma_lowpass_out: ChromaLowpass,
}

impl Default for NtscEffect {
    fn default() -> Self {
        Self {
            chroma_lowpass_in: ChromaLowpass::Full,
            chroma_lowpass_out: ChromaLowpass::Full,
            composite_preemphasis: 1.0,
            video_scanline_phase_shift: PhaseShift::Degrees90,
            video_scanline_phase_shift_offset: 0,
            head_switching: Some(HeadSwitchingSettings::default()),
            tracking_noise: Some(HeadSwitchingNoiseSettings::default()),
            ringing: Some(RingingSettings::default()),
            snow_intensity: 0.00001,
            composite_noise_intensity: 0.01,
            chroma_noise_intensity: 0.1,
            chroma_phase_noise_intensity: 0.001,
            chroma_delay: (0.0, 0),
            vhs_settings: Some(VHSSettings::default()),
        }
    }
}

impl NtscEffect {
    pub fn apply_effect(&self, input_frame: &RgbImage, frame_num: usize, seed: u64) -> RgbImage {
        let mut yiq = YiqPlanar::from_image(input_frame, YiqField::Lower);
        let (width, _) = yiq.resolution;

        match self.chroma_lowpass_in {
            ChromaLowpass::Full => {
                composite_chroma_lowpass(&mut yiq);
            }
            ChromaLowpass::Light => {
                composite_chroma_lowpass_lite(&mut yiq);
            }
            ChromaLowpass::None => {}
        };

        self.chroma_into_luma(&mut yiq, 50.0, 1);

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
            composite_noise(
                &mut yiq,
                seed,
                0.25,
                self.composite_noise_intensity,
                frame_num,
            );
        }

        if self.snow_intensity > 0.0 {
            snow(&mut yiq, seed, self.snow_intensity, frame_num);
        }

        if let Some(HeadSwitchingSettings {
            height,
            offset,
            horiz_shift,
        }) = self.head_switching
        {
            head_switching(&mut yiq, height, offset, horiz_shift, seed, frame_num);
        }

        if let Some(HeadSwitchingNoiseSettings {
            height,
            wave_intensity,
            snow_intensity,
            noise_intensity,
        }) = self.tracking_noise
        {
            tracking_noise(
                &mut yiq,
                seed,
                height,
                wave_intensity,
                snow_intensity,
                noise_intensity,
                frame_num,
            );
        }

        self.luma_into_chroma(&mut yiq, 50.0, 1);

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
            chroma_noise(&mut yiq, seed, 0.05, self.chroma_noise_intensity, frame_num);
        }

        if self.chroma_phase_noise_intensity > 0.0 {
            chroma_phase_noise(&mut yiq, seed, self.chroma_phase_noise_intensity, frame_num);
        }

        if self.chroma_delay.0 != 0.0 || self.chroma_delay.1 != 0 {
            chroma_delay(&mut yiq, self.chroma_delay);
        }

        if let Some(vhs_settings) = &self.vhs_settings {
            if vhs_settings.edge_wave > 0.0 {
                vhs_edge_wave(
                    &mut yiq,
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
                chroma_loss(&mut yiq, vhs_settings.chroma_loss, seed, frame_num);
            }

            if vhs_settings.chroma_vert_blend {
                chroma_vert_blend(&mut yiq);
            }

            if vhs_settings.sharpen > 0.0 {
                if let Some(tape_speed) = &vhs_settings.tape_speed {
                    let VHSTapeParams { luma_cut, .. } = tape_speed.filter_params();
                    let luma_sharpen_filter = make_lowpass_triple(luma_cut * 4.0, NTSC_RATE);
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
                composite_chroma_lowpass(&mut yiq);
            }
            ChromaLowpass::Light => {
                composite_chroma_lowpass_lite(&mut yiq);
            }
            ChromaLowpass::None => {}
        };

        RgbImage::from(&yiq)
    }

    // Modulate the chrominance signal into the luminance plane.
    fn chroma_into_luma(&self, yiq: &mut YiqPlanar, subcarrier_amplitude: f32, fieldno: usize) {
        let width = yiq.resolution.0;

        let y_lines = yiq.y.chunks_mut(width);
        let i_lines = yiq.i.chunks_mut(width);
        let q_lines = yiq.q.chunks_mut(width);

        y_lines
            .zip(i_lines.zip(q_lines))
            .enumerate()
            .for_each(|(index, (y, (i, q)))| {
                let xi = chroma_luma_line_offset(
                    self.video_scanline_phase_shift,
                    self.video_scanline_phase_shift_offset,
                    fieldno,
                    index * 2,
                );

                chroma_into_luma_line(y, i, q, xi, subcarrier_amplitude);
            });
    }

    fn luma_into_chroma(&self, yiq: &mut YiqPlanar, subcarrier_amplitude: f32, fieldno: usize) {
        let width = yiq.resolution.0;

        let y_lines = yiq.y.chunks_mut(width);
        let i_lines = yiq.i.chunks_mut(width);
        let q_lines = yiq.q.chunks_mut(width);

        y_lines
            .zip(i_lines.zip(q_lines))
            .enumerate()
            .for_each(|(index, (y, (i, q)))| {
                let xi = chroma_luma_line_offset(
                    self.video_scanline_phase_shift,
                    self.video_scanline_phase_shift_offset,
                    fieldno,
                    index * 2,
                );

                luma_into_chroma_line(y, i, q, xi, subcarrier_amplitude);
            });
    }
}
