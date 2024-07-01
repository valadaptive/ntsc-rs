use std::collections::VecDeque;
use std::convert::identity;

use core::f32::consts::PI;
use rand::{Rng, RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use simdnoise::{NoiseBuilder, Settings, SimplexSettings};

use crate::{
    filter::TransferFunction,
    random::{Geometric, Seeder},
    shift::{shift_row, shift_row_to, BoundaryHandling},
    yiq_fielding::{BlitInfo, PixelFormat, YiqField, YiqOwned, YiqView},
};

pub use crate::settings::*;

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
    make_lowpass(cutoff, rate).cascade_self(3)
}

fn make_lowpass_for_type(cutoff: f32, rate: f32, filter_type: FilterType) -> TransferFunction {
    match filter_type {
        FilterType::ConstantK => make_lowpass_triple(cutoff, rate),
        FilterType::Butterworth => make_butterworth_filter(cutoff, rate),
    }
}

/// Create an IIR notch filter.
pub fn make_notch_filter(freq: f32, quality: f32) -> TransferFunction {
    // Adapted from scipy and simplified
    // https://github.com/scipy/scipy/blob/686422c4f0a71be1b4258309590fd3e9de102e18/scipy/signal/_filter_design.py#L5099-L5171
    if !(0.0..=1.0).contains(&freq) {
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

pub fn make_butterworth_filter(cutoff: f32, rate: f32) -> TransferFunction {
    let coeffs = biquad::Coefficients::<f32>::from_params(
        biquad::Type::LowPass,
        biquad::Hertz::<f32>::from_hz(rate).unwrap(),
        biquad::Hertz::<f32>::from_hz(cutoff.min(rate * 0.5)).unwrap(),
        biquad::Q_BUTTERWORTH_F32,
    )
    .unwrap();
    TransferFunction::new(
        [coeffs.b0, coeffs.b1, coeffs.b2],
        [1.0, coeffs.a1, coeffs.a2],
    )
}

/// Filter initial condition.
#[allow(dead_code)]
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
    if filter.should_use_row_chunks() {
        filter_plane_with_rows::<8>(plane, width, filter, initial, scale, delay)
    } else {
        filter_plane_with_rows::<1>(plane, width, filter, initial, scale, delay)
    }
}

fn filter_plane_with_rows<const ROWS: usize>(
    plane: &mut [f32],
    width: usize,
    filter: &TransferFunction,
    initial: InitialCondition,
    scale: f32,
    delay: usize,
) {
    let mut row_chunks = plane.par_chunks_exact_mut(width * ROWS);
    row_chunks
        .take_remainder()
        .par_chunks_exact_mut(width)
        .for_each(|row| {
            let initial = match initial {
                InitialCondition::Zero => 0.0,
                InitialCondition::Constant(c) => c,
                InitialCondition::FirstSample => row[0],
            };
            filter.filter_signal_in_place::<1>(&mut [row], [initial], scale, delay)
        });

    row_chunks.for_each(|rows| {
        let mut row_chunks: Vec<&mut [f32]> = rows.chunks_exact_mut(width).into_iter().collect();
        let rows: &mut [&mut [f32]; ROWS] = row_chunks.as_mut_slice().try_into().unwrap();

        let mut initial_rows: [f32; ROWS] = [0f32; ROWS];
        for i in 0..ROWS {
            initial_rows[i] = match initial {
                InitialCondition::Zero => 0.0,
                InitialCondition::Constant(c) => c,
                InitialCondition::FirstSample => rows[i][0],
            };
        }

        filter.filter_signal_in_place::<ROWS>(rows, initial_rows, scale, delay);
    });
}

/// Settings common to each invocation of the effect. Passed to each individual effect function.
struct CommonInfo {
    seed: u64,
    frame_num: usize,
    bandwidth_scale: f32,
}

fn luma_filter(frame: &mut YiqView, filter_mode: LumaLowpass) {
    match filter_mode {
        LumaLowpass::None => {}
        LumaLowpass::Box => {
            frame.y.par_chunks_mut(frame.dimensions.0).for_each(|y| {
                let mut delay = VecDeque::<f32>::with_capacity(4);
                delay.push_back(16.0 / 255.0);
                delay.push_back(16.0 / 255.0);
                delay.push_back(y[0]);
                delay.push_back(y[1]);
                let mut sum: f32 = delay.iter().sum();
                let width = y.len();

                for index in 0..width {
                    // Box-blur the signal.
                    let c = y[usize::min(index + 2, width - 1)];
                    sum -= delay.pop_front().unwrap();
                    delay.push_back(c);
                    sum += c;
                    y[index] = sum * 0.25;
                }
            });
        }
        LumaLowpass::Notch => filter_plane(
            frame.y,
            frame.dimensions.0,
            &make_notch_filter(0.5, 2.0),
            InitialCondition::FirstSample,
            1.0,
            0,
        ),
    }
}

/// Apply a lowpass filter to the input chroma, emulating broadcast NTSC's bandwidth cutoffs.
/// (Well, almost--Wikipedia (https://en.wikipedia.org/wiki/YIQ) puts the Q bandwidth at 0.4 MHz, not 0.6. Although
/// that statement seems unsourced and I can't find any info on it...
fn composite_chroma_lowpass(frame: &mut YiqView, info: &CommonInfo, filter_type: FilterType) {
    let i_filter = make_lowpass_for_type(1300000.0, NTSC_RATE * info.bandwidth_scale, filter_type);
    let q_filter = make_lowpass_for_type(600000.0, NTSC_RATE * info.bandwidth_scale, filter_type);

    let width = frame.dimensions.0;

    filter_plane(frame.i, width, &i_filter, InitialCondition::Zero, 1.0, 2);
    filter_plane(frame.q, width, &q_filter, InitialCondition::Zero, 1.0, 4);
}

/// Apply a less intense lowpass filter to the input chroma.
fn composite_chroma_lowpass_lite(frame: &mut YiqView, info: &CommonInfo, filter_type: FilterType) {
    let filter = make_lowpass_for_type(2600000.0, NTSC_RATE * info.bandwidth_scale, filter_type);

    let width = frame.dimensions.0;

    filter_plane(frame.i, width, &filter, InitialCondition::Zero, 1.0, 1);
    filter_plane(frame.q, width, &filter, InitialCondition::Zero, 1.0, 1);
}

/// Calculate the chroma subcarrier phase for a given row/field
fn chroma_phase_shift(
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

fn chroma_into_luma_line(y: &mut [f32], i: &mut [f32], q: &mut [f32], xi: usize) {
    y.iter_mut()
        .zip(i.iter_mut().zip(q))
        .enumerate()
        .for_each(|(index, (y, (i, q)))| {
            let phase = (index + (xi & 3)) & 3;
            *y += *i * I_MULT[phase] + *q * Q_MULT[phase];
            // *i = 0.0;
            // *q = 0.0;
        });
}

/// Modulate the chrominance signal (I and Q planes) into the Y (luminance) plane.
/// TODO: sample rate
fn chroma_into_luma(
    yiq: &mut YiqView,
    info: &CommonInfo,
    phase_shift: PhaseShift,
    phase_offset: i32,
) {
    let width = yiq.dimensions.0;

    let y_lines = yiq.y.par_chunks_mut(width);
    let i_lines = yiq.i.par_chunks_mut(width);
    let q_lines = yiq.q.par_chunks_mut(width);

    y_lines
        .zip(i_lines.zip(q_lines))
        .enumerate()
        .for_each(|(index, (y, (i, q)))| {
            let xi = chroma_phase_shift(phase_shift, phase_offset, info.frame_num, index * 2);

            chroma_into_luma_line(y, i, q, xi);
        });
}

#[inline(always)]
fn demodulate_chroma(chroma: f32, index: usize, xi: usize, i: &mut [f32], q: &mut [f32]) {
    let width = i.len();

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

/// Demodulate the chrominance signal using a box filter to separate it out.
fn luma_into_chroma_line_box(
    y: &mut [f32],
    i: &mut [f32],
    q: &mut [f32],
    scratch: &mut [f32],
    xi: usize,
) {
    let width = y.len();
    for index in 0..width {
        let c = y[usize::min(index + 2, width - 1)];
        let area = [
            y.get(index.wrapping_sub(1))
                .cloned()
                .unwrap_or(16.0 / 255.0),
            y[index],
            y[(index + 1).min(width - 1)],
            y[(index + 2).min(width - 1)],
        ];
        scratch[index] = area.iter().sum::<f32>() * 0.25;
        let chroma = c - scratch[index];
        demodulate_chroma(chroma, index, xi, i, q);
    }
    y.copy_from_slice(scratch);
}

/// Demodulate the chroma signal from the Y (luma) plane back into the I and Q planes.
/// TODO: sample rate
fn luma_into_chroma(
    yiq: &mut YiqView,
    info: &CommonInfo,
    filter_mode: ChromaDemodulationFilter,
    phase_shift: PhaseShift,
    phase_offset: i32,
) {
    let width = yiq.dimensions.0;

    match filter_mode {
        ChromaDemodulationFilter::Box => {
            let y_lines = yiq.y.par_chunks_mut(width);
            let i_lines = yiq.i.par_chunks_mut(width);
            let q_lines = yiq.q.par_chunks_mut(width);
            let scratch_lines = yiq.scratch.par_chunks_mut(width);

            y_lines
                .zip(i_lines.zip(q_lines.zip(scratch_lines)))
                .enumerate()
                .for_each(|(index, (y, (i, (q, scratch))))| {
                    let xi =
                        chroma_phase_shift(phase_shift, phase_offset, info.frame_num, index * 2);

                    luma_into_chroma_line_box(y, i, q, scratch, xi);
                });
        }
        ChromaDemodulationFilter::Notch => {
            let scratch = &mut yiq.scratch;
            let filter: TransferFunction = make_notch_filter(0.5, 2.0);
            scratch.copy_from_slice(yiq.y);
            filter_plane(yiq.y, width, &filter, InitialCondition::Zero, 1.0, 0);

            let y_lines = yiq.y.par_chunks_mut(width);
            let i_lines = yiq.i.par_chunks_mut(width);
            let q_lines = yiq.q.par_chunks_mut(width);
            let scratch_lines = scratch.par_chunks_mut(width);

            y_lines
                .zip(i_lines.zip(q_lines.zip(scratch_lines)))
                .enumerate()
                .for_each(|(index, (y, (i, (q, scratch))))| {
                    let xi =
                        chroma_phase_shift(phase_shift, phase_offset, info.frame_num, index * 2);

                    for index in 0..width {
                        let chroma = y[index] - scratch[index];
                        demodulate_chroma(chroma, index, xi, i, q);
                    }
                });
        }
        ChromaDemodulationFilter::OneLineComb => {
            let delay = &mut yiq.scratch;
            // "Reflect" line 2 to line 0, so that the chroma is properly demodulated.
            // A comb filter requires the phase of the chroma carrier to alternate per line, so simply repeating line 1
            // wouldn't work.
            delay[0..width].copy_from_slice(&yiq.y[width..width * 2]);
            delay[width..].copy_from_slice(&yiq.y[0..yiq.y.len() - width]);

            let y_lines = yiq.y.par_chunks_mut(width);
            let i_lines = yiq.i.par_chunks_mut(width);
            let q_lines = yiq.q.par_chunks_mut(width);
            let delay_lines = delay.par_chunks_mut(width);
            y_lines
                .zip(i_lines.zip(q_lines.zip(delay_lines)))
                .enumerate()
                .for_each(|(line_index, (y, (i, (q, delay))))| {
                    for index in 0..width {
                        let blended = (y[index] + delay[index]) * 0.5;
                        let chroma = blended - y[index];
                        y[index] = blended;
                        let xi = chroma_phase_shift(
                            phase_shift,
                            phase_offset,
                            info.frame_num,
                            line_index * 2,
                        );
                        demodulate_chroma(chroma, index, xi, i, q);
                    }
                });
        }
        ChromaDemodulationFilter::TwoLineComb => {
            let height = yiq.num_rows();
            let modulated = &mut yiq.scratch;
            modulated.copy_from_slice(yiq.y);

            let y_lines = yiq.y.par_chunks_mut(width);
            let i_lines = yiq.i.par_chunks_mut(width);
            let q_lines = yiq.q.par_chunks_mut(width);
            y_lines
                .zip(i_lines.zip(q_lines))
                .enumerate()
                .for_each(|(line_index, (y, (i, q)))| {
                    // For the first line, both prev_line and next_line point to the second line. This effecively makes it a
                    // one-line comb filter for that line.
                    let prev_index = if line_index == 0 { 1 } else { line_index - 1 };

                    // Similar for the last line.
                    let next_index = if line_index == height - 1 {
                        height - 2
                    } else {
                        line_index + 1
                    };

                    let prev_line = &modulated[prev_index * width..(prev_index + 1) * width];
                    let cur_line = &modulated[line_index * width..(line_index + 1) * width];
                    let next_line = &modulated[next_index * width..(next_index + 1) * width];

                    for sample_index in 0..width {
                        let cur_sample = cur_line[sample_index];
                        let blended = (cur_sample * 0.5)
                            + (prev_line[sample_index] * 0.25)
                            + (next_line[sample_index] * 0.25);
                        let chroma = blended - cur_sample;
                        y[sample_index] = blended;

                        let xi = chroma_phase_shift(
                            phase_shift,
                            phase_offset,
                            info.frame_num,
                            line_index * 2,
                        );
                        demodulate_chroma(chroma, sample_index, xi, i, q);
                    }
                });
        }
    };
}

fn luma_smear(yiq: &mut YiqView, info: &CommonInfo, amount: f32) {
    let lowpass = make_lowpass(f32::exp2(-4.0 * amount) * 0.25, info.bandwidth_scale);
    filter_plane(
        yiq.y,
        yiq.dimensions.0,
        &lowpass,
        InitialCondition::Zero,
        1.0,
        0,
    );
}

/// We use a seeded RNG to generate random noise deterministically, but we don't want every pass which uses noise to use
/// the *same* noise. Each pass gets its own random seed which is mixed into the RNG.
mod noise_seeds {
    pub const VIDEO_COMPOSITE: u64 = 0;
    pub const HEAD_SWITCHING: u64 = 2;
    pub const TRACKING_NOISE: u64 = 3;
    pub const VIDEO_CHROMA_PHASE: u64 = 4;
    pub const EDGE_WAVE: u64 = 5;
    pub const SNOW: u64 = 6;
    pub const CHROMA_LOSS: u64 = 7;
    pub const HEAD_SWITCHING_MID_LINE_JITTER: u64 = 8;

    pub const VIDEO_CHROMA_I: u64 = 1;
    pub const VIDEO_CHROMA_Q: u64 = 9;
    pub const VIDEO_LUMA: u64 = 10;
}

/// Helper function to apply gradient noise to a single row of a single plane.
fn video_noise_line(
    row: &mut [f32],
    seeder: &Seeder,
    index: usize,
    frequency: f32,
    intensity: f32,
    detail: u32,
) {
    let width = row.len();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seeder.clone().mix(index as u64).finalize());
    let noise_seed = rng.next_u32();
    let offset = rng.gen::<f32>() * width as f32;

    let noise = NoiseBuilder::fbm_1d_offset(offset, width)
        .with_seed(noise_seed as i32)
        .with_freq(frequency)
        .with_octaves(detail.clamp(1, 5) as u8)
        // Yes, they got the lacunarity backwards by making it apply to frequency instead of scale.
        // 2.0 *halves* the scale each time because it doubles the frequency.
        .with_lacunarity(2.0)
        .generate()
        .0;

    row.iter_mut().enumerate().for_each(|(x, pixel)| {
        *pixel += noise[x] * 0.25 * intensity;
    });
}

/// Add noise to an NTSC-encoded signal.
fn composite_noise(yiq: &mut YiqView, info: &CommonInfo, noise_settings: &FbmNoiseSettings) {
    let width = yiq.dimensions.0;
    let seeder = Seeder::new(info.seed)
        .mix(noise_seeds::VIDEO_COMPOSITE)
        .mix(info.frame_num);

    yiq.y
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(index, row)| {
            video_noise_line(
                row,
                &seeder,
                index,
                noise_settings.frequency / info.bandwidth_scale,
                noise_settings.intensity,
                noise_settings.detail,
            );
        });
}

/// Add noise to a color plane of a de-modulated signal.
fn plane_noise(
    plane: &mut [f32],
    width: usize,
    info: &CommonInfo,
    settings: &FbmNoiseSettings,
    noise_seed: u64,
) {
    let seeder = Seeder::new(info.seed).mix(noise_seed).mix(info.frame_num);

    plane
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(index, row)| {
            video_noise_line(
                row,
                &seeder,
                index,
                settings.frequency / info.bandwidth_scale,
                settings.intensity,
                settings.detail,
            );
        });
}

fn chroma_phase_offset_line(i: &mut [f32], q: &mut [f32], offset: f32) {
    // Phase shift angle in radians. Mapped so that an intensity of 1.0 is a phase shift ranging from a full
    // rotation to the left - a full rotation to the right.
    let phase_shift = offset * PI * 2.0;
    let (sin_angle, cos_angle) = phase_shift.sin_cos();

    for (i, q) in i.iter_mut().zip(q.iter_mut()) {
        // Treat (i, q) as a 2D vector and rotate it by the phase shift amount.
        let rotated_i = (*i * cos_angle) - (*q * sin_angle);
        let rotated_q = (*i * sin_angle) + (*q * cos_angle);

        *i = rotated_i;
        *q = rotated_q;
    }
}

fn chroma_phase_error(yiq: &mut YiqView, intensity: f32) {
    let width = yiq.dimensions.0;

    yiq.i
        .par_chunks_mut(width)
        .zip(yiq.q.par_chunks_mut(width))
        .for_each(|(i, q)| {
            chroma_phase_offset_line(i, q, intensity);
        });
}

/// Add per-scanline chroma phase error.
fn chroma_phase_noise(yiq: &mut YiqView, info: &CommonInfo, intensity: f32) {
    let width = yiq.dimensions.0;
    let seeder = Seeder::new(info.seed)
        .mix(noise_seeds::VIDEO_CHROMA_PHASE)
        .mix(info.frame_num);

    yiq.i
        .par_chunks_mut(width)
        .zip(yiq.q.par_chunks_mut(width))
        .enumerate()
        .for_each(|(index, (i, q))| {
            // Phase shift angle in radians. Mapped so that an intensity of 1.0 is a phase shift ranging from a full
            // rotation to the left - a full rotation to the right.
            let phase_shift = (seeder.clone().mix(index).finalize::<f32>() - 0.5) * 2.0 * intensity;

            chroma_phase_offset_line(i, q, phase_shift);
        });
}

/// Emulate VHS head-switching at the bottom of the image.
fn head_switching(
    yiq: &mut YiqView,
    info: &CommonInfo,
    num_rows: usize,
    offset: usize,
    shift: f32,
    mid_line: Option<&HeadSwitchingMidLineSettings>,
) {
    if offset > num_rows {
        return;
    }
    let num_affected_rows = num_rows - offset;

    let width = yiq.dimensions.0;
    let height = yiq.num_rows();
    // Handle cases where the number of affected rows exceeds the number of actual rows in the image
    let start_row = height.max(num_affected_rows) - num_affected_rows;
    let affected_rows = &mut yiq.y[start_row * width..];
    let cut_off_rows = if num_affected_rows > height {
        num_affected_rows - height
    } else {
        0
    };

    let seeder = Seeder::new(info.seed)
        .mix(noise_seeds::HEAD_SWITCHING)
        .mix(info.frame_num);

    affected_rows
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(index, row)| {
            let index = num_affected_rows - (index + cut_off_rows);
            let row_shift = shift * ((index + offset) as f32 / num_rows as f32).powf(1.5);
            let noisy_shift = (row_shift + (seeder.clone().mix(index).finalize::<f32>() - 0.5))
                * info.bandwidth_scale;

            // because if-let chains are unstable :(
            if index == num_affected_rows && mid_line.is_some() {
                let mid_line = mid_line.unwrap();
                // Shift the entire row, but only copy back a portion of it.
                let mut tmp_row = vec![0.0; width];
                shift_row_to(
                    row,
                    &mut tmp_row,
                    noisy_shift,
                    BoundaryHandling::Constant(0.0),
                );

                let seeder = Seeder::new(info.seed)
                    .mix(noise_seeds::HEAD_SWITCHING_MID_LINE_JITTER)
                    .mix(info.frame_num);

                // Average two random numbers to bias the result towards the middle
                let jitter_rand = (seeder.clone().mix(0).finalize::<f32>()
                    + seeder.clone().mix(1).finalize::<f32>())
                    * 0.5;
                let jitter = (jitter_rand - 0.5) * mid_line.jitter;

                let copy_start = (width as f32 * (mid_line.position + jitter)) as usize;
                if copy_start > width {
                    return;
                }
                row[copy_start..].copy_from_slice(&tmp_row[copy_start..]);

                // Add a transient where the head switch is supposed to start
                let transient_intensity = (seeder.clone().mix(0).finalize::<f32>() + 0.5) * 0.5;
                let transient_len = 16.0 * info.bandwidth_scale;

                for i in copy_start..(copy_start + transient_len.ceil() as usize).min(width) {
                    let x = (i - copy_start) as f32;
                    row[i] += (1.0 - (x / transient_len as f32)).powi(3) * transient_intensity;
                }
            } else {
                shift_row(row, noisy_shift, BoundaryHandling::Constant(0.0));
            }
        });
}

/// Helper function for generating "snow".
fn row_speckles(
    row: &mut [f32],
    rng: &mut Xoshiro256PlusPlus,
    intensity: f32,
    anisotropy: f32,
    bandwidth_scale: f32,
) {
    let intensity = intensity as f64;
    let anisotropy = anisotropy as f64;

    // Transition smoothly from a flat function that always returns `intensity`, to a step function that returns
    // 1.0 with a probability of `intensity` and 0.0 with a probability of `1.0 - intensity`. In-between states
    // look like S-curves with increasing sharpness.
    // As a bonus, the integral of this function over (0, 1) as we transition from 0% to 100% anisotropy is *almost*
    // constant, meaning there's approximately the same amount of snow each time.
    let logistic_factor = ((rng.gen::<f64>() - intensity)
        / (intensity * (1.0 - intensity) * (1.0 - anisotropy)))
        .exp();
    let mut line_snow_intensity =
        anisotropy / (1.0 + logistic_factor) + intensity * (1.0 - anisotropy);

    // At maximum intensity (1.0), the line just gets completely whited out. 0.125 intensity looks more reasonable.
    line_snow_intensity *= 0.125;
    line_snow_intensity = line_snow_intensity.clamp(0.0, 1.0);
    if line_snow_intensity <= 0.0 {
        return;
    }

    // Turn each pixel into "snow" with probability snow_intensity * intensity_scale
    // We can simulate the distance between each "snow" pixel with a geometric distribution which avoids having to
    // loop over every pixel
    let dist = Geometric::new(line_snow_intensity);
    let mut pixel_idx = 0usize;
    loop {
        pixel_idx += rng.sample(&dist);
        if pixel_idx >= row.len() {
            break;
        }

        let transient_len: f32 = rng.gen_range(8.0..=64.0) * bandwidth_scale;
        let transient_freq = rng.gen_range(transient_len * 3.0..=transient_len * 5.0);

        // Each transient gets its own RNG to determine the intensity of each pixel within it.
        // This is to prevent the length of each transient from affecting the random state of the subsequent
        // transient, which can cause the snow to "jitter" when changing the "bandwidth scale" setting.
        rng.jump();
        let mut transient_rng = rng.clone();

        for i in pixel_idx..(pixel_idx + transient_len.ceil() as usize).min(row.len()) {
            let x = (i - pixel_idx) as f32;
            // Simulate transient with sin(pi*x / 4) * (1 - x/len)^2
            row[i] += ((x * PI) / transient_freq).cos()
                * (1.0 - x / transient_len).powi(2)
                * transient_rng.gen_range(-1.0..2.0);
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
    info: &CommonInfo,
    num_rows: usize,
    wave_intensity: f32,
    snow_intensity: f32,
    snow_anisotropy: f32,
    noise_intensity: f32,
) {
    let width = yiq.dimensions.0;
    let height = yiq.num_rows();

    let mut seeder = Seeder::new(info.seed)
        .mix(noise_seeds::TRACKING_NOISE)
        .mix(info.frame_num);
    let noise_seed = seeder.clone().mix(0).finalize::<i32>();
    let offset = seeder.clone().mix(1).finalize::<f32>() * yiq.num_rows() as f32;
    seeder = seeder.mix(2);
    let shift_noise = NoiseBuilder::gradient_1d_offset(offset, num_rows)
        .with_seed(noise_seed)
        .with_freq(0.5)
        .generate()
        .0;

    // Handle cases where the number of affected rows exceeds the number of actual rows in the image
    let start_row = height.max(num_rows) - num_rows;
    let affected_rows = &mut yiq.y[start_row * width..];
    let cut_off_rows = if num_rows > height {
        num_rows - height
    } else {
        0
    };

    affected_rows
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(index, row)| {
            let index = index + cut_off_rows;
            // This iterates from the top down. Increase the intensity as we approach the bottom of the picture.
            let intensity_scale = index as f32 / num_rows as f32;
            shift_row(
                row,
                shift_noise[index] * intensity_scale * wave_intensity * 0.25 * info.bandwidth_scale,
                BoundaryHandling::Constant(0.0),
            );

            video_noise_line(
                row,
                &seeder,
                index,
                0.25 / info.bandwidth_scale,
                intensity_scale.powi(2) * noise_intensity * 4.0,
                1,
            );

            row_speckles(
                row,
                &mut Xoshiro256PlusPlus::seed_from_u64(seeder.clone().mix(index).finalize()),
                snow_intensity * intensity_scale.powi(2),
                snow_anisotropy,
                info.bandwidth_scale,
            );
        });
}

/// Add random bits of "snow" to an NTSC-encoded signal.
fn snow(yiq: &mut YiqView, info: &CommonInfo, intensity: f32, anisotropy: f32) {
    let seeder = Seeder::new(info.seed)
        .mix(noise_seeds::SNOW)
        .mix(info.frame_num);

    yiq.y
        .par_chunks_mut(yiq.dimensions.0)
        .enumerate()
        .for_each(|(index, row)| {
            let line_seed = seeder.clone().mix(index);

            row_speckles(
                row,
                &mut Xoshiro256PlusPlus::seed_from_u64(line_seed.finalize()),
                intensity,
                anisotropy,
                info.bandwidth_scale,
            );
        });
}

/// Offset the chrominance (I and Q) planes horizontally and/or vertically.
/// Note how the horizontal shift is a float (the signal is continuous), but the vertical shift is an int (each scanline
/// is discrete).
fn chroma_delay(yiq: &mut YiqView, info: &CommonInfo, offset: (f32, isize)) {
    let horiz_shift = offset.0 * info.bandwidth_scale;
    let copy_or_shift = |src: &mut [f32], dst: &mut [f32]| {
        if offset.0.abs() == 0.0 {
            dst.copy_from_slice(src);
        } else {
            shift_row_to(src, dst, horiz_shift, BoundaryHandling::Constant(0.0));
        }
    };

    let width = yiq.dimensions.0;
    let height = yiq.num_rows();

    match offset.1.cmp(&0) {
        std::cmp::Ordering::Less => {
            let offset = (-offset.1) as usize;
            // Starting from the top, copy (or write a horizontally-shifted copy of) each row upwards.
            for dst_row_idx in 0..height {
                let (i_dst_part, i_src_part) = yiq.i.split_at_mut((dst_row_idx + 1) * width);
                let (q_dst_part, q_src_part) = yiq.q.split_at_mut((dst_row_idx + 1) * width);

                let dst_row_range = dst_row_idx * width..(dst_row_idx + 1) * width;
                let dst_i = &mut i_dst_part[dst_row_range.clone()];
                let dst_q = &mut q_dst_part[dst_row_range.clone()];

                if dst_row_idx >= height.max(offset) - offset {
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
        std::cmp::Ordering::Equal => {
            // Only a horizontal shift is necessary. We can do this in-place easily.
            yiq.i
                .par_chunks_mut(width)
                .zip(yiq.q.par_chunks_mut(width))
                .for_each(|(i, q)| {
                    shift_row(i, horiz_shift, BoundaryHandling::Constant(0.0));
                    shift_row(q, horiz_shift, BoundaryHandling::Constant(0.0));
                });
        }
        std::cmp::Ordering::Greater => {
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
        }
    };
}

/// Emulate VHS waviness / horizontal shift noise.
fn vhs_edge_wave(yiq: &mut YiqView, info: &CommonInfo, settings: &VHSEdgeWaveSettings) {
    let width = yiq.dimensions.0;
    let height = yiq.num_rows();

    let seeder = Seeder::new(info.seed).mix(noise_seeds::EDGE_WAVE);
    let noise_seed: i32 = seeder.clone().mix(0).finalize();
    let offset = seeder.mix(1).finalize::<f32>() * yiq.num_rows() as f32;
    let noise =
        NoiseBuilder::fbm_2d_offset(offset, height, info.frame_num as f32 * settings.speed, 1)
            .with_seed(noise_seed)
            .with_freq(settings.frequency)
            .with_octaves(settings.detail.clamp(1, 5) as u8)
            // Yes, they got the lacunarity backwards by making it apply to frequency instead of scale.
            // 2.0 *halves* the scale each time because it doubles the frequency.
            .with_lacunarity(2.0)
            .with_gain(std::f32::consts::FRAC_1_SQRT_2)
            .generate()
            .0;

    for plane in [&mut yiq.y, &mut yiq.i, &mut yiq.q] {
        plane
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(index, row)| {
                let shift =
                    (noise[index] / 0.022) * settings.intensity * 0.5 * info.bandwidth_scale;
                shift_row(row, shift, BoundaryHandling::Extend);
            })
    }
}

/// Drop out the chrominance signal from random lines.
fn chroma_loss(yiq: &mut YiqView, info: &CommonInfo, intensity: f32) {
    let width = yiq.dimensions.0;
    let height = yiq.num_rows();

    let seed = Seeder::new(info.seed)
        .mix(noise_seeds::CHROMA_LOSS)
        .mix(info.frame_num)
        .finalize();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    // We blank out each row with a probability of `intensity` (0 to 1). Instead of going over each row and checking
    // whether to blank out the chroma, use a geometric distribution to simulate that process and tell us which rows
    // to blank.
    let dist = Geometric::new(intensity as f64);

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
    let width = yiq.dimensions.0;
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
    fn apply_effect_to_yiq_field(&self, yiq: &mut YiqView, frame_num: usize) {
        let width = yiq.dimensions.0;

        let seed = self.random_seed as u32 as u64;

        let info = CommonInfo {
            seed,
            frame_num,
            bandwidth_scale: self.bandwidth_scale,
        };

        luma_filter(yiq, self.input_luma_filter);

        match self.chroma_lowpass_in {
            ChromaLowpass::Full => {
                composite_chroma_lowpass(yiq, &info, self.filter_type);
            }
            ChromaLowpass::Light => {
                composite_chroma_lowpass_lite(yiq, &info, self.filter_type);
            }
            ChromaLowpass::None => {}
        };

        chroma_into_luma(
            yiq,
            &info,
            self.video_scanline_phase_shift,
            self.video_scanline_phase_shift_offset,
        );

        if self.composite_preemphasis != 0.0 {
            let preemphasis_filter = make_lowpass(
                (315000000.0 / 88.0 / 2.0) * self.bandwidth_scale,
                NTSC_RATE * self.bandwidth_scale,
            );
            filter_plane(
                yiq.y,
                width,
                &preemphasis_filter,
                InitialCondition::Zero,
                -self.composite_preemphasis,
                0,
            );
        }

        if let Some(noise) = &self.composite_noise {
            composite_noise(yiq, &info, &noise);
        }

        if self.snow_intensity > 0.0 && self.bandwidth_scale > 0.0 {
            snow(yiq, &info, self.snow_intensity * 0.01, self.snow_anisotropy);
        }

        if let Some(HeadSwitchingSettings {
            height,
            offset,
            horiz_shift,
            mid_line,
        }) = &self.head_switching
        {
            head_switching(
                yiq,
                &info,
                *height as usize,
                *offset as usize,
                *horiz_shift,
                mid_line.as_ref(),
            );
        }

        if let Some(TrackingNoiseSettings {
            height,
            wave_intensity,
            snow_intensity,
            snow_anisotropy,
            noise_intensity,
        }) = self.tracking_noise
        {
            tracking_noise(
                yiq,
                &info,
                height as usize,
                wave_intensity,
                snow_intensity,
                snow_anisotropy,
                noise_intensity,
            );
        }

        luma_into_chroma(
            yiq,
            &info,
            self.chroma_demodulation,
            self.video_scanline_phase_shift,
            self.video_scanline_phase_shift_offset,
        );

        if self.luma_smear > 0.0 {
            luma_smear(yiq, &info, self.luma_smear);
        }

        if let Some(ringing) = &self.ringing {
            let notch_filter = make_notch_filter(
                (ringing.frequency / self.bandwidth_scale).clamp(0.0, 1.0),
                ringing.power,
            );
            filter_plane(
                yiq.y,
                width,
                &notch_filter,
                InitialCondition::FirstSample,
                ringing.intensity,
                1,
            );
        }

        if let Some(luma_noise_settings) = &self.luma_noise {
            plane_noise(
                &mut yiq.y,
                yiq.dimensions.0,
                &info,
                &luma_noise_settings,
                noise_seeds::VIDEO_LUMA,
            );
        }

        if let Some(chroma_noise_settings) = &self.chroma_noise {
            plane_noise(
                &mut yiq.i,
                yiq.dimensions.0,
                &info,
                &chroma_noise_settings,
                noise_seeds::VIDEO_CHROMA_I,
            );
            plane_noise(
                &mut yiq.q,
                yiq.dimensions.0,
                &info,
                &chroma_noise_settings,
                noise_seeds::VIDEO_CHROMA_Q,
            );
        }

        if self.chroma_phase_error > 0.0 {
            chroma_phase_error(yiq, self.chroma_phase_error);
        }

        if self.chroma_phase_noise_intensity > 0.0 {
            chroma_phase_noise(yiq, &info, self.chroma_phase_noise_intensity);
        }

        if self.chroma_delay.0 != 0.0 || self.chroma_delay.1 != 0 {
            chroma_delay(
                yiq,
                &info,
                (self.chroma_delay.0, self.chroma_delay.1 as isize),
            );
        }

        if let Some(vhs_settings) = &self.vhs_settings {
            if let Some(edge_wave) = &vhs_settings.edge_wave {
                if edge_wave.intensity > 0.0 {
                    vhs_edge_wave(yiq, &info, edge_wave);
                }
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
                let luma_filter = make_lowpass_for_type(
                    luma_cut,
                    NTSC_RATE * self.bandwidth_scale,
                    self.filter_type,
                );
                let chroma_filter = make_lowpass_for_type(
                    chroma_cut,
                    NTSC_RATE * self.bandwidth_scale,
                    self.filter_type,
                );
                filter_plane(yiq.y, width, &luma_filter, InitialCondition::Zero, 1.0, 0);
                filter_plane(
                    yiq.i,
                    width,
                    &chroma_filter,
                    InitialCondition::Zero,
                    1.0,
                    chroma_delay,
                );
                filter_plane(
                    yiq.q,
                    width,
                    &chroma_filter,
                    InitialCondition::Zero,
                    1.0,
                    chroma_delay,
                );
                let luma_filter_single = make_lowpass(luma_cut, NTSC_RATE * self.bandwidth_scale);
                filter_plane(
                    yiq.y,
                    width,
                    &luma_filter_single,
                    InitialCondition::Zero,
                    -1.6,
                    0,
                );
            }

            if vhs_settings.chroma_loss > 0.0 {
                chroma_loss(yiq, &info, vhs_settings.chroma_loss);
            }

            if let Some(sharpen) = &vhs_settings.sharpen {
                if let Some(tape_speed) = &vhs_settings.tape_speed {
                    let VHSTapeParams { luma_cut, .. } = tape_speed.filter_params();
                    let frequency_extra_multiplier = match self.filter_type {
                        FilterType::ConstantK => 4.0,
                        FilterType::Butterworth => 1.0,
                    };
                    let luma_sharpen_filter = make_lowpass_for_type(
                        luma_cut * frequency_extra_multiplier * sharpen.frequency,
                        NTSC_RATE * self.bandwidth_scale,
                        self.filter_type,
                    );
                    // The composite-video-simulator code sharpens the chroma plane, but ntscqt and this effect do not.
                    // I'm not sure if I'm implementing it wrong, but chroma sharpening looks awful.
                    // let chroma_sharpen_filter = make_lowpass_triple(chroma_cut * 4.0, 0.0, NTSC_RATE);
                    filter_plane(
                        yiq.y,
                        width,
                        &luma_sharpen_filter,
                        InitialCondition::Zero,
                        -sharpen.intensity * 2.0 * sharpen.frequency,
                        0,
                    );
                    // filter_plane_scaled(&mut yiq.i, width, &chroma_sharpen_filter, -vhs_settings.sharpen * 0.85);
                    // filter_plane_scaled(&mut yiq.q, width, &chroma_sharpen_filter, -vhs_settings.sharpen * 0.85);
                }
            }
        }

        if self.chroma_vert_blend {
            chroma_vert_blend(yiq);
        }

        match self.chroma_lowpass_out {
            ChromaLowpass::Full => {
                composite_chroma_lowpass(yiq, &info, self.filter_type);
            }
            ChromaLowpass::Light => {
                composite_chroma_lowpass_lite(yiq, &info, self.filter_type);
            }
            ChromaLowpass::None => {}
        };
    }

    pub fn apply_effect_to_yiq(&self, yiq: &mut YiqView, frame_num: usize) {
        // On Windows debug builds, the stack overflows with the default stack size
        let pool = rayon::ThreadPoolBuilder::new()
            .stack_size(2 * 1024 * 1024)
            .build()
            .unwrap();
        pool.scope(|_| match yiq.field {
            YiqField::Upper | YiqField::Lower | YiqField::Both => {
                self.apply_effect_to_yiq_field(yiq, frame_num);
            }
            YiqField::InterleavedUpper | YiqField::InterleavedLower => {
                let (mut yiq_upper, mut yiq_lower, frame_num_upper, frame_num_lower) =
                    match yiq.field {
                        YiqField::InterleavedUpper => {
                            let num_upper_rows = YiqField::Upper.num_image_rows(yiq.dimensions.1);
                            let (upper, lower) = yiq.split_at_row(num_upper_rows);
                            (upper, lower, frame_num * 2, frame_num * 2 + 1)
                        }
                        YiqField::InterleavedLower => {
                            let num_lower_rows = YiqField::Lower.num_image_rows(yiq.dimensions.1);
                            let (lower, upper) = yiq.split_at_row(num_lower_rows);
                            (upper, lower, frame_num * 2 + 1, frame_num * 2)
                        }
                        _ => unreachable!(),
                    };
                yiq_upper.field = YiqField::Upper;
                yiq_lower.field = YiqField::Lower;
                self.apply_effect_to_yiq_field(&mut yiq_upper, frame_num_upper);
                self.apply_effect_to_yiq_field(&mut yiq_lower, frame_num_lower);
            }
        })
    }

    pub fn apply_effect_to_buffer<S: PixelFormat>(
        &self,
        dimensions: (usize, usize),
        input_frame: &mut [S::DataFormat],
        frame_num: usize,
    ) {
        let field = self.use_field.to_yiq_field(frame_num);
        let row_bytes = dimensions.0 * S::pixel_bytes();
        let mut yiq = YiqOwned::from_strided_buffer::<S>(
            input_frame,
            row_bytes,
            dimensions.0,
            dimensions.1,
            field,
        );
        let mut view = YiqView::from(&mut yiq);
        self.apply_effect_to_yiq(&mut view, frame_num);
        view.write_to_strided_buffer::<S, _>(
            input_frame,
            BlitInfo::from_full_frame(dimensions.0, dimensions.1, row_bytes),
            crate::yiq_fielding::DeinterlaceMode::Bob,
            identity,
        );
    }
}
