//! Metal command planning. Random decisions and filter coefficients use the same Rust
//! implementation as the CPU renderer; image-sized work is encoded on the host's queue.
use super::*;
use alloc::{vec, vec::Vec};
use core::ffi::{c_char, c_void};

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Params {
    width: u32,
    height: u32,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    args: [u32; 8],
    values: [f32; 16],
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Command {
    kernel: u32,
    metadata: u32,
    p: Params,
}
const _: () = assert!(core::mem::size_of::<Params>() == 120);
const _: () = assert!(core::mem::size_of::<Command>() == 128);

unsafe extern "C" {
    fn ntscrs_metal_test_images(
        source: *const c_void,
        bytes: usize,
        queue: *mut *mut c_void,
        src: *mut *mut c_void,
        dst: *mut *mut c_void,
    ) -> *mut c_void;
    fn ntscrs_metal_test_wait(handle: *mut c_void, destination: *mut c_void, bytes: usize) -> i32;
    fn ntscrs_metal_test_free(handle: *mut c_void);
    fn ntscrs_metal_render(
        queue: *mut c_void,
        source: *mut c_void,
        destination: *mut c_void,
        source_bytes: usize,
        destination_bytes: usize,
        arena_bytes: usize,
        commands: *const Command,
        command_count: usize,
        metadata: *const f32,
        metadata_count: usize,
        shader: *const c_char,
        shader_len: usize,
        gpu_ms: *mut f64,
    ) -> i32;
}

/// Image layout. Rows are positive-strided; `flip_y` converts OFX's bottom-up coordinates.
#[derive(Clone, Copy)]
pub struct ImageLayout {
    pub row_bytes: usize,
    pub components: u32,
    pub bytes_per_component: u32,
    pub flip_y: bool,
}

/// Render with host-owned MTLBuffers on a host-owned MTLCommandQueue, or ordinary CPU
/// buffers with a null queue (synchronous, intended for verification and CPU-only hosts).
///
/// # Safety
/// Non-null queue/source/destination must be valid Objective-C Metal objects on the same
/// device. With a null queue, source/destination must address accessible memory covering
/// their layouts. Metal retains all resources until submitted work completes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn render(
    ctx: &Context,
    effect: &NtscEffect,
    dimensions: (usize, usize),
    frame: usize,
    scale: [f32; 2],
    gamma: bool,
    queue: *mut c_void,
    source: *mut c_void,
    src: ImageLayout,
    destination: *mut c_void,
    dst: ImageLayout,
) -> Result<f64, &'static str> {
    let (width, height) = dimensions;
    if width == 0
        || height == 0
        || width > 16384
        || height > 16384
        || ![src, dst].iter().all(|l| {
            matches!(l.components, 3 | 4)
                && matches!(l.bytes_per_component, 1 | 2 | 4)
                && l.row_bytes >= width * l.components as usize * l.bytes_per_component as usize
        })
        || source.is_null()
        || destination.is_null()
    {
        return Err("unsupported Metal image layout");
    }
    let mut plan = Plan::new(width, height);
    let field = effect.use_field.to_yiq_field(frame);
    let fields: Vec<(YiqField, usize)> = match field {
        YiqField::InterleavedUpper => vec![
            (YiqField::Upper, frame * 2),
            (YiqField::Lower, frame * 2 + 1),
        ],
        YiqField::InterleavedLower => vec![
            (YiqField::Lower, frame * 2),
            (YiqField::Upper, frame * 2 + 1),
        ],
        f => vec![(f, frame)],
    };
    for (f, field_frame) in fields {
        if f.num_actual_image_rows(height) == 0
            && matches!(
                field,
                YiqField::InterleavedUpper | YiqField::InterleavedLower
            )
        {
            continue;
        }
        plan.rows = f.num_image_rows(height);
        plan.planes = [0, 1, 2];
        let ctx = EffectCtx::new(ctx, effect, height, field_frame, scale);
        plan.convert(0, src, f, false, gamma);
        plan.effect(&ctx, effect);
        plan.convert(
            1,
            dst,
            f,
            matches!(field, YiqField::Upper | YiqField::Lower),
            gamma,
        );
    }
    // Lower-only on a one-line frame has no source field in the CPU implementation.
    if plan.commands.is_empty() {
        return Err("empty source field");
    }
    let shader = include_str!("metal/kernels.metal");
    let mut gpu_ms = 0.0;
    let status = unsafe {
        ntscrs_metal_render(
            queue,
            source,
            destination,
            src.row_bytes * height,
            dst.row_bytes * height,
            width * height * 6 * 4,
            plan.commands.as_ptr(),
            plan.commands.len(),
            plan.metadata.as_ptr(),
            plan.metadata.len(),
            shader.as_ptr().cast(),
            shader.len(),
            &mut gpu_ms,
        )
    };
    if status == 0 {
        Ok(gpu_ms)
    } else {
        Err("Metal render failed; see host log")
    }
}

struct Plan {
    width: usize,
    height: usize,
    rows: usize,
    serial_filters: bool,
    planes: [u32; 3],
    commands: Vec<Command>,
    metadata: Vec<f32>,
}
impl Plan {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            rows: height,
            serial_filters: false,
            planes: [0, 1, 2],
            commands: Vec::new(),
            metadata: Vec::new(),
        }
    }
    fn offset(&self, plane: u32) -> u32 {
        plane * (self.width * self.height) as u32
    }
    fn params(&self) -> Params {
        Params {
            width: self.width as u32,
            height: self.rows as u32,
            a: self.offset(self.planes[0]),
            b: self.offset(self.planes[1]),
            c: self.offset(self.planes[2]),
            ..Default::default()
        }
    }
    fn spare(&self, extra: Option<u32>) -> u32 {
        (0..6)
            .find(|p| !self.planes.contains(p) && Some(*p) != extra)
            .unwrap()
    }
    fn emit(&mut self, kernel: u32, p: Params, metadata: &[f32]) {
        // Metal buffer offsets must be aligned; 256 also works on discrete Mac GPUs.
        self.metadata
            .resize(self.metadata.len().div_ceil(64) * 64, 0.0);
        let offset = self.metadata.len() as u32 * 4;
        self.metadata.extend_from_slice(metadata);
        self.commands.push(Command {
            kernel,
            metadata: offset,
            p,
        });
    }
    fn convert(
        &mut self,
        kernel: u32,
        layout: ImageLayout,
        field: YiqField,
        bob: bool,
        gamma: bool,
    ) {
        let mut p = self.params();
        p.args = [
            self.height as u32,
            layout.row_bytes as u32,
            layout.components,
            layout.bytes_per_component,
            layout.flip_y as u32,
            match field {
                YiqField::Both => 2,
                YiqField::Upper => 0,
                _ => 1,
            },
            bob as u32,
            gamma as u32,
        ];
        self.emit(kernel, p, &[]);
    }
    fn filter(
        &mut self,
        channel: usize,
        filter: &TransferFunction,
        initial: bool,
        delay: usize,
        reserved: Option<u32>,
    ) {
        let dest = self.spare(reserved);
        let mut p = self.params();
        p.a = self.offset(self.planes[channel]);
        p.b = self.offset(dest);
        p.args[0] = delay as u32;
        p.args[1] = initial as u32;
        p.args[2] = filter.len() as u32;
        let coeff = filter.metal_coefficients();
        p.values[..7].copy_from_slice(&coeff);
        // Each SIMD lane independently filters a block, followed by an affine prefix scan.
        let block = (self.width + delay).div_ceil(32).max(1);
        p.args[3] = block as u32;
        let a = [
            [-coeff[4] as f64, 1.0, 0.0],
            [-coeff[5] as f64, 0.0, 1.0],
            [-coeff[6] as f64, 0.0, 0.0],
        ];
        fn mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
            core::array::from_fn(|i| {
                core::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum())
            })
        }
        let mut power = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
        let mut base = a;
        let mut n = block;
        while n != 0 {
            if n & 1 != 0 {
                power = mul(power, base);
            }
            base = mul(base, base);
            n >>= 1;
        }
        let mut transitions = Vec::with_capacity(45);
        for _ in 0..5 {
            transitions.extend(power.into_iter().flatten().map(|v| v as f32));
            power = mul(power, power);
        }
        // Near-unit poles amplify regrouping error. Preserve serial FMA order for
        // these extreme settings instead of changing the appearance for speed.
        let near_unit = (1.0 + coeff[4] + coeff[5] + coeff[6]).abs() < 0.001
            || (1.0 - coeff[4] + coeff[5] - coeff[6]).abs() < 0.001;
        self.emit(
            if near_unit || self.serial_filters {
                15
            } else {
                match filter.len() {
                    2 => 13,
                    3 => 14,
                    _ => 2,
                }
            },
            p,
            &transitions,
        );
        self.planes[channel] = dest;
    }
    fn lowpass(&mut self, ctx: &EffectCtx, mode: ChromaLowpass, kind: FilterType) {
        let (i, q, di, dq) = match mode {
            ChromaLowpass::Full => (1300000., 600000., 2, 4),
            ChromaLowpass::Light => (2600000., 2600000., 1, 1),
            ChromaLowpass::None => return,
        };
        self.filter(
            1,
            &make_lowpass_for_type(i, NTSC_RATE * ctx.horizontal_scale, kind),
            false,
            di,
            None,
        );
        self.filter(
            2,
            &make_lowpass_for_type(q, NTSC_RATE * ctx.horizontal_scale, kind),
            false,
            dq,
            None,
        );
    }
    fn noise(&mut self, ctx: &EffectCtx, channel: usize, s: &FbmNoiseSettings, seed: u64) {
        let rng = SplitMix64::new(ctx.seed)
            .mix(seed)
            .mix(ctx.frame_num as u64);
        let mut rows = Vec::with_capacity(self.rows * 4);
        for row in 0..self.rows {
            let mut rng = rng.clone().mix(row as u64);
            rows.extend([
                f32::from_bits(rng.random::<i32>() as u32),
                rng.random::<f32>() * self.width as f32,
                s.frequency / ctx.horizontal_scale,
                s.intensity * 0.25,
            ]);
        }
        self.noise_rows(channel, s.detail.clamp(1, 5) as u32, &rows);
    }
    fn noise_rows(&mut self, channel: usize, detail: u32, rows: &[f32]) {
        let mut p = self.params();
        p.a = self.offset(self.planes[channel]);
        p.args[0] = detail;
        self.emit(6, p, rows);
    }
    fn rotate(&mut self, angles: &[f32]) {
        let rows: Vec<f32> = angles
            .iter()
            .flat_map(|a| {
                let (s, c) = (a * PI * 2.0).sin_cos();
                [s, c]
            })
            .collect();
        self.emit(7, self.params(), &rows);
    }
    fn shift(&mut self, channel: usize, rows: &[f32], extend: bool, vertical: isize) {
        let dest = self.spare(None);
        let mut p = self.params();
        p.a = self.offset(self.planes[channel]);
        p.b = self.offset(dest);
        p.args[0] = extend as u32;
        p.args[1] = vertical as i32 as u32;
        self.emit(8, p, rows);
        self.planes[channel] = dest;
    }
    fn speckles(
        &mut self,
        ctx: &EffectCtx,
        rng: &SplitMix64,
        start: usize,
        cut: usize,
        total: usize,
        intensity: f32,
        anisotropy: f32,
        tracking: bool,
    ) {
        if intensity <= 0.0 || ctx.horizontal_scale <= 0.0 {
            return;
        }
        let mut events: Vec<(usize, f32)> = Vec::new();
        let mut sparse = Vec::new();
        for y in start..self.rows {
            let index = y - start + cut;
            events.clear();
            let scale = if tracking {
                (index as f32 / total as f32).powi(2)
            } else {
                1.0
            };
            ctx.for_each_speckle(
                self.width,
                &mut rng.clone().mix(index as u64),
                intensity * scale,
                anisotropy,
                |x, v| events.push((x, v)),
            );
            events.sort_by_key(|&(x, _)| x);
            let mut start = 0;
            while start < events.len() {
                let x = events[start].0;
                let mut value = 0.0;
                let mut end = start;
                while end < events.len() && events[end].0 == x {
                    value += events[end].1;
                    end += 1;
                }
                if value != 0.0 {
                    sparse.extend([f32::from_bits((y * self.width + x) as u32), value]);
                }
                start = end;
            }
        }
        if !sparse.is_empty() {
            let mut p = self.params();
            p.args[0] = (sparse.len() / 2) as u32;
            self.emit(9, p, &sparse);
        }
    }
    fn head(&mut self, ctx: &EffectCtx, s: &HeadSwitchingSettings) {
        let total = ((s.height.max(0) as usize) as f32 * ctx.vertical_scale).round() as usize;
        let offset = ((s.offset.max(0) as usize) as f32 * ctx.vertical_scale).round() as usize;
        if offset >= total {
            return;
        }
        let affected = total - offset;
        let start = self.rows.saturating_sub(affected);
        let cut = affected.saturating_sub(self.rows);
        let rng = SplitMix64::new(ctx.seed)
            .mix(noise_seeds::HEAD_SWITCHING)
            .mix(ctx.frame_num as u64);
        let mut rows = vec![0.0; self.rows * 4];
        for y in start..self.rows {
            let index = affected - (y - start + cut);
            rows[y * 4] = (s.horiz_shift * ((index + offset) as f32 / total as f32).powf(1.5)
                + (rng.clone().mix(index as u64).random::<f32>() - 0.5))
                * ctx.horizontal_scale;
            if index == affected
                && let Some(mid) = s.mid_line.as_option()
            {
                let mut r = SplitMix64::new(ctx.seed)
                    .mix(noise_seeds::HEAD_SWITCHING_MID_LINE_JITTER)
                    .mix(ctx.frame_num as u64);
                let jitter = ((r.random::<f32>() + r.random::<f32>()) * 0.5 - 0.5) * mid.jitter;
                rows[y * 4 + 1] = (self.width as f32 * (mid.position + jitter)) as usize as f32;
                rows[y * 4 + 2] = 16.0 * ctx.horizontal_scale;
                rows[y * 4 + 3] = (r.random::<f32>() + 0.5) * 0.5;
            }
        }
        self.shift(0, &rows, false, 0);
    }
    fn tracking(&mut self, ctx: &EffectCtx, s: &TrackingNoiseSettings) {
        let total = (s.height.max(0) as f32 * ctx.vertical_scale).round() as usize;
        if total == 0 {
            return;
        }
        let start = self.rows.saturating_sub(total);
        let cut = total.saturating_sub(self.rows);
        let mut rng = SplitMix64::new(ctx.seed)
            .mix(noise_seeds::TRACKING_NOISE)
            .mix(ctx.frame_num as u64);
        let seed = rng.random::<i32>();
        let offset = rng.random::<f32>() * self.rows as f32;
        let mut wave = vec![0.0; total.min(self.rows)];
        sample_noise_1d::<Simplex1d, _>(
            ctx.level,
            &Simplex {
                seed,
                frequency: 0.5,
            },
            [offset],
            [wave.len()],
            &mut wave,
        );
        let mut shifts = vec![0.0; self.rows * 4];
        let mut noise = vec![0.0; self.rows * 4];
        for y in start..self.rows {
            let index = y - start + cut;
            let intensity = index as f32 / total as f32;
            shifts[y * 4] =
                wave[y - start] * intensity * s.wave_intensity * 0.25 * ctx.horizontal_scale;
            let mut r = rng.clone().mix(index as u64);
            noise[y * 4] = f32::from_bits(r.random::<i32>() as u32);
            noise[y * 4 + 1] = r.random::<f32>() * self.width as f32;
            noise[y * 4 + 2] = 0.25 / ctx.horizontal_scale;
            noise[y * 4 + 3] = intensity.powi(2) * s.noise_intensity;
        }
        self.shift(0, &shifts, false, 0);
        self.noise_rows(0, 1, &noise);
        self.speckles(
            ctx,
            &rng,
            start,
            cut,
            total,
            s.snow_intensity,
            s.snow_anisotropy,
            true,
        );
    }
    fn edge_wave(&mut self, ctx: &EffectCtx, s: &VHSEdgeWaveSettings) {
        let mut rng = SplitMix64::new(ctx.seed).mix(noise_seeds::EDGE_WAVE);
        let seed = rng.random::<i32>();
        let offset = rng.random::<f32>() * self.rows as f32;
        let mut wave = vec![0.0; self.rows];
        sample_noise_2d::<Simplex2d, _>(
            ctx.level,
            &Fbm {
                seed,
                octaves: s.detail.clamp(1, 5) as usize,
                gain: FRAC_1_SQRT_2,
                lacunarity: 2.0,
                frequency: s.frequency / ctx.vertical_scale,
            },
            [offset, ctx.frame_num as f32 * s.speed],
            [self.rows, 1],
            &mut wave,
        );
        let rows: Vec<f32> = wave
            .iter()
            .flat_map(|v| {
                [
                    (v / 0.022) * s.intensity * 0.5 * ctx.horizontal_scale,
                    0.,
                    0.,
                    0.,
                ]
            })
            .collect();
        for c in 0..3 {
            self.shift(c, &rows, true, 0);
        }
    }
    fn effect(&mut self, ctx: &EffectCtx, s: &NtscEffect) {
        self.serial_filters = s.filter_type == FilterType::ConstantK && ctx.horizontal_scale > 8.0;
        match s.input_luma_filter {
            LumaLowpass::None => {}
            LumaLowpass::Notch => self.filter(0, &make_notch_filter(0.5, 2.0), true, 0, None),
            LumaLowpass::Box => {
                let dest = self.spare(None);
                let mut p = self.params();
                p.b = self.offset(dest);
                self.emit(12, p, &[]);
                self.planes[0] = dest;
            }
        }
        self.lowpass(ctx, s.chroma_lowpass_in, s.filter_type);
        let phases: Vec<f32> = (0..self.rows)
            .map(|r| ctx.chroma_phase_shift(r) as f32)
            .collect();
        self.emit(3, self.params(), &phases);
        if s.composite_sharpening != 0.0 {
            self.filter(
                0,
                &make_lowpass(
                    (315000000.0 / 88.0 / 2.0) * ctx.horizontal_scale,
                    NTSC_RATE * ctx.horizontal_scale,
                )
                .with_scale(-s.composite_sharpening),
                false,
                0,
                None,
            );
        }
        if s.composite_noise.enabled {
            self.noise(
                ctx,
                0,
                &s.composite_noise.settings,
                noise_seeds::VIDEO_COMPOSITE,
            );
        }
        if s.snow_intensity > 0.0 {
            self.speckles(
                ctx,
                &SplitMix64::new(ctx.seed)
                    .mix(noise_seeds::SNOW)
                    .mix(ctx.frame_num as u64),
                0,
                0,
                self.rows,
                s.snow_intensity * 0.01,
                s.snow_anisotropy,
                false,
            );
        }
        if s.head_switching.enabled {
            self.head(ctx, &s.head_switching.settings);
        }
        if s.tracking_noise.enabled {
            self.tracking(ctx, &s.tracking_noise.settings);
        }
        let modulated = self.planes[0];
        let mode = if (self.rows == 1
            && matches!(
                s.chroma_demodulation,
                ChromaDemodulationFilter::OneLineComb
                    | ChromaDemodulationFilter::TwoLineComb
                    | ChromaDemodulationFilter::TwoD
                    | ChromaDemodulationFilter::TwoDAdaptive
            ))
            || (self.width < 3
                && matches!(
                    s.chroma_demodulation,
                    ChromaDemodulationFilter::TwoD | ChromaDemodulationFilter::TwoDAdaptive
                )) {
            ChromaDemodulationFilter::Notch
        } else {
            s.chroma_demodulation
        };
        if mode == ChromaDemodulationFilter::Notch {
            self.filter(0, &make_notch_filter(0.5, 2.0), false, 0, Some(modulated));
        } else {
            let dest = self.spare(Some(modulated));
            let mut p = self.params();
            p.b = self.offset(dest);
            p.args[0] = match mode {
                ChromaDemodulationFilter::Box => 0,
                ChromaDemodulationFilter::OneLineComb => 1,
                ChromaDemodulationFilter::TwoLineComb => 2,
                ChromaDemodulationFilter::TwoD => 3,
                ChromaDemodulationFilter::TwoDAdaptive => 4,
                ChromaDemodulationFilter::Notch => unreachable!(),
            };
            self.emit(4, p, &[]);
            self.planes[0] = dest;
        }
        let mut p = self.params();
        p.d = self.offset(modulated);
        self.emit(5, p, &phases);
        if s.luma_smear > 0.0 {
            self.filter(
                0,
                &make_lowpass(f32::exp2(-4.0 * s.luma_smear) * 0.25, ctx.horizontal_scale),
                false,
                0,
                None,
            );
        }
        if s.ringing.enabled {
            let r = &s.ringing.settings;
            self.filter(
                0,
                &make_notch_filter((r.frequency / ctx.horizontal_scale).clamp(0., 1.), r.power)
                    .with_scale(r.intensity),
                true,
                1,
                None,
            );
        }
        if s.luma_noise.enabled {
            self.noise(ctx, 0, &s.luma_noise.settings, noise_seeds::VIDEO_LUMA);
        }
        if s.chroma_noise.enabled {
            self.noise(
                ctx,
                1,
                &s.chroma_noise.settings,
                noise_seeds::VIDEO_CHROMA_I,
            );
            self.noise(
                ctx,
                2,
                &s.chroma_noise.settings,
                noise_seeds::VIDEO_CHROMA_Q,
            );
        }
        if s.chroma_phase_error > 0.0 {
            self.rotate(&vec![s.chroma_phase_error; self.rows]);
        }
        if s.chroma_phase_noise_intensity > 0.0 {
            let rng = SplitMix64::new(ctx.seed)
                .mix(noise_seeds::VIDEO_CHROMA_PHASE)
                .mix(ctx.frame_num as u64);
            let angles: Vec<f32> = (0..self.rows)
                .map(|r| {
                    (rng.clone().mix(r as u64).random::<f32>() - 0.5)
                        * 2.0
                        * s.chroma_phase_noise_intensity
                })
                .collect();
            self.rotate(&angles);
        }
        if s.chroma_delay_horizontal != 0.0 || s.chroma_delay_vertical != 0 {
            let rows: Vec<f32> = (0..self.rows)
                .flat_map(|_| [s.chroma_delay_horizontal * ctx.horizontal_scale, 0., 0., 0.])
                .collect();
            let vertical = (s.chroma_delay_vertical as f32 * ctx.vertical_scale).round() as isize;
            self.shift(1, &rows, false, vertical);
            self.shift(2, &rows, false, vertical);
        }
        if let Some(v) = s.vhs_settings.as_option() {
            if v.edge_wave.enabled && v.edge_wave.settings.intensity > 0.0 {
                self.edge_wave(ctx, &v.edge_wave.settings);
            }
            if let Some(t) = v.tape_speed.filter_params() {
                let delay = (t.chroma_delay as f32 * ctx.horizontal_scale).round() as usize;
                self.filter(
                    0,
                    &make_lowpass_for_type(
                        t.luma_cut,
                        NTSC_RATE * ctx.horizontal_scale,
                        s.filter_type,
                    ),
                    false,
                    0,
                    None,
                );
                let c = make_lowpass_for_type(
                    t.chroma_cut,
                    NTSC_RATE * ctx.horizontal_scale,
                    s.filter_type,
                );
                self.filter(1, &c, false, delay, None);
                self.filter(2, &c, false, delay, None);
                self.filter(
                    0,
                    &make_lowpass(t.luma_cut, NTSC_RATE * ctx.horizontal_scale).with_scale(-1.6),
                    false,
                    0,
                    None,
                );
            }
            if v.chroma_loss > 0.0 {
                let mut rng = SplitMix64::new(ctx.seed)
                    .mix(noise_seeds::CHROMA_LOSS)
                    .mix(ctx.frame_num as u64);
                let dist = geometric_lambda(v.chroma_loss as f64);
                let mut rows = vec![0.0; self.rows];
                let mut r = 0usize;
                loop {
                    r = r.saturating_add(rng.random_geometric(dist));
                    if r >= self.rows {
                        break;
                    }
                    rows[r] = 1.;
                    r += 1;
                }
                self.emit(11, self.params(), &rows);
            }
            if v.sharpen.enabled
                && let Some(t) = v.tape_speed.filter_params()
            {
                let sh = &v.sharpen.settings;
                let mult = if s.filter_type == FilterType::ConstantK {
                    4.0
                } else {
                    1.0
                };
                self.filter(
                    0,
                    &make_lowpass_for_type(
                        t.luma_cut * mult * sh.frequency,
                        NTSC_RATE * ctx.horizontal_scale,
                        s.filter_type,
                    )
                    .with_scale(-sh.intensity * 2.0 * sh.frequency),
                    false,
                    0,
                    None,
                );
            }
        }
        if s.chroma_vert_blend && self.height >= 2 {
            for c in [1, 2] {
                let dest = self.spare(None);
                let mut p = self.params();
                p.a = self.offset(self.planes[c]);
                p.b = self.offset(dest);
                self.emit(10, p, &[]);
                self.planes[c] = dest;
            }
        }
        self.lowpass(ctx, s.chroma_lowpass_out, s.filter_type);
    }
}

/// Reusable images for measuring the same direct-buffer path used by an OFX host.
#[doc(hidden)]
pub struct TestImages {
    handle: *mut c_void,
    pub queue: *mut c_void,
    pub source: *mut c_void,
    pub destination: *mut c_void,
    len: usize,
}
impl TestImages {
    pub fn new(source: &[f32]) -> Self {
        let mut s = Self {
            handle: core::ptr::null_mut(),
            queue: core::ptr::null_mut(),
            source: core::ptr::null_mut(),
            destination: core::ptr::null_mut(),
            len: source.len(),
        };
        s.handle = unsafe {
            ntscrs_metal_test_images(
                source.as_ptr().cast(),
                core::mem::size_of_val(source),
                &mut s.queue,
                &mut s.source,
                &mut s.destination,
            )
        };
        assert!(!s.handle.is_null());
        s
    }
    pub fn wait(&self) {
        assert_eq!(
            unsafe { ntscrs_metal_test_wait(self.handle, core::ptr::null_mut(), 0) },
            0
        );
    }
    pub fn read(&self, destination: &mut [f32]) {
        assert_eq!(destination.len(), self.len);
        assert_eq!(
            unsafe {
                ntscrs_metal_test_wait(
                    self.handle,
                    destination.as_mut_ptr().cast(),
                    core::mem::size_of_val(destination),
                )
            },
            0
        );
    }
}
impl Drop for TestImages {
    fn drop(&mut self) {
        self.wait();
        unsafe { ntscrs_metal_test_free(self.handle) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yiq_fielding::{DeinterlaceMode, Rect, Rgb, Rgbx};

    fn check<P: PixelFormat>(
        w: usize,
        h: usize,
        frame: usize,
        effect: &NtscEffect,
        flip: bool,
        gamma: bool,
    ) {
        let ctx = Context::new();
        let components = P::NUM_COMPONENTS;
        let stride = w * components + 4;
        let mut input = vec![0.0f32; stride * h];
        for y in 0..h {
            for x in 0..w {
                for c in 0..components {
                    input[y * stride + x * components + c] =
                        ((x * 13 + y * 19 + c * 47) % 251) as f32 / 250.0;
                }
            }
        }
        let mut cpu = vec![0.0f32; stride * h];
        let mut gpu = vec![0.0f32; stride * h];
        let field = effect.use_field.to_yiq_field(frame);
        let mut scratch = vec![0.0; YiqView::buf_length_for((w, h), field)];
        let mut view = YiqView::from_parts(&mut scratch, (w, h), field);
        let blit = BlitInfo::new(Rect::from_width_height(w, h), (0, 0), stride * 4, h, flip);
        let to_srgb = |rgb: [f32; 3]| {
            rgb.map(|v| {
                if gamma {
                    if v <= 0.0031308 {
                        v * 12.92
                    } else {
                        1.055 * v.powf(1.0 / 2.4) - 0.055
                    }
                } else {
                    v
                }
            })
        };
        let from_srgb = |rgb: [f32; 3]| {
            rgb.map(|v| {
                if gamma {
                    if v <= 0.04045 {
                        v / 12.92
                    } else {
                        ((v + 0.055) / 1.055).powf(2.4)
                    }
                } else {
                    v
                }
            })
        };
        view.set_from_strided_buffer::<P, f32, _>(&ctx, &input, blit, to_srgb);
        effect.apply_effect_to_yiq(&ctx, &mut view, frame, [1.0, 1.0]);
        view.write_to_strided_buffer::<P, f32, _>(
            &ctx,
            &mut cpu,
            blit,
            DeinterlaceMode::Bob,
            from_srgb,
        );
        let layout = ImageLayout {
            row_bytes: stride * 4,
            components: components as u32,
            bytes_per_component: 4,
            flip_y: flip,
        };
        unsafe {
            render(
                &ctx,
                effect,
                (w, h),
                frame,
                [1., 1.],
                gamma,
                core::ptr::null_mut(),
                input.as_ptr() as _,
                layout,
                gpu.as_mut_ptr().cast(),
                layout,
            )
        }
        .unwrap();
        let mut max = 0f32;
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let i = y * stride + x * components + c;
                    assert!(
                        cpu[i].is_finite() && gpu[i].is_finite(),
                        "nonfinite {w}x{h} field {:?} at {i}",
                        effect.use_field
                    );
                    max = max.max((cpu[i] - gpu[i]).abs());
                }
            }
        }
        assert!(
            max < 0.0005,
            "max error {max}, {w}x{h}, frame {frame}, field {:?}, gamma {gamma}, effect {effect:?}",
            effect.use_field
        );
    }

    #[test]
    #[ignore = "requires a Metal GPU; run explicitly on a Mac"]
    fn metal_fields_strides_gamma() {
        for field in [
            UseField::Both,
            UseField::Upper,
            UseField::Lower,
            UseField::Alternating,
            UseField::InterleavedUpper,
            UseField::InterleavedLower,
        ] {
            let mut effect = NtscEffect::default();
            effect.use_field = field;
            for frame in [0, 1, 47] {
                for gamma in [false, true] {
                    check::<Rgbx>(257, 17, frame, &effect, true, gamma);
                    check::<Rgb>(31, 9, frame, &effect, false, gamma);
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a Metal GPU; run explicitly on a Mac"]
    fn metal_scaled_filters() {
        for kind in [FilterType::Butterworth, FilterType::ConstantK] {
            for scale in [0.125, 0.5, 1., 4.5, 8., 36.] {
                let mut effect = NtscEffect::default();
                effect.filter_type = kind;
                effect.scale.enabled = true;
                effect.scale.settings.horizontal_scale = scale;
                effect.scale.settings.scale_with_video_size = false;
                effect.vhs_settings.enabled = true;
                effect.vhs_settings.settings.sharpen.enabled = true;
                effect.ringing.enabled = true;
                check::<Rgbx>(721, 25, 31, &effect, false, false);
            }
        }
    }

    #[test]
    #[ignore = "requires a Metal GPU; run explicitly on a Mac"]
    fn metal_concurrent_renders() {
        std::thread::scope(|scope| {
            for frame in 10..14 {
                scope.spawn(move || {
                    check::<Rgbx>(129, 19, frame, &NtscEffect::default(), true, false)
                });
            }
        });
    }

    #[test]
    #[ignore = "requires a Metal GPU; run explicitly on a Mac"]
    fn metal_two_d_filters() {
        for mode in [
            ChromaDemodulationFilter::TwoD,
            ChromaDemodulationFilter::TwoDAdaptive,
        ] {
            let mut effect = NtscEffect::default();
            effect.chroma_demodulation = mode;
            effect.use_field = UseField::Both;
            for (w, h) in [(1, 7), (2, 7), (5, 1), (5, 7), (257, 17)] {
                for frame in [0, 1, 47] {
                    check::<Rgbx>(w, h, frame, &effect, true, false);
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a Metal GPU; run explicitly on a Mac"]
    fn metal_integer_formats() {
        fn check<T: Normalize + Default + Into<f64>>(input: Vec<T>, width: usize, height: usize) {
            let ctx = Context::new();
            let effect = NtscEffect::default();
            let mut cpu = vec![T::default(); input.len()];
            let mut gpu = vec![T::default(); input.len()];
            let field = effect.use_field.to_yiq_field(7);
            let stride = width * 4 * core::mem::size_of::<T>();
            let mut scratch = vec![0.0; YiqView::buf_length_for((width, height), field)];
            let mut view = YiqView::from_parts(&mut scratch, (width, height), field);
            let blit = BlitInfo::from_full_frame(width, height, stride);
            view.set_from_strided_buffer::<Rgbx, T, _>(&ctx, &input, blit, ());
            effect.apply_effect_to_yiq(&ctx, &mut view, 7, [1.0, 1.0]);
            view.write_to_strided_buffer::<Rgbx, T, _>(
                &ctx,
                &mut cpu,
                blit,
                DeinterlaceMode::Bob,
                (),
            );
            let layout = ImageLayout {
                row_bytes: stride,
                components: 4,
                bytes_per_component: core::mem::size_of::<T>() as u32,
                flip_y: false,
            };
            unsafe {
                render(
                    &ctx,
                    &effect,
                    (width, height),
                    7,
                    [1., 1.],
                    false,
                    core::ptr::null_mut(),
                    input.as_ptr() as _,
                    layout,
                    gpu.as_mut_ptr().cast(),
                    layout,
                )
            }
            .unwrap();
            for (a, b) in cpu.into_iter().zip(gpu) {
                assert!((a.into() - b.into()).abs() <= 1.0);
            }
        }
        let (w, h) = (37, 9);
        check((0..w * h * 4).map(|i| (i * 17 % 256) as u8).collect(), w, h);
        check(
            (0..w * h * 4).map(|i| (i * 997 % 65536) as u16).collect(),
            w,
            h,
        );
    }
}
