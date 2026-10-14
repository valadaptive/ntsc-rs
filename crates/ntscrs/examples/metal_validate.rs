//! Compare complete Metal renders with the CPU renderer without writing raw frames.
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "metal"))]
fn main() {
    use ntsc_rs::{
        Context, NtscEffect,
        metal::{self, ImageLayout},
        settings::standard::*,
        yiq_fielding::Rgbx,
    };
    use std::{ptr, time::Instant};
    let args: Vec<_> = std::env::args().collect();
    let w = args.get(1).and_then(|x| x.parse().ok()).unwrap_or(257usize);
    let h = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(33usize);
    let rounds = args.get(3).and_then(|x| x.parse().ok()).unwrap_or(1usize);
    let ctx = Context::new();
    let source: Vec<f32> = (0..w * h)
        .flat_map(|i| {
            let x = i % w;
            let y = i / w;
            [
                x as f32 / w as f32,
                y as f32 / h as f32,
                ((x * 19 + y * 31) % 257) as f32 / 256.,
                1.,
            ]
        })
        .collect();
    let images = metal::TestImages::new(&source);
    let direct = std::env::var_os("NTSC_METAL_DIRECT").is_some();
    let layout = ImageLayout {
        row_bytes: w * 16,
        components: 4,
        bytes_per_component: 4,
        flip_y: false,
    };
    let mut cases = Vec::new();
    cases.push(("default".to_string(), NtscEffect::default()));
    if rounds == 1 {
        for field in [
            UseField::Both,
            UseField::Upper,
            UseField::Lower,
            UseField::Alternating,
            UseField::InterleavedLower,
        ] {
            let mut s = NtscEffect::default();
            s.use_field = field;
            cases.push((format!("field-{field:?}"), s));
        }
        for kind in [FilterType::ConstantK, FilterType::Butterworth] {
            for demod in [
                ChromaDemodulationFilter::Box,
                ChromaDemodulationFilter::Notch,
                ChromaDemodulationFilter::OneLineComb,
                ChromaDemodulationFilter::TwoLineComb,
                ChromaDemodulationFilter::TwoD,
                ChromaDemodulationFilter::TwoDAdaptive,
            ] {
                let mut s = NtscEffect::default();
                s.filter_type = kind;
                s.chroma_demodulation = demod;
                s.vhs_settings.enabled = true;
                s.vhs_settings.settings.edge_wave.enabled = true;
                s.vhs_settings.settings.sharpen.enabled = true;
                s.head_switching.enabled = true;
                s.head_switching.settings.mid_line.enabled = true;
                s.tracking_noise.enabled = true;
                s.ringing.enabled = true;
                s.snow_intensity = 2.0;
                s.chroma_phase_error = 0.04;
                s.chroma_delay_horizontal = -2.7;
                s.chroma_delay_vertical = 2;
                cases.push((format!("all-{kind:?}-{demod:?}"), s));
            }
        }
        for mode in [LumaLowpass::Box, LumaLowpass::None] {
            let mut s = NtscEffect::default();
            s.input_luma_filter = mode;
            cases.push((format!("input-{mode:?}"), s));
        }
    }
    println!("case,width,height,max_abs,rms,cpu_ms,metal_wall_ms,gpu_ms");
    let mut failures = 0;
    for (name, s) in cases {
        let mut cpu = source.clone();
        let mut gpu = vec![0.0f32; source.len()];
        unsafe {
            metal::render(
                &ctx,
                &s,
                (w, h),
                0,
                [1., 1.],
                false,
                ptr::null_mut(),
                source.as_ptr() as _,
                layout,
                gpu.as_mut_ptr().cast(),
                layout,
            )
        }
        .unwrap();
        let mut cpu_times = Vec::new();
        let mut gpu_times = Vec::new();
        let mut device_times = Vec::new();
        for frame in 0..rounds {
            cpu.copy_from_slice(&source);
            let t = Instant::now();
            s.apply_effect_to_buffer::<Rgbx, f32>(&ctx, (w, h), &mut cpu, frame, [1., 1.]);
            cpu_times.push(t.elapsed().as_secs_f64() * 1000.);
            let t = Instant::now();
            let device = if direct {
                let result = unsafe {
                    metal::render(
                        &ctx,
                        &s,
                        (w, h),
                        frame,
                        [1., 1.],
                        false,
                        images.queue,
                        images.source,
                        layout,
                        images.destination,
                        layout,
                    )
                }
                .unwrap();
                images.wait();
                result
            } else {
                unsafe {
                    metal::render(
                        &ctx,
                        &s,
                        (w, h),
                        frame,
                        [1., 1.],
                        false,
                        ptr::null_mut(),
                        source.as_ptr() as _,
                        layout,
                        gpu.as_mut_ptr().cast(),
                        layout,
                    )
                }
                .unwrap()
            };
            gpu_times.push(t.elapsed().as_secs_f64() * 1000.);
            device_times.push(device);
        }
        if direct {
            images.read(&mut gpu);
        }
        let mut max = 0f64;
        let mut sum = 0f64;
        let mut index = 0;
        for (i, (a, b)) in cpu.iter().zip(&gpu).enumerate() {
            let diff = (*a as f64 - *b as f64).abs();
            if diff > max {
                max = diff;
                index = i;
            }
            sum += diff * diff;
            assert!(b.is_finite(), "nonfinite {name} {i}");
        }
        cpu_times.sort_by(f64::total_cmp);
        gpu_times.sort_by(f64::total_cmp);
        device_times.sort_by(f64::total_cmp);
        println!(
            "{name},{w},{h},{max:.9},{:.9},{:.3},{:.3},{}",
            (sum / cpu.len() as f64).sqrt(),
            cpu_times[rounds / 2],
            gpu_times[rounds / 2],
            if direct {
                "NA".to_owned()
            } else {
                format!("{:.3}", device_times[rounds / 2])
            }
        );
        if max > 0.0005 {
            eprintln!(
                "FAIL {name}: index {index} CPU={} GPU={}",
                cpu[index], gpu[index]
            );
            failures += 1;
        }
    }
    assert_eq!(failures, 0, "Metal parity failures");
}
#[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "metal")))]
fn main() {
    eprintln!("Requires macOS and --features metal");
}
