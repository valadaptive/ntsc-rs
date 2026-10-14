//! Compare serial and parallel processing with OFX-style float RGBX frames.
//! Run with --release; add --features rayon for the parallel version.
//! Optional arguments: width height measured_frames output_file field_mode.

use std::{hint::black_box, io::Write, time::Instant};

use ntsc_rs::{
    Context, NtscEffect,
    settings::UseField,
    yiq_fielding::{BlitInfo, DeinterlaceMode, Rgbx, YiqView},
};

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let number = |index: usize, default| {
        args.get(index)
            .map(|s| s.parse::<usize>().expect("expected a positive integer"))
            .unwrap_or(default)
    };
    let (width, height, frames) = (number(0, 3840), number(1, 2160), number(2, 12));
    assert!(width > 0 && height > 0 && frames > 0);
    let field_mode = args.get(4).map(String::as_str).unwrap_or("interleaved");
    let mut effect = NtscEffect::default();
    effect.use_field = match field_mode {
        "interleaved" => UseField::InterleavedUpper,
        "both" => UseField::Both,
        "alternating" => UseField::Alternating,
        _ => panic!("field mode must be interleaved, both, or alternating"),
    };
    let ctx = Context::new();
    // Deterministic gradients plus fine detail, with no file decoding in the timed path.
    let mut src = vec![0.0f32; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 4;
            src[offset] = x as f32 / width as f32;
            src[offset + 1] = y as f32 / height as f32;
            src[offset + 2] = ((x * 17 + y * 31) % 256) as f32 / 255.0;
            src[offset + 3] = 1.0;
        }
    }
    let mut dest = vec![0.0f32; src.len()];
    let blit = BlitInfo::from_full_frame(width, height, width * 4 * size_of::<f32>());
    let mut timings = Vec::with_capacity(frames);
    for frame in 0..frames + 3 {
        let start = Instant::now();
        let field = effect.use_field.to_yiq_field(frame);
        // The OFX plugin allocates and clears this scratch buffer on every render.
        let mut scratch = vec![0.0f32; YiqView::buf_length_for((width, height), field)];
        // Force physical zeroing, as the OFX host allocation path does.
        black_box(&mut scratch).fill(0.0);
        let allocated = Instant::now();
        let mut yiq = YiqView::from_parts(&mut scratch, (width, height), field);
        yiq.set_from_strided_buffer::<Rgbx, f32, _>(&ctx, &src, blit, ());
        let converted = Instant::now();
        effect.apply_effect_to_yiq(&ctx, &mut yiq, frame, [1.0, 1.0]);
        let processed = Instant::now();
        yiq.write_to_strided_buffer::<Rgbx, f32, _>(
            &ctx,
            &mut dest,
            blit,
            DeinterlaceMode::Bob,
            (),
        );
        black_box(&dest);
        drop(scratch);
        let finished = Instant::now();
        if frame >= 3 {
            timings.push([
                (allocated - start).as_secs_f64() * 1000.0,
                (converted - allocated).as_secs_f64() * 1000.0,
                (processed - converted).as_secs_f64() * 1000.0,
                (finished - processed).as_secs_f64() * 1000.0,
                (finished - start).as_secs_f64() * 1000.0,
            ]);
        }
    }
    let mut medians = [0.0; 5];
    for column in 0..5 {
        let mut values: Vec<_> = timings.iter().map(|row| row[column]).collect();
        values.sort_by(f64::total_cmp);
        medians[column] = if frames % 2 == 0 {
            (values[frames / 2 - 1] + values[frames / 2]) * 0.5
        } else {
            values[frames / 2]
        };
    }
    // Alpha is an unused channel in Rgbx. Compare exactly the RGB channels written by the effect.
    let mut checksum = 0xcbf29ce484222325u64;
    for pixel in dest.chunks_exact(4) {
        for value in &pixel[..3] {
            assert!(value.is_finite(), "non-finite output");
            for byte in value.to_le_bytes() {
                checksum = (checksum ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
    }
    if let Some(path) = args.get(3).filter(|s| s.as_str() != "-") {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        for pixel in dest.chunks_exact(4) {
            for value in &pixel[..3] {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
        }
        file.flush().unwrap();
    }
    println!(
        "width,height,field,rayon,threads,frames,alloc_ms,input_ms,effect_ms,output_free_ms,total_ms,fps,checksum"
    );
    println!(
        "{width},{height},{field_mode},{},{},{frames},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{checksum:016x}",
        cfg!(feature = "rayon"),
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "auto".into()),
        medians[0],
        medians[1],
        medians[2],
        medians[3],
        medians[4],
        1000.0 / medians[4],
    );
}
