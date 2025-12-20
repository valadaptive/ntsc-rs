extern crate criterion;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ntscrs::{
    NtscEffect,
    yiq_fielding::{BlitInfo, Rgb, YiqView, pixel_bytes_for},
};
#[cfg(not(target_os = "windows"))]
use pprof::criterion::{Output, PProfProfiler};

const BENCH_IMAGE: &'static [u8] = include_bytes!("./balloons.png");

fn criterion_benchmark(c: &mut Criterion) {
    let img = image::load_from_memory_with_format(BENCH_IMAGE, image::ImageFormat::Png).unwrap();
    let img = img.to_rgb8();
    c.bench_function("full effect", |b| {
        b.iter_batched_ref(
            || {
                let img = img.clone();
                let width = img.width() as usize;
                let height = img.height() as usize;
                let effect = NtscEffect::default();
                let data = img.into_raw();
                let scratch =
                    vec![
                        0f32;
                        YiqView::buf_length_for((width, height), effect.use_field.to_yiq_field(0))
                    ];
                let dest = vec![0u8; data.len()];

                (width, height, effect, data, scratch, dest)
            },
            |(width, height, effect, buf, scratch, dest)| {
                let mut yiq = YiqView::from_parts(
                    scratch,
                    (*width, *height),
                    effect.use_field.to_yiq_field(0),
                );
                let row_bytes = *width * pixel_bytes_for::<Rgb, u8>();
                let blit_info = BlitInfo::from_full_frame(*width, *height, row_bytes);
                yiq.set_from_strided_buffer::<Rgb, u8, _>(buf, blit_info, ());
                effect.apply_effect_to_yiq(&mut yiq, 0, [1.0, 1.0]);
                yiq.write_to_strided_buffer::<Rgb, u8, _>(
                    dest,
                    blit_info,
                    ntscrs::yiq_fielding::DeinterlaceMode::Bob,
                    (),
                );
                black_box(dest);
            },
            criterion::BatchSize::LargeInput,
        );
    });
}

criterion_group! {
    name = benches;
    config = {
        #[cfg(not(target_os="windows"))]
        let config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
        #[cfg(target_os="windows")]
        let config = Criterion::default();

        config
    };
    targets = criterion_benchmark
}
criterion_main!(benches);
