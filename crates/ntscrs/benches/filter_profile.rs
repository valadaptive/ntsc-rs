extern crate criterion;
use std::convert::identity;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ntscrs::{
    ntsc::NtscEffect,
    yiq_fielding::{BlitInfo, PixelFormat as _, Rgb8, YiqView},
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

                (width, height, effect, data, scratch)
            },
            |(width, height, effect, buf, scratch)| {
                let mut yiq = YiqView::from_parts(
                    scratch,
                    (*width, *height),
                    effect.use_field.to_yiq_field(0),
                );
                let row_bytes = *width * Rgb8::pixel_bytes();
                yiq.set_from_strided_buffer::<Rgb8, _>(
                    buf,
                    BlitInfo::from_full_frame(*width, *height, row_bytes),
                    identity,
                );
                effect.apply_effect_to_yiq(&mut yiq, 0, [1.0, 1.0]);
                black_box(buf);
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
