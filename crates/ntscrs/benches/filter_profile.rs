extern crate criterion;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::ImageReader;
use ntscrs::{ntsc::NtscEffect, yiq_fielding::Rgb8};
#[cfg(not(target_os = "windows"))]
use pprof::criterion::{Output, PProfProfiler};

fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("./benches/balloons.png")
        .unwrap()
        .decode()
        .unwrap();
    let img = img.to_rgb8();
    c.bench_function("full effect", |b| {
        b.iter(|| {
            let img = img.clone();
            let width = img.width() as usize;
            let height = img.height() as usize;
            let mut buf = img.into_raw();
            NtscEffect::default().apply_effect_to_buffer::<Rgb8>((width, height), &mut buf, 0);
            black_box(&mut buf);
        })
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
