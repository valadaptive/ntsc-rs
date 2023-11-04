extern crate criterion;
use criterion::{criterion_group, criterion_main, Criterion};
use image::io::Reader as ImageReader;
use ntscrs::ntsc::NtscEffect;
#[cfg(not(target_os="windows"))]
use pprof::criterion::{Output, PProfProfiler};

fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("./benches/balloons.png")
        .unwrap()
        .decode()
        .unwrap();
    let img = img.to_rgb8();
    c.bench_function("full effect", |b| {
        b.iter(|| {
            NtscEffect::default().apply_effect(&img, 0);
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
