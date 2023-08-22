extern crate criterion;
use criterion::{criterion_group, criterion_main, Criterion};
use image::io::Reader as ImageReader;
use ntscrs::ntsc::NtscEffect;
use pprof::criterion::{Output, PProfProfiler};

fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("/home/va_erie/Pictures/ntsc-test-1.png").unwrap().decode().unwrap();
    let img = img.as_rgb8().unwrap();
    c.bench_function("full effect", |b| b.iter(|| {
        NtscEffect::default().apply_effect(img, 0, 0);
    }));
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = criterion_benchmark
}
criterion_main!(benches);