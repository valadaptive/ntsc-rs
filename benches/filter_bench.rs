use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::io::Reader as ImageReader;
use ntscrs::filter::{TransferFunction, StateSpace};
use ntscrs::ntsc::{NtscEffect};

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n-1) + fibonacci(n-2),
    }
}

/// Create a lowpass filter with the given parameters, which can then be used to filter a signal.
fn make_lowpass(cutoff: f64, reset: f64, rate: f64) -> StateSpace {
    let time_interval = 1.0 / rate;
    let tau = (cutoff * 2.0 * core::f64::consts::PI).recip();
    let alpha = time_interval / (tau + time_interval);

    StateSpace::try_from(&TransferFunction::new(
        vec![alpha, 0.0],
        vec![1.0, -(1.0 - alpha)],
    ))
    .unwrap()
}

/// Create a lowpass filter with the given parameters, which can then be used to filter a signal.
fn make_lowpass_triple(cutoff: f64, reset: f64, rate: f64) -> StateSpace {
    let time_interval = 1.0 / rate;
    let tau = (cutoff * 2.0 * core::f64::consts::PI).recip();
    let alpha = time_interval / (tau + time_interval);

    let tf = TransferFunction::new(
        vec![alpha, 0.0],
        vec![1.0, -(1.0 - alpha)],
    );

    StateSpace::try_from(&(&(&tf * &tf) * &tf))
    .unwrap()
}

fn triple_filter_row(mut row: &mut Vec<f64>) {
    let filter = make_lowpass(600000.0, 0.0, (315000000.00 / 88.0) * 4.0);
    filter.filter_signal_in_place(row.into_iter());
    filter.filter_signal_in_place(row.into_iter());
    filter.filter_signal_in_place(row.into_iter());
}
fn single_filter_row(row: &mut Vec<f64>) {
    let filter = make_lowpass_triple(600000.0, 0.0, (315000000.00 / 88.0) * 4.0);
    filter.filter_signal_in_place(row.into_iter());
}

fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("/home/va_erie/Pictures/ntsc-test-1.png").unwrap().decode().unwrap();
    let img = img.as_rgb8().unwrap();
    /*let mut row: Vec<f64> = vec![];
    img.rows().next().unwrap().for_each(|px| {
        row.push(px.0[0] as f64 / 255.0);
    });

    c.bench_function("filter triple", |b| b.iter(|| {
        triple_filter_row(&mut row.clone())
    }));

    c.bench_function("filter single", |b| b.iter(|| {
        single_filter_row(&mut row.clone())
    }));*/
    c.bench_function("full effect", |b| b.iter(|| {
        NtscEffect::default().apply_effect(img, 0, 0);
    }));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);