use ntscrs::filter::{TransferFunction, StateSpace, polynomial_multiply};
use ntscrs::ntsc::{NtscEffect, make_lowpass};

use image::ImageFormat;
use image::io::Reader as ImageReader;

fn lptf(cutoff: f64, reset: f64, rate: f64) -> TransferFunction {
    let time_interval = 1.0 / rate;
    let tau = (cutoff * 2.0 * core::f64::consts::PI).recip();
    let alpha = time_interval / (tau + time_interval);

    TransferFunction::new(
        // TODO: I'm no filter expert but I'm think this is correct and ntsc-qt improperly implemented a
        // constant-k filter? Alpha *should* be the z^0 term in the numerator, right?
        vec![alpha, 0.0],
        vec![1.0, -(1.0 - alpha)],
    )
}

fn main() {
    let filt = TransferFunction::new(
        vec![0.5],
        vec![2.3, -1.2],
    );

    /*let butter = TransferFunction {
        num: vec![0.16666667, 0.5       , 0.5       , 0.16666667],
        den: vec![ 1.00000000e+00, -2.77555756e-16,  3.33333333e-01, -1.85037171e-17]
    };

    let filt = StateSpace::try_from(&butter).unwrap();
    let filt_tf = butter;*/

    let filt = make_lowpass(600000.0, (315000000.00 / 88.0) * 4.0);
    let filt_tf = lptf(600000.0, 0.0, (315000000.00 / 88.0) * 4.0);

    // let filt2 = make_lowpass(320000.0, 0.0, (315000000.00 / 88.0) * 4.0);

    // let res = StateSpace::try_from(filt).unwrap();

    // println!("{:?}", res);

    /*let butter = TransferFunction {
        num: vec![0.16666667, 0.5       , 0.5       , 0.16666667],
        den: vec![ 1.00000000e+00, -2.77555756e-16,  3.33333333e-01, -1.85037171e-17]
    };

    let butter_ss = StateSpace::try_from(butter).unwrap();*/

    let signal = vec![0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0];

    let filtered = filt.filter_signal(&signal);
    let filtered_alt = filt_tf.filter_signal(&signal);
    println!("{:?}", filtered);
    println!("{:?}", filtered_alt);

    let butter = TransferFunction::new(
        [0.26393202, 0.52786405, 0.52786405, 0.26393202, 0.0527864],
        [1.00000000e+00, -4.16333634e-16, 6.33436854e-01, -1.31860534e-16, 5.57280900e-02, -3.09353043e-18]
    );

    let ic = butter.steady_state_condition(1.0);
    println!("ic: {:?}", ic);

    /*let filtered = res.filter_signal(signal.clone());
    let triple_filtered = res.filter_signal(res.filter_signal(res.filter_signal(signal.clone())));

    let filt_tf = lptf(600000.0, 0.0, (315000000.00 / 88.0) * 4.0);
    let triple_filter = StateSpace::try_from(&(&filt_tf * &filt_tf) * &filt_tf).unwrap();
    let triple_filtered_alt = triple_filter.filter_signal(signal.clone());
    println!("{:?}", filtered);
    println!("{:?}", triple_filtered);
    println!("{:?}", triple_filtered_alt);*/

    //println!("{:?}", polynomial_multiply(vec![2.0, 1.0], vec![-5.0, 2.0]));

    let img = ImageReader::open("/home/va_erie/Pictures/ntsc-test-1.png").unwrap().decode().unwrap();
    let img = img.as_rgb8().unwrap();

    println!("Decoded image");
    let filtered = NtscEffect::default().apply_effect(img, 0, 456);

    filtered.save("/home/va_erie/Pictures/ntsc-out.png").unwrap();
}
