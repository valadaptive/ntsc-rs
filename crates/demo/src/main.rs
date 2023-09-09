use image::io::Reader as ImageReader;
use ntscrs::ntsc::NtscEffect;

fn main() {
    let img = ImageReader::open("/home/va_erie/Pictures/ntsc-test-1.png")
        .unwrap()
        .decode()
        .unwrap();
    let img = img.to_rgb8();

    println!("Decoded image");
    let filtered = NtscEffect::default().apply_effect(&img, 0);

    filtered
        .save("/home/va_erie/Pictures/ntsc-out.png")
        .unwrap();
}
