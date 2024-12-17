use std::fmt::Write;

pub fn format_eta(
    mut dest: impl Write,
    time_remaining: f64,
    units: [[&str; 2]; 3],
    separator: &str,
) {
    let mut time_remaining = time_remaining.ceil() as u64;
    let hrs = time_remaining / (60 * 60);
    time_remaining %= 60 * 60;
    let min = time_remaining / 60;
    time_remaining %= 60;
    let sec = time_remaining;

    let mut write_unit = |value, singular, plural, separator| {
        let unit_label = match value {
            1 => singular,
            _ => plural,
        };
        write!(dest, "{value}{unit_label}{separator}").unwrap();
    };

    let [[hours_singular, hours_plural], [minutes_singular, minutes_plural], [seconds_singular, seconds_plural]] =
        units;
    if hrs > 0 {
        write_unit(hrs, hours_singular, hours_plural, separator);
    }
    if min > 0 {
        write_unit(min, minutes_singular, minutes_plural, separator);
    }
    write_unit(sec, seconds_singular, seconds_plural, "");
}
