use std::ops::RangeInclusive;

pub fn clock_time_formatter(value: f64, _: RangeInclusive<usize>) -> String {
    clock_time_format(value as u64)
}

pub fn clock_time_format(value: u64) -> String {
    let display_duration = gstreamer::ClockTime::from_nseconds(value);
    format!("{:.*}", 2, display_duration)
}

pub fn clock_time_parser(input: &str) -> Option<f64> {
    let mut out_value: Option<u64> = None;
    const MULTIPLIERS: &[u64] = &[1_000, 60 * 1_000, 60 * 60 * 1_000];
    input
        .rsplit(':')
        .enumerate()
        .try_for_each(|(index, item)| -> Option<()> {
            let multiplier = MULTIPLIERS.get(index)?;
            if item.contains('.') {
                if index != 0 {
                    return None;
                }
                *out_value.get_or_insert(0) +=
                    (item.parse::<f64>().ok()? * *multiplier as f64) as u64;
            } else {
                *out_value.get_or_insert(0) += item.parse::<u64>().ok()? * *multiplier;
            }
            Some(())
        });
    out_value.map(|value| value as f64)
}
