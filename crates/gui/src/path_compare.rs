use std::{cmp::Ordering, path::Path};

struct MaybeReplacementChar(bool);

impl Iterator for MaybeReplacementChar {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 {
            self.0 = false;
            Some('\u{FFFD}')
        } else {
            None
        }
    }
}

fn chars_lossy(bytes: &[u8]) -> impl Iterator<Item = char> + '_ {
    bytes.utf8_chunks().flat_map(|chunk| {
        chunk
            .valid()
            .chars()
            .chain(MaybeReplacementChar(!chunk.invalid().is_empty()))
    })
}

pub fn cmp_paths(a: &Path, b: &Path) -> Ordering {
    let [a_bytes, b_bytes] = [a, b].map(|path| path.as_os_str().as_encoded_bytes());

    // Case-insensitive compare first
    for (a_char, b_char) in chars_lossy(a_bytes)
        .flat_map(|c| c.to_lowercase())
        .zip(chars_lossy(b_bytes).flat_map(|c| c.to_lowercase()))
    {
        let ordering = a_char.cmp(&b_char);
        if !ordering.is_eq() {
            return ordering;
        }
    }

    // Then case-sensitive
    for (a_char, b_char) in chars_lossy(a_bytes).zip(chars_lossy(b_bytes)) {
        let ordering = a_char.cmp(&b_char);
        if !ordering.is_eq() {
            return ordering;
        }
    }

    // Fall back to comparing lengths
    a_bytes.len().cmp(&b_bytes.len())
}
