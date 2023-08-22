pub enum BoundaryHandling {
    /// Repeat the boundary pixel over and over.
    Extend,
    /// Use a specific constant for the boundary.
    Constant(f64),
}

/// Shift a row by a non-integer amount using linear interpolation.
pub fn shift_row(row: &mut [f64], shift: f64, boundary_handling: BoundaryHandling) {
    // floor the shift (conversions round towards zero)
    let shift_int = shift as i64 - if shift < 0.0 { 1 } else { 0 };

    let width = row.len();
    let boundary_value = match boundary_handling {
        BoundaryHandling::Extend => {
            if shift > 0.0 {
                row[0]
            } else {
                row[width - 1]
            }
        }
        BoundaryHandling::Constant(value) => value,
    };

    // Do the integer part of the shift
    if shift_int > 0 {
        let offset = shift_int as usize;
        for i in (0..width).rev() {
            row[i] = if i >= offset {
                row[i - offset]
            } else {
                boundary_value
            }
        }
    } else {
        let offset = (-shift_int) as usize;
        for i in 0..width {
            row[i] = if i + offset < width {
                row[i + offset]
            } else {
                boundary_value
            }
        }
    }

    let shift_frac = if shift < 0.0 {
        1.0 - shift.fract().abs()
    } else {
        shift.fract()
    };

    // Interpolate
    let mut prev: f64 = row[0];
    for i in 0..width - 1 {
        let old_value = row[i];
        row[i] = (prev * shift_frac) + (row[i] * (1.0 - shift_frac));
        prev = old_value;
    }
}
