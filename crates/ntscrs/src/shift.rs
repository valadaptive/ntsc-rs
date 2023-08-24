pub enum BoundaryHandling {
    /// Repeat the boundary pixel over and over.
    Extend,
    /// Use a specific constant for the boundary.
    Constant(f32),
}

fn shift_row_initial_conditions(row: &[f32], shift: f32, boundary_handling: BoundaryHandling) -> (i64, f32, f32) {
    // Floor the shift (conversions round towards zero)
    let shift_int = shift as i64 - if shift < 0.0 { 1 } else { 0 };

    let width = row.len();
    let boundary_value = match boundary_handling {
        BoundaryHandling::Extend => {
            if shift_int >= 0 {
                row[0]
            } else {
                row[width - 1]
            }
        }
        BoundaryHandling::Constant(value) => value,
    };
    let shift_frac = if shift < 0.0 {
        1.0 - shift.fract().abs()
    } else {
        shift.fract()
    };

    (shift_int, shift_frac, boundary_value)
}

/// Shift a row by a non-integer amount using linear interpolation.
pub fn shift_row(row: &mut [f32], shift: f32, boundary_handling: BoundaryHandling) {
    let width = row.len();
    let (shift_int, shift_frac, boundary_value) = shift_row_initial_conditions(row, shift, boundary_handling);

    let mut prev = if shift_int >= 0 {
        row[width - shift_int as usize - 1]
    } else {
        row[-shift_int as usize - 1]
    };

    if shift_int >= 0 {
        // Shift forwards; iterate the list backwards
        let offset = shift_int as usize + 1;
        for i in (0..width).rev() {
            let old_value = if i >= offset {
                row[i - offset]
            } else {
                boundary_value
            };
            row[i] = (prev * (1.0 - shift_frac)) + (old_value * shift_frac);
            prev = old_value;
        }
    } else {
        // Shift backwards; iterate the list backwards
        let offset = (-shift_int) as usize;
        for i in 0..width {
            let old_value = if i + offset < width {
                row[i + offset]
            } else {
                boundary_value
            };
            row[i] = (prev * shift_frac) + (old_value * (1.0 - shift_frac));
            prev = old_value;
        }
    }
}

/// Shift a row by a non-integer amount using linear interpolation.
pub fn shift_row_to(src: &[f32], dst: &mut [f32], shift: f32, boundary_handling: BoundaryHandling) {
    let width = src.len();
    let (shift_int, shift_frac, boundary_value) = shift_row_initial_conditions(src, shift, boundary_handling);

    let mut prev = if shift_int >= 0 {
        boundary_value
    } else {
        src[-shift_int as usize - 1]
    };

    for i in 0..width {
        let old_value = if i.wrapping_sub(shift_int as usize) < width {
            src[i.wrapping_sub(shift_int as usize)]
        } else {
            boundary_value
        };
        dst[i] = (prev * shift_frac) + (old_value * (1.0 - shift_frac));
        prev = old_value;
    }
}
