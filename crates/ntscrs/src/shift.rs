#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundaryHandling {
    /// Repeat the boundary pixel over and over.
    Extend,
    /// Use a specific constant for the boundary.
    Constant(f32),
}

fn shift_row_initial_conditions(row: &[f32], shift: f32, boundary_handling: BoundaryHandling) -> (isize, f32, f32, f32) {
    // Floor the shift (conversions round towards zero)
    let shift_int = shift as isize - if shift < 0.0 { 1 } else { 0 };

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

    let prev = if shift_int >= width as isize || shift_int < -(width as isize) {
        boundary_value
    } else if shift_int >= 0 {
        row[(width - shift_int as usize) - 1]
    } else {
        row[-shift_int as usize - 1]
    };

    (shift_int, shift_frac, boundary_value, prev)
}

/// Shift a row by a non-integer amount using linear interpolation.
pub fn shift_row(row: &mut [f32], shift: f32, boundary_handling: BoundaryHandling) {
    let width = row.len();
    let (shift_int, shift_frac, boundary_value, mut prev) = shift_row_initial_conditions(row, shift, boundary_handling);

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
        // Shift backwards; iterate the list forwards
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
    let (shift_int, shift_frac, boundary_value, mut prev) = shift_row_initial_conditions(src, shift, boundary_handling);
    if shift_int >= 0 {
        // Shift forwards; iterate the list backwards
        let offset = shift_int as usize + 1;
        for i in (0..width).rev() {
            let old_value = if i >= offset {
                src[i - offset]
            } else {
                boundary_value
            };
            dst[i] = (prev * (1.0 - shift_frac)) + (old_value * shift_frac);
            prev = old_value;
        }
    } else {
        // Shift backwards; iterate the list forwards
        let offset = (-shift_int) as usize;
        for i in 0..width {
            let old_value = if i + offset < width {
                src[i + offset]
            } else {
                boundary_value
            };
            dst[i] = (prev * shift_frac) + (old_value * (1.0 - shift_frac));
            prev = old_value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_DATA: &[f32] = &[1.0, 2.5, -0.7, 0.0, 0.0, 2.2, 0.3];

    fn assert_almost_eq(a: &[f32], b: &[f32]) {
        let all_almost_equal = a.into_iter().zip(b.into_iter()).all(|(a, b)| {
            (a - b).abs() <= 0.01
        });
        assert!(all_almost_equal, "{a:?} is almost equal to {b:?}");
    }

    fn test_case(shift: f32, boundary_handling: BoundaryHandling, expected: &[f32]) {
        let mut shifted = TEST_DATA.to_vec();
        shift_row(&mut shifted, shift, boundary_handling);
        assert_almost_eq(&shifted, expected);

        let mut shifted = TEST_DATA.to_vec();
        shift_row_to(TEST_DATA, &mut shifted, shift, boundary_handling);
        assert_almost_eq(&shifted, expected);
    }

    #[test]
    fn test_shift_pos_1() {
        test_case(0.5, BoundaryHandling::Extend, &[
            1.0,
            1.75,
            0.9,
            -0.35,
            0.0,
            1.1,
            1.25,
        ]);
    }

    #[test]
    fn test_shift_pos_2() {
        test_case(5.5, BoundaryHandling::Extend, &[
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.75,
        ]);
    }

    #[test]
    fn test_shift_neg_1() {
        test_case(-0.01, BoundaryHandling::Extend, &[
            1.015,
            2.468,
            -0.693,
            0.0,
            0.02199998,
            2.181,
            0.3,
        ]);
    }

    #[test]
    fn test_shift_neg_2() {
        test_case(-1.01, BoundaryHandling::Extend, &[
            2.468,
            -0.693,
            0.0,
            0.02199998,
            2.181,
            0.3,
            0.3,
        ]);
    }

    #[test]
    fn test_shift_neg_full() {
        test_case(-6.0, BoundaryHandling::Extend, &[
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
        ]);
    }

    #[test]
    fn test_shift_neg_full_ext() {
        test_case(-7.0, BoundaryHandling::Extend, &[
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
        ]);
    }

    #[test]
    fn test_shift_pos_full() {
        test_case(6.0, BoundaryHandling::Extend, &[
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]);
    }

    #[test]
    fn test_shift_pos_full_ext() {
        test_case(7.0, BoundaryHandling::Extend, &[
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]);
    }
}