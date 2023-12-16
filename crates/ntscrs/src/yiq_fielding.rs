use glam::{Mat3, Vec3};
use image::RgbImage;
use rayon::prelude::*;

const YIQ_MATRIX: Mat3 = Mat3 {
    x_axis: Vec3 {
        x: 0.299,
        y: 0.5959,
        z: 0.2115,
    },
    y_axis: Vec3 {
        x: 0.587,
        y: -0.2746,
        z: -0.5227,
    },
    z_axis: Vec3 {
        x: 0.114,
        y: -0.3213,
        z: 0.3112,
    },
};

const RGB_MATRIX: Mat3 = Mat3 {
    x_axis: Vec3 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    },
    y_axis: Vec3 {
        x: 0.956,
        y: -0.272,
        z: -1.106,
    },
    z_axis: Vec3 {
        x: 0.619,
        y: -0.647,
        z: 1.703,
    },
};

#[inline(always)]
pub fn rgb_to_yiq(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (YIQ_MATRIX * Vec3::new(r, g, b)).into()
}

#[inline(always)]
pub fn yiq_to_rgb(y: f32, i: f32, q: f32) -> (f32, f32, f32) {
    (RGB_MATRIX * Vec3::new(y, i, q)).into()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum YiqField {
    Upper,
    Lower,
    Both,
}

impl YiqField {
    pub fn num_image_rows(&self, image_height: usize) -> usize {
        // On an image with an odd input height, we do ceiling division if we render upper-field-first
        // (take an image 3 pixels tall. it goes render, skip, render--that's 2 renders) but floor division if we
        // render lower-field-first (skip, render, skip--only 1 render).
        match self {
            Self::Upper => (image_height + 1) / 2,
            Self::Lower => (image_height / 2).max(1),
            Self::Both => image_height,
        }
    }
}

pub trait Normalize: Sized + Copy + Send + Sync {
    fn from_norm(value: f32) -> Self;
    fn to_norm(self) -> f32;
}

impl Normalize for f32 {
    #[inline(always)]
    fn from_norm(value: f32) -> Self {
        value
    }

    #[inline(always)]
    fn to_norm(self) -> f32 {
        self
    }
}

impl Normalize for u16 {
    #[inline(always)]
    fn from_norm(value: f32) -> Self {
        (value.clamp(0.0, 1.0) * Self::MAX as f32) as Self
    }

    #[inline(always)]
    fn to_norm(self) -> f32 {
        (self as f32) / Self::MAX as f32
    }
}

impl Normalize for u8 {
    #[inline(always)]
    fn from_norm(value: f32) -> Self {
        (value.clamp(0.0, 1.0) * Self::MAX as f32) as Self
    }

    #[inline(always)]
    fn to_norm(self) -> f32 {
        (self as f32) / Self::MAX as f32
    }
}

/// Borrowed YIQ data in a planar format.
/// Each plane is densely packed with regards to rows--if we skip fields, we just leave them out of these planes, which
/// squashes them vertically.
pub struct YiqView<'a> {
    pub y: &'a mut [f32],
    pub i: &'a mut [f32],
    pub q: &'a mut [f32],
    pub dimensions: (usize, usize),
    /// The source field that this data is for.
    pub field: YiqField,
}

impl<'a> YiqView<'a> {
    pub fn num_rows(&self) -> usize {
        self.field.num_image_rows(self.dimensions.1)
    }

    pub fn set_from_strided_buffer<S: PixelFormat>(
        &mut self,
        buf: &[S::DataFormat],
        row_bytes: usize,
    ) {
        let num_components = S::ORDER.num_components();
        let (r_idx, g_idx, b_idx) = S::ORDER.rgb_indices();
        assert!(num_components >= 3);
        assert_eq!(row_bytes % std::mem::size_of::<S::DataFormat>(), 0);

        // We write into the destination array differently depending on whether we're using the upper field, lower
        // field, or both. row_lshift determines whether we left-shift the source row index (doubling it). When we use
        // only one of the fields, the source row index needs to be double the destination row index so we take every
        // other row. When we use both fields, we just use the source row index as-is.
        // The row_offset determines whether we skip the first row (when using the lower field).
        let (row_lshift, row_offset): (usize, usize) = match self.field {
            YiqField::Upper => (1, 0),
            YiqField::Lower => (1, 1),
            YiqField::Both => (0, 0),
        };

        let Self { y, i, q, .. } = self;
        let (width, ..) = self.dimensions;

        y.par_chunks_mut(width)
            .zip(i.par_chunks_mut(width).zip(q.par_chunks_mut(width)))
            .enumerate()
            .for_each(|(row_idx, (y, (i, q)))| {
                let src_row_idx = (row_idx << row_lshift) + row_offset;
                let src_offset = src_row_idx * (row_bytes / std::mem::size_of::<S::DataFormat>());
                for pixel_idx in 0..width {
                    let yiq_pixel = YIQ_MATRIX
                        * Vec3::new(
                            (buf[((pixel_idx * num_components) + src_offset) + r_idx]).to_norm(),
                            (buf[((pixel_idx * num_components) + src_offset) + g_idx]).to_norm(),
                            (buf[((pixel_idx * num_components) + src_offset) + b_idx]).to_norm(),
                        );
                    y[pixel_idx] = yiq_pixel[0];
                    i[pixel_idx] = yiq_pixel[1];
                    q[pixel_idx] = yiq_pixel[2];
                }
            });
    }

    pub fn write_to_strided_buffer<S: PixelFormat>(
        &self,
        dst: &mut [S::DataFormat],
        row_bytes: usize,
    ) {
        let num_components = S::ORDER.num_components();
        let (r_idx, g_idx, b_idx) = S::ORDER.rgb_indices();
        assert!(num_components >= 3);
        assert_eq!(row_bytes % std::mem::size_of::<S::DataFormat>(), 0);

        let width = self.dimensions.0;
        let output_height = self.dimensions.1;

        // If the row index modulo 2 equals this number, that row was not rendered in the source data and we need to
        // interpolate between the rows above and beneath it.
        let skip_field: usize = match self.field {
            YiqField::Upper => 1,
            YiqField::Lower => 0,
            // The row index modulo 2 never reaches 2, meaning we don't skip any rows
            YiqField::Both => 2,
        };

        let row_rshift = match self.field {
            YiqField::Both => 0,
            YiqField::Upper | YiqField::Lower => 1,
        };

        let num_rows = self.num_rows();

        dst.par_chunks_exact_mut(row_bytes / std::mem::size_of::<S::DataFormat>())
            .enumerate()
            .for_each(|(row_idx, dst_row)| {
                // Inner fields with lines above and below them. Interpolate between those fields
                if (row_idx & 1) == skip_field && row_idx != 0 && row_idx != output_height - 1 {
                    for (pix_idx, pixel) in dst_row.chunks_mut(num_components).enumerate() {
                        let src_idx_lower = ((row_idx - 1) >> 1) * width + pix_idx;
                        let src_idx_upper = ((row_idx + 1) >> 1) * width + pix_idx;

                        let interp_pixel = Vec3::new(
                            (self.y[src_idx_lower] + self.y[src_idx_upper]) * 0.5,
                            (self.i[src_idx_lower] + self.i[src_idx_upper]) * 0.5,
                            (self.q[src_idx_lower] + self.q[src_idx_upper]) * 0.5,
                        );

                        let rgb = RGB_MATRIX * interp_pixel;
                        pixel[r_idx] = S::DataFormat::from_norm(rgb[0]);
                        pixel[g_idx] = S::DataFormat::from_norm(rgb[1]);
                        pixel[b_idx] = S::DataFormat::from_norm(rgb[2]);
                    }
                } else {
                    // Copy the field directly
                    for (pix_idx, pixel) in dst_row.chunks_mut(num_components).enumerate() {
                        let src_idx = (row_idx >> row_rshift).min(num_rows - 1) * width + pix_idx;
                        let rgb = RGB_MATRIX
                            * Vec3::new(self.y[src_idx], self.i[src_idx], self.q[src_idx]);
                        pixel[r_idx] = S::DataFormat::from_norm(rgb[0]);
                        pixel[g_idx] = S::DataFormat::from_norm(rgb[1]);
                        pixel[b_idx] = S::DataFormat::from_norm(rgb[2]);
                    }
                }
            });
    }
}

/// Owned YIQ data.
pub struct YiqOwned {
    /// Densely-packed planar YUV data. The Y plane comes first in memory, then I, then Q.
    data: Box<[f32]>,
    /// This refers to the "logical" dimensions, meaning that the number of scanlines is the same no matter whether any
    /// fields are being skipped.
    dimensions: (usize, usize),
    /// The source field that this data is for.
    field: YiqField,
}

pub enum SwizzleOrder {
    Rgbx,
    Xrgb,
    Bgrx,
    Xbgr,
    Rgb,
    Bgr,
}

impl SwizzleOrder {
    #[inline(always)]
    const fn num_components(&self) -> usize {
        match self {
            Self::Rgbx | Self::Xrgb | Self::Bgrx | Self::Xbgr => 4,
            Self::Rgb | Self::Bgr => 3,
        }
    }

    #[inline(always)]
    const fn rgb_indices(&self) -> (usize, usize, usize) {
        match self {
            Self::Rgbx | Self::Rgb => (0, 1, 2),
            Self::Xrgb => (1, 2, 3),
            Self::Bgrx | Self::Bgr => (2, 1, 0),
            Self::Xbgr => (3, 2, 1),
        }
    }
}

pub trait PixelFormat {
    const ORDER: SwizzleOrder;
    type DataFormat: Normalize;
}

macro_rules! impl_pix_fmt {
    ($ty: ident, $order: expr, $format: ty) => {
        pub struct $ty();
        impl PixelFormat for $ty {
            const ORDER: SwizzleOrder = $order;
            type DataFormat = $format;
        }
    };
}

impl_pix_fmt!(Rgbx8, SwizzleOrder::Rgbx, u8);
impl_pix_fmt!(Rgbx16, SwizzleOrder::Rgbx, u16);
impl_pix_fmt!(Xrgb8, SwizzleOrder::Xrgb, u8);
impl_pix_fmt!(Xrgb16, SwizzleOrder::Xrgb, u16);
impl_pix_fmt!(Bgrx8, SwizzleOrder::Bgrx, u8);
impl_pix_fmt!(Bgrx16, SwizzleOrder::Bgrx, u16);
impl_pix_fmt!(Xbgr8, SwizzleOrder::Xbgr, u8);
impl_pix_fmt!(Xbgr16, SwizzleOrder::Xbgr, u16);
impl_pix_fmt!(Rgb8, SwizzleOrder::Rgb, u8);
impl_pix_fmt!(Rgb16, SwizzleOrder::Rgb, u16);
impl_pix_fmt!(Bgr8, SwizzleOrder::Bgr, u8);
impl_pix_fmt!(Bgr16, SwizzleOrder::Bgr, u16);

impl YiqOwned {
    pub fn num_rows(&self) -> usize {
        self.field.num_image_rows(self.dimensions.1)
    }

    pub fn from_strided_buffer<S: PixelFormat>(
        buf: &[S::DataFormat],
        row_bytes: usize,
        width: usize,
        height: usize,
        field: YiqField,
    ) -> Self {
        let num_rows = field.num_image_rows(height);
        let num_pixels = width * num_rows;

        let mut data = vec![0f32; num_pixels * 3];
        let (y, iq) = data.split_at_mut(num_pixels);
        let (i, q) = iq.split_at_mut(num_pixels);

        let mut view = YiqView {
            y,
            i,
            q,
            dimensions: (width, height),
            field,
        };

        view.set_from_strided_buffer::<S>(buf, row_bytes);

        YiqOwned {
            data: data.into_boxed_slice(),
            dimensions: (width, height),
            field,
        }
    }

    pub fn from_image(image: &RgbImage, field: YiqField) -> Self {
        let width = image.width() as usize;
        let height = image.height() as usize;

        let num_rows = field.num_image_rows(height);

        let num_pixels = width * num_rows;

        let mut data = vec![0f32; num_pixels * 3];
        let (y, iq) = data.split_at_mut(num_pixels);
        let (i, q) = iq.split_at_mut(num_pixels);

        let mut view = YiqView {
            y,
            i,
            q,
            dimensions: (width, height),
            field,
        };

        view.set_from_strided_buffer::<Rgb8>(image.as_raw(), width * 3);

        YiqOwned {
            data: data.into_boxed_slice(),
            dimensions: (width, height),
            field,
        }
    }
}

impl<'a> From<&'a mut YiqOwned> for YiqView<'a> {
    fn from(value: &'a mut YiqOwned) -> Self {
        let num_pixels = value.dimensions.0 * value.num_rows();
        let (y, iq) = value.data.split_at_mut(num_pixels);
        let (i, q) = iq.split_at_mut(num_pixels);
        YiqView {
            y,
            i,
            q,
            dimensions: value.dimensions,
            field: value.field,
        }
    }
}

impl From<&YiqView<'_>> for RgbImage {
    fn from(image: &YiqView) -> Self {
        let (width, output_height) = image.dimensions;
        let num_pixels = width * output_height;
        let mut dst = vec![0u8; num_pixels * 3];
        image.write_to_strided_buffer::<Rgb8>(&mut dst, width * 3);

        RgbImage::from_raw(width as u32, output_height as u32, dst).unwrap()
    }
}
