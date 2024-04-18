use std::{convert::identity, mem::MaybeUninit};

use glam::{Mat3A, Vec3A};
use image::RgbImage;
use rayon::prelude::*;

#[inline(always)]
pub fn rgb_to_yiq([r, g, b]: [f32; 3]) -> [f32; 3] {
    const YIQ_MATRIX: Mat3A = Mat3A::from_cols(
        Vec3A::new(0.299, 0.5959, 0.2115),
        Vec3A::new(0.587, -0.2746, -0.5227),
        Vec3A::new(0.114, -0.3213, 0.3112),
    );

    (YIQ_MATRIX * Vec3A::new(r, g, b)).into()
}

#[inline(always)]
pub fn yiq_to_rgb([y, i, q]: [f32; 3]) -> [f32; 3] {
    const RGB_MATRIX: Mat3A = Mat3A::from_cols(
        Vec3A::new(1.0, 1.0, 1.0),
        Vec3A::new(0.956, -0.272, -1.106),
        Vec3A::new(0.619, -0.647, 1.703),
    );

    (RGB_MATRIX * Vec3A::new(y, i, q)).into()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum YiqField {
    Upper,
    Lower,
    Both,
    InterleavedUpper,
    InterleavedLower,
}

impl YiqField {
    pub fn num_image_rows(&self, image_height: usize) -> usize {
        // On an image with an odd input height, we do ceiling division if we render upper-field-first
        // (take an image 3 pixels tall. it goes render, skip, render--that's 2 renders) but floor division if we
        // render lower-field-first (skip, render, skip--only 1 render).
        match self {
            Self::Upper => (image_height + 1) / 2,
            Self::Lower => (image_height / 2).max(1),
            Self::Both | Self::InterleavedUpper | Self::InterleavedLower => image_height,
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

impl Normalize for i16 {
    #[inline(always)]
    fn from_norm(value: f32) -> Self {
        (value * Self::MAX as f32).clamp(Self::MIN as f32, Self::MAX as f32) as Self
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
    pub const fn num_components(&self) -> usize {
        match self {
            Self::Rgbx | Self::Xrgb | Self::Bgrx | Self::Xbgr => 4,
            Self::Rgb | Self::Bgr => 3,
        }
    }

    #[inline(always)]
    pub const fn rgba_indices(&self) -> (usize, usize, usize, Option<usize>) {
        match self {
            Self::Rgbx => (0, 1, 2, Some(3)),
            Self::Rgb => (0, 1, 2, None),
            Self::Xrgb => (1, 2, 3, Some(0)),
            Self::Bgrx => (2, 1, 0, Some(3)),
            Self::Bgr => (2, 1, 0, None),
            Self::Xbgr => (3, 2, 1, Some(0)),
        }
    }
}

pub trait PixelFormat {
    const ORDER: SwizzleOrder;
    type DataFormat: Normalize;
}

macro_rules! impl_pix_fmt {
    ($ty: ident, $order: expr, $format: ty) => {
        pub struct $ty;
        impl PixelFormat for $ty {
            const ORDER: SwizzleOrder = $order;
            type DataFormat = $format;
        }
    };
}

impl_pix_fmt!(Rgbx8, SwizzleOrder::Rgbx, u8);
impl_pix_fmt!(Xrgb8, SwizzleOrder::Xrgb, u8);
impl_pix_fmt!(Bgrx8, SwizzleOrder::Bgrx, u8);
impl_pix_fmt!(Xbgr8, SwizzleOrder::Xbgr, u8);
impl_pix_fmt!(Rgb8, SwizzleOrder::Rgb, u8);
impl_pix_fmt!(Bgr8, SwizzleOrder::Bgr, u8);

impl_pix_fmt!(Rgbx16, SwizzleOrder::Rgbx, u16);
impl_pix_fmt!(Xrgb16, SwizzleOrder::Xrgb, u16);
impl_pix_fmt!(Bgrx16, SwizzleOrder::Bgrx, u16);
impl_pix_fmt!(Xbgr16, SwizzleOrder::Xbgr, u16);
impl_pix_fmt!(Rgb16, SwizzleOrder::Rgb, u16);
impl_pix_fmt!(Bgr16, SwizzleOrder::Bgr, u16);

impl_pix_fmt!(Rgbx16s, SwizzleOrder::Rgbx, i16);
impl_pix_fmt!(Xrgb16s, SwizzleOrder::Xrgb, i16);
impl_pix_fmt!(Bgrx16s, SwizzleOrder::Bgrx, i16);
impl_pix_fmt!(Xbgr16s, SwizzleOrder::Xbgr, i16);
impl_pix_fmt!(Rgb16s, SwizzleOrder::Rgb, i16);
impl_pix_fmt!(Bgr16s, SwizzleOrder::Bgr, i16);

impl_pix_fmt!(Rgbx32f, SwizzleOrder::Rgbx, f32);
impl_pix_fmt!(Xrgb32f, SwizzleOrder::Xrgb, f32);
impl_pix_fmt!(Bgrx32f, SwizzleOrder::Bgrx, f32);
impl_pix_fmt!(Xbgr32f, SwizzleOrder::Xbgr, f32);
impl_pix_fmt!(Rgb32f, SwizzleOrder::Rgb, f32);
impl_pix_fmt!(Bgr32f, SwizzleOrder::Bgr, f32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeinterlaceMode {
    /// Interpolate between the given fields.
    Bob,
    /// Don't write absent fields at all--just leave whatever was already in the buffer.
    Skip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub top: usize,
    pub left: usize,
    pub bottom: usize,
    pub right: usize,
}

impl Rect {
    pub fn new(top: usize, left: usize, bottom: usize, right: usize) -> Self {
        assert!(
            bottom >= top && right >= left,
            "Invalid rectangle (top: {top}, bottom: {bottom}, left: {left}, right: {right})"
        );
        Self {
            top,
            left,
            bottom,
            right,
        }
    }

    pub fn from_width_height(width: usize, height: usize) -> Self {
        Self {
            top: 0,
            left: 0,
            bottom: height,
            right: width,
        }
    }

    pub fn width(&self) -> usize {
        self.right - self.left
    }

    pub fn height(&self) -> usize {
        self.bottom - self.top
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlitInfo {
    /// The rectangular area which will be read out of or written into the other buffer.
    pub rect: Rect,
    /// Number of bytes per pixel row in the other buffer. May include padding.
    pub row_bytes: usize,
    /// True if the source buffer is y-up instead of y-down.
    pub flip_y: bool,
}

impl BlitInfo {
    pub fn from_full_frame(width: usize, height: usize, row_bytes: usize) -> Self {
        BlitInfo {
            rect: Rect::new(0, 0, height, width),
            row_bytes,
            flip_y: false,
        }
    }

    pub fn new(rect: Rect, row_bytes: usize, flip_y: bool) -> Self {
        Self {
            rect,
            row_bytes,
            flip_y,
        }
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

fn slice_to_maybe_uninit<T>(slice: &[T]) -> &[MaybeUninit<T>] {
    // Safety: we know these are all initialized, so it's fine to transmute into a type that makes fewer assumptions
    unsafe { std::mem::transmute(slice) }
}

fn slice_to_maybe_uninit_mut<T>(slice: &mut [T]) -> &mut [MaybeUninit<T>] {
    // Safety: we know these are all initialized, so it's fine to transmute into a type that makes fewer assumptions
    unsafe { std::mem::transmute(slice) }
}

pub trait PixelTransform: Fn([f32; 3]) -> [f32; 3] + Send + Sync {}
impl<T: Fn([f32; 3]) -> [f32; 3] + Send + Sync> PixelTransform for T {}

impl<'a> YiqView<'a> {
    pub fn split_at_row(&mut self, idx: usize) -> (YiqView<'_>, YiqView<'_>) {
        let (y1, y2) = self.y.split_at_mut(idx * self.dimensions.0);
        let (i1, i2) = self.i.split_at_mut(idx * self.dimensions.0);
        let (q1, q2) = self.q.split_at_mut(idx * self.dimensions.0);
        (
            YiqView {
                y: y1,
                i: i1,
                q: q1,
                dimensions: (self.dimensions.0, self.dimensions.1),
                field: self.field,
            },
            YiqView {
                y: y2,
                i: i2,
                q: q2,
                dimensions: (self.dimensions.0, self.dimensions.1),
                field: self.field,
            },
        )
    }

    pub fn num_rows(&self) -> usize {
        self.field.num_image_rows(self.dimensions.1)
    }

    /// Safety:
    /// - `buf` must be a valid pointer to a buffer of length `len`.
    /// - All data within the portions of `buf` within each row, as specified by `row_bytes` and this view's dimensions,
    ///   must be initialized and valid. Data outside of those portions need not be valid.
    pub unsafe fn set_from_strided_buffer_maybe_uninit<S: PixelFormat, F: PixelTransform>(
        &mut self,
        buf: &[MaybeUninit<S::DataFormat>],
        blit_info: BlitInfo,
        pixel_transform: F,
    ) {
        let num_components = S::ORDER.num_components();
        let (r_idx, g_idx, b_idx, ..) = S::ORDER.rgba_indices();

        assert!(num_components >= 3);
        assert!(
            blit_info.row_bytes / std::mem::size_of::<S::DataFormat>() * S::ORDER.num_components()
                >= blit_info.rect.width(),
            "Blit rectangle width exceeds rowbytes"
        );
        assert_eq!(self.dimensions.0, blit_info.rect.width());
        assert_eq!(self.dimensions.1, blit_info.rect.height());
        assert_eq!(
            blit_info.row_bytes % std::mem::size_of::<S::DataFormat>(),
            0,
            "Rowbytes not aligned to datatype"
        );

        let row_length = blit_info.row_bytes / std::mem::size_of::<S::DataFormat>();
        let Self { y, i, q, .. } = self;
        let (width, height) = self.dimensions;

        y.par_chunks_mut(width)
            .zip(i.par_chunks_mut(width).zip(q.par_chunks_mut(width)))
            .enumerate()
            .for_each(|(row_idx, (y, (i, q)))| {
                // For interleaved fields, we write the first field into the first half of the buffer,
                // and the second field into the second half.
                let mut src_row_idx = match self.field {
                    YiqField::Upper => row_idx * 2,
                    YiqField::Lower => (row_idx * 2) + 1,
                    YiqField::Both => row_idx,
                    YiqField::InterleavedUpper => {
                        let idx = row_idx * 2;
                        if idx >= height {
                            let parity = 1 - (height % 2);
                            idx + parity - height
                        } else {
                            idx
                        }
                    }
                    YiqField::InterleavedLower => {
                        let idx = row_idx * 2 + 1;
                        if idx >= height {
                            let parity = 1 - (height % 2);
                            idx - parity - height
                        } else {
                            idx
                        }
                    }
                };
                src_row_idx += blit_info.rect.top;
                if blit_info.flip_y {
                    src_row_idx = height - src_row_idx - 1;
                }
                if self.dimensions.1 == 1 {
                    src_row_idx = 0;
                }
                let src_offset = src_row_idx * row_length;
                for pixel_idx in 0..width {
                    let yiq_pixel = rgb_to_yiq(pixel_transform([
                        buf[(((pixel_idx + blit_info.rect.left) * num_components) + src_offset) + r_idx]
                            .assume_init()
                            .to_norm(),
                        buf[(((pixel_idx + blit_info.rect.left) * num_components) + src_offset) + g_idx]
                            .assume_init()
                            .to_norm(),
                        buf[(((pixel_idx + blit_info.rect.left) * num_components) + src_offset) + b_idx]
                            .assume_init()
                            .to_norm(),
                    ]));
                    y[pixel_idx] = yiq_pixel[0];
                    i[pixel_idx] = yiq_pixel[1];
                    q[pixel_idx] = yiq_pixel[2];
                }
            });
    }

    pub fn set_from_strided_buffer<S: PixelFormat, F: PixelTransform>(
        &mut self,
        buf: &[S::DataFormat],
        blit_info: BlitInfo,
        pixel_transform: F,
    ) {
        // Safety: We know this data is valid because it's a slice.
        unsafe {
            self.set_from_strided_buffer_maybe_uninit::<S, F>(
                slice_to_maybe_uninit(buf),
                blit_info,
                pixel_transform,
            )
        }
    }

    pub fn write_to_strided_buffer_maybe_uninit<S: PixelFormat, F: PixelTransform>(
        &self,
        dst: &mut [MaybeUninit<S::DataFormat>],
        blit_info: BlitInfo,
        deinterlace_mode: DeinterlaceMode,
        fill_alpha: bool,
        pixel_transform: F,
    ) {
        let num_components = S::ORDER.num_components();
        let (r_idx, g_idx, b_idx, a_idx) = S::ORDER.rgba_indices();
        let a_idx = a_idx.unwrap_or(0);

        assert!(num_components >= 3);
        assert!(
            blit_info.row_bytes / std::mem::size_of::<S::DataFormat>() * S::ORDER.num_components()
                >= blit_info.rect.width(),
            "Blit rectangle width exceeds rowbytes"
        );
        assert_eq!(
            blit_info.row_bytes % std::mem::size_of::<S::DataFormat>(),
            0,
            "Rowbytes not aligned to datatype"
        );

        let row_length = blit_info.row_bytes / std::mem::size_of::<S::DataFormat>();
        let width = self.dimensions.0;
        let output_height = self.dimensions.1;

        // If the row index modulo 2 equals this number, that row was not rendered in the source data and we need to
        // interpolate between the rows above and beneath it.
        let skip_field: usize = match self.field {
            YiqField::Upper => 1,
            YiqField::Lower => 0,
            // The row index modulo 2 never reaches 2, meaning we don't skip any rows
            YiqField::Both | YiqField::InterleavedUpper | YiqField::InterleavedLower => 2,
        };

        let num_rows = self.num_rows();

        let chunks = dst
            .par_chunks_exact_mut(row_length)
            .skip(blit_info.rect.top)
            .take(blit_info.rect.height())
            .enumerate();

        match (deinterlace_mode, self.field) {
            (DeinterlaceMode::Bob, YiqField::Upper | YiqField::Lower) => {
                chunks.for_each(|(mut row_idx, dst_row)| {
                    // Limit to the actual width of the output (rowbytes may include trailing padding)
                    let dst_row = &mut dst_row[blit_info.rect.left * num_components
                        ..blit_info.rect.right * num_components];
                    if blit_info.flip_y {
                        row_idx = output_height - row_idx - 1;
                    }
                    // Inner fields with lines above and below them. Interpolate between those fields
                    if (row_idx & 1) == skip_field && row_idx != 0 && row_idx != output_height - 1 {
                        for (pix_idx, pixel) in dst_row.chunks_mut(num_components).enumerate() {
                            let src_idx_lower = ((row_idx - 1) >> 1) * width + pix_idx;
                            let src_idx_upper = ((row_idx + 1) >> 1) * width + pix_idx;

                            let interp_pixel = [
                                (self.y[src_idx_lower] + self.y[src_idx_upper]) * 0.5,
                                (self.i[src_idx_lower] + self.i[src_idx_upper]) * 0.5,
                                (self.q[src_idx_lower] + self.q[src_idx_upper]) * 0.5,
                            ];

                            let rgb = pixel_transform(yiq_to_rgb(interp_pixel));
                            pixel[r_idx] = MaybeUninit::new(S::DataFormat::from_norm(rgb[0]));
                            pixel[g_idx] = MaybeUninit::new(S::DataFormat::from_norm(rgb[1]));
                            pixel[b_idx] = MaybeUninit::new(S::DataFormat::from_norm(rgb[2]));
                            if fill_alpha {
                                pixel[a_idx] = MaybeUninit::new(S::DataFormat::from_norm(1.0));
                            }
                        }
                    } else {
                        // Copy the field directly
                        for (pix_idx, pixel) in dst_row.chunks_mut(num_components).enumerate() {
                            let src_idx = (row_idx >> 1).min(num_rows - 1) * width + pix_idx;
                            let rgb = pixel_transform(yiq_to_rgb([
                                self.y[src_idx],
                                self.i[src_idx],
                                self.q[src_idx],
                            ]));
                            pixel[r_idx] = MaybeUninit::new(S::DataFormat::from_norm(rgb[0]));
                            pixel[g_idx] = MaybeUninit::new(S::DataFormat::from_norm(rgb[1]));
                            pixel[b_idx] = MaybeUninit::new(S::DataFormat::from_norm(rgb[2]));
                            if fill_alpha {
                                pixel[a_idx] = MaybeUninit::new(S::DataFormat::from_norm(1.0));
                            }
                        }
                    }
                });
            }
            (DeinterlaceMode::Skip, YiqField::Upper | YiqField::Lower) => {
                chunks.for_each(|(mut row_idx, dst_row)| {
                    // Limit to the actual width of the output (rowbytes may include trailing padding)
                    let dst_row = &mut dst_row[blit_info.rect.left * num_components
                        ..blit_info.rect.right * num_components];
                    if blit_info.flip_y {
                        row_idx = output_height - row_idx - 1;
                    }
                    if (row_idx & 1) == skip_field {
                        return;
                    }
                    for (pix_idx, pixel) in dst_row.chunks_mut(num_components).enumerate() {
                        let src_idx = (row_idx >> 1).min(num_rows - 1) * width + pix_idx;
                        let rgb = pixel_transform(yiq_to_rgb([
                            self.y[src_idx],
                            self.i[src_idx],
                            self.q[src_idx],
                        ]));
                        pixel[r_idx] = MaybeUninit::new(S::DataFormat::from_norm(rgb[0]));
                        pixel[g_idx] = MaybeUninit::new(S::DataFormat::from_norm(rgb[1]));
                        pixel[b_idx] = MaybeUninit::new(S::DataFormat::from_norm(rgb[2]));
                        if fill_alpha {
                            pixel[a_idx] = MaybeUninit::new(S::DataFormat::from_norm(1.0));
                        }
                    }
                });
            }
            (_, YiqField::InterleavedUpper | YiqField::InterleavedLower) => {
                chunks.for_each(|(mut row_idx, dst_row)| {
                    // Limit to the actual width of the output (rowbytes may include trailing padding)
                    let dst_row = &mut dst_row[blit_info.rect.left * num_components
                        ..blit_info.rect.right * num_components];
                    if blit_info.flip_y {
                        row_idx = output_height - row_idx - 1;
                    }
                    let row_offset = match self.field {
                        YiqField::InterleavedUpper => {
                            YiqField::Upper.num_image_rows(self.dimensions.1) * (row_idx & 1)
                        }
                        YiqField::InterleavedLower => {
                            YiqField::Lower.num_image_rows(self.dimensions.1) * (1 - (row_idx & 1))
                        }
                        _ => unreachable!(),
                    };
                    // handle edge case where there's only one row and the mode is InterleavedLower
                    let interleaved_row_idx =
                        ((row_idx >> 1) + row_offset).min(self.dimensions.1 - 1);
                    let src_idx = interleaved_row_idx * width;
                    for (pix_idx, pixel) in dst_row.chunks_mut(num_components).enumerate() {
                        let rgb = pixel_transform(yiq_to_rgb([
                            self.y[src_idx + pix_idx],
                            self.i[src_idx + pix_idx],
                            self.q[src_idx + pix_idx],
                        ]));
                        pixel[r_idx] = MaybeUninit::new(S::DataFormat::from_norm(rgb[0]));
                        pixel[g_idx] = MaybeUninit::new(S::DataFormat::from_norm(rgb[1]));
                        pixel[b_idx] = MaybeUninit::new(S::DataFormat::from_norm(rgb[2]));
                        if fill_alpha {
                            pixel[a_idx] = MaybeUninit::new(S::DataFormat::from_norm(1.0));
                        }
                    }
                });
            }
            _ => {
                chunks.for_each(|(mut row_idx, dst_row)| {
                    // Limit to the actual width of the output (rowbytes may include trailing padding)
                    let dst_row = &mut dst_row[blit_info.rect.left * num_components
                        ..blit_info.rect.right * num_components];
                    if blit_info.flip_y {
                        row_idx = output_height - row_idx - 1;
                    }
                    for (pix_idx, pixel) in dst_row.chunks_mut(num_components).enumerate() {
                        let src_idx = row_idx.min(num_rows - 1) * width + pix_idx;
                        let rgb = pixel_transform(yiq_to_rgb([
                            self.y[src_idx],
                            self.i[src_idx],
                            self.q[src_idx],
                        ]));
                        pixel[r_idx] = MaybeUninit::new(S::DataFormat::from_norm(rgb[0]));
                        pixel[g_idx] = MaybeUninit::new(S::DataFormat::from_norm(rgb[1]));
                        pixel[b_idx] = MaybeUninit::new(S::DataFormat::from_norm(rgb[2]));
                        if fill_alpha {
                            pixel[a_idx] = MaybeUninit::new(S::DataFormat::from_norm(1.0));
                        }
                    }
                });
            }
        }
    }

    pub fn write_to_strided_buffer<S: PixelFormat, F: PixelTransform>(
        &self,
        dst: &mut [S::DataFormat],
        blit_info: BlitInfo,
        deinterlace_mode: DeinterlaceMode,
        pixel_transform: F,
    ) {
        self.write_to_strided_buffer_maybe_uninit::<S, F>(
            slice_to_maybe_uninit_mut(dst),
            blit_info,
            deinterlace_mode,
            false,
            pixel_transform,
        )
    }

    pub fn from_parts(buf: &'a mut [f32], dimensions: (usize, usize), field: YiqField) -> Self {
        let num_pixels = dimensions.0 * field.num_image_rows(dimensions.1);
        let (y, iq) = buf.split_at_mut(num_pixels);
        let (i, q) = iq.split_at_mut(num_pixels);
        YiqView {
            y,
            i,
            q,
            dimensions,
            field,
        }
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

        view.set_from_strided_buffer::<S, _>(
            buf,
            BlitInfo::from_full_frame(width, height, row_bytes),
            identity,
        );

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

        view.set_from_strided_buffer::<Rgb8, _>(
            image.as_raw(),
            BlitInfo::from_full_frame(width, height, width * height * 3),
            identity,
        );

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
        image.write_to_strided_buffer::<Rgb8, _>(
            &mut dst,
            BlitInfo::from_full_frame(width, output_height, width * 3),
            DeinterlaceMode::Bob,
            identity,
        );

        RgbImage::from_raw(width as u32, output_height as u32, dst).unwrap()
    }
}
