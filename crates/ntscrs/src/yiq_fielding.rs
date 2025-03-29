use std::{
    convert::identity,
    mem::{self, MaybeUninit},
};

use glam::{Mat3A, Vec3A};
use rayon::prelude::*;

use crate::thread_pool::with_thread_pool;

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

/// How are the fields being stored? This is similar to the `UseField` enum in the settings module, but doesn't include
/// `Alternating`--that is turned into either `Upper` or `Lower` depending on the frame number.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum YiqField {
    /// Use the upper (even-numbered, when indexing from 0) fields from the frame.
    Upper,
    /// Use the lower (odd-numbered, when indexing from 0) fields from the frame.
    Lower,
    /// Use both fields from the frame--somewhat inaccurate due to the lack of interlacing but may look nicer.
    Both,
    /// Use the upper fields and then the lower fields, in effect interlacing then combining them.
    InterleavedUpper,
    /// Use the lower fields and then the upper fields, in effect interlacing then combining them.
    InterleavedLower,
}

impl YiqField {
    /// The number of rows needed in the YIQ buffer to store data for a given field setting.
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

    /// The number of rows that correspond to this field for a given vertical resolution. Can be 0 unlike
    /// `num_image_rows`.
    pub fn num_actual_image_rows(&self, image_height: usize) -> usize {
        // On an image with an odd input height, we do ceiling division if we render upper-field-first
        // (take an image 3 pixels tall. it goes render, skip, render--that's 2 renders) but floor division if we
        // render lower-field-first (skip, render, skip--only 1 render).
        match self {
            Self::Upper => (image_height + 1) / 2,
            Self::Lower => image_height / 2,
            Self::Both | Self::InterleavedUpper | Self::InterleavedLower => image_height,
        }
    }

    /// Flips the field parity--upper becomes lower and vice versa.
    pub fn flip(&self) -> Self {
        match self {
            Self::Upper => Self::Lower,
            Self::Lower => Self::Upper,
            Self::Both => Self::Both,
            Self::InterleavedUpper => Self::InterleavedLower,
            Self::InterleavedLower => Self::InterleavedUpper,
        }
    }
}

/// Trait for converting various pixel formats to and from the f32 representation used when processing the image.
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

#[repr(transparent)]
#[derive(Clone, Copy)]
/// Special u16 pixel format for After Effects.
/// Ranges from 0 to 32768--anything outside of that will be wrapped.
/// That's right! *Not* 0 to 32767, the maximum for an i16, but one *above* that.
/// As far as I can tell, the values 32769-65535 are entirely unused and wasted. Why, Adobe, why?
pub struct AfterEffectsU16(u16);

impl Normalize for AfterEffectsU16 {
    #[inline(always)]
    fn from_norm(value: f32) -> Self {
        Self((value.clamp(0.0, 1.0) * 32768.0) as u16)
    }

    #[inline(always)]
    fn to_norm(self) -> f32 {
        (self.0 as f32) / 32768.0
    }
}

impl Normalize for i16 {
    #[inline(always)]
    fn from_norm(value: f32) -> Self {
        // Don't allow negative values; even though it's allowed, it causes problems with AE
        (value * Self::MAX as f32).clamp(0.0, Self::MAX as f32) as Self
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

/// Order in which the pixels are laid out for a given pixel buffer format.
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

/// The data format of a given pixel buffer.
pub trait PixelFormat {
    const ORDER: SwizzleOrder;
    type DataFormat: Normalize;

    /// Number of bytes that a single pixel in this format takes up.
    fn pixel_bytes() -> usize {
        Self::ORDER.num_components() * std::mem::size_of::<Self::DataFormat>()
    }
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

impl_pix_fmt!(Xrgb16AE, SwizzleOrder::Xrgb, AfterEffectsU16);

/// How to handle writing back fields that we *didn't* process if we used YiqField::Upper or YiqField::Lower.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeinterlaceMode {
    /// Interpolate between the given fields.
    Bob,
    /// Don't write absent fields at all--just leave whatever was already in the buffer.
    Skip,
}

/// Clip rectangle for copying to/from the YIQ buffer. Cannot be negative, and must be in bounds of both the source and
/// destination--you'll need to do some clamping and coordinate-space transforms yourself. Sorry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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

/// Settings for how to copy the image to and from the YIQ buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlitInfo {
    /// The rectangular area which will be read out of or written into the other buffer.
    pub rect: Rect,
    /// The coordinates at which to place the image in the buffer being written to, whether the YIQ buffer or output
    /// buffer.
    pub destination: (usize, usize),
    /// Number of bytes per pixel row in the other buffer. May include padding.
    pub row_bytes: usize,
    /// Height of the non-YIQ buffer.
    pub other_buffer_height: usize,
    /// True if the source buffer is y-up instead of y-down.
    pub flip_y: bool,
}

impl BlitInfo {
    /// When you don't need to process a specific rectangle of pixels, you can just use this to process the entire
    /// frame at once.
    pub fn from_full_frame(width: usize, height: usize, row_bytes: usize) -> Self {
        BlitInfo {
            rect: Rect::new(0, 0, height, width),
            destination: (0, 0),
            row_bytes,
            other_buffer_height: height,
            flip_y: false,
        }
    }

    pub fn new(
        rect: Rect,
        destination: (usize, usize),
        row_bytes: usize,
        other_buffer_height: usize,
        flip_y: bool,
    ) -> Self {
        Self {
            rect,
            destination,
            row_bytes,
            other_buffer_height,
            flip_y,
        }
    }
}

/// Borrowed YIQ data in a planar format.
/// Each plane is densely packed with regards to rows--if we skip fields, we just leave them out of these planes, which
/// squashes them vertically.
pub struct YiqView<'a> {
    /// Y (luma) plane.
    pub y: &'a mut [f32],
    /// I (in-phase chroma) plane.
    pub i: &'a mut [f32],
    /// Q (quadrature chroma) plane.
    pub q: &'a mut [f32],
    /// Scratch buffer; used to accelerate some operations which would be much slower in-place.
    pub scratch: &'a mut [f32],
    /// Logical dimensions of the image, counting skipped fields. This does *not* depend on the field setting, and will
    /// *not* tell you how many rows of pixels are being stored in the buffers (use the `num_rows` method instead).
    pub dimensions: (usize, usize),
    /// The source field that this data is for.
    pub field: YiqField,
}

fn slice_to_maybe_uninit<T>(slice: &[T]) -> &[MaybeUninit<T>] {
    // Safety: we know these are all initialized, so it's fine to transmute into a type that makes fewer assumptions
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as _, slice.len()) }
}

/// # Safety:
/// - You must only write initialized values into the slice.
unsafe fn slice_to_maybe_uninit_mut<T>(slice: &mut [T]) -> &mut [MaybeUninit<T>] {
    // Safety: we know these are all initialized, so it's fine to transmute into a type that makes fewer assumptions
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as _, slice.len()) }
}

pub trait PixelTransform: Fn([f32; 3]) -> [f32; 3] + Send + Sync + Copy {}
impl<T: Fn([f32; 3]) -> [f32; 3] + Send + Sync + Copy> PixelTransform for T {}

impl<'a> YiqView<'a> {
    /// Split this `YiqView` into two `YiqView`s vertically at a given row.
    pub fn split_at_row(&mut self, idx: usize) -> (Option<YiqView<'_>>, Option<YiqView<'_>>) {
        let (y1, y2) = self.y.split_at_mut(idx * self.dimensions.0);
        let (i1, i2) = self.i.split_at_mut(idx * self.dimensions.0);
        let (q1, q2) = self.q.split_at_mut(idx * self.dimensions.0);
        let (s1, s2) = self.scratch.split_at_mut(idx * self.dimensions.0);
        (
            if !y1.is_empty() {
                Some(YiqView {
                    y: y1,
                    i: i1,
                    q: q1,
                    scratch: s1,
                    dimensions: (self.dimensions.0, self.dimensions.1),
                    field: self.field,
                })
            } else {
                None
            },
            if !y2.is_empty() {
                Some(YiqView {
                    y: y2,
                    i: i2,
                    q: q2,
                    scratch: s2,
                    dimensions: (self.dimensions.0, self.dimensions.1),
                    field: self.field,
                })
            } else {
                None
            },
        )
    }

    /// Number of rows of pixels being stored in this view. This will be smaller than `dimensions.1` if some
    /// fields are being skipped.
    pub fn num_rows(&self) -> usize {
        self.field.num_image_rows(self.dimensions.1)
    }

    /// Convert (a given part of) the input pixel buffer into YIQ planar format, and optionally apply a color transform
    /// to the pixels beforehand. This method allows padding bytes of the source buffer to be uninitialized, which *may*
    /// be the case for effect plugin APIs (OpenFX and After Effects, which both leave it ambiguous).
    ///
    /// # Safety
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
        assert_eq!(
            blit_info.row_bytes % std::mem::size_of::<S::DataFormat>(),
            0,
            "Rowbytes not aligned to datatype"
        );
        let row_length = blit_info.row_bytes / std::mem::size_of::<S::DataFormat>();
        assert!(num_components >= 3);
        assert!(
            row_length * S::ORDER.num_components() >= blit_info.rect.width(),
            "Blit rectangle width exceeds rowbytes"
        );

        assert!(blit_info.rect.width() + blit_info.destination.0 <= self.dimensions.0);
        assert!(blit_info.rect.height() + blit_info.destination.1 <= self.dimensions.1);

        unsafe fn blit_single_field<S: PixelFormat, F: PixelTransform>(
            y: &mut [f32],
            i: &mut [f32],
            q: &mut [f32],
            dimensions: (usize, usize),
            mut field: YiqField,
            buf: &[MaybeUninit<S::DataFormat>],
            blit_info: BlitInfo,
            pixel_transform: F,
        ) {
            let row_length = blit_info.row_bytes / std::mem::size_of::<S::DataFormat>();
            let num_components = S::ORDER.num_components();
            let (r_idx, g_idx, b_idx, ..) = S::ORDER.rgba_indices();
            let (width, _) = dimensions;

            let num_skipped_rows = match field {
                YiqField::Upper => (blit_info.destination.1 + 1) / 2,
                YiqField::Lower => blit_info.destination.1 / 2,
                _ => blit_info.destination.1,
            };

            if blit_info.destination.1 & 1 == 1 {
                field = field.flip();
            }

            let num_rect_rows = field.num_image_rows(blit_info.rect.height());

            y.par_chunks_exact_mut(width)
                .zip(
                    i.par_chunks_exact_mut(width)
                        .zip(q.par_chunks_exact_mut(width)),
                )
                .skip(num_skipped_rows)
                .take(num_rect_rows)
                .enumerate()
                .for_each(|(row_idx, (y, (i, q)))| {
                    // For interleaved fields, we write the first field into the first half of the buffer,
                    // and the second field into the second half.
                    let mut src_row_idx = match field {
                        YiqField::Upper => row_idx * 2,
                        YiqField::Lower => (row_idx * 2) + 1,
                        YiqField::Both => row_idx,
                        YiqField::InterleavedUpper | YiqField::InterleavedLower => {
                            panic!("blit_single_field doesn't operate on interleaved stuff")
                        }
                    };
                    src_row_idx += blit_info.rect.top;
                    if blit_info.flip_y {
                        src_row_idx = blit_info.other_buffer_height - src_row_idx - 1;
                    }
                    if blit_info.rect.height() == 1 {
                        src_row_idx = 0;
                    }
                    let src_offset = src_row_idx * row_length;
                    for idx in 0..blit_info.rect.width() {
                        let src_pixel_idx = idx + blit_info.rect.left;
                        let rgb = unsafe {
                            [
                                buf[((src_pixel_idx * num_components) + src_offset) + r_idx]
                                    .assume_init()
                                    .to_norm(),
                                buf[((src_pixel_idx * num_components) + src_offset) + g_idx]
                                    .assume_init()
                                    .to_norm(),
                                buf[((src_pixel_idx * num_components) + src_offset) + b_idx]
                                    .assume_init()
                                    .to_norm(),
                            ]
                        };
                        let yiq_pixel = rgb_to_yiq(pixel_transform(rgb));
                        let dst_pixel_idx = idx + blit_info.destination.0;
                        y[dst_pixel_idx] = yiq_pixel[0];
                        i[dst_pixel_idx] = yiq_pixel[1];
                        q[dst_pixel_idx] = yiq_pixel[2];
                    }
                });
        }

        match self.field {
            YiqField::Upper | YiqField::Lower | YiqField::Both => {
                let Self { y, i, q, .. } = self;
                with_thread_pool(|| unsafe {
                    blit_single_field::<S, F>(
                        y,
                        i,
                        q,
                        self.dimensions,
                        self.field,
                        buf,
                        blit_info,
                        pixel_transform,
                    )
                })
            }
            YiqField::InterleavedUpper => {
                let num_upper_rows = YiqField::Upper.num_actual_image_rows(self.dimensions.1);
                let (mut upper, mut lower) = self.split_at_row(num_upper_rows);
                if let Some(upper) = upper.as_mut() {
                    upper.field = YiqField::Upper;
                    unsafe {
                        upper.set_from_strided_buffer_maybe_uninit::<S, F>(
                            buf,
                            blit_info,
                            pixel_transform,
                        )
                    };
                };

                if let Some(lower) = lower.as_mut() {
                    lower.field = YiqField::Lower;
                    unsafe {
                        lower.set_from_strided_buffer_maybe_uninit::<S, F>(
                            buf,
                            blit_info,
                            pixel_transform,
                        )
                    };
                };
            }
            YiqField::InterleavedLower => {
                let num_lower_rows = YiqField::Lower.num_actual_image_rows(self.dimensions.1);
                let (mut lower, mut upper) = self.split_at_row(num_lower_rows);
                if let Some(upper) = upper.as_mut() {
                    upper.field = YiqField::Upper;
                    unsafe {
                        upper.set_from_strided_buffer_maybe_uninit::<S, F>(
                            buf,
                            blit_info,
                            pixel_transform,
                        )
                    };
                };

                if let Some(lower) = lower.as_mut() {
                    lower.field = YiqField::Lower;
                    unsafe {
                        lower.set_from_strided_buffer_maybe_uninit::<S, F>(
                            buf,
                            blit_info,
                            pixel_transform,
                        )
                    };
                };
            }
        }
    }

    /// Convert (a given part of) the input pixel buffer into YIQ planar format, and optionally apply a color transform
    /// to the pixels beforehand.
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

    /// Convert (a given part of) the YIQ planar data back into the given pixel fornat, and optionally apply a color
    /// transform to the pixels, before writing it into the destination buffer. This method allows you to write into a
    /// buffer which may not be initialized beforehand.
    pub fn write_to_strided_buffer_maybe_uninit<S: PixelFormat, F: PixelTransform>(
        &self,
        dst: &mut [MaybeUninit<S::DataFormat>],
        mut blit_info: BlitInfo,
        deinterlace_mode: DeinterlaceMode,
        fill_alpha: bool,
        pixel_transform: F,
    ) {
        let num_components = S::ORDER.num_components();
        let (r_idx, g_idx, b_idx, a_idx) = S::ORDER.rgba_indices();
        let a_idx = a_idx.unwrap_or(0);

        // If we flip the Y coordinate, we need to flip the blit rectangle and destination coords as well. If we were
        // doing "for each source pixel, write to the destination", we could just flip the coordinate of the pixel we
        // write to, but we want to do this in parallel which requires "for each destination pixel, *read* from the
        // source".
        if blit_info.flip_y {
            blit_info.rect.top = blit_info.other_buffer_height - blit_info.rect.top;
            blit_info.rect.bottom = blit_info.other_buffer_height - blit_info.rect.bottom;
            mem::swap(&mut blit_info.rect.bottom, &mut blit_info.rect.top);

            let distance_to_bottom =
                blit_info.other_buffer_height - (blit_info.rect.height() + blit_info.destination.1);
            blit_info.destination.1 = distance_to_bottom;
        }

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
        assert!(blit_info.rect.width() + blit_info.destination.0 <= self.dimensions.0);
        assert!(blit_info.rect.height() + blit_info.destination.1 <= self.dimensions.1);

        let row_length = blit_info.row_bytes / std::mem::size_of::<S::DataFormat>();
        let width = self.dimensions.0;
        let output_height = blit_info.other_buffer_height;

        // If the row index modulo 2 equals this number, that row was not rendered in the source data and we need to
        // interpolate between the rows above and beneath it.
        let skip_field: usize = match self.field {
            YiqField::Upper => 1,
            YiqField::Lower => 0,
            // The row index modulo 2 never reaches 2, meaning we don't skip any rows
            YiqField::Both | YiqField::InterleavedUpper | YiqField::InterleavedLower => 2,
        };

        let num_rows = self.num_rows();

        with_thread_pool(|| {
            let chunks = dst
                .par_chunks_exact_mut(row_length)
                .skip(blit_info.destination.1)
                .take(blit_info.rect.height())
                .enumerate();
            match (deinterlace_mode, self.field) {
                (DeinterlaceMode::Bob, YiqField::Upper | YiqField::Lower) => {
                    chunks.for_each(|(mut dst_row_idx, dst_row)| {
                        dst_row_idx += blit_info.rect.top;
                        // Limit to the actual width of the output (rowbytes may include trailing padding)
                        let dst_row = &mut dst_row[blit_info.destination.0 * num_components
                            ..(blit_info.destination.0 + blit_info.rect.width()) * num_components];
                        if blit_info.flip_y {
                            dst_row_idx = output_height - dst_row_idx - 1;
                        }
                        // Inner fields with lines above and below them. Interpolate between those fields
                        if (dst_row_idx & 1) == skip_field
                            && dst_row_idx != 0
                            && dst_row_idx != output_height - 1
                        {
                            for (pix_idx, pixel) in
                                dst_row.chunks_exact_mut(num_components).enumerate()
                            {
                                let src_idx_lower = ((dst_row_idx - 1) >> 1) * width
                                    + pix_idx
                                    + blit_info.rect.left;
                                let src_idx_upper = ((dst_row_idx + 1) >> 1) * width
                                    + pix_idx
                                    + blit_info.rect.left;

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
                            for (pix_idx, pixel) in
                                dst_row.chunks_exact_mut(num_components).enumerate()
                            {
                                let src_idx = (dst_row_idx >> 1).min(num_rows - 1) * width
                                    + pix_idx
                                    + blit_info.rect.left;
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
                        row_idx += blit_info.rect.top;
                        // Limit to the actual width of the output (rowbytes may include trailing padding)
                        let dst_row = &mut dst_row[blit_info.destination.0 * num_components
                            ..(blit_info.destination.0 + blit_info.rect.width()) * num_components];
                        if blit_info.flip_y {
                            row_idx = output_height - row_idx - 1;
                        }
                        if (row_idx & 1) == skip_field {
                            return;
                        }
                        for (pix_idx, pixel) in dst_row.chunks_exact_mut(num_components).enumerate()
                        {
                            let src_idx = (row_idx >> 1).min(num_rows - 1) * width
                                + pix_idx
                                + blit_info.rect.left;
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
                        row_idx += blit_info.rect.top;
                        // Limit to the actual width of the output (rowbytes may include trailing padding)
                        let dst_row = &mut dst_row[blit_info.destination.0 * num_components
                            ..(blit_info.destination.0 + blit_info.rect.width()) * num_components];
                        if blit_info.flip_y {
                            row_idx = output_height - row_idx - 1;
                        }
                        let row_offset = match self.field {
                            YiqField::InterleavedUpper => {
                                YiqField::Upper.num_image_rows(self.dimensions.1) * (row_idx & 1)
                            }
                            YiqField::InterleavedLower => {
                                YiqField::Lower.num_image_rows(self.dimensions.1)
                                    * (1 - (row_idx & 1))
                            }
                            _ => unreachable!(),
                        };
                        // handle edge case where there's only one row and the mode is InterleavedLower
                        let interleaved_row_idx =
                            ((row_idx >> 1) + row_offset).min(self.dimensions.1 - 1);
                        let src_idx = interleaved_row_idx * width;
                        for (pix_idx, pixel) in dst_row.chunks_exact_mut(num_components).enumerate()
                        {
                            let rgb = pixel_transform(yiq_to_rgb([
                                self.y[src_idx + pix_idx + blit_info.rect.left],
                                self.i[src_idx + pix_idx + blit_info.rect.left],
                                self.q[src_idx + pix_idx + blit_info.rect.left],
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
                        row_idx += blit_info.rect.top;
                        // Limit to the actual width of the output (rowbytes may include trailing padding)
                        let dst_row = &mut dst_row[blit_info.destination.0 * num_components
                            ..(blit_info.destination.0 + blit_info.rect.width()) * num_components];
                        if blit_info.flip_y {
                            row_idx = output_height - row_idx - 1;
                        }
                        for (pix_idx, pixel) in dst_row.chunks_exact_mut(num_components).enumerate()
                        {
                            let src_idx =
                                row_idx.min(num_rows - 1) * width + pix_idx + blit_info.rect.left;
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
        });
    }

    pub fn write_to_strided_buffer<S: PixelFormat, F: PixelTransform>(
        &self,
        dst: &mut [S::DataFormat],
        blit_info: BlitInfo,
        deinterlace_mode: DeinterlaceMode,
        pixel_transform: F,
    ) {
        self.write_to_strided_buffer_maybe_uninit::<S, F>(
            unsafe { slice_to_maybe_uninit_mut(dst) },
            blit_info,
            deinterlace_mode,
            false,
            pixel_transform,
        )
    }

    pub fn from_parts(buf: &'a mut [f32], dimensions: (usize, usize), field: YiqField) -> Self {
        let num_pixels = dimensions.0 * field.num_image_rows(dimensions.1);
        assert_eq!(
            buf.len(),
            num_pixels * 4,
            "buffer length: {}, expected buffer length: {}",
            buf.len(),
            num_pixels * 4
        );
        let (y, iqs) = buf.split_at_mut(num_pixels);
        let (i, qs) = iqs.split_at_mut(num_pixels);
        let (q, s) = qs.split_at_mut(num_pixels);
        YiqView {
            y,
            i,
            q,
            scratch: s,
            dimensions,
            field,
        }
    }

    /// Calculate the length (in elements, not bytes) of a buffer needed to hold a YiqView with the given dimensions and
    /// field.
    pub fn buf_length_for(dimensions: (usize, usize), field: YiqField) -> usize {
        dimensions.0 * field.num_image_rows(dimensions.1) * 4
    }
}

/// Owned YIQ data. If you bring your own buffer, you probably don't need this.
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
    pub fn from_strided_buffer<S: PixelFormat>(
        buf: &[S::DataFormat],
        row_bytes: usize,
        width: usize,
        height: usize,
        field: YiqField,
    ) -> Self {
        let num_rows = field.num_image_rows(height);
        let num_pixels = width * num_rows;

        let mut data = vec![0f32; num_pixels * 4];
        let mut view = YiqView::from_parts(&mut data, (width, height), field);

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
}

impl<'a> From<&'a mut YiqOwned> for YiqView<'a> {
    fn from(value: &'a mut YiqOwned) -> Self {
        YiqView::from_parts(&mut value.data, value.dimensions, value.field)
    }
}
