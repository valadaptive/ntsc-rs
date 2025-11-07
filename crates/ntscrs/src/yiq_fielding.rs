use std::mem::{self, MaybeUninit};

use macros::simd_dispatch;

use crate::{
    f32x4::F32x4,
    thread_pool::{ZipChunks, with_thread_pool},
};

#[inline(always)]
fn rgb_to_yiq<S: F32x4>(rgb: S) -> S {
    // This is a matrix multiply with the matrix being:
    // [
    //     [0.299, 0.5959, 0.2115],
    //     [0.587, -0.2746, -0.5227],
    //     [0.114, -0.3213, 0.3112],
    // ]

    let rr = rgb.swizzle(0, 0, 0, 0);
    let gg = rgb.swizzle(1, 1, 1, 1);
    let bb = rgb.swizzle(2, 2, 2, 2);

    unsafe {
        rr.mul_add(
            S::load4(&[0.299, 0.5959, 0.2115, 0.0]),
            gg.mul_add(
                S::load4(&[0.587, -0.2746, -0.5227, 0.0]),
                bb * S::load4(&[0.114, -0.3213, 0.3112, 0.0]),
            ),
        )
    }
}

#[inline(always)]
fn yiq_to_rgb<S: F32x4>(yiq: S) -> S {
    // This is a matrix multiply with the matrix being:
    // [
    //    [1.0, 1.0, 1.0],
    //    [0.956, -0.272, -1.106],
    //    [0.619, -0.647, 1.703],
    // ]
    // Since the top row is all ones, we can skip it.

    let yy = yiq.swizzle(0, 0, 0, 0);
    let ii = yiq.swizzle(1, 1, 1, 1);
    let qq = yiq.swizzle(2, 2, 2, 2);

    unsafe {
        qq.mul_add(
            S::load4(&[0.619, -0.647, 1.703, 0.0]),
            ii.mul_add(S::load4(&[0.956, -0.272, -1.106, 0.0]), yy),
        )
    }
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
    #[inline(always)]
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
    #[inline(always)]
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

mod private {
    pub trait Sealed {}

    // Implement for those same types, but no others.
    impl Sealed for f32 {}
    impl Sealed for u16 {}
    impl Sealed for super::AfterEffectsU16 {}
    impl Sealed for i16 {}
    impl Sealed for u8 {}
}

/// Trait for converting various pixel formats to and from the f32 representation used when processing the image.
pub trait Normalize: Sized + Copy + Send + Sync + private::Sealed {
    const ONE: Self;
    fn from_norm<S: F32x4>(value: S) -> [Self; 4];
    unsafe fn to_norm<S: F32x4>(value: [Self; 4]) -> S;
}

impl Normalize for f32 {
    const ONE: Self = 1.0;
    #[inline(always)]
    fn from_norm<S: F32x4>(value: S) -> [Self; 4] {
        value.as_array()
    }

    #[inline(always)]
    unsafe fn to_norm<S: F32x4>(value: [Self; 4]) -> S {
        unsafe { S::load4(&value) }
    }
}

impl Normalize for u16 {
    const ONE: Self = Self::MAX;
    #[inline(always)]
    fn from_norm<S: F32x4>(value: S) -> [Self; 4] {
        let min = unsafe { S::set1(Self::MIN as f32) };
        let max = unsafe { S::set1(Self::MAX as f32) };
        let multiplied = unsafe { (value * max).min(max).max(min).as_signed_ints() };
        [
            multiplied[0] as u16,
            multiplied[1] as u16,
            multiplied[2] as u16,
            multiplied[3] as u16,
        ]
    }

    #[inline(always)]
    unsafe fn to_norm<S: F32x4>(value: [Self; 4]) -> S {
        let values = unsafe {
            S::from_signed_ints(&[
                value[0] as i32,
                value[1] as i32,
                value[2] as i32,
                value[3] as i32,
            ])
        };
        values / unsafe { S::set1(Self::MAX as f32) }
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
    const ONE: Self = Self(32768);
    #[inline(always)]
    fn from_norm<S: F32x4>(value: S) -> [Self; 4] {
        let min = unsafe { S::set1(0.0) };
        let max = unsafe { S::set1(32768.0) };
        let multiplied = unsafe { (value * max).min(max).max(min).as_signed_ints() };
        [
            Self(multiplied[0] as u16),
            Self(multiplied[1] as u16),
            Self(multiplied[2] as u16),
            Self(multiplied[3] as u16),
        ]
    }

    #[inline(always)]
    unsafe fn to_norm<S: F32x4>(value: [Self; 4]) -> S {
        let values = unsafe {
            S::from_signed_ints(&[
                value[0].0 as i32,
                value[1].0 as i32,
                value[2].0 as i32,
                value[3].0 as i32,
            ])
        };
        values / unsafe { S::set1(32768.0) }
    }
}

impl Normalize for i16 {
    const ONE: Self = Self::MAX;
    #[inline(always)]
    fn from_norm<S: F32x4>(value: S) -> [Self; 4] {
        let min = unsafe { S::set1(Self::MIN as f32) };
        let max = unsafe { S::set1(Self::MAX as f32) };
        let multiplied = unsafe { (value * max).min(max).max(min).as_signed_ints() };
        [
            multiplied[0] as i16,
            multiplied[1] as i16,
            multiplied[2] as i16,
            multiplied[3] as i16,
        ]
    }

    #[inline(always)]
    unsafe fn to_norm<S: F32x4>(value: [Self; 4]) -> S {
        let values = unsafe {
            S::from_signed_ints(&[
                value[0] as i32,
                value[1] as i32,
                value[2] as i32,
                value[3] as i32,
            ])
        };
        values / unsafe { S::set1(Self::MAX as f32) }
    }
}

impl Normalize for u8 {
    const ONE: Self = Self::MAX;
    #[inline(always)]
    fn from_norm<S: F32x4>(value: S) -> [Self; 4] {
        let min = unsafe { S::set1(Self::MIN as f32) };
        let max = unsafe { S::set1(Self::MAX as f32) };
        let multiplied = unsafe { (value * max).min(max).max(min).as_signed_ints() };
        [
            multiplied[0] as u8,
            multiplied[1] as u8,
            multiplied[2] as u8,
            multiplied[3] as u8,
        ]
    }

    #[inline(always)]
    unsafe fn to_norm<S: F32x4>(value: [Self; 4]) -> S {
        let values = unsafe {
            S::from_signed_ints(&[
                value[0] as i32,
                value[1] as i32,
                value[2] as i32,
                value[3] as i32,
            ])
        };
        values / unsafe { S::set1(Self::MAX as f32) }
    }
}

/// The data format of a given pixel buffer.
pub trait PixelFormat {
    const NUM_COMPONENTS: usize;
    const RGBA_INDICES: (usize, usize, usize, Option<usize>);
}

macro_rules! impl_pix_fmt {
    ($ty: ident, $num_components: expr, $rgba_indices: expr) => {
        pub struct $ty;
        impl PixelFormat for $ty {
            const NUM_COMPONENTS: usize = $num_components;
            const RGBA_INDICES: (usize, usize, usize, Option<usize>) = $rgba_indices;
        }
    };
}

impl_pix_fmt!(Rgbx, 4, (0, 1, 2, Some(3)));
impl_pix_fmt!(Xrgb, 4, (1, 2, 3, Some(0)));
impl_pix_fmt!(Bgrx, 4, (2, 1, 0, Some(3)));
impl_pix_fmt!(Xbgr, 4, (3, 2, 1, Some(0)));
impl_pix_fmt!(Rgb, 3, (0, 1, 2, None));
impl_pix_fmt!(Bgr, 3, (2, 1, 0, None));

pub fn pixel_bytes_for<S: PixelFormat, T: Normalize>() -> usize {
    S::NUM_COMPONENTS * std::mem::size_of::<T>()
}

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

pub trait PixelTransform: Send + Sync + Copy {
    fn transform_pixel<S: F32x4>(&self, pixel: S) -> S;
}
impl<T: Fn([f32; 3]) -> [f32; 3] + Send + Sync + Copy> PixelTransform for T {
    #[inline(always)]
    fn transform_pixel<S: F32x4>(&self, pixel: S) -> S {
        let mut tmp = pixel.as_array();
        let transformed = self([tmp[0], tmp[1], tmp[2]]);
        tmp[0] = transformed[0];
        tmp[1] = transformed[1];
        tmp[2] = transformed[2];
        unsafe { S::load4(&tmp) }
    }
}
impl PixelTransform for () {
    #[inline(always)]
    fn transform_pixel<S: F32x4>(&self, pixel: S) -> S {
        pixel
    }
}

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
    #[inline(always)]
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
    pub unsafe fn set_from_strided_buffer_maybe_uninit<
        S: PixelFormat,
        T: Normalize,
        F: PixelTransform,
    >(
        &mut self,
        buf: &[MaybeUninit<T>],
        blit_info: BlitInfo,
        pixel_transform: F,
    ) {
        let num_components = S::NUM_COMPONENTS;
        assert_eq!(
            blit_info.row_bytes % std::mem::size_of::<T>(),
            0,
            "Rowbytes not aligned to datatype"
        );
        let row_length = blit_info.row_bytes / std::mem::size_of::<T>();
        assert!(num_components >= 3);
        assert!(
            row_length * S::NUM_COMPONENTS >= blit_info.rect.width(),
            "Blit rectangle width exceeds rowbytes"
        );

        assert!(blit_info.rect.width() + blit_info.destination.0 <= self.dimensions.0);
        assert!(blit_info.rect.height() + blit_info.destination.1 <= self.dimensions.1);

        unsafe fn blit_row_simd_inner<
            S: PixelFormat,
            T: Normalize,
            F: PixelTransform,
            Simd: F32x4,
        >(
            y: &mut [f32],
            i: &mut [f32],
            q: &mut [f32],
            buf: &[MaybeUninit<T>],
            blit_info: &BlitInfo,
            src_row_idx: usize,
            pixel_transform: F,
        ) {
            let row_length = blit_info.row_bytes / std::mem::size_of::<T>();
            let src_offset = src_row_idx * row_length;
            let (r_idx, g_idx, b_idx, ..) = S::RGBA_INDICES;
            for idx in 0..blit_info.rect.width() {
                let src_pixel_idx = idx + blit_info.rect.left;
                let rgba: Simd = unsafe {
                    T::to_norm([
                        buf[((src_pixel_idx * S::NUM_COMPONENTS) + src_offset) + r_idx]
                            .assume_init(),
                        buf[((src_pixel_idx * S::NUM_COMPONENTS) + src_offset) + g_idx]
                            .assume_init(),
                        buf[((src_pixel_idx * S::NUM_COMPONENTS) + src_offset) + b_idx]
                            .assume_init(),
                        T::ONE,
                    ])
                };
                let transformed = pixel_transform.transform_pixel(rgba);
                let yiq_pixel = rgb_to_yiq(transformed);
                let dst_pixel_idx = idx + blit_info.destination.0;
                let yiq_channels = yiq_pixel.as_array();
                y[dst_pixel_idx] = yiq_channels[0];
                i[dst_pixel_idx] = yiq_channels[1];
                q[dst_pixel_idx] = yiq_channels[2];
            }
        }

        #[simd_dispatch(Simd, scalar_fallback)]
        unsafe fn blit_row_simd<S: PixelFormat, T: Normalize, F: PixelTransform>(
            y: &mut [f32],
            i: &mut [f32],
            q: &mut [f32],
            buf: &[MaybeUninit<T>],
            blit_info: &BlitInfo,
            src_row_idx: usize,
            pixel_transform: F,
        ) {
            blit_row_simd_inner::<S, T, F, Simd>(
                y,
                i,
                q,
                buf,
                blit_info,
                src_row_idx,
                pixel_transform,
            )
        }

        match self.field {
            YiqField::Upper | YiqField::Lower | YiqField::Both => {
                let Self { y, i, q, .. } = self;
                with_thread_pool(|| {
                    let (width, _) = self.dimensions;
                    let mut field = self.field;

                    let num_skipped_rows = match field {
                        YiqField::Upper => (blit_info.destination.1 + 1) / 2,
                        YiqField::Lower => blit_info.destination.1 / 2,
                        _ => blit_info.destination.1,
                    };

                    if blit_info.destination.1 & 1 == 1 {
                        field = field.flip();
                    }

                    let num_rect_rows = field.num_image_rows(blit_info.rect.height());

                    let lines = [y, i, q].map(|plane| {
                        &mut plane
                            [num_skipped_rows * width..(num_skipped_rows + num_rect_rows) * width]
                    });
                    ZipChunks::new(lines, width).par_for_each(|row_idx, [y, i, q]| {
                        // For interleaved fields, we write the first field into the first half of the buffer,
                        // and the second field into the second half.
                        let mut src_row_idx = match field {
                            YiqField::Upper => row_idx * 2,
                            YiqField::Lower => (row_idx * 2) + 1,
                            YiqField::Both => row_idx,
                            _ => unreachable!(),
                        };
                        src_row_idx += blit_info.rect.top;
                        if blit_info.flip_y {
                            src_row_idx = blit_info.other_buffer_height - src_row_idx - 1;
                        }
                        if blit_info.rect.height() == 1 {
                            src_row_idx = 0;
                        }
                        unsafe {
                            blit_row_simd::<S, T, F>(
                                y,
                                i,
                                q,
                                buf,
                                &blit_info,
                                src_row_idx,
                                pixel_transform,
                            );
                        }
                    });
                })
            }
            YiqField::InterleavedUpper => {
                let num_upper_rows = YiqField::Upper.num_actual_image_rows(self.dimensions.1);
                let (mut upper, mut lower) = self.split_at_row(num_upper_rows);
                if let Some(upper) = upper.as_mut() {
                    upper.field = YiqField::Upper;
                    unsafe {
                        upper.set_from_strided_buffer_maybe_uninit::<S, T, F>(
                            buf,
                            blit_info,
                            pixel_transform,
                        )
                    };
                };

                if let Some(lower) = lower.as_mut() {
                    lower.field = YiqField::Lower;
                    unsafe {
                        lower.set_from_strided_buffer_maybe_uninit::<S, T, F>(
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
                        upper.set_from_strided_buffer_maybe_uninit::<S, T, F>(
                            buf,
                            blit_info,
                            pixel_transform,
                        )
                    };
                };

                if let Some(lower) = lower.as_mut() {
                    lower.field = YiqField::Lower;
                    unsafe {
                        lower.set_from_strided_buffer_maybe_uninit::<S, T, F>(
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
    pub fn set_from_strided_buffer<S: PixelFormat, T: Normalize, F: PixelTransform>(
        &mut self,
        buf: &[T],
        blit_info: BlitInfo,
        pixel_transform: F,
    ) {
        // Safety: We know this data is valid because it's a slice.
        unsafe {
            self.set_from_strided_buffer_maybe_uninit::<S, T, F>(
                slice_to_maybe_uninit(buf),
                blit_info,
                pixel_transform,
            )
        }
    }

    /// Convert (a given part of) the YIQ planar data back into the given pixel fornat, and optionally apply a color
    /// transform to the pixels, before writing it into the destination buffer. This method allows you to write into a
    /// buffer which may not be initialized beforehand.
    pub fn write_to_strided_buffer_maybe_uninit<S: PixelFormat, T: Normalize, F: PixelTransform>(
        &self,
        dst: &mut [MaybeUninit<T>],
        mut blit_info: BlitInfo,
        deinterlace_mode: DeinterlaceMode,
        pixel_transform: F,
    ) {
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

        assert!(S::NUM_COMPONENTS >= 3);
        assert!(
            blit_info.row_bytes / std::mem::size_of::<T>() * S::NUM_COMPONENTS
                >= blit_info.rect.width(),
            "Blit rectangle width exceeds rowbytes"
        );
        assert_eq!(
            blit_info.row_bytes % std::mem::size_of::<T>(),
            0,
            "Rowbytes not aligned to datatype"
        );
        assert!(blit_info.rect.width() + blit_info.destination.0 <= self.dimensions.0);
        assert!(blit_info.rect.height() + blit_info.destination.1 <= self.dimensions.1);

        unsafe fn write_single_row_simd_inner<
            S: PixelFormat,
            T: Normalize,
            F: PixelTransform,
            Simd: F32x4,
        >(
            view: &YiqView,
            blit_info: &BlitInfo,
            deinterlace_mode: DeinterlaceMode,
            mut dst_row_idx: usize,
            dst_row: &mut [MaybeUninit<T>],
            pixel_transform: F,
        ) {
            let (r_idx, g_idx, b_idx, a_idx) = S::RGBA_INDICES;
            let width = view.dimensions.0;
            let output_height = blit_info.other_buffer_height;
            let num_rows = view.num_rows();
            // If the row index modulo 2 equals this number, that row was not rendered in the source data and we need to
            // interpolate between the rows above and beneath it.
            let skip_field: usize = match view.field {
                YiqField::Upper => 1,
                YiqField::Lower => 0,
                // The row index modulo 2 never reaches 2, meaning we don't skip any rows
                YiqField::Both | YiqField::InterleavedUpper | YiqField::InterleavedLower => 2,
            };

            match (deinterlace_mode, view.field) {
                (DeinterlaceMode::Bob, YiqField::Upper | YiqField::Lower) => {
                    // Limit to the actual width of the output (rowbytes may include trailing padding)
                    let dst_row = &mut dst_row[blit_info.destination.0 * S::NUM_COMPONENTS
                        ..(blit_info.destination.0 + blit_info.rect.width()) * S::NUM_COMPONENTS];
                    dst_row_idx += blit_info.rect.top;
                    if blit_info.flip_y {
                        dst_row_idx = output_height - dst_row_idx - 1;
                    }
                    // Inner fields with lines above and below them. Interpolate between those fields
                    if (dst_row_idx & 1) == skip_field
                        && dst_row_idx != 0
                        && dst_row_idx != output_height - 1
                    {
                        for (pix_idx, pixel) in
                            dst_row.chunks_exact_mut(S::NUM_COMPONENTS).enumerate()
                        {
                            let src_idx_lower =
                                ((dst_row_idx - 1) >> 1) * width + pix_idx + blit_info.rect.left;
                            let src_idx_upper =
                                ((dst_row_idx + 1) >> 1) * width + pix_idx + blit_info.rect.left;

                            let upper_pixel = unsafe {
                                Simd::load4(&[
                                    view.y[src_idx_upper],
                                    view.i[src_idx_upper],
                                    view.q[src_idx_upper],
                                    0.0,
                                ])
                            };
                            let lower_pixel = unsafe {
                                Simd::load4(&[
                                    view.y[src_idx_lower],
                                    view.i[src_idx_lower],
                                    view.q[src_idx_lower],
                                    0.0,
                                ])
                            };

                            let interp_pixel =
                                (upper_pixel + lower_pixel) * unsafe { Simd::set1(0.5) };

                            let rgba = T::from_norm(
                                pixel_transform.transform_pixel(yiq_to_rgb(interp_pixel)),
                            );
                            pixel[r_idx] = MaybeUninit::new(rgba[0]);
                            pixel[g_idx] = MaybeUninit::new(rgba[1]);
                            pixel[b_idx] = MaybeUninit::new(rgba[2]);
                            if let Some(a_idx) = a_idx {
                                pixel[a_idx] = MaybeUninit::new(T::ONE);
                            }
                        }
                    } else {
                        // Copy the field directly
                        for (pix_idx, pixel) in
                            dst_row.chunks_exact_mut(S::NUM_COMPONENTS).enumerate()
                        {
                            let src_idx = (dst_row_idx >> 1).min(num_rows - 1) * width
                                + pix_idx
                                + blit_info.rect.left;
                            let rgba =
                                T::from_norm(pixel_transform.transform_pixel(yiq_to_rgb(unsafe {
                                    Simd::load4(&[
                                        view.y[src_idx],
                                        view.i[src_idx],
                                        view.q[src_idx],
                                        0.0,
                                    ])
                                })));
                            pixel[r_idx] = MaybeUninit::new(rgba[0]);
                            pixel[g_idx] = MaybeUninit::new(rgba[1]);
                            pixel[b_idx] = MaybeUninit::new(rgba[2]);
                            if let Some(a_idx) = a_idx {
                                pixel[a_idx] = MaybeUninit::new(T::ONE);
                            }
                        }
                    }
                }
                (DeinterlaceMode::Skip, YiqField::Upper | YiqField::Lower) => {
                    // Limit to the actual width of the output (rowbytes may include trailing padding)
                    let dst_row = &mut dst_row[blit_info.destination.0 * S::NUM_COMPONENTS
                        ..(blit_info.destination.0 + blit_info.rect.width()) * S::NUM_COMPONENTS];
                    dst_row_idx += blit_info.rect.top;
                    if blit_info.flip_y {
                        dst_row_idx = output_height - dst_row_idx - 1;
                    }
                    if (dst_row_idx & 1) == skip_field {
                        return;
                    }
                    for (pix_idx, pixel) in dst_row.chunks_exact_mut(S::NUM_COMPONENTS).enumerate()
                    {
                        let src_idx = (dst_row_idx >> 1).min(num_rows - 1) * width
                            + pix_idx
                            + blit_info.rect.left;
                        let rgba =
                            T::from_norm(pixel_transform.transform_pixel(yiq_to_rgb(unsafe {
                                Simd::load4(&[
                                    view.y[src_idx],
                                    view.i[src_idx],
                                    view.q[src_idx],
                                    0.0,
                                ])
                            })));
                        pixel[r_idx] = MaybeUninit::new(rgba[0]);
                        pixel[g_idx] = MaybeUninit::new(rgba[1]);
                        pixel[b_idx] = MaybeUninit::new(rgba[2]);
                        if let Some(a_idx) = a_idx {
                            pixel[a_idx] = MaybeUninit::new(T::ONE);
                        }
                    }
                }
                (_, YiqField::InterleavedUpper | YiqField::InterleavedLower) => {
                    // Limit to the actual width of the output (rowbytes may include trailing padding)
                    let dst_row = &mut dst_row[blit_info.destination.0 * S::NUM_COMPONENTS
                        ..(blit_info.destination.0 + blit_info.rect.width()) * S::NUM_COMPONENTS];
                    dst_row_idx += blit_info.rect.top;
                    if blit_info.flip_y {
                        dst_row_idx = output_height - dst_row_idx - 1;
                    }
                    let row_offset = match view.field {
                        YiqField::InterleavedUpper => {
                            YiqField::Upper.num_image_rows(view.dimensions.1) * (dst_row_idx & 1)
                        }
                        YiqField::InterleavedLower => {
                            YiqField::Lower.num_image_rows(view.dimensions.1)
                                * (1 - (dst_row_idx & 1))
                        }
                        _ => unreachable!(),
                    };
                    // handle edge case where there's only one row and the mode is InterleavedLower
                    let interleaved_row_idx =
                        ((dst_row_idx >> 1) + row_offset).min(view.dimensions.1 - 1);
                    let src_idx = interleaved_row_idx * width;
                    for (pix_idx, pixel) in dst_row.chunks_exact_mut(S::NUM_COMPONENTS).enumerate()
                    {
                        let rgba =
                            T::from_norm(pixel_transform.transform_pixel(yiq_to_rgb(unsafe {
                                Simd::load4(&[
                                    view.y[src_idx + pix_idx + blit_info.rect.left],
                                    view.i[src_idx + pix_idx + blit_info.rect.left],
                                    view.q[src_idx + pix_idx + blit_info.rect.left],
                                    0.0,
                                ])
                            })));
                        pixel[r_idx] = MaybeUninit::new(rgba[0]);
                        pixel[g_idx] = MaybeUninit::new(rgba[1]);
                        pixel[b_idx] = MaybeUninit::new(rgba[2]);
                        if let Some(a_idx) = a_idx {
                            pixel[a_idx] = MaybeUninit::new(T::ONE);
                        }
                    }
                }
                _ => {
                    // Limit to the actual width of the output (rowbytes may include trailing padding)
                    let dst_row = &mut dst_row[blit_info.destination.0 * S::NUM_COMPONENTS
                        ..(blit_info.destination.0 + blit_info.rect.width()) * S::NUM_COMPONENTS];
                    dst_row_idx += blit_info.rect.top;
                    if blit_info.flip_y {
                        dst_row_idx = output_height - dst_row_idx - 1;
                    }
                    for (pix_idx, pixel) in dst_row.chunks_exact_mut(S::NUM_COMPONENTS).enumerate()
                    {
                        let src_idx =
                            dst_row_idx.min(num_rows - 1) * width + pix_idx + blit_info.rect.left;
                        let rgba =
                            T::from_norm(pixel_transform.transform_pixel(yiq_to_rgb(unsafe {
                                Simd::load4(&[
                                    view.y[src_idx],
                                    view.i[src_idx],
                                    view.q[src_idx],
                                    0.0,
                                ])
                            })));
                        pixel[r_idx] = MaybeUninit::new(rgba[0]);
                        pixel[g_idx] = MaybeUninit::new(rgba[1]);
                        pixel[b_idx] = MaybeUninit::new(rgba[2]);
                        if let Some(a_idx) = a_idx {
                            pixel[a_idx] = MaybeUninit::new(T::ONE);
                        }
                    }
                }
            }
        }

        #[simd_dispatch(Simd, scalar_fallback)]
        fn write_single_row_simd<S: PixelFormat, T: Normalize, F: PixelTransform>(
            view: &YiqView,
            blit_info: &BlitInfo,
            deinterlace_mode: DeinterlaceMode,
            dst_row_idx: usize,
            dst_row: &mut [MaybeUninit<T>],
            pixel_transform: F,
        ) {
            write_single_row_simd_inner::<S, T, F, Simd>(
                view,
                blit_info,
                deinterlace_mode,
                dst_row_idx,
                dst_row,
                pixel_transform,
            )
        }

        with_thread_pool(|| {
            let row_length = blit_info.row_bytes / std::mem::size_of::<T>();

            let skip_rows = blit_info.destination.1;
            let take_rows = blit_info.rect.height();
            let chunks = ZipChunks::new(
                [&mut dst[skip_rows * row_length..(skip_rows + take_rows) * row_length]],
                row_length,
            );

            chunks.par_for_each(|dst_row_idx, [dst_row]| {
                write_single_row_simd::<S, T, F>(
                    self,
                    &blit_info,
                    deinterlace_mode,
                    dst_row_idx,
                    dst_row,
                    pixel_transform,
                );
            });
        });
    }

    pub fn write_to_strided_buffer<S: PixelFormat, T: Normalize, F: PixelTransform>(
        &self,
        dst: &mut [T],
        blit_info: BlitInfo,
        deinterlace_mode: DeinterlaceMode,
        pixel_transform: F,
    ) {
        self.write_to_strided_buffer_maybe_uninit::<S, T, F>(
            unsafe { slice_to_maybe_uninit_mut(dst) },
            blit_info,
            deinterlace_mode,
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
    pub fn from_strided_buffer<S: PixelFormat, T: Normalize>(
        buf: &[T],
        row_bytes: usize,
        width: usize,
        height: usize,
        field: YiqField,
    ) -> Self {
        let mut data = vec![0f32; YiqView::buf_length_for((width, height), field)];
        let mut view = YiqView::from_parts(&mut data, (width, height), field);

        view.set_from_strided_buffer::<S, T, _>(
            buf,
            BlitInfo::from_full_frame(width, height, row_bytes),
            (),
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
