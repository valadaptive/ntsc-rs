use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
    sync::OnceLock,
};

/// A trait for 4-wide single-precision floating point SIMD vectors, abstracting over different architectures and
/// intrinsics. This is implemented for NEON, SSE4.1, and AVX2. It only contains the ops that I need to implement the
/// IIR filter.
///
/// Soundness is upheld by making all ways to *construct* a SIMD vector (e.g. load, set1) unsafe. SIMD operations are
/// unsafe because the intrinsics are bogus instructions with undefined behavior on platforms that don't support
/// them--therefore, we treat the creation of a SIMD vector for a given instruction set as an "assertion" that the CPU
/// supports said instruction set.
#[allow(dead_code)]
pub trait F32x4:
    Sized
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Copy
    + Debug
{
    /// Safety:
    /// You must ensure that whatever flavor of SIMD vector you're creating is supported by the current CPU.
    unsafe fn load(src: &[f32]) -> Self;

    /// Safety:
    /// You must ensure that whatever flavor of SIMD vector you're creating is supported by the current CPU.
    #[inline(always)]
    unsafe fn load4(src: &[f32; 4]) -> Self {
        Self::load(src.as_slice())
    }

    /// Safety:
    /// You must ensure that whatever flavor of SIMD vector you're creating is supported by the current CPU.
    unsafe fn load1(src: &f32) -> Self;

    fn store(self, dst: &mut [f32]);
    fn store1(self, dst: &mut f32);

    /// Safety:
    /// You must ensure that whatever flavor of SIMD vector you're creating is supported by the current CPU.
    unsafe fn set1(src: f32) -> Self;

    fn mul_add(self, a: Self, b: Self) -> Self;
    fn mul_sub(self, a: Self, b: Self) -> Self;
    fn neg_mul_add(self, a: Self, b: Self) -> Self;
    fn neg_mul_sub(self, a: Self, b: Self) -> Self;
    fn swizzle(self, x: i32, y: i32, z: i32, w: i32) -> Self;
    fn insert<const INDEX: i32>(self, value: f32) -> Self;
}

#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
    use std::arch::x86_64::{
        __m128, _mm_add_ps, _mm_broadcast_ss, _mm_castps_si128, _mm_castsi128_ps, _mm_div_ps,
        _mm_fmadd_ps, _mm_fmsub_ps, _mm_fnmadd_ps, _mm_fnmsub_ps, _mm_insert_epi32, _mm_loadu_ps,
        _mm_mul_ps, _mm_permutevar_ps, _mm_set1_ps, _mm_set_epi32, _mm_set_epi8, _mm_shuffle_epi8,
        _mm_store_ss, _mm_storeu_ps, _mm_sub_ps,
    };
    use std::{
        fmt::Debug,
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
    };

    use super::F32x4;

    // Most intrinsics can be shared between the AVX2 and SSE4.1 implementations, and they both operate on the __m128
    // type. We can reuse most of the code, and just check `if USE_AVX2` on those ops which differ (mostly fused
    // multiply-add, with load and swizzle as well).
    #[derive(Clone, Copy, Debug)]
    #[repr(transparent)]
    pub struct IntelF32x4<const USE_AVX2: bool>(__m128);

    pub type AvxF32x4 = IntelF32x4<true>;
    pub type SseF32x4 = IntelF32x4<false>;

    impl<const USE_AVX2: bool> From<__m128> for IntelF32x4<USE_AVX2> {
        #[inline(always)]
        fn from(src: __m128) -> Self {
            IntelF32x4(src)
        }
    }

    impl<const USE_AVX2: bool> From<IntelF32x4<USE_AVX2>> for __m128 {
        #[inline(always)]
        fn from(src: IntelF32x4<USE_AVX2>) -> Self {
            src.0
        }
    }

    impl<const USE_AVX2: bool> Add for IntelF32x4<USE_AVX2> {
        type Output = Self;

        #[inline(always)]
        fn add(self, rhs: Self) -> Self::Output {
            unsafe { _mm_add_ps(self.into(), rhs.into()).into() }
        }
    }

    impl<const USE_AVX2: bool> Sub for IntelF32x4<USE_AVX2> {
        type Output = Self;

        #[inline(always)]
        fn sub(self, rhs: Self) -> Self::Output {
            unsafe { _mm_sub_ps(self.into(), rhs.into()).into() }
        }
    }

    impl<const USE_AVX2: bool> Mul for IntelF32x4<USE_AVX2> {
        type Output = Self;

        #[inline(always)]
        fn mul(self, rhs: Self) -> Self::Output {
            unsafe { _mm_mul_ps(self.into(), rhs.into()).into() }
        }
    }

    impl<const USE_AVX2: bool> Div for IntelF32x4<USE_AVX2> {
        type Output = Self;

        #[inline(always)]
        fn div(self, rhs: Self) -> Self::Output {
            unsafe { _mm_div_ps(self.into(), rhs.into()).into() }
        }
    }

    impl<const USE_AVX2: bool> AddAssign for IntelF32x4<USE_AVX2> {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl<const USE_AVX2: bool> SubAssign for IntelF32x4<USE_AVX2> {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl<const USE_AVX2: bool> MulAssign for IntelF32x4<USE_AVX2> {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl<const USE_AVX2: bool> DivAssign for IntelF32x4<USE_AVX2> {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }

    impl<const USE_AVX2: bool> F32x4 for IntelF32x4<USE_AVX2> {
        #[inline(always)]
        unsafe fn load(src: &[f32]) -> Self {
            // SAFETY: the range operator ensures that the slice is at least 4 elements long
            unsafe { _mm_loadu_ps(src[0..4].as_ptr()).into() }
        }

        #[inline(always)]
        unsafe fn load1(src: &f32) -> Self {
            if USE_AVX2 {
                unsafe { _mm_broadcast_ss(src).into() }
            } else {
                unsafe { _mm_set1_ps(*src).into() }
            }
        }

        #[inline(always)]
        fn store(self, dst: &mut [f32]) {
            // SAFETY: the range operator ensures that the slice is at least 4 elements long
            unsafe { _mm_storeu_ps(dst[0..4].as_mut_ptr(), self.into()) }
        }

        #[inline(always)]
        fn store1(self, dst: &mut f32) {
            unsafe { _mm_store_ss(dst, self.into()) }
        }

        #[inline(always)]
        unsafe fn set1(src: f32) -> Self {
            unsafe { _mm_set1_ps(src).into() }
        }

        #[inline(always)]
        fn mul_add(self, a: Self, b: Self) -> Self {
            if USE_AVX2 {
                unsafe { _mm_fmadd_ps(self.into(), a.into(), b.into()).into() }
            } else {
                (self * a) + b
            }
        }

        #[inline(always)]
        fn mul_sub(self, a: Self, b: Self) -> Self {
            if USE_AVX2 {
                unsafe { _mm_fmsub_ps(self.into(), a.into(), b.into()).into() }
            } else {
                (self * a) - b
            }
        }

        #[inline(always)]
        fn neg_mul_add(self, a: Self, b: Self) -> Self {
            if USE_AVX2 {
                unsafe { _mm_fnmadd_ps(self.into(), a.into(), b.into()).into() }
            } else {
                b - (self * a)
            }
        }

        #[inline(always)]
        fn neg_mul_sub(self, a: Self, b: Self) -> Self {
            if USE_AVX2 {
                unsafe { _mm_fnmsub_ps(self.into(), a.into(), b.into()).into() }
            } else {
                (unsafe { Self::set1(0.0) } - (self * a)) - b
            }
        }

        #[inline(always)]
        fn swizzle(self, x: i32, y: i32, z: i32, w: i32) -> Self {
            assert!((x | y | z | w) & !3 == 0, "Invalid swizzle indices");
            if USE_AVX2 {
                unsafe { _mm_permutevar_ps(self.into(), _mm_set_epi32(w, z, y, x)).into() }
            } else {
                let x = (x << 2) as i8;
                let y = (y << 2) as i8;
                let z = (z << 2) as i8;
                let w = (w << 2) as i8;

                unsafe {
                    let control_mask = _mm_set_epi8(
                        w + 3,
                        w + 2,
                        w + 1,
                        w + 0,
                        z + 3,
                        z + 2,
                        z + 1,
                        z + 0,
                        y + 3,
                        y + 2,
                        y + 1,
                        y + 0,
                        x + 3,
                        x + 2,
                        x + 1,
                        x + 0,
                    );
                    _mm_castsi128_ps(_mm_shuffle_epi8(
                        _mm_castps_si128(self.into()),
                        control_mask,
                    ))
                    .into()
                }
            }
        }

        #[inline(always)]
        fn insert<const INDEX: i32>(self, value: f32) -> Self {
            assert!(INDEX & !3 == 0, "Invalid insert index");
            unsafe {
                _mm_castsi128_ps(_mm_insert_epi32::<INDEX>(
                    _mm_castps_si128(self.into()),
                    value.to_bits() as i32,
                ))
                .into()
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
    use std::arch::aarch64::{
        float32x4_t, uint8x16_t, vaddq_f32, vcombine_u8, vcreate_u8, vdivq_f32, vdupq_n_f32,
        vfmaq_f32, vfmsq_f32, vld1q_dup_f32, vld1q_f32, vmulq_f32, vnegq_f32, vqtbl1q_u8,
        vreinterpretq_f32_u8, vreinterpretq_u8_f32, vsetq_lane_f32, vst1q_f32, vst1q_lane_f32,
        vsubq_f32,
    };
    use std::{
        fmt::Debug,
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
    };

    use super::F32x4;

    #[derive(Clone, Copy, Debug)]
    #[repr(transparent)]
    pub struct ArmF32x4(float32x4_t);

    impl From<float32x4_t> for ArmF32x4 {
        #[inline(always)]
        fn from(src: float32x4_t) -> Self {
            ArmF32x4(src)
        }
    }

    impl From<ArmF32x4> for float32x4_t {
        #[inline(always)]
        fn from(src: ArmF32x4) -> Self {
            src.0
        }
    }

    impl Add for ArmF32x4 {
        type Output = Self;

        #[inline(always)]
        fn add(self, rhs: Self) -> Self::Output {
            unsafe { vaddq_f32(self.into(), rhs.into()).into() }
        }
    }

    impl Sub for ArmF32x4 {
        type Output = Self;

        #[inline(always)]
        fn sub(self, rhs: Self) -> Self::Output {
            unsafe { vsubq_f32(self.into(), rhs.into()).into() }
        }
    }

    impl Mul for ArmF32x4 {
        type Output = Self;

        #[inline(always)]
        fn mul(self, rhs: Self) -> Self::Output {
            unsafe { vmulq_f32(self.into(), rhs.into()).into() }
        }
    }

    impl Div for ArmF32x4 {
        type Output = Self;

        #[inline(always)]
        fn div(self, rhs: Self) -> Self::Output {
            unsafe { vdivq_f32(self.into(), rhs.into()).into() }
        }
    }

    impl AddAssign for ArmF32x4 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl SubAssign for ArmF32x4 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl MulAssign for ArmF32x4 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl DivAssign for ArmF32x4 {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }

    impl F32x4 for ArmF32x4 {
        #[inline(always)]
        unsafe fn load(src: &[f32]) -> Self {
            // SAFETY: the range operator ensures that the slice is at least 4 elements long
            unsafe { vld1q_f32(src[0..4].as_ptr()).into() }
        }

        #[inline(always)]
        unsafe fn load1(src: &f32) -> Self {
            unsafe { vld1q_dup_f32(src as *const _).into() }
        }

        #[inline(always)]
        fn store(self, dst: &mut [f32]) {
            // SAFETY: the range operator ensures that the slice is at least 4 elements long
            unsafe { vst1q_f32(dst[0..4].as_mut_ptr(), self.into()) }
        }

        #[inline(always)]
        fn store1(self, dst: &mut f32) {
            unsafe { vst1q_lane_f32::<0>(dst as *mut _, self.into()) };
        }

        #[inline(always)]
        unsafe fn set1(src: f32) -> Self {
            unsafe { vdupq_n_f32(src).into() }
        }

        #[inline(always)]
        fn mul_add(self, a: Self, b: Self) -> Self {
            unsafe { vfmaq_f32(b.into(), self.into(), a.into()).into() }
        }

        #[inline(always)]
        fn mul_sub(self, a: Self, b: Self) -> Self {
            unsafe { vnegq_f32(vfmsq_f32(b.into(), self.into(), a.into())).into() }
        }

        #[inline(always)]
        fn neg_mul_add(self, a: Self, b: Self) -> Self {
            unsafe { vfmsq_f32(b.into(), self.into(), a.into()).into() }
        }

        #[inline(always)]
        fn neg_mul_sub(self, a: Self, b: Self) -> Self {
            unsafe { vnegq_f32(vfmaq_f32(b.into(), self.into(), a.into())).into() }
        }

        #[inline(always)]
        fn swizzle(self, x: i32, y: i32, z: i32, w: i32) -> Self {
            assert!((x | y | z | w) & !3 == 0, "Invalid swizzle indices");
            let x = (x << 2) as u64;
            let y = (y << 2) as u64;
            let z = (z << 2) as u64;
            let w = (w << 2) as u64;
            unsafe {
                let indexes: uint8x16_t = vcombine_u8(
                    vcreate_u8(
                        (x + 0)
                            | (x + 1) << 8
                            | (x + 2) << 16
                            | (x + 3) << 24
                            | (y + 0) << 32
                            | (y + 1) << 40
                            | (y + 2) << 48
                            | (y + 3) << 56,
                    ),
                    vcreate_u8(
                        (z + 0)
                            | (z + 1) << 8
                            | (z + 2) << 16
                            | (z + 3) << 24
                            | (w + 0) << 32
                            | (w + 1) << 40
                            | (w + 2) << 48
                            | (w + 3) << 56,
                    ),
                );
                vreinterpretq_f32_u8(vqtbl1q_u8(vreinterpretq_u8_f32(self.into()), indexes)).into()
            }
        }

        #[inline(always)]
        fn insert<const INDEX: i32>(self, value: f32) -> Self {
            assert!(INDEX & !3 == 0, "Invalid insert index");
            unsafe { vsetq_lane_f32::<INDEX>(value, self.into()).into() }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm32 {
    use std::{
        arch::wasm32::{
            f32x4_add, f32x4_div, f32x4_mul, f32x4_replace_lane, f32x4_splat, f32x4_sub,
            i8x16_swizzle, u8x16, v128, v128_load, v128_store, v128_store32_lane,
        },
        fmt::Debug,
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
    };

    use super::F32x4;

    #[derive(Clone, Copy, Debug)]
    #[repr(transparent)]
    pub struct WasmF32x4(v128);

    impl From<v128> for WasmF32x4 {
        #[inline(always)]
        fn from(src: v128) -> Self {
            WasmF32x4(src)
        }
    }

    impl From<WasmF32x4> for v128 {
        #[inline(always)]
        fn from(src: WasmF32x4) -> Self {
            src.0
        }
    }

    impl Add for WasmF32x4 {
        type Output = Self;

        #[inline(always)]
        fn add(self, rhs: Self) -> Self::Output {
            f32x4_add(self.into(), rhs.into()).into()
        }
    }

    impl Sub for WasmF32x4 {
        type Output = Self;

        #[inline(always)]
        fn sub(self, rhs: Self) -> Self::Output {
            f32x4_sub(self.into(), rhs.into()).into()
        }
    }

    impl Mul for WasmF32x4 {
        type Output = Self;

        #[inline(always)]
        fn mul(self, rhs: Self) -> Self::Output {
            f32x4_mul(self.into(), rhs.into()).into()
        }
    }

    impl Div for WasmF32x4 {
        type Output = Self;

        #[inline(always)]
        fn div(self, rhs: Self) -> Self::Output {
            f32x4_div(self.into(), rhs.into()).into()
        }
    }

    impl AddAssign for WasmF32x4 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl SubAssign for WasmF32x4 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl MulAssign for WasmF32x4 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl DivAssign for WasmF32x4 {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }

    impl F32x4 for WasmF32x4 {
        unsafe fn load(src: &[f32]) -> Self {
            v128_load(src[0..4].as_ptr() as _).into()
        }

        unsafe fn load1(src: &f32) -> Self {
            f32x4_splat(*src).into()
        }

        fn store(self, dst: &mut [f32]) {
            // SAFETY: the range operator ensures that the slice is at least 4 elements long
            unsafe { v128_store(dst[0..4].as_mut_ptr() as _, self.into()) };
        }

        fn store1(self, dst: &mut f32) {
            unsafe { v128_store32_lane::<0>(self.into(), dst as *mut f32 as *mut _) };
        }

        unsafe fn set1(src: f32) -> Self {
            f32x4_splat(src).into()
        }

        fn mul_add(self, a: Self, b: Self) -> Self {
            (self * a) + b
        }

        fn mul_sub(self, a: Self, b: Self) -> Self {
            (self * a) - b
        }

        fn neg_mul_add(self, a: Self, b: Self) -> Self {
            b - (self * a)
        }

        fn neg_mul_sub(self, a: Self, b: Self) -> Self {
            (unsafe { Self::set1(0.0) } - (self * a)) - b
        }

        fn swizzle(self, x: i32, y: i32, z: i32, w: i32) -> Self {
            assert!((x | y | z | w) & !3 == 0, "Invalid swizzle indices");
            let x = (x << 2) as u8;
            let y = (y << 2) as u8;
            let z = (z << 2) as u8;
            let w = (w << 2) as u8;
            let indexes = u8x16(
                x + 0,
                x + 1,
                x + 2,
                x + 3,
                y + 0,
                y + 1,
                y + 2,
                y + 3,
                z + 0,
                z + 1,
                z + 2,
                z + 3,
                w + 0,
                w + 1,
                w + 2,
                w + 3,
            );
            i8x16_swizzle(self.into(), indexes).into()
        }

        fn insert<const INDEX: i32>(self, value: f32) -> Self {
            // Unlike the Intel and ARM SIMD ops, WASM takes a usize as the lane index! Thanks Rust!!!
            match INDEX {
                0 => f32x4_replace_lane::<0>(self.into(), value).into(),
                1 => f32x4_replace_lane::<1>(self.into(), value).into(),
                2 => f32x4_replace_lane::<2>(self.into(), value).into(),
                3 => f32x4_replace_lane::<3>(self.into(), value).into(),
                _ => panic!("Invalid insert index"),
            }
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupportedSimdType {
    Avx2,
    Sse41,
    Neon,
    Wasm,
    None,
}

static SUPPORTED_SIMD_TYPE: OnceLock<SupportedSimdType> = OnceLock::new();

pub fn get_supported_simd_type() -> SupportedSimdType {
    #[allow(unreachable_code)]
    *SUPPORTED_SIMD_TYPE.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            return if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                SupportedSimdType::Avx2
            } else if is_x86_feature_detected!("sse4.1") {
                SupportedSimdType::Sse41
            } else {
                SupportedSimdType::None
            };
        }
        #[cfg(target_arch = "aarch64")]
        {
            return SupportedSimdType::Neon;
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            return SupportedSimdType::Wasm;
        }
        SupportedSimdType::None
    })
}
