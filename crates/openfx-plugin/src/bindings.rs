#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unused)]
#![allow(clippy::all)]

#[must_use]
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct OfxStatus(pub std::ffi::c_int);

impl OfxStatus {
    pub fn ofx_ok(self) -> OfxResult<()> {
        if self.0 == 0 { Ok(()) } else { Err(self) }
    }
}

impl std::fmt::Debug for OfxStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            0 => f.write_str("kOfxStatOK"),
            1 => f.write_str("kOfxStatFailed"),
            2 => f.write_str("kOfxStatErrFatal"),
            3 => f.write_str("kOfxStatErrUnknown"),
            4 => f.write_str("kOfxStatErrMissingHostFeature"),
            5 => f.write_str("kOfxStatErrUnsupported"),
            6 => f.write_str("kOfxStatErrExists"),
            7 => f.write_str("kOfxStatErrFormat"),
            8 => f.write_str("kOfxStatErrMemory"),
            9 => f.write_str("kOfxStatErrBadHandle"),
            10 => f.write_str("kOfxStatErrBadIndex"),
            11 => f.write_str("kOfxStatErrValue"),
            12 => f.write_str("kOfxStatReplyYes"),
            13 => f.write_str("kOfxStatReplyNo"),
            14 => f.write_str("kOfxStatReplyDefault"),
            _ => f.debug_tuple("OfxStatus").field(&self.0).finish(),
        }
    }
}

impl From<std::ffi::c_int> for OfxStatus {
    fn from(value: std::ffi::c_int) -> Self {
        Self(value)
    }
}

pub type OfxResult<T> = Result<T, OfxStatus>;

// bindgen can't import these
#[allow(dead_code)]
pub mod OfxStat {
    use crate::bindings::OfxStatus;

    pub const kOfxStatOK: OfxStatus = OfxStatus(0);
    pub const kOfxStatFailed: OfxStatus = OfxStatus(1);
    pub const kOfxStatErrFatal: OfxStatus = OfxStatus(2);
    pub const kOfxStatErrUnknown: OfxStatus = OfxStatus(3);
    pub const kOfxStatErrMissingHostFeature: OfxStatus = OfxStatus(4);
    pub const kOfxStatErrUnsupported: OfxStatus = OfxStatus(5);
    pub const kOfxStatErrExists: OfxStatus = OfxStatus(6);
    pub const kOfxStatErrFormat: OfxStatus = OfxStatus(7);
    pub const kOfxStatErrMemory: OfxStatus = OfxStatus(8);
    pub const kOfxStatErrBadHandle: OfxStatus = OfxStatus(9);
    pub const kOfxStatErrBadIndex: OfxStatus = OfxStatus(10);
    pub const kOfxStatErrValue: OfxStatus = OfxStatus(11);
    pub const kOfxStatReplyYes: OfxStatus = OfxStatus(12);
    pub const kOfxStatReplyNo: OfxStatus = OfxStatus(13);
    pub const kOfxStatReplyDefault: OfxStatus = OfxStatus(14);
}

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
