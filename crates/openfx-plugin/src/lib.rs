#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use core::slice;
use std::{
    borrow::{Borrow, BorrowMut},
    ffi::{c_char, c_int, c_void, CStr, CString, FromBytesWithNulError},
    mem::{self, size_of, MaybeUninit},
    ptr::{self, NonNull},
    sync::{OnceLock, RwLock},
};

use ntscrs::settings::{NtscEffectFullSettings, SettingDescriptor, SettingKind, SettingsList};
use ntscrs::ToPrimitive;
use ntscrs::{
    ntsc::NtscEffect,
    settings::UseField,
    yiq_fielding::{rgb_to_yiq, yiq_to_rgb, YiqField, YiqView},
};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// https://stackoverflow.com/questions/53611161/how-do-i-expose-a-compile-time-generated-static-c-string-through-ffi
macro_rules! static_cstr {
    ($l:expr) => {
        unsafe { ::std::ffi::CStr::from_bytes_with_nul_unchecked(concat!($l, "\0").as_bytes()) }
    };
}

macro_rules! ofx_str {
    ($l:expr) => {
        $l as *const u8 as *const i8
    };
}

static mut PLUGIN_INFO: OnceLock<OfxPlugin> = OnceLock::new();
static shared_data: RwLock<Option<SharedData>> = RwLock::new(None);

struct HostInfo {
    // From the OpenFX api docs:
    // "This pointer will be valid while the binary containing the plug-in is loaded."
    // https://openfx.readthedocs.io/en/master/Reference/ofxHostStruct.html
    host: &'static OfxPropertySetStruct,
    fetchSuite: unsafe extern "C" fn(
        host: OfxPropertySetHandle,
        suiteName: *const c_char,
        suiteVersion: c_int,
    ) -> *const c_void,
}

struct SharedData {
    host_info: HostInfo,
    property_suite: &'static OfxPropertySuiteV1,
    image_effect_suite: &'static OfxImageEffectSuiteV1,
    memory_suite: &'static OfxMemorySuiteV1,
    parameter_suite: &'static OfxParameterSuiteV1,
    settings_list: SettingsList,
    supports_multiple_clip_depths: bool,
}

type OfxResult<T> = Result<T, OfxStatus>;

// bindgen can't import these
#[allow(dead_code)]
mod OfxStat {
    use std::ffi::c_int;

    pub const kOfxStatFailed: c_int = 1;
    pub const kOfxStatErrFatal: c_int = 2;
    pub const kOfxStatErrUnknown: c_int = 3;
    pub const kOfxStatErrMissingHostFeature: c_int = 4;
    pub const kOfxStatErrUnsupported: c_int = 5;
    pub const kOfxStatErrExists: c_int = 6;
    pub const kOfxStatErrFormat: c_int = 7;
    pub const kOfxStatErrMemory: c_int = 8;
    pub const kOfxStatErrBadHandle: c_int = 9;
    pub const kOfxStatErrBadIndex: c_int = 10;
    pub const kOfxStatErrValue: c_int = 11;
    pub const kOfxStatReplyYes: c_int = 12;
    pub const kOfxStatReplyNo: c_int = 13;
    pub const kOfxStatReplyDefault: c_int = 14;
}

impl SharedData {
    pub unsafe fn new(host_info: HostInfo) -> OfxResult<Self> {
        let property_suite = (host_info.fetchSuite)(
            host_info.host as *const _ as *mut _,
            ofx_str!(kOfxPropertySuite),
            1,
        ) as *const OfxPropertySuiteV1;
        let image_effect_suite = (host_info.fetchSuite)(
            host_info.host as *const _ as *mut _,
            ofx_str!(kOfxImageEffectSuite),
            1,
        ) as *const OfxImageEffectSuiteV1;
        let memory_suite = (host_info.fetchSuite)(
            host_info.host as *const _ as *mut _,
            ofx_str!(kOfxMemorySuite),
            1,
        ) as *const OfxMemorySuiteV1;
        let parameter_suite = (host_info.fetchSuite)(
            host_info.host as *const _ as *mut _,
            ofx_str!(kOfxParameterSuite),
            1,
        ) as *const OfxParameterSuiteV1;

        Ok(SharedData {
            host_info,
            property_suite: property_suite
                .as_ref()
                .ok_or(OfxStat::kOfxStatErrMissingHostFeature)?,
            image_effect_suite: image_effect_suite
                .as_ref()
                .ok_or(OfxStat::kOfxStatErrMissingHostFeature)?,
            memory_suite: memory_suite
                .as_ref()
                .ok_or(OfxStat::kOfxStatErrMissingHostFeature)?,
            parameter_suite: parameter_suite
                .as_ref()
                .ok_or(OfxStat::kOfxStatErrMissingHostFeature)?,
            settings_list: SettingsList::new(),
            supports_multiple_clip_depths: false,
        })
    }
}

unsafe fn set_host_info_inner(host: *mut OfxHost) -> OfxResult<()> {
    if let Some(host_struct) = host.as_ref() {
        let mut data = shared_data.write().map_err(|_| OfxStat::kOfxStatFailed)?;
        data.get_or_insert(SharedData::new(HostInfo {
            host: host_struct.host.as_ref().ok_or(OfxStat::kOfxStatFailed)?,
            fetchSuite: host_struct.fetchSuite.ok_or(OfxStat::kOfxStatFailed)?,
        })?);
        Ok(())
    } else {
        Err(OfxStat::kOfxStatFailed)
    }
}

unsafe extern "C" fn set_host_info(host: *mut OfxHost) {
    set_host_info_inner(host).unwrap();
}

unsafe fn action_load() -> OfxResult<()> {
    let mut data = shared_data.write().map_err(|_| OfxStat::kOfxStatFailed)?;
    let data = data.as_mut().ok_or(OfxStat::kOfxStatFailed)?;
    let propGetInt = data
        .property_suite
        .propGetInt
        .ok_or(OfxStat::kOfxStatFailed)?;
    let mut supports_multiple_clip_depths: c_int = 0;
    propGetInt(
        data.host_info.host as *const _ as *mut _,
        ofx_str!(kOfxImageEffectPropSupportsMultipleClipDepths),
        0,
        &mut supports_multiple_clip_depths,
    );
    data.supports_multiple_clip_depths = supports_multiple_clip_depths != 0;

    Ok(())
}

unsafe fn action_describe(descriptor: OfxImageEffectHandle) -> OfxResult<()> {
    let data = shared_data.read().map_err(|_| OfxStat::kOfxStatFailed)?;
    let data = data.as_ref().ok_or(OfxStat::kOfxStatFailed)?;
    let mut effectProps: OfxPropertySetHandle = ptr::null_mut();
    (data
        .image_effect_suite
        .getPropertySet
        .ok_or(OfxStat::kOfxStatFailed)?)(descriptor, &mut effectProps);

    let propSetString = data
        .property_suite
        .propSetString
        .ok_or(OfxStat::kOfxStatFailed)?;
    let propSetInt = data
        .property_suite
        .propSetInt
        .ok_or(OfxStat::kOfxStatFailed)?;

    propSetString(
        effectProps,
        ofx_str!(kOfxPropLabel),
        0,
        static_cstr!("NTSC-rs").as_ptr(),
    );

    propSetString(
        effectProps,
        ofx_str!(kOfxImageEffectPluginPropGrouping),
        0,
        static_cstr!("Filter").as_ptr(),
    );

    propSetString(
        effectProps,
        ofx_str!(kOfxImageEffectPropSupportedContexts),
        0,
        ofx_str!(kOfxImageEffectContextFilter),
    );
    // TODO needed for resolve support(?)
    propSetString(
        effectProps,
        ofx_str!(kOfxImageEffectPropSupportedContexts),
        1,
        ofx_str!(kOfxImageEffectContextGeneral),
    );

    propSetString(
        effectProps,
        ofx_str!(kOfxImageEffectPropSupportedPixelDepths),
        0,
        ofx_str!(kOfxBitDepthFloat),
    );
    propSetString(
        effectProps,
        ofx_str!(kOfxImageEffectPropSupportedPixelDepths),
        1,
        ofx_str!(kOfxBitDepthShort),
    );
    propSetString(
        effectProps,
        ofx_str!(kOfxImageEffectPropSupportedPixelDepths),
        2,
        ofx_str!(kOfxBitDepthByte),
    );

    // TODO: is this wrong?
    propSetString(
        effectProps,
        ofx_str!(kOfxImageEffectPluginRenderThreadSafety),
        0,
        ofx_str!(kOfxImageEffectRenderFullySafe),
    );
    // We'll manage threading ourselves
    propSetInt(
        effectProps,
        ofx_str!(kOfxImageEffectPluginPropHostFrameThreading),
        0,
        0,
    );
    // We need to operate on the whole image at once
    propSetInt(
        effectProps,
        ofx_str!(kOfxImageEffectPropSupportsTiles),
        0,
        0,
    );

    Ok(())
}

fn ofx_err(code: c_int) -> OfxResult<()> {
    match code {
        0 => Ok(()),
        _ => Err(code),
    }
}

unsafe fn setup_params(
    property_suite: &OfxPropertySuiteV1,
    param_suite: &OfxParameterSuiteV1,
    param_set: OfxParamSetHandle,
    setting_descriptors: &[SettingDescriptor],
    parent: &CStr,
) -> OfxResult<()> {
    let paramDefine = param_suite.paramDefine.ok_or(OfxStat::kOfxStatFailed)?;
    let propSetDouble = property_suite
        .propSetDouble
        .ok_or(OfxStat::kOfxStatFailed)?;
    let propSetInt = property_suite.propSetInt.ok_or(OfxStat::kOfxStatFailed)?;
    let propSetString = property_suite
        .propSetString
        .ok_or(OfxStat::kOfxStatFailed)?;
    for descriptor in setting_descriptors {
        let mut paramProps: OfxPropertySetHandle = ptr::null_mut();
        // the official OFX host support library clones this string (via the std::string constructor) and I really hope
        // all other OFX hosts do too, because the documentation is silent on wtf the lifetime of these strings is
        // supposed to be
        let descriptor_id_str = (descriptor.id.to_u32() as Option<u32>).unwrap().to_string();
        let descriptor_id_cstr = CString::new(descriptor_id_str.clone()).unwrap();

        match &descriptor.kind {
            SettingKind::Enumeration {
                options,
                default_value,
            } => {
                ofx_err(paramDefine(
                    param_set,
                    ofx_str!(kOfxParamTypeChoice),
                    descriptor_id_cstr.as_ptr(),
                    &mut paramProps,
                ))?;
                let mut default_idx: usize = 0;
                for (i, menu_item) in options.iter().enumerate() {
                    let item_label_cstr = CString::new(menu_item.label).unwrap();
                    ofx_err(propSetString(
                        paramProps,
                        ofx_str!(kOfxParamPropChoiceOption),
                        i as i32,
                        item_label_cstr.as_ptr(),
                    ))?;

                    if menu_item.index == *default_value {
                        default_idx = i;
                    }
                }
                ofx_err(propSetInt(
                    paramProps,
                    ofx_str!(kOfxParamPropDefault),
                    0,
                    default_idx as i32,
                ))?;
            }
            SettingKind::Percentage { default_value, .. } => {
                ofx_err(paramDefine(
                    param_set,
                    ofx_str!(kOfxParamTypeDouble),
                    descriptor_id_cstr.as_ptr(),
                    &mut paramProps,
                ))?;
                ofx_err(propSetString(
                    paramProps,
                    ofx_str!(kOfxParamPropDoubleType),
                    0,
                    ofx_str!(kOfxParamDoubleTypeScale),
                ))?;
                ofx_err(propSetDouble(
                    paramProps,
                    ofx_str!(kOfxParamPropDefault),
                    0,
                    *default_value as f64,
                ))?;
                ofx_err(propSetDouble(
                    paramProps,
                    ofx_str!(kOfxParamPropMin),
                    0,
                    0.0,
                ))?;
                ofx_err(propSetDouble(
                    paramProps,
                    ofx_str!(kOfxParamPropDisplayMin),
                    0,
                    0.0,
                ))?;
                ofx_err(propSetDouble(
                    paramProps,
                    ofx_str!(kOfxParamPropMax),
                    0,
                    1.0,
                ))?;
                ofx_err(propSetDouble(
                    paramProps,
                    ofx_str!(kOfxParamPropDisplayMax),
                    0,
                    1.0,
                ))?;
            }
            SettingKind::IntRange {
                range,
                default_value,
            } => {
                ofx_err(paramDefine(
                    param_set,
                    ofx_str!(kOfxParamTypeInteger),
                    descriptor_id_cstr.as_ptr(),
                    &mut paramProps,
                ))?;
                ofx_err(propSetInt(
                    paramProps,
                    ofx_str!(kOfxParamPropDefault),
                    0,
                    *default_value,
                ))?;
                ofx_err(propSetInt(
                    paramProps,
                    ofx_str!(kOfxParamPropMin),
                    0,
                    *range.start(),
                ))?;
                ofx_err(propSetInt(
                    paramProps,
                    ofx_str!(kOfxParamPropDisplayMin),
                    0,
                    *range.start(),
                ))?;
                ofx_err(propSetInt(
                    paramProps,
                    ofx_str!(kOfxParamPropMax),
                    0,
                    *range.end(),
                ))?;
                ofx_err(propSetInt(
                    paramProps,
                    ofx_str!(kOfxParamPropDisplayMax),
                    0,
                    *range.end(),
                ))?;
            }
            SettingKind::FloatRange {
                range,
                default_value,
                ..
            } => {
                ofx_err(paramDefine(
                    param_set,
                    ofx_str!(kOfxParamTypeDouble),
                    descriptor_id_cstr.as_ptr(),
                    &mut paramProps,
                ))?;
                ofx_err(propSetDouble(
                    paramProps,
                    ofx_str!(kOfxParamPropDefault),
                    0,
                    *default_value as f64,
                ))?;
                ofx_err(propSetDouble(
                    paramProps,
                    ofx_str!(kOfxParamPropMin),
                    0,
                    *range.start() as f64,
                ))?;
                ofx_err(propSetDouble(
                    paramProps,
                    ofx_str!(kOfxParamPropDisplayMin),
                    0,
                    *range.start() as f64,
                ))?;
                ofx_err(propSetDouble(
                    paramProps,
                    ofx_str!(kOfxParamPropMax),
                    0,
                    *range.end() as f64,
                ))?;
                ofx_err(propSetDouble(
                    paramProps,
                    ofx_str!(kOfxParamPropDisplayMax),
                    0,
                    *range.end() as f64,
                ))?;
            }
            SettingKind::Boolean { default_value } => {
                ofx_err(paramDefine(
                    param_set,
                    ofx_str!(kOfxParamTypeBoolean),
                    descriptor_id_cstr.as_ptr(),
                    &mut paramProps,
                ))?;
                ofx_err(propSetInt(
                    paramProps,
                    ofx_str!(kOfxParamPropDefault),
                    0,
                    *default_value as i32,
                ))?;
            }
            SettingKind::Group {
                children,
                default_value,
            } => {
                let group_name = descriptor_id_str.clone() + "_group";
                let group_name_cstr = CString::new(group_name).unwrap();
                ofx_err(paramDefine(
                    param_set,
                    ofx_str!(kOfxParamTypeGroup),
                    group_name_cstr.as_ptr(),
                    &mut paramProps,
                ))?;

                let mut checkboxProps: OfxPropertySetHandle = ptr::null_mut();
                ofx_err(paramDefine(
                    param_set,
                    ofx_str!(kOfxParamTypeBoolean),
                    descriptor_id_cstr.as_ptr(),
                    &mut checkboxProps,
                ))?;
                ofx_err(propSetString(
                    checkboxProps,
                    ofx_str!(kOfxPropLabel),
                    0,
                    static_cstr!("Enabled").as_ptr(),
                ))?;
                ofx_err(propSetInt(
                    checkboxProps,
                    ofx_str!(kOfxParamPropDefault),
                    0,
                    *default_value as i32,
                ))?;
                ofx_err(propSetString(
                    checkboxProps,
                    ofx_str!(kOfxParamPropParent),
                    0,
                    group_name_cstr.as_ptr(),
                ))?;

                ofx_err(propSetInt(
                    checkboxProps,
                    ofx_str!(kOfxParamPropAnimates),
                    0,
                    0,
                ))?;

                setup_params(
                    property_suite,
                    param_suite,
                    param_set,
                    children,
                    &group_name_cstr,
                )?;
            }
        }
        if !paramProps.is_null() {
            let descriptor_label_cstr = CString::new(descriptor.label).unwrap();
            ofx_err(propSetString(
                paramProps,
                ofx_str!(kOfxPropLabel),
                0,
                descriptor_label_cstr.as_ptr(),
            ))?;
            if let Some(description) = descriptor.description {
                let descriptor_desc_cstr = CString::new(description).unwrap();
                ofx_err(propSetString(
                    paramProps,
                    ofx_str!(kOfxParamPropHint),
                    0,
                    descriptor_desc_cstr.as_ptr(),
                ))?;
            }
            ofx_err(propSetString(
                paramProps,
                ofx_str!(kOfxParamPropParent),
                0,
                parent.as_ptr(),
            ))?;
        }
    }

    Ok(())
}

unsafe fn apply_params(
    param_suite: &OfxParameterSuiteV1,
    param_set: OfxParamSetHandle,
    time: f64,
    setting_descriptors: &[SettingDescriptor],
    dst: &mut NtscEffectFullSettings,
) -> OfxResult<()> {
    let paramGetHandle = param_suite.paramGetHandle.ok_or(OfxStat::kOfxStatFailed)?;
    let paramGetValueAtTime = param_suite
        .paramGetValueAtTime
        .ok_or(OfxStat::kOfxStatFailed)?;

    for descriptor in setting_descriptors {
        let descriptor_id_str = (descriptor.id.to_u32() as Option<u32>).unwrap().to_string();
        let descriptor_id_cstr = CString::new(descriptor_id_str.clone()).unwrap();

        let mut param: OfxParamHandle = ptr::null_mut();
        ofx_err(paramGetHandle(
            param_set,
            descriptor_id_cstr.as_ptr(),
            &mut param,
            ptr::null_mut(),
        ))?;

        match &descriptor.kind {
            SettingKind::Enumeration { options, .. } => {
                let mut selected_idx = 0;
                ofx_err(paramGetValueAtTime(param, time, &mut selected_idx))?;
                descriptor
                    .id
                    .set_field_enum(dst, options[selected_idx as usize].index)
                    .unwrap();
            }
            SettingKind::IntRange { .. } => {
                let mut int_value: i32 = 0;
                ofx_err(paramGetValueAtTime(param, time, &mut int_value))?;
                if let Some(field_ref) = descriptor.id.get_field_mut::<i32>(dst) {
                    *field_ref = int_value;
                } else if let Some(field_ref) = descriptor.id.get_field_mut::<u32>(dst) {
                    *field_ref = int_value as u32;
                }
            }
            SettingKind::FloatRange { .. } | SettingKind::Percentage { .. } => {
                let mut float_value: f64 = 0.0;
                ofx_err(paramGetValueAtTime(param, time, &mut float_value))?;
                let field_ref = descriptor
                    .id
                    .get_field_mut::<f32>(dst)
                    .ok_or(OfxStat::kOfxStatFailed)?;
                *field_ref = float_value as f32;
            }
            SettingKind::Boolean { .. } => {
                let mut bool_value: i32 = 0;
                ofx_err(paramGetValueAtTime(param, time, &mut bool_value))?;
                let field_ref = descriptor
                    .id
                    .get_field_mut::<bool>(dst)
                    .ok_or(OfxStat::kOfxStatFailed)?;
                *field_ref = bool_value != 0;
            }
            SettingKind::Group { children, .. } => {
                // The fetched handle refers to the group's checkbox
                let mut bool_value: i32 = 0;
                ofx_err(paramGetValueAtTime(param, time, &mut bool_value))?;
                let field_ref = descriptor
                    .id
                    .get_field_mut::<bool>(dst)
                    .ok_or(OfxStat::kOfxStatFailed)?;
                *field_ref = bool_value != 0;

                apply_params(param_suite, param_set, time, children, dst)?;
            }
        }
    }

    Ok(())
}

const SRGB_GAMMA_NAME: &CStr = static_cstr!("SrgbGammaCorrect");

unsafe fn action_describe_in_context(descriptor: OfxImageEffectHandle) -> OfxResult<()> {
    let data = shared_data.read().map_err(|_| OfxStat::kOfxStatFailed)?;
    let data = data.as_ref().ok_or(OfxStat::kOfxStatFailed)?;
    let clipDefine = data
        .image_effect_suite
        .clipDefine
        .ok_or(OfxStat::kOfxStatFailed)?;
    let getParamSet = data
        .image_effect_suite
        .getParamSet
        .ok_or(OfxStat::kOfxStatFailed)?;
    let param_suite = data.parameter_suite;
    let property_suite = data.property_suite;

    let paramDefine = param_suite.paramDefine.ok_or(OfxStat::kOfxStatFailed)?;
    let propSetString = property_suite
        .propSetString
        .ok_or(OfxStat::kOfxStatFailed)?;

    let mut props: OfxPropertySetHandle = ptr::null_mut();
    // TODO: all these functions return errors
    ofx_err(clipDefine(
        descriptor,
        static_cstr!("Output").as_ptr(),
        &mut props,
    ))?;
    if props.is_null() {
        return Err(OfxStat::kOfxStatFailed);
    }
    ofx_err(propSetString(
        props,
        ofx_str!(kOfxImageEffectPropSupportedComponents),
        0,
        ofx_str!(kOfxImageComponentRGBA),
    ))?;
    ofx_err(propSetString(
        props,
        ofx_str!(kOfxImageEffectPropSupportedComponents),
        1,
        ofx_str!(kOfxImageComponentRGB),
    ))?;

    clipDefine(descriptor, static_cstr!("Source").as_ptr(), &mut props);
    if props.is_null() {
        return Err(OfxStat::kOfxStatFailed);
    }
    ofx_err(propSetString(
        props,
        ofx_str!(kOfxImageEffectPropSupportedComponents),
        0,
        ofx_str!(kOfxImageComponentRGBA),
    ))?;
    ofx_err(propSetString(
        props,
        ofx_str!(kOfxImageEffectPropSupportedComponents),
        1,
        ofx_str!(kOfxImageComponentRGB),
    ))?;

    let mut param_set: OfxParamSetHandle = ptr::null_mut();
    ofx_err(getParamSet(descriptor, &mut param_set))?;

    setup_params(
        property_suite,
        param_suite,
        param_set,
        &data.settings_list.settings,
        static_cstr!(""),
    )?;

    let mut checkboxProps: OfxPropertySetHandle = ptr::null_mut();
    ofx_err(paramDefine(
        param_set,
        ofx_str!(kOfxParamTypeBoolean),
        SRGB_GAMMA_NAME.as_ptr(),
        &mut checkboxProps,
    ))?;
    ofx_err(propSetString(
        checkboxProps,
        ofx_str!(kOfxPropLabel),
        0,
        static_cstr!("Apply sRGB gamma").as_ptr(),
    ))?;

    Ok(())
}

unsafe fn action_get_regions_of_interest(
    descriptor: OfxImageEffectHandle,
    inArgs: OfxPropertySetHandle,
    outArgs: OfxPropertySetHandle,
) -> OfxResult<()> {
    let data = shared_data.read().map_err(|_| OfxStat::kOfxStatFailed)?;
    let data = data.as_ref().ok_or(OfxStat::kOfxStatFailed)?;
    let propGetDouble = data
        .property_suite
        .propGetDouble
        .ok_or(OfxStat::kOfxStatFailed)?;
    let propSetDoubleN = data
        .property_suite
        .propSetDoubleN
        .ok_or(OfxStat::kOfxStatFailed)?;
    let clipGetHandle = data
        .image_effect_suite
        .clipGetHandle
        .ok_or(OfxStat::kOfxStatFailed)?;
    let clipGetRegionOfDefinition = data
        .image_effect_suite
        .clipGetRegionOfDefinition
        .ok_or(OfxStat::kOfxStatFailed)?;

    let mut sourceClip: OfxImageClipHandle = ptr::null_mut();
    clipGetHandle(
        descriptor,
        static_cstr!("Source").as_ptr(),
        &mut sourceClip,
        ptr::null_mut(),
    );
    let mut sourceRoD = OfxRectD {
        x1: 0.0,
        x2: 0.0,
        y1: 0.0,
        y2: 0.0,
    };
    let mut time: OfxTime = 0.0;
    propGetDouble(inArgs, ofx_str!(kOfxPropTime), 0, &mut time);
    clipGetRegionOfDefinition(sourceClip, time, &mut sourceRoD);

    propSetDoubleN(
        outArgs,
        static_cstr!("OfxImageClipPropRoI_Source").as_ptr(),
        4,
        ptr::addr_of_mut!(sourceRoD) as *mut _,
    );

    Ok(())
}

unsafe fn action_get_clip_preferences(outArgs: OfxPropertySetHandle) -> OfxResult<()> {
    let data = shared_data.read().map_err(|_| OfxStat::kOfxStatFailed)?;
    let data = data.as_ref().ok_or(OfxStat::kOfxStatFailed)?;
    let propSetInt = data
        .property_suite
        .propSetInt
        .ok_or(OfxStat::kOfxStatFailed)?;
    let propSetString = data
        .property_suite
        .propSetString
        .ok_or(OfxStat::kOfxStatFailed)?;

    propSetInt(outArgs, ofx_str!(kOfxImageEffectFrameVarying), 0, 1);
    propSetString(
        outArgs,
        ofx_str!(kOfxImageEffectPropPreMultiplication),
        0,
        ofx_str!(kOfxImageOpaque),
    );

    Ok(())
}

unsafe fn update_controls_disabled(
    data: &SharedData,
    param_set: OfxParamSetHandle,
    setting_descriptors: &[SettingDescriptor],
    time: f64,
    enabled: bool,
) -> OfxResult<()> {
    let propSetInt = data
        .property_suite
        .propSetInt
        .ok_or(OfxStat::kOfxStatFailed)?;
    let paramGetHandle = data
        .parameter_suite
        .paramGetHandle
        .ok_or(OfxStat::kOfxStatFailed)?;
    let paramGetValueAtTime = data
        .parameter_suite
        .paramGetValueAtTime
        .ok_or(OfxStat::kOfxStatFailed)?;
    let paramGetPropertySet = data
        .parameter_suite
        .paramGetPropertySet
        .ok_or(OfxStat::kOfxStatFailed)?;

    for descriptor in setting_descriptors {
        let descriptor_id_str = (descriptor.id.to_u32() as Option<u32>).unwrap().to_string();
        let descriptor_id_cstr = CString::new(descriptor_id_str.clone()).unwrap();

        let mut param: OfxParamHandle = ptr::null_mut();
        ofx_err(paramGetHandle(
            param_set,
            descriptor_id_cstr.as_ptr(),
            &mut param,
            ptr::null_mut(),
        ))?;

        match &descriptor.kind {
            SettingKind::Group { children, .. } => {
                // The fetched handle refers to the group's checkbox
                let mut bool_value: i32 = 0;
                ofx_err(paramGetValueAtTime(param, time, &mut bool_value))?;
                let group_enabled = bool_value != 0;

                update_controls_disabled(data, param_set, children, time, group_enabled)?;
            }
            _ => {
                let mut prop_set: OfxPropertySetHandle = ptr::null_mut();
                ofx_err(paramGetPropertySet(param, &mut prop_set))?;
                propSetInt(prop_set, ofx_str!(kOfxParamPropEnabled), 0, enabled as i32);
            }
        }
    }

    Ok(())
}

unsafe fn action_instance_changed(
    descriptor: OfxImageEffectHandle,
    inArgs: OfxPropertySetHandle,
) -> OfxResult<()> {
    let data = shared_data.read().map_err(|_| OfxStat::kOfxStatFailed)?;
    let data = data.as_ref().ok_or(OfxStat::kOfxStatFailed)?;
    let getParamSet = data
        .image_effect_suite
        .getParamSet
        .ok_or(OfxStat::kOfxStatFailed)?;
    let propGetDouble = data
        .property_suite
        .propGetDouble
        .ok_or(OfxStat::kOfxStatFailed)?;

    let mut param_set: OfxParamSetHandle = ptr::null_mut();
    ofx_err(getParamSet(descriptor, &mut param_set))?;

    let mut time: f64 = 0.0;
    propGetDouble(inArgs, ofx_str!(kOfxPropTime), 0, &mut time);

    update_controls_disabled(data, param_set, &data.settings_list.settings, time, true)?;

    Ok(())
}

fn srgb_gamma(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * (value.powf(1.0 / 2.4)) - 0.055
    }
}

fn srgb_gamma_inv(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

struct OfxClipImage(OfxPropertySetHandle);

impl Drop for OfxClipImage {
    fn drop(&mut self) {
        let data = shared_data.read().unwrap();
        let data = data.as_ref().unwrap();
        let clipReleaseImage = data.image_effect_suite.clipReleaseImage.unwrap();

        unsafe { clipReleaseImage(self.0) };
    }
}

// Is this *really* safe? Probably not, but at least I'm not leaking memory
struct OfxAllocatedBuffer<T>(NonNull<[MaybeUninit<T>]>);

impl<T> OfxAllocatedBuffer<T> {
    pub fn alloc(elements: usize) -> OfxResult<Self> {
        if elements == 0 {
            return Err(OfxStat::kOfxStatErrValue);
        }
        let data = shared_data.read().map_err(|_| OfxStat::kOfxStatFailed)?;
        let data = data.as_ref().ok_or(OfxStat::kOfxStatFailed)?;
        let memoryAlloc = data
            .memory_suite
            .memoryAlloc
            .ok_or(OfxStat::kOfxStatFailed)?;

        let mut buf = ptr::null_mut();
        unsafe {
            ofx_err(memoryAlloc(
                ptr::null_mut(),
                elements * size_of::<T>(),
                &mut buf,
            ))?
        };

        let buf_slice = unsafe { slice::from_raw_parts_mut(buf as *mut MaybeUninit<T>, elements) };
        let buf_ptr = NonNull::new(buf_slice).ok_or(OfxStat::kOfxStatErrMemory)?;

        Ok(Self(buf_ptr))
    }

    #[allow(dead_code)]
    unsafe fn borrow_assume_init(&self) -> &[T] {
        mem::transmute(self.0.as_ref())
    }

    unsafe fn borrow_assume_init_mut(&mut self) -> &mut [T] {
        mem::transmute(self.0.as_mut())
    }
}

impl<T> Drop for OfxAllocatedBuffer<T> {
    fn drop(&mut self) {
        let data = shared_data.read().unwrap();
        let data = data.as_ref().unwrap();
        let memoryFree = data.memory_suite.memoryFree.unwrap();
        let ptr = self.0;

        unsafe { memoryFree(ptr.as_ptr() as *mut _) };
    }
}

impl<T> Borrow<[MaybeUninit<T>]> for OfxAllocatedBuffer<T> {
    fn borrow(&self) -> &[MaybeUninit<T>] {
        unsafe { self.0.as_ref() }
    }
}

impl<T> BorrowMut<[MaybeUninit<T>]> for OfxAllocatedBuffer<T> {
    fn borrow_mut(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe { self.0.as_mut() }
    }
}

trait Normalize {
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

struct RowInfo {
    row_lshift: usize,
    row_offset: usize,
    num_rows: usize,
}

fn getRowInfo(cur_field: YiqField, src_height: usize) -> RowInfo {
    // Always be sure to render at least 1 row
    if src_height == 1 {
        return RowInfo {
            row_lshift: 0,
            row_offset: 0,
            num_rows: 1,
        };
    }

    // We write into the destination array differently depending on whether we're using the upper field, lower
    // field, or both. row_lshift determines whether we left-shift the source row index (doubling it). When we use
    // only one of the fields, the source row index needs to be double the destination row index so we take every
    // other row. When we use both fields, we just use the source row index as-is.
    // The row_offset determines whether we skip the first row (when using the lower field).
    let row_lshift: usize = match cur_field {
        YiqField::Upper => 1,
        YiqField::Lower => 1,
        YiqField::Both => 0,
    };

    let row_offset: usize = match (src_height & 1, cur_field) {
        (0, YiqField::Upper) => 1,
        (1, YiqField::Upper) => 0,
        (0, YiqField::Lower) => 0,
        (1, YiqField::Lower) => 1,
        (_, YiqField::Both) => 0,
        _ => unreachable!(),
    };

    // On an image with an odd input height, we do ceiling division if we render upper-field-first
    // (take an image 3 pixels tall. it goes render, skip, render--that's 2 renders) but floor division if we
    // render lower-field-first (skip, render, skip--only 1 render).
    let numRows = match cur_field {
        YiqField::Upper => (src_height + 1) / 2,
        YiqField::Lower => src_height / 2,
        YiqField::Both => src_height,
    };

    RowInfo {
        row_lshift,
        row_offset,
        num_rows: numRows,
    }
}

unsafe fn pixel_processing<S: Normalize + Sized, D: Normalize + Sized>(
    srcPtr: *mut c_void,
    dstPtr: *mut c_void,
    srcRowBytes: i32,
    dstRowBytes: i32,
    srcBounds: OfxRectI,
    dstBounds: OfxRectI,
    num_source_components: usize,
    num_output_components: usize,
    effect: &NtscEffect,
    frame_num: usize,
    apply_srgb_gamma: bool,
) -> OfxResult<()> {
    let dstWidth = (dstBounds.x2 - dstBounds.x1) as usize;
    let dstHeight = (dstBounds.y2 - dstBounds.y1) as usize;
    let srcWidth = (srcBounds.x2 - srcBounds.x1) as usize;
    let srcHeight = (srcBounds.y2 - srcBounds.y1) as usize;

    let cur_field = match effect.use_field {
        UseField::Alternating => {
            if frame_num & 1 == 0 {
                YiqField::Lower
            } else {
                YiqField::Upper
            }
        }
        UseField::Upper => YiqField::Upper,
        UseField::Lower => YiqField::Lower,
        UseField::Both => YiqField::Both,
    };

    let RowInfo {
        row_lshift,
        row_offset,
        num_rows: numRows,
    } = getRowInfo(cur_field, srcHeight);

    // Pixels per YIQ plane
    let numPlanePixels = srcWidth * numRows;

    let mut ntsc_buf = OfxAllocatedBuffer::<f32>::alloc(numPlanePixels * 3)?;

    {
        let yiq_buf: &mut [MaybeUninit<f32>] = ntsc_buf.borrow_mut();
        let (y, iq) = yiq_buf.split_at_mut(numPlanePixels);
        let (i, q) = iq.split_at_mut(numPlanePixels);

        // Convert the source pixel data into YIQ normalized data
        for yi in 0..numRows {
            let row_idx = (yi << row_lshift) + row_offset;
            // We need to offset the row pointer in bytes, which is why we leave it as a *mut c_void here
            let rowPtr = srcPtr.offset((srcRowBytes * row_idx as i32) as isize);
            for x in 0..srcWidth {
                // Now that we have the row pointer, we offset by the actual datatype to get the pixel
                let pixPtr = (rowPtr as *mut S).add(x * num_source_components);
                let r = pixPtr.read().to_norm();
                let g = pixPtr.offset(1).read().to_norm();
                let b = pixPtr.offset(2).read().to_norm();
                let (pix_y, pix_i, pix_q) = if apply_srgb_gamma {
                    rgb_to_yiq(srgb_gamma(r), srgb_gamma(g), srgb_gamma(b))
                } else {
                    rgb_to_yiq(r, g, b)
                };

                y[((numRows - 1 - yi) * srcWidth) + x].write(pix_y);
                i[((numRows - 1 - yi) * srcWidth) + x].write(pix_i);
                q[((numRows - 1 - yi) * srcWidth) + x].write(pix_q);
            }
        }
    }

    // We've now initialized our buffer; construct a YiqView from it
    let yiq_buf: &mut [f32] = ntsc_buf.borrow_assume_init_mut();
    let (y, iq) = yiq_buf.split_at_mut(numPlanePixels);
    let (i, q) = iq.split_at_mut(numPlanePixels);

    let mut yiq_view = YiqView {
        y,
        i,
        q,
        dimensions: (srcWidth, srcHeight),
        field: cur_field,
    };

    effect.apply_effect_to_yiq(&mut yiq_view, frame_num);

    let (y, i, q) = (yiq_view.y, yiq_view.i, yiq_view.q);

    // Convert back and copy into the destination buffer
    for yi in 0..dstHeight {
        // We need to offset the row pointer in bytes, which is why we leave it as a *mut c_void here
        let dstRowPtr = dstPtr.offset((dstRowBytes * yi as i32) as isize);
        let src_y = yi as i32 + (dstBounds.y1 - srcBounds.y1);
        let interp_row = cur_field != YiqField::Both
            && (yi & 1) != row_offset
            && src_y > 0
            && src_y < (srcHeight - 1) as i32;

        for x in 0..dstWidth {
            // Now that we have the row pointer, we offset by the actual datatype to get the pixel
            let pixPtr = (dstRowPtr as *mut D).add(x * num_output_components);
            let src_x = x as i32 + (dstBounds.x1 - srcBounds.x1);
            if src_x < 0 || src_x >= srcWidth as i32 || src_y < 0 || src_y >= srcHeight as i32 {
                pixPtr.write(D::from_norm(0.0));
                pixPtr.offset(1).write(D::from_norm(0.0));
                pixPtr.offset(2).write(D::from_norm(0.0));
                if num_output_components == 4 {
                    pixPtr.offset(3).write(D::from_norm(0.0));
                }
                continue;
            }

            let (pix_y, pix_i, pix_q) = if interp_row {
                // This row was not processed this frame. Interpolate from the rows above and below it.
                let row_idx_bottom = (srcHeight as i32 - 1 - src_y + 1) as usize >> row_lshift;
                let row_idx_top = (srcHeight as i32 - 1 - src_y - 1) as usize >> row_lshift;
                let idx_top = (row_idx_top * srcWidth) + src_x as usize;
                let idx_bottom = (row_idx_bottom * srcWidth) + src_x as usize;
                (
                    (y[idx_top] + y[idx_bottom]) * 0.5,
                    (i[idx_top] + i[idx_bottom]) * 0.5,
                    (q[idx_top] + q[idx_bottom]) * 0.5,
                )
            } else {
                let row_idx =
                    ((srcHeight as i32 - 1 - src_y) as usize >> row_lshift).min(numRows - 1);
                let idx = (row_idx * srcWidth) + src_x as usize;
                (y[idx], i[idx], q[idx])
            };

            let (mut r, mut g, mut b) = yiq_to_rgb(pix_y, pix_i, pix_q);
            if apply_srgb_gamma {
                r = srgb_gamma_inv(r);
                g = srgb_gamma_inv(g);
                b = srgb_gamma_inv(b);
            }
            pixPtr.write(D::from_norm(r));
            pixPtr.offset(1).write(D::from_norm(g));
            pixPtr.offset(2).write(D::from_norm(b));
            if num_output_components == 4 {
                pixPtr.offset(3).write(D::from_norm(1.0));
            }
        }
    }

    Ok(())
}

const PIXEL_DEPTH_BYTE: Result<&'static CStr, FromBytesWithNulError> =
    CStr::from_bytes_with_nul(kOfxBitDepthByte);
const PIXEL_DEPTH_SHORT: Result<&'static CStr, FromBytesWithNulError> =
    CStr::from_bytes_with_nul(kOfxBitDepthShort);
const PIXEL_DEPTH_FLOAT: Result<&'static CStr, FromBytesWithNulError> =
    CStr::from_bytes_with_nul(kOfxBitDepthFloat);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SupportedPixelDepth {
    Byte,
    Short,
    Float,
}

impl TryFrom<&CStr> for SupportedPixelDepth {
    type Error = OfxStatus;

    fn try_from(value: &CStr) -> OfxResult<Self> {
        if value == PIXEL_DEPTH_BYTE.unwrap() {
            Ok(Self::Byte)
        } else if value == PIXEL_DEPTH_SHORT.unwrap() {
            Ok(Self::Short)
        } else if value == PIXEL_DEPTH_FLOAT.unwrap() {
            Ok(Self::Float)
        } else {
            Err(OfxStat::kOfxStatErrUnsupported)
        }
    }
}

unsafe fn action_render(
    descriptor: OfxImageEffectHandle,
    inArgs: OfxPropertySetHandle,
) -> OfxResult<()> {
    let data = shared_data.read().map_err(|_| OfxStat::kOfxStatFailed)?;
    let data = data.as_ref().ok_or(OfxStat::kOfxStatFailed)?;
    let propGetString = data
        .property_suite
        .propGetString
        .ok_or(OfxStat::kOfxStatFailed)?;
    let propGetDouble = data
        .property_suite
        .propGetDouble
        .ok_or(OfxStat::kOfxStatFailed)?;
    let propGetInt = data
        .property_suite
        .propGetInt
        .ok_or(OfxStat::kOfxStatFailed)?;
    let propGetIntN = data
        .property_suite
        .propGetIntN
        .ok_or(OfxStat::kOfxStatFailed)?;
    let propGetPointer = data
        .property_suite
        .propGetPointer
        .ok_or(OfxStat::kOfxStatFailed)?;
    let clipGetHandle = data
        .image_effect_suite
        .clipGetHandle
        .ok_or(OfxStat::kOfxStatFailed)?;
    let clipGetImage = data
        .image_effect_suite
        .clipGetImage
        .ok_or(OfxStat::kOfxStatFailed)?;
    let getParamSet = data
        .image_effect_suite
        .getParamSet
        .ok_or(OfxStat::kOfxStatFailed)?;
    let paramGetHandle = data
        .parameter_suite
        .paramGetHandle
        .ok_or(OfxStat::kOfxStatFailed)?;
    let paramGetValueAtTime = data
        .parameter_suite
        .paramGetValueAtTime
        .ok_or(OfxStat::kOfxStatFailed)?;

    let mut time: OfxTime = 0.0;
    let mut renderWindow = OfxRectI {
        x1: 0,
        x2: 0,
        y1: 0,
        y2: 0,
    };

    propGetDouble(inArgs, ofx_str!(kOfxPropTime), 0, &mut time);
    // I'm sure nothing bad will happen here as a result of propGetIntN writing past the pointer it was given
    propGetIntN(
        inArgs,
        ofx_str!(kOfxImageEffectPropRenderWindow),
        4,
        ptr::addr_of_mut!(renderWindow) as *mut _,
    );

    let mut outputClip: OfxImageClipHandle = ptr::null_mut();
    clipGetHandle(
        descriptor,
        static_cstr!("Output").as_ptr(),
        &mut outputClip,
        ptr::null_mut(),
    );
    let mut sourceClip: OfxImageClipHandle = ptr::null_mut();
    clipGetHandle(
        descriptor,
        static_cstr!("Source").as_ptr(),
        &mut sourceClip,
        ptr::null_mut(),
    );

    let mut outputImg: OfxPropertySetHandle = ptr::null_mut();
    clipGetImage(outputClip, time, ptr::null_mut(), &mut outputImg);
    let mut sourceImg: OfxPropertySetHandle = ptr::null_mut();
    clipGetImage(sourceClip, time, ptr::null_mut(), &mut sourceImg);
    let outputImg = OfxClipImage(outputImg);
    let sourceImg = OfxClipImage(sourceImg);

    let num_source_components = {
        let mut cstr: *mut c_char = ptr::null_mut();
        propGetString(
            sourceImg.0,
            ofx_str!(kOfxImageEffectPropComponents),
            0,
            &mut cstr,
        );
        let components = CStr::from_ptr(cstr);
        if components == CStr::from_ptr(ofx_str!(kOfxImageComponentRGBA)) {
            4
        } else if components == CStr::from_ptr(ofx_str!(kOfxImageComponentRGB)) {
            3
        } else {
            return Err(OfxStat::kOfxStatErrFormat);
        }
    };

    let num_output_components = {
        let mut cstr: *mut c_char = ptr::null_mut();
        propGetString(
            outputImg.0,
            ofx_str!(kOfxImageEffectPropComponents),
            0,
            &mut cstr,
        );
        let components = CStr::from_ptr(cstr);
        if components == CStr::from_ptr(ofx_str!(kOfxImageComponentRGBA)) {
            4
        } else if components == CStr::from_ptr(ofx_str!(kOfxImageComponentRGB)) {
            3
        } else {
            return Err(OfxStat::kOfxStatErrFormat);
        }
    };

    let source_pixel_depth = {
        let mut cstr: *mut c_char = ptr::null_mut();
        propGetString(
            sourceImg.0,
            ofx_str!(kOfxImageEffectPropPixelDepth),
            0,
            &mut cstr,
        );
        SupportedPixelDepth::try_from(CStr::from_ptr(cstr))
    }?;

    let output_pixel_depth = {
        let mut cstr: *mut c_char = ptr::null_mut();
        propGetString(
            outputImg.0,
            ofx_str!(kOfxImageEffectPropPixelDepth),
            0,
            &mut cstr,
        );
        SupportedPixelDepth::try_from(CStr::from_ptr(cstr))
    }?;

    let mut dstRowBytes: c_int = 0;
    let mut dstBounds = OfxRectI {
        x1: 0,
        x2: 0,
        y1: 0,
        y2: 0,
    };
    let mut dstPtr: *mut c_void = ptr::null_mut();
    propGetInt(
        outputImg.0,
        ofx_str!(kOfxImagePropRowBytes),
        0,
        &mut dstRowBytes,
    );
    propGetIntN(
        outputImg.0,
        ofx_str!(kOfxImagePropBounds),
        4,
        ptr::addr_of_mut!(dstBounds) as *mut _,
    );
    propGetPointer(outputImg.0, ofx_str!(kOfxImagePropData), 0, &mut dstPtr);

    let mut srcRowBytes: c_int = 0;
    let mut srcBounds = OfxRectI {
        x1: 0,
        x2: 0,
        y1: 0,
        y2: 0,
    };
    let mut srcPtr: *mut c_void = ptr::null_mut();
    propGetInt(
        sourceImg.0,
        ofx_str!(kOfxImagePropRowBytes),
        0,
        &mut srcRowBytes,
    );
    propGetIntN(
        sourceImg.0,
        ofx_str!(kOfxImagePropBounds),
        4,
        ptr::addr_of_mut!(srcBounds) as *mut _,
    );
    propGetPointer(sourceImg.0, ofx_str!(kOfxImagePropData), 0, &mut srcPtr);

    let mut param_set: OfxParamSetHandle = ptr::null_mut();
    ofx_err(getParamSet(descriptor, &mut param_set))?;
    let mut out_settings: NtscEffectFullSettings = NtscEffectFullSettings::default();
    apply_params(
        data.parameter_suite,
        param_set,
        time,
        &data.settings_list.settings,
        &mut out_settings,
    )?;

    let mut srgb_param: OfxParamHandle = ptr::null_mut();
    ofx_err(paramGetHandle(
        param_set,
        SRGB_GAMMA_NAME.as_ptr(),
        &mut srgb_param,
        ptr::null_mut(),
    ))?;
    let mut srgb_bool_value: i32 = 0;
    ofx_err(paramGetValueAtTime(srgb_param, time, &mut srgb_bool_value))?;
    let apply_srgb_gamma = srgb_bool_value != 0;

    let effect: NtscEffect = out_settings.into();

    let frame_num = time as usize;

    if source_pixel_depth != output_pixel_depth {
        return Err(OfxStat::kOfxStatErrUnsupported);
    }

    match output_pixel_depth {
        SupportedPixelDepth::Byte => {
            pixel_processing::<u8, u8>(
                srcPtr,
                dstPtr,
                srcRowBytes,
                dstRowBytes,
                srcBounds,
                dstBounds,
                num_source_components,
                num_output_components,
                &effect,
                frame_num,
                apply_srgb_gamma,
            )?;
        }
        SupportedPixelDepth::Short => {
            pixel_processing::<u16, u16>(
                srcPtr,
                dstPtr,
                srcRowBytes,
                dstRowBytes,
                srcBounds,
                dstBounds,
                num_source_components,
                num_output_components,
                &effect,
                frame_num,
                apply_srgb_gamma,
            )?;
        }
        SupportedPixelDepth::Float => {
            pixel_processing::<f32, f32>(
                srcPtr,
                dstPtr,
                srcRowBytes,
                dstRowBytes,
                srcBounds,
                dstBounds,
                num_source_components,
                num_output_components,
                &effect,
                frame_num,
                apply_srgb_gamma,
            )?;
        }
    }

    Ok(())
}

const OFX_ACTION_LOAD: Result<&'static CStr, FromBytesWithNulError> =
    CStr::from_bytes_with_nul(kOfxActionLoad);
const OFX_ACTION_DESCRIBE: Result<&'static CStr, FromBytesWithNulError> =
    CStr::from_bytes_with_nul(kOfxActionDescribe);
const OFX_ACTION_DESCRIBE_IN_CONTEXT: Result<&'static CStr, FromBytesWithNulError> =
    CStr::from_bytes_with_nul(kOfxImageEffectActionDescribeInContext);
const OFX_ACTION_GET_REGIONS_OF_INTEREST: Result<&'static CStr, FromBytesWithNulError> =
    CStr::from_bytes_with_nul(kOfxImageEffectActionGetRegionsOfInterest);
const OFX_ACTION_GET_CLIP_PREFERENCES: Result<&'static CStr, FromBytesWithNulError> =
    CStr::from_bytes_with_nul(kOfxImageEffectActionGetClipPreferences);
const OFX_ACTION_INSTANCE_CHANGED: Result<&'static CStr, FromBytesWithNulError> =
    CStr::from_bytes_with_nul(kOfxActionInstanceChanged);
const OFX_ACTION_RENDER: Result<&'static CStr, FromBytesWithNulError> =
    CStr::from_bytes_with_nul(kOfxImageEffectActionRender);

const OFX_ACTION_CREATE_INSTANCE: Result<&'static CStr, FromBytesWithNulError> =
    CStr::from_bytes_with_nul(kOfxActionCreateInstance);
const OFX_ACTION_DESTROY_INSTANCE: Result<&'static CStr, FromBytesWithNulError> =
    CStr::from_bytes_with_nul(kOfxActionDestroyInstance);

unsafe extern "C" fn main_entry(
    action: *const c_char,
    handle: *const c_void,
    inArgs: OfxPropertySetHandle,
    outArgs: OfxPropertySetHandle,
) -> OfxStatus {
    let effect = handle as OfxImageEffectHandle;
    let action = CStr::from_ptr(action);

    // Needed so Resolve doesn't swallow the panic info
    std::panic::set_hook(Box::new(|info| {
        println!("{:?}", info);
    }));

    let return_status = if action == OFX_ACTION_LOAD.unwrap() {
        action_load()
    } else if action == OFX_ACTION_DESCRIBE.unwrap() {
        action_describe(effect)
    } else if action == OFX_ACTION_DESCRIBE_IN_CONTEXT.unwrap() {
        action_describe_in_context(effect)
    } else if action == OFX_ACTION_GET_REGIONS_OF_INTEREST.unwrap() {
        action_get_regions_of_interest(effect, inArgs, outArgs)
    } else if action == OFX_ACTION_GET_CLIP_PREFERENCES.unwrap() {
        action_get_clip_preferences(outArgs)
    } else if action == OFX_ACTION_INSTANCE_CHANGED.unwrap() {
        action_instance_changed(effect, inArgs)
    } else if action == OFX_ACTION_RENDER.unwrap() {
        action_render(effect, inArgs)
    } else if action == OFX_ACTION_CREATE_INSTANCE.unwrap()
        || action == OFX_ACTION_DESTROY_INSTANCE.unwrap()
    {
        // We need to handle these actions (even if it's just a no-op) for DaVinci resolve to properly load our plugin
        // If not handled, it'll load the plugin but will never show the controls or actually render anything
        // TODO: try to preallocate buffer here
        Ok(())
    } else {
        OfxResult::Err(OfxStat::kOfxStatReplyDefault)
    };

    match return_status {
        Ok(_) => kOfxStatOK as i32,
        Err(e) => e,
    }
}

#[no_mangle]
pub extern "C" fn OfxGetNumberOfPlugins() -> c_int {
    1
}

#[no_mangle]
pub extern "C" fn OfxGetPlugin(nth: c_int) -> *const OfxPlugin {
    if nth != 0 {
        return ptr::null();
    }

    // Safety: We're synchronizing access to the OfxPlugin using a OnceLock, and I *think* the reason it has to be
    // `static mut` is that some fields in it are raw pointers, which could theoretically be messed with in an
    // unsynchronized manner, which OFX hosts probably shouldn't do?
    #[allow(unused_unsafe)]
    let plugin_info: &'static OfxPlugin = unsafe {
        PLUGIN_INFO.get_or_init(|| {
            OfxPlugin {
                // I think this cast is OK?
                pluginApi: ofx_str!(kOfxImageEffectPluginApi),
                apiVersion: 1,
                pluginIdentifier: static_cstr!("wtf.vala:NtscRs").as_ptr(),
                pluginVersionMajor: 1,
                pluginVersionMinor: 0,
                setHost: Some(set_host_info),
                mainEntry: Some(main_entry),
            }
        })
    };
    plugin_info as *const _
}
