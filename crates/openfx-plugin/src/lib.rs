#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use core::slice;
use std::{
    convert::identity,
    ffi::{c_char, c_int, c_void, CStr, CString},
    mem::{self, MaybeUninit},
    ptr::{self, NonNull},
    sync::{OnceLock, RwLock},
};

use allocator_api2::{
    alloc::{AllocError, Allocator, Layout},
    boxed::Box as AllocBox,
};

use ntscrs::{
    ntsc::NtscEffect,
    yiq_fielding::{yiq_to_rgb, YiqField, YiqView},
};
use ntscrs::{
    settings::{NtscEffectFullSettings, SettingDescriptor, SettingKind, SettingsList},
    yiq_fielding::{Normalize, PixelFormat, Rgb16, Rgb32f, Rgb8, Rgbx16, Rgbx32f, Rgbx8},
};
use ntscrs::{
    yiq_fielding::{BlitInfo, Rect},
    ToPrimitive,
};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// https://stackoverflow.com/questions/53611161/how-do-i-expose-a-compile-time-generated-static-c-string-through-ffi
macro_rules! static_cstr {
    ($l:expr) => {
        unsafe { ::std::ffi::CStr::from_bytes_with_nul_unchecked(concat!($l, "\0").as_bytes()) }
    };
}

// SAFETY: The host promises not to mess with the raw string pointers in this struct
unsafe impl Send for OfxPlugin {}
unsafe impl Sync for OfxPlugin {}

static PLUGIN_INFO: OnceLock<OfxPlugin> = OnceLock::new();
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
            kOfxPropertySuite.as_ptr(),
            1,
        ) as *const OfxPropertySuiteV1;
        let image_effect_suite = (host_info.fetchSuite)(
            host_info.host as *const _ as *mut _,
            kOfxImageEffectSuite.as_ptr(),
            1,
        ) as *const OfxImageEffectSuiteV1;
        let memory_suite = (host_info.fetchSuite)(
            host_info.host as *const _ as *mut _,
            kOfxMemorySuite.as_ptr(),
            1,
        ) as *const OfxMemorySuiteV1;
        let parameter_suite = (host_info.fetchSuite)(
            host_info.host as *const _ as *mut _,
            kOfxParameterSuite.as_ptr(),
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
        kOfxImageEffectPropSupportsMultipleClipDepths.as_ptr(),
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
        kOfxPropLabel.as_ptr(),
        0,
        static_cstr!("NTSC-rs").as_ptr(),
    );

    propSetString(
        effectProps,
        kOfxImageEffectPluginPropGrouping.as_ptr(),
        0,
        static_cstr!("Filter").as_ptr(),
    );

    propSetString(
        effectProps,
        kOfxImageEffectPropSupportedContexts.as_ptr(),
        0,
        kOfxImageEffectContextFilter.as_ptr(),
    );
    // TODO needed for resolve support(?)
    propSetString(
        effectProps,
        kOfxImageEffectPropSupportedContexts.as_ptr(),
        1,
        kOfxImageEffectContextGeneral.as_ptr(),
    );

    propSetString(
        effectProps,
        kOfxImageEffectPropSupportedPixelDepths.as_ptr(),
        0,
        kOfxBitDepthFloat.as_ptr(),
    );
    propSetString(
        effectProps,
        kOfxImageEffectPropSupportedPixelDepths.as_ptr(),
        1,
        kOfxBitDepthShort.as_ptr(),
    );
    propSetString(
        effectProps,
        kOfxImageEffectPropSupportedPixelDepths.as_ptr(),
        2,
        kOfxBitDepthByte.as_ptr(),
    );

    // TODO: is this wrong?
    propSetString(
        effectProps,
        kOfxImageEffectPluginRenderThreadSafety.as_ptr(),
        0,
        kOfxImageEffectRenderFullySafe.as_ptr(),
    );
    // We'll manage threading ourselves
    propSetInt(
        effectProps,
        kOfxImageEffectPluginPropHostFrameThreading.as_ptr(),
        0,
        0,
    );
    // We need to operate on the whole image at once
    propSetInt(effectProps, kOfxImageEffectPropSupportsTiles.as_ptr(), 0, 0);

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
                    kOfxParamTypeChoice.as_ptr(),
                    descriptor_id_cstr.as_ptr(),
                    &mut paramProps,
                ))?;
                let mut default_idx: usize = 0;
                for (i, menu_item) in options.iter().enumerate() {
                    let item_label_cstr = CString::new(menu_item.label).unwrap();
                    ofx_err(propSetString(
                        paramProps,
                        kOfxParamPropChoiceOption.as_ptr(),
                        i as i32,
                        item_label_cstr.as_ptr(),
                    ))?;

                    if menu_item.index == *default_value {
                        default_idx = i;
                    }
                }
                ofx_err(propSetInt(
                    paramProps,
                    kOfxParamPropDefault.as_ptr(),
                    0,
                    default_idx as i32,
                ))?;
            }
            SettingKind::Percentage { default_value, .. } => {
                ofx_err(paramDefine(
                    param_set,
                    kOfxParamTypeDouble.as_ptr(),
                    descriptor_id_cstr.as_ptr(),
                    &mut paramProps,
                ))?;
                ofx_err(propSetString(
                    paramProps,
                    kOfxParamPropDoubleType.as_ptr(),
                    0,
                    kOfxParamDoubleTypeScale.as_ptr(),
                ))?;
                ofx_err(propSetDouble(
                    paramProps,
                    kOfxParamPropDefault.as_ptr(),
                    0,
                    *default_value as f64,
                ))?;
                ofx_err(propSetDouble(paramProps, kOfxParamPropMin.as_ptr(), 0, 0.0))?;
                ofx_err(propSetDouble(
                    paramProps,
                    kOfxParamPropDisplayMin.as_ptr(),
                    0,
                    0.0,
                ))?;
                ofx_err(propSetDouble(paramProps, kOfxParamPropMax.as_ptr(), 0, 1.0))?;
                ofx_err(propSetDouble(
                    paramProps,
                    kOfxParamPropDisplayMax.as_ptr(),
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
                    kOfxParamTypeInteger.as_ptr(),
                    descriptor_id_cstr.as_ptr(),
                    &mut paramProps,
                ))?;
                ofx_err(propSetInt(
                    paramProps,
                    kOfxParamPropDefault.as_ptr(),
                    0,
                    *default_value,
                ))?;
                ofx_err(propSetInt(
                    paramProps,
                    kOfxParamPropMin.as_ptr(),
                    0,
                    *range.start(),
                ))?;
                ofx_err(propSetInt(
                    paramProps,
                    kOfxParamPropDisplayMin.as_ptr(),
                    0,
                    *range.start(),
                ))?;
                ofx_err(propSetInt(
                    paramProps,
                    kOfxParamPropMax.as_ptr(),
                    0,
                    *range.end(),
                ))?;
                ofx_err(propSetInt(
                    paramProps,
                    kOfxParamPropDisplayMax.as_ptr(),
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
                    kOfxParamTypeDouble.as_ptr(),
                    descriptor_id_cstr.as_ptr(),
                    &mut paramProps,
                ))?;
                ofx_err(propSetDouble(
                    paramProps,
                    kOfxParamPropDefault.as_ptr(),
                    0,
                    *default_value as f64,
                ))?;
                ofx_err(propSetDouble(
                    paramProps,
                    kOfxParamPropMin.as_ptr(),
                    0,
                    *range.start() as f64,
                ))?;
                ofx_err(propSetDouble(
                    paramProps,
                    kOfxParamPropDisplayMin.as_ptr(),
                    0,
                    *range.start() as f64,
                ))?;
                ofx_err(propSetDouble(
                    paramProps,
                    kOfxParamPropMax.as_ptr(),
                    0,
                    *range.end() as f64,
                ))?;
                ofx_err(propSetDouble(
                    paramProps,
                    kOfxParamPropDisplayMax.as_ptr(),
                    0,
                    *range.end() as f64,
                ))?;
            }
            SettingKind::Boolean { default_value } => {
                ofx_err(paramDefine(
                    param_set,
                    kOfxParamTypeBoolean.as_ptr(),
                    descriptor_id_cstr.as_ptr(),
                    &mut paramProps,
                ))?;
                ofx_err(propSetInt(
                    paramProps,
                    kOfxParamPropDefault.as_ptr(),
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
                    kOfxParamTypeGroup.as_ptr(),
                    group_name_cstr.as_ptr(),
                    &mut paramProps,
                ))?;

                let mut checkboxProps: OfxPropertySetHandle = ptr::null_mut();
                ofx_err(paramDefine(
                    param_set,
                    kOfxParamTypeBoolean.as_ptr(),
                    descriptor_id_cstr.as_ptr(),
                    &mut checkboxProps,
                ))?;
                ofx_err(propSetString(
                    checkboxProps,
                    kOfxPropLabel.as_ptr(),
                    0,
                    static_cstr!("Enabled").as_ptr(),
                ))?;
                ofx_err(propSetInt(
                    checkboxProps,
                    kOfxParamPropDefault.as_ptr(),
                    0,
                    *default_value as i32,
                ))?;
                ofx_err(propSetString(
                    checkboxProps,
                    kOfxParamPropParent.as_ptr(),
                    0,
                    group_name_cstr.as_ptr(),
                ))?;

                ofx_err(propSetInt(
                    checkboxProps,
                    kOfxParamPropAnimates.as_ptr(),
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
                kOfxPropLabel.as_ptr(),
                0,
                descriptor_label_cstr.as_ptr(),
            ))?;
            if let Some(description) = descriptor.description {
                let descriptor_desc_cstr = CString::new(description).unwrap();
                ofx_err(propSetString(
                    paramProps,
                    kOfxParamPropHint.as_ptr(),
                    0,
                    descriptor_desc_cstr.as_ptr(),
                ))?;
            }
            ofx_err(propSetString(
                paramProps,
                kOfxParamPropParent.as_ptr(),
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
        kOfxImageEffectPropSupportedComponents.as_ptr(),
        0,
        kOfxImageComponentRGBA.as_ptr(),
    ))?;
    ofx_err(propSetString(
        props,
        kOfxImageEffectPropSupportedComponents.as_ptr(),
        1,
        kOfxImageComponentRGB.as_ptr(),
    ))?;

    clipDefine(descriptor, static_cstr!("Source").as_ptr(), &mut props);
    if props.is_null() {
        return Err(OfxStat::kOfxStatFailed);
    }
    ofx_err(propSetString(
        props,
        kOfxImageEffectPropSupportedComponents.as_ptr(),
        0,
        kOfxImageComponentRGBA.as_ptr(),
    ))?;
    ofx_err(propSetString(
        props,
        kOfxImageEffectPropSupportedComponents.as_ptr(),
        1,
        kOfxImageComponentRGB.as_ptr(),
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
        kOfxParamTypeBoolean.as_ptr(),
        SRGB_GAMMA_NAME.as_ptr(),
        &mut checkboxProps,
    ))?;
    ofx_err(propSetString(
        checkboxProps,
        kOfxPropLabel.as_ptr(),
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
    propGetDouble(inArgs, kOfxPropTime.as_ptr(), 0, &mut time);
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

    propSetInt(outArgs, kOfxImageEffectFrameVarying.as_ptr(), 0, 1);
    propSetString(
        outArgs,
        kOfxImageEffectPropPreMultiplication.as_ptr(),
        0,
        kOfxImageOpaque.as_ptr(),
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

        if let SettingKind::Group { children, .. } = &descriptor.kind {
            // The fetched handle refers to the group's checkbox
            let mut bool_value: i32 = 0;
            ofx_err(paramGetValueAtTime(param, time, &mut bool_value))?;
            let group_enabled = bool_value != 0;

            update_controls_disabled(data, param_set, children, time, group_enabled && enabled)?;
        }
        let mut prop_set: OfxPropertySetHandle = ptr::null_mut();
        ofx_err(paramGetPropertySet(param, &mut prop_set))?;
        propSetInt(prop_set, kOfxParamPropEnabled.as_ptr(), 0, enabled as i32);
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
    propGetDouble(inArgs, kOfxPropTime.as_ptr(), 0, &mut time);

    update_controls_disabled(data, param_set, &data.settings_list.settings, time, true)?;

    Ok(())
}

#[inline(always)]
fn srgb_gamma_single(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * (value.powf(1.0 / 2.4)) - 0.055
    }
}

#[inline(always)]
fn srgb_gamma_inv_single(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[inline(always)]
fn srgb_gamma(value: [f32; 3]) -> [f32; 3] {
    [
        srgb_gamma_single(value[0]),
        srgb_gamma_single(value[1]),
        srgb_gamma_single(value[2]),
    ]
}

#[inline(always)]
fn srgb_gamma_inv(value: [f32; 3]) -> [f32; 3] {
    [
        srgb_gamma_inv_single(value[0]),
        srgb_gamma_inv_single(value[1]),
        srgb_gamma_inv_single(value[2]),
    ]
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

struct OfxAllocator;

unsafe impl Allocator for OfxAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let data = shared_data.read().map_err(|_| AllocError)?;
        let data = data.as_ref().ok_or(AllocError)?;
        let memoryAlloc = data.memory_suite.memoryAlloc.ok_or(AllocError)?;

        if layout.size() == 0 {
            panic!("zero-size allocations are not supported");
        }

        let mut buf = ptr::null_mut();
        unsafe {
            ofx_err(memoryAlloc(
                ptr::null_mut(), // effect instance handle (we don't care)
                layout.size(),
                &mut buf,
            ))
            .map_err(|_| AllocError)?
        };

        let buf_slice = unsafe { slice::from_raw_parts_mut(buf as *mut u8, layout.size()) };
        NonNull::new(buf_slice).ok_or(AllocError)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        let data = shared_data.read().unwrap();
        let data = data.as_ref().unwrap();
        let memoryFree = data.memory_suite.memoryFree.unwrap();
        memoryFree(ptr.as_ptr() as *mut _);
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
        YiqField::Both | YiqField::InterleavedUpper | YiqField::InterleavedLower => 0,
    };

    let row_offset: usize = match (src_height & 1, cur_field) {
        (0, YiqField::Upper) => 1,
        (1, YiqField::Upper) => 0,
        (0, YiqField::Lower) => 0,
        (1, YiqField::Lower) => 1,
        (_, YiqField::Both | YiqField::InterleavedUpper | YiqField::InterleavedLower) => 0,
        _ => unreachable!(),
    };

    // On an image with an odd input height, we do ceiling division if we render upper-field-first
    // (take an image 3 pixels tall. it goes render, skip, render--that's 2 renders) but floor division if we
    // render lower-field-first (skip, render, skip--only 1 render).
    let numRows = match cur_field {
        YiqField::Upper => (src_height + 1) / 2,
        YiqField::Lower => src_height / 2,
        YiqField::Both | YiqField::InterleavedUpper | YiqField::InterleavedLower => src_height,
    };

    RowInfo {
        row_lshift,
        row_offset,
        num_rows: numRows,
    }
}

struct EffectApplicationParams<'a> {
    src_ptr: *mut c_void,
    src_row_bytes: i32,
    src_bounds: OfxRectI,
    dst_ptr: *mut c_void,
    dst_row_bytes: i32,
    dst_bounds: OfxRectI,
    effect: &'a NtscEffect,
    frame_num: usize,
    apply_srgb_gamma: bool,
}

struct EffectStorageParams<'a> {
    yiq_data: AllocBox<[f32], OfxAllocator>,
    dst_ptr: *mut c_void,
    dst_row_bytes: i32,
    src_bounds: OfxRectI,
    dst_bounds: OfxRectI,
    effect: &'a NtscEffect,
    frame_num: usize,
    apply_srgb_gamma: bool,
}

impl<'a> EffectApplicationParams<'a> {
    unsafe fn apply<S: PixelFormat>(self) -> OfxResult<EffectStorageParams<'a>> {
        let srcWidth = (self.src_bounds.x2 - self.src_bounds.x1) as usize;
        let srcHeight = (self.src_bounds.y2 - self.src_bounds.y1) as usize;

        let cur_field = self.effect.use_field.to_yiq_field(self.frame_num);

        let yiqBufLength = YiqView::buf_length_for((srcWidth, srcHeight), cur_field);

        let mut ntsc_buf =
            AllocBox::<[f32], _>::new_zeroed_slice_in(yiqBufLength, OfxAllocator).assume_init();

        let (srcFirstRowPtr, flip_y) = if self.src_row_bytes < 0 {
            // Currently untested because I can't find an OFX host that uses negative rowbytes. Fingers crossed it works!
            let row_size = self.src_row_bytes / mem::size_of::<S::DataFormat>() as i32;
            (
                self.src_ptr.sub(-row_size as usize * (srcHeight - 1)),
                false,
            )
        } else {
            (self.src_ptr, true)
        };
        let srcStride = self.src_row_bytes.abs() as usize;

        let mut yiq_view = YiqView::from_parts(&mut ntsc_buf, (srcWidth, srcHeight), cur_field);

        let srcData = slice::from_raw_parts(
            srcFirstRowPtr as *const MaybeUninit<S::DataFormat>,
            srcStride / std::mem::size_of::<S::DataFormat>() * srcHeight as usize,
        );
        // TODO: ported from code written under the assumption that we go from (0, 0) to (width, height).
        // I could change this now to use the actual source rect (in case we need that), but then I'd have to convert
        // from positive Y = up to positive Y = down.
        let blit_info = BlitInfo::new(
            Rect::from_width_height(srcWidth, srcHeight),
            (0, 0),
            srcStride,
            srcHeight,
            flip_y,
        );
        if self.apply_srgb_gamma {
            yiq_view.set_from_strided_buffer_maybe_uninit::<S, _>(srcData, blit_info, srgb_gamma);
        } else {
            yiq_view.set_from_strided_buffer_maybe_uninit::<S, _>(srcData, blit_info, identity);
        }

        self.effect
            .apply_effect_to_yiq(&mut yiq_view, self.frame_num);

        Ok(EffectStorageParams {
            yiq_data: ntsc_buf,
            dst_ptr: self.dst_ptr,
            dst_row_bytes: self.dst_row_bytes,
            src_bounds: self.src_bounds,
            dst_bounds: self.dst_bounds,
            effect: self.effect,
            frame_num: self.frame_num,
            apply_srgb_gamma: self.apply_srgb_gamma,
        })
    }
}

impl<'a> EffectStorageParams<'a> {
    unsafe fn write_to_output<D: PixelFormat>(mut self) -> OfxResult<()> {
        let dstWidth = (self.dst_bounds.x2 - self.dst_bounds.x1) as usize;
        let dstHeight = (self.dst_bounds.y2 - self.dst_bounds.y1) as usize;
        let srcWidth = (self.src_bounds.x2 - self.src_bounds.x1) as usize;
        let srcHeight = (self.src_bounds.y2 - self.src_bounds.y1) as usize;

        let cur_field = self.effect.use_field.to_yiq_field(self.frame_num);
        let yiq_view = YiqView::from_parts(&mut self.yiq_data, (srcWidth, srcHeight), cur_field);
        let (y, i, q) = (yiq_view.y, yiq_view.i, yiq_view.q);

        let RowInfo {
            row_lshift,
            row_offset,
            num_rows: numRows,
        } = getRowInfo(cur_field, srcHeight);

        // Convert back and copy into the destination buffer
        for yi in 0..dstHeight {
            // We need to offset the row pointer in bytes, which is why we leave it as a *mut c_void here
            let dstRowPtr = self
                .dst_ptr
                .offset((self.dst_row_bytes * yi as i32) as isize);
            let src_y = yi as i32 + (self.dst_bounds.y1 - self.src_bounds.y1);

            for x in 0..dstWidth {
                let (r_idx, g_idx, b_idx, a_idx) = D::ORDER.rgba_indices();
                // Now that we have the row pointer, we offset by the actual datatype to get the pixel
                let pixPtr = (dstRowPtr as *mut D::DataFormat).add(x * D::ORDER.num_components());
                let src_x = x as i32 + (self.dst_bounds.x1 - self.src_bounds.x1);
                if src_x < 0 || src_x >= srcWidth as i32 || src_y < 0 || src_y >= srcHeight as i32 {
                    pixPtr.add(0).write(D::DataFormat::from_norm(0.0));
                    pixPtr.add(1).write(D::DataFormat::from_norm(0.0));
                    pixPtr.add(2).write(D::DataFormat::from_norm(0.0));
                    if let Some(a_idx) = a_idx {
                        pixPtr.add(a_idx).write(D::DataFormat::from_norm(0.0));
                    }
                    continue;
                }

                let yiq = match cur_field {
                    YiqField::Upper | YiqField::Lower => {
                        let interp_row = cur_field != YiqField::Both
                            && (yi & 1) != row_offset
                            && src_y > 0
                            && src_y < (srcHeight - 1) as i32;

                        if interp_row {
                            // This row was not processed this frame. Interpolate from the rows above and below it.
                            let row_idx_bottom =
                                (srcHeight as i32 - 1 - src_y + 1) as usize >> row_lshift;
                            let row_idx_top =
                                (srcHeight as i32 - 1 - src_y - 1) as usize >> row_lshift;
                            let idx_top = (row_idx_top * srcWidth) + src_x as usize;
                            let idx_bottom = (row_idx_bottom * srcWidth) + src_x as usize;
                            [
                                (y[idx_top] + y[idx_bottom]) * 0.5,
                                (i[idx_top] + i[idx_bottom]) * 0.5,
                                (q[idx_top] + q[idx_bottom]) * 0.5,
                            ]
                        } else {
                            let row_idx = ((srcHeight as i32 - 1 - src_y) as usize >> row_lshift)
                                .min(numRows - 1);
                            let idx = (row_idx * srcWidth) + src_x as usize;
                            [y[idx], i[idx], q[idx]]
                        }
                    }
                    YiqField::Both => {
                        let row_idx = (srcHeight as i32 - 1 - src_y) as usize;
                        let idx = (row_idx * srcWidth) + src_x as usize;
                        [y[idx], i[idx], q[idx]]
                    }
                    YiqField::InterleavedUpper | YiqField::InterleavedLower => {
                        let row_idx = (srcHeight as i32 - 1 - src_y) as usize;

                        let row_offset = match cur_field {
                            YiqField::InterleavedUpper => {
                                YiqField::Upper.num_image_rows(srcHeight) * (row_idx & 1)
                            }
                            YiqField::InterleavedLower => {
                                YiqField::Lower.num_image_rows(srcHeight) * (1 - (row_idx & 1))
                            }
                            _ => unreachable!(),
                        };
                        let interleaved_row_idx = ((row_idx >> 1) + row_offset).min(srcHeight - 1);

                        let idx = (interleaved_row_idx * srcWidth) + src_x as usize;
                        [y[idx], i[idx], q[idx]]
                    }
                };

                let mut rgb = yiq_to_rgb(yiq);
                if self.apply_srgb_gamma {
                    rgb = srgb_gamma_inv(rgb);
                }
                pixPtr.add(r_idx).write(D::DataFormat::from_norm(rgb[0]));
                pixPtr.add(g_idx).write(D::DataFormat::from_norm(rgb[1]));
                pixPtr.add(b_idx).write(D::DataFormat::from_norm(rgb[2]));
                if let Some(a_idx) = a_idx {
                    pixPtr.add(a_idx).write(D::DataFormat::from_norm(1.0));
                }
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SupportedPixelDepth {
    Byte,
    Short,
    Float,
}

impl TryFrom<&CStr> for SupportedPixelDepth {
    type Error = OfxStatus;

    fn try_from(value: &CStr) -> OfxResult<Self> {
        if value == kOfxBitDepthByte {
            Ok(Self::Byte)
        } else if value == kOfxBitDepthShort {
            Ok(Self::Short)
        } else if value == kOfxBitDepthFloat {
            Ok(Self::Float)
        } else {
            Err(OfxStat::kOfxStatErrUnsupported)
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SupportedImageComponents {
    Rgb,
    Rgba,
}

impl TryFrom<&CStr> for SupportedImageComponents {
    type Error = OfxStatus;

    fn try_from(value: &CStr) -> OfxResult<Self> {
        if value == kOfxImageComponentRGB {
            Ok(Self::Rgb)
        } else if value == kOfxImageComponentRGBA {
            Ok(Self::Rgba)
        } else {
            Err(OfxStat::kOfxStatErrUnsupported)
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SupportedPixelFormat {
    Rgb8,
    Rgba8,
    Rgb16,
    Rgba16,
    Rgb32f,
    Rgba32f,
}

impl From<(SupportedImageComponents, SupportedPixelDepth)> for SupportedPixelFormat {
    fn from(value: (SupportedImageComponents, SupportedPixelDepth)) -> Self {
        match value {
            (SupportedImageComponents::Rgb, SupportedPixelDepth::Byte) => Self::Rgb8,
            (SupportedImageComponents::Rgba, SupportedPixelDepth::Byte) => Self::Rgba8,
            (SupportedImageComponents::Rgb, SupportedPixelDepth::Short) => Self::Rgb16,
            (SupportedImageComponents::Rgba, SupportedPixelDepth::Short) => Self::Rgba16,
            (SupportedImageComponents::Rgb, SupportedPixelDepth::Float) => Self::Rgb32f,
            (SupportedImageComponents::Rgba, SupportedPixelDepth::Float) => Self::Rgba32f,
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

    propGetDouble(inArgs, kOfxPropTime.as_ptr(), 0, &mut time);
    // I'm sure nothing bad will happen here as a result of propGetIntN writing past the pointer it was given
    propGetIntN(
        inArgs,
        kOfxImageEffectPropRenderWindow.as_ptr(),
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
            kOfxImageEffectPropComponents.as_ptr(),
            0,
            &mut cstr,
        );
        SupportedImageComponents::try_from(CStr::from_ptr(cstr))
    }?;

    let num_output_components = {
        let mut cstr: *mut c_char = ptr::null_mut();
        propGetString(
            outputImg.0,
            kOfxImageEffectPropComponents.as_ptr(),
            0,
            &mut cstr,
        );
        SupportedImageComponents::try_from(CStr::from_ptr(cstr))
    }?;

    let source_pixel_depth = {
        let mut cstr: *mut c_char = ptr::null_mut();
        propGetString(
            sourceImg.0,
            kOfxImageEffectPropPixelDepth.as_ptr(),
            0,
            &mut cstr,
        );
        SupportedPixelDepth::try_from(CStr::from_ptr(cstr))
    }?;

    let output_pixel_depth = {
        let mut cstr: *mut c_char = ptr::null_mut();
        propGetString(
            outputImg.0,
            kOfxImageEffectPropPixelDepth.as_ptr(),
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
        kOfxImagePropRowBytes.as_ptr(),
        0,
        &mut dstRowBytes,
    );
    propGetIntN(
        outputImg.0,
        kOfxImagePropBounds.as_ptr(),
        4,
        ptr::addr_of_mut!(dstBounds) as *mut _,
    );
    propGetPointer(outputImg.0, kOfxImagePropData.as_ptr(), 0, &mut dstPtr);

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
        kOfxImagePropRowBytes.as_ptr(),
        0,
        &mut srcRowBytes,
    );
    propGetIntN(
        sourceImg.0,
        kOfxImagePropBounds.as_ptr(),
        4,
        ptr::addr_of_mut!(srcBounds) as *mut _,
    );
    propGetPointer(sourceImg.0, kOfxImagePropData.as_ptr(), 0, &mut srcPtr);

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

    let application_params = EffectApplicationParams {
        src_ptr: srcPtr,
        src_row_bytes: srcRowBytes,
        src_bounds: srcBounds,
        dst_ptr: dstPtr,
        dst_row_bytes: dstRowBytes,
        dst_bounds: dstBounds,
        effect: &effect,
        frame_num,
        apply_srgb_gamma,
    };

    let storage_params =
        match SupportedPixelFormat::from((num_source_components, source_pixel_depth)) {
            SupportedPixelFormat::Rgb8 => application_params.apply::<Rgb8>()?,
            SupportedPixelFormat::Rgb16 => application_params.apply::<Rgb16>()?,
            SupportedPixelFormat::Rgb32f => application_params.apply::<Rgb32f>()?,

            SupportedPixelFormat::Rgba8 => application_params.apply::<Rgbx8>()?,
            SupportedPixelFormat::Rgba16 => application_params.apply::<Rgbx16>()?,
            SupportedPixelFormat::Rgba32f => application_params.apply::<Rgbx32f>()?,
        };

    match SupportedPixelFormat::from((num_output_components, output_pixel_depth)) {
        SupportedPixelFormat::Rgb8 => storage_params.write_to_output::<Rgb8>()?,
        SupportedPixelFormat::Rgb16 => storage_params.write_to_output::<Rgb16>()?,
        SupportedPixelFormat::Rgb32f => storage_params.write_to_output::<Rgb32f>()?,

        SupportedPixelFormat::Rgba8 => storage_params.write_to_output::<Rgbx8>()?,
        SupportedPixelFormat::Rgba16 => storage_params.write_to_output::<Rgbx16>()?,
        SupportedPixelFormat::Rgba32f => storage_params.write_to_output::<Rgbx32f>()?,
    };

    Ok(())
}

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

    let return_status = if action == kOfxActionLoad {
        action_load()
    } else if action == kOfxActionDescribe {
        action_describe(effect)
    } else if action == kOfxImageEffectActionDescribeInContext {
        action_describe_in_context(effect)
    } else if action == kOfxImageEffectActionGetRegionsOfInterest {
        action_get_regions_of_interest(effect, inArgs, outArgs)
    } else if action == kOfxImageEffectActionGetClipPreferences {
        action_get_clip_preferences(outArgs)
    } else if action == kOfxActionInstanceChanged {
        action_instance_changed(effect, inArgs)
    } else if action == kOfxImageEffectActionRender {
        action_render(effect, inArgs)
    } else if action == kOfxActionCreateInstance || action == kOfxActionDestroyInstance {
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

    let plugin_info: &'static OfxPlugin = PLUGIN_INFO.get_or_init(|| {
        OfxPlugin {
            // I think this cast is OK?
            pluginApi: kOfxImageEffectPluginApi.as_ptr(),
            apiVersion: 1,
            pluginIdentifier: static_cstr!("wtf.vala:NtscRs").as_ptr(),
            pluginVersionMajor: 1,
            pluginVersionMinor: 4,
            setHost: Some(set_host_info),
            mainEntry: Some(main_entry),
        }
    });
    plugin_info as *const _
}
