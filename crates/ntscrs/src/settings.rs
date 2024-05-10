use std::{
    any::Any,
    borrow::{Borrow, BorrowMut},
    collections::HashMap,
    error::Error,
    fmt::Display,
    ops::RangeInclusive,
};

use crate::{yiq_fielding::YiqField, FromPrimitive, ToPrimitive};
use macros::FullSettings;
use tinyjson::{JsonParseError, JsonValue};

/// This is used to dynamically inform API consumers of the settings that can be passed to ntsc-rs. This lets various
/// UIs and effect plugins to query this set of settings and display them in their preferred format without having to
/// duplicate a bunch of code.

// TODO: replace with a bunch of metaprogramming macro magic?

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromPrimitive, ToPrimitive)]
pub enum UseField {
    Alternating = 0,
    Upper,
    Lower,
    Both,
    InterleavedUpper,
    InterleavedLower,
}

impl UseField {
    pub fn to_yiq_field(&self, frame_num: usize) -> YiqField {
        match self {
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
            UseField::InterleavedUpper => YiqField::InterleavedUpper,
            UseField::InterleavedLower => YiqField::InterleavedLower,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromPrimitive, ToPrimitive)]
pub enum FilterType {
    ConstantK = 0,
    Butterworth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromPrimitive, ToPrimitive)]
pub enum LumaLowpass {
    None,
    Box,
    Notch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromPrimitive, ToPrimitive)]
pub enum PhaseShift {
    Degrees0,
    Degrees90,
    Degrees180,
    Degrees270,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromPrimitive, ToPrimitive)]
pub enum VHSTapeSpeed {
    SP = 1,
    LP,
    EP,
}

pub(crate) struct VHSTapeParams {
    pub luma_cut: f32,
    pub chroma_cut: f32,
    pub chroma_delay: usize,
}

impl VHSTapeSpeed {
    pub(crate) fn filter_params(&self) -> VHSTapeParams {
        match self {
            VHSTapeSpeed::SP => VHSTapeParams {
                luma_cut: 2400000.0,
                chroma_cut: 320000.0,
                chroma_delay: 4,
            },
            VHSTapeSpeed::LP => VHSTapeParams {
                luma_cut: 1900000.0,
                chroma_cut: 300000.0,
                chroma_delay: 5,
            },
            VHSTapeSpeed::EP => VHSTapeParams {
                luma_cut: 1400000.0,
                chroma_cut: 280000.0,
                chroma_delay: 6,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VHSSharpenSettings {
    pub intensity: f32,
    pub frequency: f32,
}

impl Default for VHSSharpenSettings {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            frequency: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VHSEdgeWaveSettings {
    pub intensity: f32,
    pub speed: f32,
    pub frequency: f32,
    pub detail: i32,
}

impl Default for VHSEdgeWaveSettings {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            speed: 4.0,
            frequency: 0.05,
            detail: 1,
        }
    }
}

#[derive(FullSettings, Debug, Clone, PartialEq)]
pub struct VHSSettings {
    pub tape_speed: Option<VHSTapeSpeed>,
    pub chroma_loss: f32,
    #[settings_block]
    pub sharpen: Option<VHSSharpenSettings>,
    #[settings_block]
    pub edge_wave: Option<VHSEdgeWaveSettings>,
}

impl Default for VHSSettings {
    fn default() -> Self {
        Self {
            tape_speed: Some(VHSTapeSpeed::LP),
            chroma_loss: 0.0,
            sharpen: Some(VHSSharpenSettings::default()),
            edge_wave: Some(VHSEdgeWaveSettings::default()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromPrimitive, ToPrimitive)]
pub enum ChromaLowpass {
    None,
    Light,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromPrimitive, ToPrimitive)]
pub enum ChromaDemodulationFilter {
    Box,
    Notch,
    OneLineComb,
    TwoLineComb,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HeadSwitchingSettings {
    pub height: u32,
    pub offset: u32,
    pub horiz_shift: f32,
}

impl Default for HeadSwitchingSettings {
    fn default() -> Self {
        Self {
            height: 8,
            offset: 3,
            horiz_shift: 72.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrackingNoiseSettings {
    pub height: u32,
    pub wave_intensity: f32,
    pub snow_intensity: f32,
    pub snow_anisotropy: f32,
    pub noise_intensity: f32,
}

impl Default for TrackingNoiseSettings {
    fn default() -> Self {
        Self {
            height: 24,
            wave_intensity: 5.0,
            snow_intensity: 0.05,
            snow_anisotropy: 0.5,
            noise_intensity: 0.005,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RingingSettings {
    pub frequency: f32,
    pub power: f32,
    pub intensity: f32,
}

impl Default for RingingSettings {
    fn default() -> Self {
        Self {
            frequency: 0.45,
            power: 4.0,
            intensity: 4.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChromaNoiseSettings {
    pub frequency: f32,
    pub intensity: f32,
    pub detail: u32,
}

impl Default for ChromaNoiseSettings {
    fn default() -> Self {
        Self {
            frequency: 0.05,
            intensity: 0.1,
            detail: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SettingsBlock<T> {
    pub enabled: bool,
    pub settings: T,
}

impl<T: Default + Clone> From<&Option<T>> for SettingsBlock<T> {
    fn from(opt: &Option<T>) -> Self {
        Self {
            enabled: opt.is_some(),
            settings: match opt {
                Some(v) => v.clone(),
                None => T::default(),
            },
        }
    }
}

impl<T: Default> From<Option<T>> for SettingsBlock<T> {
    fn from(opt: Option<T>) -> Self {
        Self {
            enabled: opt.is_some(),
            settings: opt.unwrap_or_default(),
        }
    }
}

impl<T> From<SettingsBlock<T>> for Option<T> {
    fn from(value: SettingsBlock<T>) -> Self {
        if value.enabled {
            Some(value.settings)
        } else {
            None
        }
    }
}

impl<T: Clone> From<&SettingsBlock<T>> for Option<T> {
    fn from(value: &SettingsBlock<T>) -> Self {
        if value.enabled {
            Some(value.settings.clone())
        } else {
            None
        }
    }
}

impl<T: Default> Default for SettingsBlock<T> {
    fn default() -> Self {
        Self {
            enabled: true,
            settings: T::default(),
        }
    }
}

#[derive(FullSettings, Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct NtscEffect {
    pub random_seed: i32,
    pub use_field: UseField,
    pub filter_type: FilterType,
    pub input_luma_filter: LumaLowpass,
    pub chroma_lowpass_in: ChromaLowpass,
    pub chroma_demodulation: ChromaDemodulationFilter,
    pub luma_smear: f32,
    pub composite_preemphasis: f32,
    pub video_scanline_phase_shift: PhaseShift,
    pub video_scanline_phase_shift_offset: i32,
    #[settings_block]
    pub head_switching: Option<HeadSwitchingSettings>,
    #[settings_block]
    pub tracking_noise: Option<TrackingNoiseSettings>,
    pub composite_noise_intensity: f32,
    #[settings_block]
    pub ringing: Option<RingingSettings>,
    #[settings_block]
    pub chroma_noise: Option<ChromaNoiseSettings>,
    pub snow_intensity: f32,
    pub snow_anisotropy: f32,
    pub chroma_phase_noise_intensity: f32,
    pub chroma_phase_error: f32,
    pub chroma_delay: (f32, i32),
    #[settings_block(nested)]
    pub vhs_settings: Option<VHSSettings>,
    pub chroma_vert_blend: bool,
    pub chroma_lowpass_out: ChromaLowpass,
    pub bandwidth_scale: f32,
}

impl Default for NtscEffect {
    fn default() -> Self {
        Self {
            random_seed: 0,
            use_field: UseField::Alternating,
            filter_type: FilterType::ConstantK,
            input_luma_filter: LumaLowpass::Notch,
            chroma_lowpass_in: ChromaLowpass::Full,
            chroma_demodulation: ChromaDemodulationFilter::Box,
            luma_smear: 0.0,
            chroma_lowpass_out: ChromaLowpass::Full,
            composite_preemphasis: 1.0,
            video_scanline_phase_shift: PhaseShift::Degrees180,
            video_scanline_phase_shift_offset: 0,
            head_switching: Some(HeadSwitchingSettings::default()),
            tracking_noise: Some(TrackingNoiseSettings::default()),
            ringing: Some(RingingSettings::default()),
            snow_intensity: 0.003,
            snow_anisotropy: 0.5,
            composite_noise_intensity: 0.01,
            chroma_noise: Some(ChromaNoiseSettings::default()),
            chroma_phase_noise_intensity: 0.001,
            chroma_phase_error: 0.0,
            chroma_delay: (0.0, 0),
            vhs_settings: Some(VHSSettings::default()),
            chroma_vert_blend: true,
            bandwidth_scale: 1.0,
        }
    }
}

#[derive(Debug)]
pub struct MenuItem {
    pub label: &'static str,
    pub description: Option<&'static str>,
    pub index: u32,
}

#[derive(Debug)]
pub enum SettingKind {
    Enumeration {
        options: Vec<MenuItem>,
        default_value: u32,
    },
    Percentage {
        logarithmic: bool,
        default_value: f32,
    },
    IntRange {
        range: RangeInclusive<i32>,
        default_value: i32,
    },
    FloatRange {
        range: RangeInclusive<f32>,
        logarithmic: bool,
        default_value: f32,
    },
    Boolean {
        default_value: bool,
    },
    Group {
        children: Vec<SettingDescriptor>,
        default_value: bool,
    },
}

#[derive(Debug)]
pub struct SettingDescriptor {
    pub label: &'static str,
    pub description: Option<&'static str>,
    pub kind: SettingKind,
    pub id: SettingID,
}

/// These setting IDs uniquely identify each setting. They are all unique and cannot be reused.
#[allow(non_camel_case_types)]
#[derive(Debug, FromPrimitive, ToPrimitive, Clone, Copy, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum SettingID {
    CHROMA_LOWPASS_IN,
    COMPOSITE_PREEMPHASIS,
    VIDEO_SCANLINE_PHASE_SHIFT,
    VIDEO_SCANLINE_PHASE_SHIFT_OFFSET,
    COMPOSITE_NOISE_INTENSITY,
    CHROMA_NOISE_INTENSITY,
    SNOW_INTENSITY,
    CHROMA_PHASE_NOISE_INTENSITY,
    CHROMA_DELAY_HORIZONTAL,
    CHROMA_DELAY_VERTICAL,
    CHROMA_LOWPASS_OUT,

    HEAD_SWITCHING,
    HEAD_SWITCHING_HEIGHT,
    HEAD_SWITCHING_OFFSET,
    HEAD_SWITCHING_HORIZONTAL_SHIFT,

    TRACKING_NOISE,
    TRACKING_NOISE_HEIGHT,
    TRACKING_NOISE_WAVE_INTENSITY,
    TRACKING_NOISE_SNOW_INTENSITY,

    RINGING,
    RINGING_FREQUENCY,
    RINGING_POWER,
    RINGING_SCALE,

    VHS_SETTINGS,
    VHS_TAPE_SPEED,

    CHROMA_VERT_BLEND,

    VHS_CHROMA_LOSS,
    VHS_SHARPEN_INTENSITY,
    VHS_EDGE_WAVE_INTENSITY,
    VHS_EDGE_WAVE_SPEED,

    USE_FIELD,
    TRACKING_NOISE_NOISE_INTENSITY,
    BANDWIDTH_SCALE,
    CHROMA_DEMODULATION,
    SNOW_ANISOTROPY,
    TRACKING_NOISE_SNOW_ANISOTROPY,

    RANDOM_SEED,

    CHROMA_PHASE_ERROR,
    INPUT_LUMA_FILTER,

    VHS_EDGE_WAVE_ENABLED,
    VHS_EDGE_WAVE_FREQUENCY,
    VHS_EDGE_WAVE_DETAIL,

    CHROMA_NOISE,
    CHROMA_NOISE_FREQUENCY,
    CHROMA_NOISE_DETAIL,

    LUMA_SMEAR,

    FILTER_TYPE,

    VHS_SHARPEN_ENABLED,
    VHS_SHARPEN_FREQUENCY,
}

macro_rules! impl_get_field_ref {
    ($self:ident, $settings:ident, $borrow_op:ident) => {
        match $self {
            SettingID::USE_FIELD => $settings.use_field.$borrow_op(),
            SettingID::CHROMA_LOWPASS_IN => $settings.chroma_lowpass_in.$borrow_op(),
            SettingID::COMPOSITE_PREEMPHASIS => $settings.composite_preemphasis.$borrow_op(),
            SettingID::VIDEO_SCANLINE_PHASE_SHIFT => {
                $settings.video_scanline_phase_shift.$borrow_op()
            }
            SettingID::VIDEO_SCANLINE_PHASE_SHIFT_OFFSET => {
                $settings.video_scanline_phase_shift_offset.$borrow_op()
            }
            SettingID::COMPOSITE_NOISE_INTENSITY => {
                $settings.composite_noise_intensity.$borrow_op()
            }
            SettingID::CHROMA_NOISE_INTENSITY => {
                $settings.chroma_noise.settings.intensity.$borrow_op()
            }
            SettingID::SNOW_INTENSITY => $settings.snow_intensity.$borrow_op(),
            SettingID::SNOW_ANISOTROPY => $settings.snow_anisotropy.$borrow_op(),
            SettingID::CHROMA_DEMODULATION => $settings.chroma_demodulation.$borrow_op(),
            SettingID::CHROMA_PHASE_NOISE_INTENSITY => {
                $settings.chroma_phase_noise_intensity.$borrow_op()
            }
            SettingID::CHROMA_DELAY_HORIZONTAL => $settings.chroma_delay.0.$borrow_op(),
            SettingID::CHROMA_DELAY_VERTICAL => $settings.chroma_delay.1.$borrow_op(),
            SettingID::CHROMA_LOWPASS_OUT => $settings.chroma_lowpass_out.$borrow_op(),

            SettingID::HEAD_SWITCHING => $settings.head_switching.enabled.$borrow_op(),
            SettingID::HEAD_SWITCHING_HEIGHT => {
                $settings.head_switching.settings.height.$borrow_op()
            }
            SettingID::HEAD_SWITCHING_OFFSET => {
                $settings.head_switching.settings.offset.$borrow_op()
            }
            SettingID::HEAD_SWITCHING_HORIZONTAL_SHIFT => {
                $settings.head_switching.settings.horiz_shift.$borrow_op()
            }

            SettingID::TRACKING_NOISE => $settings.tracking_noise.enabled.$borrow_op(),
            SettingID::TRACKING_NOISE_HEIGHT => {
                $settings.tracking_noise.settings.height.$borrow_op()
            }
            SettingID::TRACKING_NOISE_WAVE_INTENSITY => $settings
                .tracking_noise
                .settings
                .wave_intensity
                .$borrow_op(),
            SettingID::TRACKING_NOISE_SNOW_INTENSITY => $settings
                .tracking_noise
                .settings
                .snow_intensity
                .$borrow_op(),
            SettingID::TRACKING_NOISE_SNOW_ANISOTROPY => $settings
                .tracking_noise
                .settings
                .snow_anisotropy
                .$borrow_op(),
            SettingID::TRACKING_NOISE_NOISE_INTENSITY => $settings
                .tracking_noise
                .settings
                .noise_intensity
                .$borrow_op(),

            SettingID::RINGING => $settings.ringing.enabled.$borrow_op(),
            SettingID::RINGING_FREQUENCY => $settings.ringing.settings.frequency.$borrow_op(),
            SettingID::RINGING_POWER => $settings.ringing.settings.power.$borrow_op(),
            SettingID::RINGING_SCALE => $settings.ringing.settings.intensity.$borrow_op(),

            SettingID::VHS_SETTINGS => $settings.vhs_settings.enabled.$borrow_op(),
            SettingID::VHS_TAPE_SPEED => $settings.vhs_settings.settings.tape_speed.$borrow_op(),
            SettingID::CHROMA_VERT_BLEND => $settings.chroma_vert_blend.$borrow_op(),
            SettingID::VHS_CHROMA_LOSS => $settings.vhs_settings.settings.chroma_loss.$borrow_op(),
            SettingID::VHS_SHARPEN_ENABLED => {
                $settings.vhs_settings.settings.sharpen.enabled.$borrow_op()
            }
            SettingID::VHS_SHARPEN_INTENSITY => $settings
                .vhs_settings
                .settings
                .sharpen
                .settings
                .intensity
                .$borrow_op(),
            SettingID::VHS_SHARPEN_FREQUENCY => $settings
                .vhs_settings
                .settings
                .sharpen
                .settings
                .frequency
                .$borrow_op(),
            SettingID::VHS_EDGE_WAVE_ENABLED => $settings
                .vhs_settings
                .settings
                .edge_wave
                .enabled
                .$borrow_op(),
            SettingID::VHS_EDGE_WAVE_SPEED => $settings
                .vhs_settings
                .settings
                .edge_wave
                .settings
                .speed
                .$borrow_op(),
            SettingID::VHS_EDGE_WAVE_INTENSITY => $settings
                .vhs_settings
                .settings
                .edge_wave
                .settings
                .intensity
                .$borrow_op(),
            SettingID::VHS_EDGE_WAVE_FREQUENCY => $settings
                .vhs_settings
                .settings
                .edge_wave
                .settings
                .frequency
                .$borrow_op(),
            SettingID::VHS_EDGE_WAVE_DETAIL => $settings
                .vhs_settings
                .settings
                .edge_wave
                .settings
                .detail
                .$borrow_op(),

            SettingID::BANDWIDTH_SCALE => $settings.bandwidth_scale.$borrow_op(),
            SettingID::RANDOM_SEED => $settings.random_seed.$borrow_op(),

            SettingID::CHROMA_PHASE_ERROR => $settings.chroma_phase_error.$borrow_op(),
            SettingID::INPUT_LUMA_FILTER => $settings.input_luma_filter.$borrow_op(),

            SettingID::CHROMA_NOISE => $settings.chroma_noise.enabled.$borrow_op(),
            SettingID::CHROMA_NOISE_FREQUENCY => {
                $settings.chroma_noise.settings.frequency.$borrow_op()
            }
            SettingID::CHROMA_NOISE_DETAIL => $settings.chroma_noise.settings.detail.$borrow_op(),

            SettingID::LUMA_SMEAR => $settings.luma_smear.$borrow_op(),

            SettingID::FILTER_TYPE => $settings.filter_type.$borrow_op(),
        }
    };
}

#[derive(Debug)]
pub enum ParseSettingsError {
    InvalidJSON(JsonParseError),
    MissingField { field: &'static str },
    UnsupportedVersion { version: f64 },
    InvalidSettingType { key: String, expected: &'static str },
    InvalidEnumValue(SetFieldEnumError),
}

impl Display for ParseSettingsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseSettingsError::InvalidJSON(e) => e.fmt(f),
            ParseSettingsError::MissingField { field } => {
                write!(f, "Missing field: {}", field)
            }
            ParseSettingsError::UnsupportedVersion { version } => {
                write!(f, "Unsupported version: {}", version)
            }
            ParseSettingsError::InvalidSettingType { key, expected } => {
                write!(f, "Setting {} is not a(n) {}", key, expected)
            }
            ParseSettingsError::InvalidEnumValue(e) => e.fmt(f),
        }
    }
}

impl Error for ParseSettingsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl From<JsonParseError> for ParseSettingsError {
    fn from(err: JsonParseError) -> Self {
        Self::InvalidJSON(err)
    }
}

impl From<SetFieldEnumError> for ParseSettingsError {
    fn from(err: SetFieldEnumError) -> Self {
        Self::InvalidEnumValue(err)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SetFieldEnumError {
    InvalidEnumValue { setting_id: SettingID, value: u32 },
    NotAnEnum { setting_id: SettingID },
}

impl SetFieldEnumError {
    pub fn invalid_enum_value(setting_id: SettingID, value: u32) -> Self {
        Self::InvalidEnumValue { setting_id, value }
    }

    pub fn not_an_enum(setting_id: SettingID) -> Self {
        Self::NotAnEnum { setting_id }
    }
}

impl Display for SetFieldEnumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SetFieldEnumError::InvalidEnumValue { setting_id, value } => {
                write!(
                    f,
                    "Invalid enum value {} for setting ID {}",
                    value,
                    setting_id.name()
                )
            }
            SetFieldEnumError::NotAnEnum { setting_id } => {
                write!(f, "Setting ID {} is not an enum", setting_id.name())
            }
        }
    }
}

impl SettingID {
    pub fn set_field_enum(
        &self,
        settings: &mut NtscEffectFullSettings,
        value: u32,
    ) -> Result<(), SetFieldEnumError> {
        // We have to handle each enum manually since FromPrimitive isn't object-safe
        let err = || SetFieldEnumError::invalid_enum_value(*self, value);
        match self {
            SettingID::INPUT_LUMA_FILTER => {
                settings.input_luma_filter = LumaLowpass::from_u32(value).ok_or_else(err)?;
            }
            SettingID::CHROMA_LOWPASS_IN => {
                settings.chroma_lowpass_in = ChromaLowpass::from_u32(value).ok_or_else(err)?;
            }
            SettingID::VIDEO_SCANLINE_PHASE_SHIFT => {
                settings.video_scanline_phase_shift =
                    PhaseShift::from_u32(value).ok_or_else(err)?;
            }
            SettingID::CHROMA_LOWPASS_OUT => {
                settings.chroma_lowpass_out = ChromaLowpass::from_u32(value).ok_or_else(err)?;
            }
            SettingID::VHS_TAPE_SPEED => {
                if value == 0 {
                    settings.vhs_settings.settings.tape_speed = None;
                    return Ok(());
                }
                settings.vhs_settings.settings.tape_speed =
                    Some(VHSTapeSpeed::from_u32(value).ok_or_else(err)?);
            }
            SettingID::USE_FIELD => {
                settings.use_field = UseField::from_u32(value).ok_or_else(err)?;
            }
            SettingID::CHROMA_DEMODULATION => {
                settings.chroma_demodulation =
                    ChromaDemodulationFilter::from_u32(value).ok_or_else(err)?;
            }
            SettingID::FILTER_TYPE => {
                settings.filter_type = FilterType::from_u32(value).ok_or_else(err)?;
            }
            _ => {
                return Err(SetFieldEnumError::not_an_enum(*self));
            }
        }
        Ok(())
    }

    pub fn get_field_enum(&self, settings: &NtscEffectFullSettings) -> Option<u32> {
        // We have to handle each enum manually since FromPrimitive isn't object-safe
        match self {
            SettingID::INPUT_LUMA_FILTER => Some(settings.input_luma_filter.to_u32().unwrap()),
            SettingID::CHROMA_LOWPASS_IN => Some(settings.chroma_lowpass_in.to_u32().unwrap()),
            SettingID::VIDEO_SCANLINE_PHASE_SHIFT => {
                Some(settings.video_scanline_phase_shift.to_u32().unwrap())
            }
            SettingID::CHROMA_LOWPASS_OUT => Some(settings.chroma_lowpass_out.to_u32().unwrap()),
            SettingID::VHS_TAPE_SPEED => Some(match settings.vhs_settings.settings.tape_speed {
                Some(tape_speed) => tape_speed.to_u32().unwrap(),
                None => 0,
            }),
            SettingID::USE_FIELD => Some(settings.use_field.to_u32().unwrap()),
            SettingID::CHROMA_DEMODULATION => Some(settings.chroma_demodulation.to_u32().unwrap()),
            SettingID::FILTER_TYPE => Some(settings.filter_type.to_u32().unwrap()),
            _ => None,
        }
    }

    pub fn get_field_ref<'a, T: 'static>(
        &self,
        settings: &'a NtscEffectFullSettings,
    ) -> Option<&'a T> {
        let field_ref: &dyn Any = impl_get_field_ref!(self, settings, borrow);

        field_ref.downcast_ref::<T>()
    }

    pub fn get_field_mut<'a, T: 'static>(
        &self,
        settings: &'a mut NtscEffectFullSettings,
    ) -> Option<&'a mut T> {
        let field_ref: &mut dyn Any = impl_get_field_ref!(self, settings, borrow_mut);

        field_ref.downcast_mut::<T>()
    }

    /// Get the fixed name for a setting ID. These are unique and will not be reused, and will not change.
    pub fn name(&self) -> &'static str {
        match self {
            SettingID::CHROMA_LOWPASS_IN => "chroma_lowpass_in",
            SettingID::COMPOSITE_PREEMPHASIS => "composite_preemphasis",
            SettingID::VIDEO_SCANLINE_PHASE_SHIFT => "video_scanline_phase_shift",
            SettingID::VIDEO_SCANLINE_PHASE_SHIFT_OFFSET => "video_scanline_phase_shift_offset",
            SettingID::COMPOSITE_NOISE_INTENSITY => "composite_noise_intensity",
            SettingID::CHROMA_NOISE_INTENSITY => "chroma_noise_intensity",
            SettingID::SNOW_INTENSITY => "snow_intensity",
            SettingID::CHROMA_PHASE_NOISE_INTENSITY => "chroma_phase_noise_intensity",
            SettingID::CHROMA_DELAY_HORIZONTAL => "chroma_delay_horizontal",
            SettingID::CHROMA_DELAY_VERTICAL => "chroma_delay_vertical",
            SettingID::CHROMA_LOWPASS_OUT => "chroma_lowpass_out",
            SettingID::HEAD_SWITCHING => "head_switching",
            SettingID::HEAD_SWITCHING_HEIGHT => "head_switching_height",
            SettingID::HEAD_SWITCHING_OFFSET => "head_switching_offset",
            SettingID::HEAD_SWITCHING_HORIZONTAL_SHIFT => "head_switching_horizontal_shift",
            SettingID::TRACKING_NOISE => "tracking_noise",
            SettingID::TRACKING_NOISE_HEIGHT => "tracking_noise_height",
            SettingID::TRACKING_NOISE_WAVE_INTENSITY => "tracking_noise_wave_intensity",
            SettingID::TRACKING_NOISE_SNOW_INTENSITY => "tracking_noise_snow_intensity",
            SettingID::RINGING => "ringing",
            SettingID::RINGING_FREQUENCY => "ringing_frequency",
            SettingID::RINGING_POWER => "ringing_power",
            SettingID::RINGING_SCALE => "ringing_scale",
            SettingID::VHS_SETTINGS => "vhs_settings",
            SettingID::VHS_TAPE_SPEED => "vhs_tape_speed",
            SettingID::CHROMA_VERT_BLEND => "vhs_chroma_vert_blend",
            SettingID::VHS_CHROMA_LOSS => "vhs_chroma_loss",
            SettingID::VHS_SHARPEN_ENABLED => "vhs_sharpen_enabled",
            SettingID::VHS_SHARPEN_INTENSITY => "vhs_sharpen",
            SettingID::VHS_SHARPEN_FREQUENCY => "vhs_sharpen_frequency",
            SettingID::VHS_EDGE_WAVE_ENABLED => "vhs_edge_wave_enabled",
            SettingID::VHS_EDGE_WAVE_INTENSITY => "vhs_edge_wave",
            SettingID::VHS_EDGE_WAVE_SPEED => "vhs_edge_wave_speed",
            SettingID::VHS_EDGE_WAVE_FREQUENCY => "vhs_edge_wave_frequency",
            SettingID::VHS_EDGE_WAVE_DETAIL => "vhs_edge_wave_detail",
            SettingID::USE_FIELD => "use_field",
            SettingID::TRACKING_NOISE_NOISE_INTENSITY => "tracking_noise_noise_intensity",
            SettingID::BANDWIDTH_SCALE => "bandwidth_scale",
            SettingID::CHROMA_DEMODULATION => "chroma_demodulation",
            SettingID::SNOW_ANISOTROPY => "snow_anisotropy",
            SettingID::TRACKING_NOISE_SNOW_ANISOTROPY => "tracking_noise_snow_anisotropy",
            SettingID::RANDOM_SEED => "random_seed",
            SettingID::CHROMA_PHASE_ERROR => "chroma_phase_error",
            SettingID::INPUT_LUMA_FILTER => "input_luma_filter",
            SettingID::CHROMA_NOISE => "chroma_noise",
            SettingID::CHROMA_NOISE_FREQUENCY => "chroma_noise_frequency",
            SettingID::CHROMA_NOISE_DETAIL => "chroma_noise_detail",
            SettingID::LUMA_SMEAR => "luma_smear",
            SettingID::FILTER_TYPE => "filter_type",
        }
    }
}

trait GetAndExpect {
    fn get_and_expect_bool(&self, key: &str) -> Result<Option<bool>, ParseSettingsError>;
    fn get_and_expect_number(&self, key: &str) -> Result<Option<f64>, ParseSettingsError>;
    fn get_and_expect_object(
        &self,
        key: &str,
    ) -> Result<Option<&HashMap<String, JsonValue>>, ParseSettingsError>;
}

impl GetAndExpect for HashMap<String, JsonValue> {
    fn get_and_expect_bool(&self, key: &str) -> Result<Option<bool>, ParseSettingsError> {
        self.get(key)
            .map(|v| {
                v.get::<bool>()
                    .cloned()
                    .ok_or_else(|| ParseSettingsError::InvalidSettingType {
                        key: key.to_owned(),
                        expected: "bool",
                    })
            })
            .transpose()
    }

    fn get_and_expect_number(&self, key: &str) -> Result<Option<f64>, ParseSettingsError> {
        self.get(key)
            .map(|v| {
                v.get::<f64>()
                    .cloned()
                    .ok_or_else(|| ParseSettingsError::InvalidSettingType {
                        key: key.to_owned(),
                        expected: "number",
                    })
            })
            .transpose()
    }

    fn get_and_expect_object(
        &self,
        key: &str,
    ) -> Result<Option<&HashMap<String, JsonValue>>, ParseSettingsError> {
        self.get(key)
            .map(|v| {
                v.get::<HashMap<_, _>>()
                    .ok_or_else(|| ParseSettingsError::InvalidSettingType {
                        key: key.to_owned(),
                        expected: "object",
                    })
            })
            .transpose()
    }
}

pub struct SettingsList {
    pub settings: Box<[SettingDescriptor]>,
    pub by_id: Box<[Option<Box<[usize]>>]>,
}

impl SettingsList {
    fn construct_id_map(
        settings: &[SettingDescriptor],
        map: &mut Vec<Option<Box<[usize]>>>,
        parent_path: &[usize],
    ) {
        for (index, descriptor) in settings.iter().enumerate() {
            let id = descriptor.id as usize;
            if id >= map.len() {
                map.resize(id + 1, None);
            }

            let mut path = parent_path.to_owned();
            path.push(index);

            if let SettingKind::Group { children, .. } = &descriptor.kind {
                Self::construct_id_map(children, map, &path);
            }
            map[id] = Some(path.into_boxed_slice());
        }
    }

    pub fn new() -> SettingsList {
        let default_settings = NtscEffectFullSettings::default();

        let v = vec![
            SettingDescriptor {
                label: "Random seed",
                description: None,
                kind: SettingKind::IntRange { range: i32::MIN..=i32::MAX, default_value: default_settings.random_seed },
                id: SettingID::RANDOM_SEED,
            },
            SettingDescriptor {
                label: "Bandwidth scale",
                description: Some("Horizontally scale the effect by this amount."),
                kind: SettingKind::FloatRange { range: 0.125..=8.0, logarithmic: false, default_value: default_settings.bandwidth_scale },
                id: SettingID::BANDWIDTH_SCALE,
            },
            SettingDescriptor {
                label: "Use field",
                description: Some("Choose which rows (\"fields\" in NTSC parlance) of the source image will be used."),
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "Alternating",
                            description: Some("Skip every other row, alternating between skipping even and odd rows."),
                            index: UseField::Alternating.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "Upper only",
                            description: Some("Skip every lower row, keeping the upper ones."),
                            index: UseField::Upper.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "Lower only",
                            description: Some("Skip every upper row, keeping the lower ones."),
                            index: UseField::Lower.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "Interleaved (upper first)",
                            description: Some("Treat the video as interlaced, with the upper field as the earlier frame."),
                            index: UseField::InterleavedUpper.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "Interleaved (lower first)",
                            description: Some("Treat the video as interlaced, with the lower field as the earlier frame."),
                            index: UseField::InterleavedLower.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "Both",
                            description: Some("Use all rows; don't skip any."),
                            index: UseField::Both.to_u32().unwrap(),
                        },
                    ],
                    default_value: default_settings.use_field.to_u32().unwrap(),
                },
                id: SettingID::USE_FIELD,
            },
            SettingDescriptor {
                label: "Lowpass filter type",
                description: Some("The low-pass filter to use throughout the effect."),
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "Constant K (blurry)",
                            description: Some("Simple constant-k filter. Produces longer, blurry results."),
                            index: FilterType::ConstantK.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "Butterworth (sharper)",
                            description: Some("Filter with a sharper falloff. Produces sharpened, less blurry results."),
                            index: FilterType::Butterworth.to_u32().unwrap(),
                        },
                    ],
                    default_value: default_settings.filter_type.to_u32().unwrap(),
                },
                id: SettingID::FILTER_TYPE,
            },
            SettingDescriptor {
                label: "Input luma filter",
                description: Some("Filter the input luminance to decrease rainbow artifacts."),
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "Notch",
                            description: Some("Apply a notch filter to the input luminance signal. Sharp, but has ringing artifacts."),
                            index: LumaLowpass::Notch.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "Box",
                            description: Some("Apply a simple box filter to the input luminance signal."),
                            index: LumaLowpass::Box.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "None",
                            description: Some("Do not filter the luminance signal. Adds rainbow artifacts."),
                            index: LumaLowpass::None.to_u32().unwrap(),
                        },
                    ],
                    default_value: default_settings.input_luma_filter.to_u32().unwrap(),
                },
                id: SettingID::INPUT_LUMA_FILTER,
            },
            SettingDescriptor {
                label: "Chroma low-pass in",
                description: Some("Apply a low-pass filter to the input chroma signal."),
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "Full",
                            description: Some("Full-intensity low-pass filter."),
                            index: ChromaLowpass::Full.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "Light",
                            description: Some("Less intense low-pass filter."),
                            index: ChromaLowpass::Light.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "None",
                            description: Some("No low-pass filter."),
                            index: ChromaLowpass::None.to_u32().unwrap(),
                        },
                    ],
                    default_value: default_settings.chroma_lowpass_in.to_u32().unwrap(),
                },
                id: SettingID::CHROMA_LOWPASS_IN,
            },
            SettingDescriptor {
                label: "Composite preemphasis",
                description: Some("Boost high frequencies in the NTSC signal, sharpening the image and intensifying colors."),
                kind: SettingKind::FloatRange {
                    range: 0.0..=2.0,
                    logarithmic: false,
                    default_value: default_settings.composite_preemphasis,
                },
                id: SettingID::COMPOSITE_PREEMPHASIS,
            },
            SettingDescriptor {
                label: "Composite noise",
                description: Some("Apply noise to the NTSC signal."),
                kind: SettingKind::Percentage {
                    logarithmic: true,
                    default_value: default_settings.composite_noise_intensity,
                },
                id: SettingID::COMPOSITE_NOISE_INTENSITY,
            },
            SettingDescriptor {
                label: "Snow",
                description: Some("Frequency of random speckles in the image."),
                kind: SettingKind::FloatRange {
                    range: 0.0..=100.0,
                    logarithmic: true,
                    default_value: default_settings.snow_intensity,
                },
                id: SettingID::SNOW_INTENSITY,
            },
            SettingDescriptor {
                label: "Snow anisotropy",
                description: Some("Determines whether the speckles are placed truly randomly or concentrated in certain rows."),
                kind: SettingKind::Percentage {
                    logarithmic: false,
                    default_value: default_settings.snow_anisotropy,
                },
                id: SettingID::SNOW_ANISOTROPY,
            },
            SettingDescriptor {
                label: "Scanline phase shift",
                description: Some("Phase shift of the chrominance signal each scanline. Usually 180 degrees."),
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "0 degrees",
                            description: None,
                            index: PhaseShift::Degrees0.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "90 degrees",
                            description: None,
                            index: PhaseShift::Degrees90.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "180 degrees",
                            description: None,
                            index: PhaseShift::Degrees180.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "270 degrees",
                            description: None,
                            index: PhaseShift::Degrees270.to_u32().unwrap(),
                        },
                    ],
                    default_value: default_settings.video_scanline_phase_shift.to_u32().unwrap(),
                },
                id: SettingID::VIDEO_SCANLINE_PHASE_SHIFT,
            },
            SettingDescriptor {
                label: "Scanline phase shift offset",
                description: None,
                kind: SettingKind::IntRange {
                    range: 0..=4,
                    default_value: default_settings.video_scanline_phase_shift_offset,
                },
                id: SettingID::VIDEO_SCANLINE_PHASE_SHIFT_OFFSET,
            },
            SettingDescriptor {
                label: "Chroma demodulation filter",
                description: Some("Filter used to modulate the chrominance (color) data out of the composite NTSC signal."),
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "Box",
                            description: Some("Simple horizontal box blur."),
                            index: ChromaDemodulationFilter::Box.to_u32().unwrap()
                        },
                        MenuItem {
                            label: "Notch",
                            description: Some("Notch filter. Sharper than a box blur, but with ringing artifacts."),
                            index: ChromaDemodulationFilter::Notch.to_u32().unwrap()
                        },
                        MenuItem {
                            label: "1-line comb",
                            description: Some("Average the current row with the previous one, phase-cancelling the chrominance signals. Only works if the scanline phase shift is 180 degrees."),
                            index: ChromaDemodulationFilter::OneLineComb.to_u32().unwrap()
                        },
                        MenuItem {
                            label: "2-line comb",
                            description: Some("Average the current row with the previous and next ones, phase-cancelling the chrominance signals. Only works if the scanline phase shift is 180 degrees."),
                            index: ChromaDemodulationFilter::TwoLineComb.to_u32().unwrap()
                        }
                    ],
                    default_value: default_settings.chroma_demodulation.to_u32().unwrap(),
                },
                id: SettingID::CHROMA_DEMODULATION,
            },
            SettingDescriptor {
                label: "Luma smear",
                description: None,
                kind: SettingKind::FloatRange { range: 0.0..=1.0, logarithmic: false, default_value: default_settings.luma_smear },
                id: SettingID::LUMA_SMEAR
            },
            SettingDescriptor {
                label: "Head switching",
                description: Some("Emulate VHS head-switching artifacts at the bottom of the image."),
                kind: SettingKind::Group {
                    children: vec![
                        SettingDescriptor {
                            label: "Height",
                            description: Some("Total height of the head-switching artifact."),
                            kind: SettingKind::IntRange { range: 0..=24, default_value: default_settings.head_switching.settings.height as i32 },
                            id: SettingID::HEAD_SWITCHING_HEIGHT
                        },
                        SettingDescriptor {
                            label: "Offset",
                            description: Some("How much of the head-switching artifact is off-screen."),
                            kind: SettingKind::IntRange { range: 0..=24, default_value: default_settings.head_switching.settings.offset as i32 },
                            id: SettingID::HEAD_SWITCHING_OFFSET
                        },
                        SettingDescriptor {
                            label: "Horizontal shift",
                            description: Some("How much the head-switching artifact shifts rows horizontally."),
                            kind: SettingKind::FloatRange { range: -100.0..=100.0, logarithmic: false, default_value: default_settings.head_switching.settings.horiz_shift },
                            id: SettingID::HEAD_SWITCHING_HORIZONTAL_SHIFT
                        },
                    ],
                    default_value: true,
                },
                id: SettingID::HEAD_SWITCHING,
            },
            SettingDescriptor {
                label: "Tracking noise",
                description: Some("Emulate noise from VHS tracking error."),
                kind: SettingKind::Group {
                    children: vec![
                        SettingDescriptor {
                            label: "Height",
                            description: Some("Total height of the tracking artifacts."),
                            kind: SettingKind::IntRange { range: 0..=120, default_value: default_settings.tracking_noise.settings.height as i32 },
                            id: SettingID::TRACKING_NOISE_HEIGHT
                        },
                        SettingDescriptor {
                            label: "Wave intensity",
                            description: Some("How much the affected scanlines \"wave\" back and forth."),
                            kind: SettingKind::FloatRange { range: -50.0..=50.0, logarithmic: false, default_value: default_settings.tracking_noise.settings.wave_intensity },
                            id: SettingID::TRACKING_NOISE_WAVE_INTENSITY
                        },
                        SettingDescriptor {
                            label: "Snow intensity",
                            description: Some("Frequency of speckle-type noise in the artifacts."),
                            kind: SettingKind::FloatRange { range: 0.0..=1.0, logarithmic: true, default_value: default_settings.tracking_noise.settings.snow_intensity },
                            id: SettingID::TRACKING_NOISE_SNOW_INTENSITY
                        },
                        SettingDescriptor {
                            label: "Snow anisotropy",
                            description: Some("How much the speckles are clustered by scanline."),
                            kind: SettingKind::Percentage { logarithmic: false, default_value: default_settings.tracking_noise.settings.snow_intensity },
                            id: SettingID::TRACKING_NOISE_SNOW_ANISOTROPY
                        },
                        SettingDescriptor {
                            label: "Noise intensity",
                            description: Some("Intensity of non-speckle noise."),
                            kind: SettingKind::Percentage { logarithmic: true, default_value: default_settings.tracking_noise.settings.noise_intensity },
                            id: SettingID::TRACKING_NOISE_NOISE_INTENSITY
                        },
                    ],
                    default_value: true,
                },
                id: SettingID::TRACKING_NOISE,
            },
            SettingDescriptor {
                label: "Ringing",
                description: Some("Additional ringing artifacts, simulated with a notch filter."),
                kind: SettingKind::Group {
                    children: vec![
                        SettingDescriptor {
                            label: "Frequency",
                            description: Some("Frequency/period of the ringing, in \"rings per pixel\"."),
                            kind: SettingKind::Percentage { logarithmic: false, default_value: default_settings.ringing.settings.frequency },
                            id: SettingID::RINGING_FREQUENCY
                        },
                        SettingDescriptor {
                            label: "Power",
                            description: Some("The power of the notch filter / how far out the ringing extends."),
                            kind: SettingKind::FloatRange { range: 1.0..=10.0, logarithmic: false, default_value: default_settings.ringing.settings.power },
                            id: SettingID::RINGING_POWER
                        },
                        SettingDescriptor {
                            label: "Scale",
                            description: Some("Intensity of the ringing."),
                            kind: SettingKind::FloatRange { range: 0.0..=10.0, logarithmic: false, default_value: default_settings.ringing.settings.intensity },
                            id: SettingID::RINGING_SCALE
                        },
                    ],
                    default_value: true,
                },
                id: SettingID::RINGING,
            },
            SettingDescriptor {
                label: "Chroma noise",
                description: Some("Noise applied to the chrominance signal."),
                kind: SettingKind::Group {
                    children: vec![
                        SettingDescriptor {
                            label: "Intensity",
                            description: Some("Intensity of the noise."),
                            kind: SettingKind::Percentage { logarithmic: true, default_value: default_settings.chroma_noise.settings.intensity },
                            id: SettingID::CHROMA_NOISE_INTENSITY
                        },
                        SettingDescriptor {
                            label: "Frequency",
                            description: Some("Base wavelength, in pixels, of the noise."),
                            kind: SettingKind::FloatRange { range: 0.0..=0.5, logarithmic: false, default_value: default_settings.chroma_noise.settings.frequency },
                            id: SettingID::CHROMA_NOISE_FREQUENCY
                        },
                        SettingDescriptor {
                            label: "Detail",
                            description: Some("Octaves of noise."),
                            kind: SettingKind::IntRange { range: 1..=5, default_value: default_settings.chroma_noise.settings.detail as i32 },
                            id: SettingID::CHROMA_NOISE_DETAIL
                        },
                    ],
                    default_value: true,
                },
                id: SettingID::CHROMA_NOISE,
            },
            SettingDescriptor {
                label: "Chroma phase error",
                description: Some("Phase error for the chrominance signal."),
                kind: SettingKind::Percentage {
                    logarithmic: false,
                    default_value: default_settings.chroma_phase_error,
                },
                id: SettingID::CHROMA_PHASE_ERROR,
            },
            SettingDescriptor {
                label: "Chroma phase noise",
                description: Some("Noise applied per-scanline to the phase of the chrominance signal."),
                kind: SettingKind::Percentage {
                    logarithmic: true,
                    default_value: default_settings.chroma_phase_noise_intensity,
                },
                id: SettingID::CHROMA_PHASE_NOISE_INTENSITY,
            },
            SettingDescriptor {
                label: "Chroma delay (horizontal)",
                description: Some("Horizontal offset of the chrominance signal."),
                kind: SettingKind::FloatRange {
                    range: -40.0..=40.0,
                    logarithmic: false,
                    default_value: default_settings.chroma_delay.0,
                },
                id: SettingID::CHROMA_DELAY_HORIZONTAL,
            },
            SettingDescriptor {
                label: "Chroma delay (vertical)",
                description: Some("Vertical offset of the chrominance signal. Usually increases with VHS generation loss."),
                kind: SettingKind::IntRange {
                    range: -20..=20,
                    default_value: default_settings.chroma_delay.1,
                },
                id: SettingID::CHROMA_DELAY_VERTICAL,
            },
            SettingDescriptor {
                label: "VHS emulation",
                description: None,
                kind: SettingKind::Group {
                    children: vec![
                        SettingDescriptor {
                            label: "Tape speed",
                            description: Some("Emulate cutoff of high-frequency data at various VHS recording speeds."),
                            kind: SettingKind::Enumeration {
                                options: vec![
                                    MenuItem {
                                        label: "SP (Standard Play)",
                                        description: None,
                                        index: VHSTapeSpeed::SP.to_u32().unwrap(),
                                    },
                                    MenuItem {
                                        label: "LP (Long Play)",
                                        description: None,
                                        index: VHSTapeSpeed::LP.to_u32().unwrap(),
                                    },
                                    MenuItem {
                                        label: "EP (Extended Play)",
                                        description: None,
                                        index: VHSTapeSpeed::EP.to_u32().unwrap(),
                                    },
                                    MenuItem {
                                        label: "None",
                                        description: None,
                                        index: 0,
                                    },
                                ],
                                default_value: default_settings.chroma_lowpass_in.to_u32().unwrap(),
                            },
                            id: SettingID::VHS_TAPE_SPEED
                        },
                        SettingDescriptor {
                            label: "Chroma loss",
                            description: Some("Chance that the chrominance signal is completely lost in each scanline."),
                            kind: SettingKind::Percentage { logarithmic: true, default_value: default_settings.vhs_settings.settings.chroma_loss },
                            id: SettingID::VHS_CHROMA_LOSS
                        },
                        SettingDescriptor {
                            label: "Sharpen",
                            description: Some("Sharpening of the image, as done by some VHS decks."),
                            kind: SettingKind::Group { children: vec![
                                SettingDescriptor {
                                    label: "Intensity",
                                    description: Some("Amount of sharpening to apply."),
                                    kind: SettingKind::FloatRange { range: 0.0..=5.0, logarithmic: false, default_value: default_settings.vhs_settings.settings.sharpen.settings.intensity },
                                    id: SettingID::VHS_SHARPEN_INTENSITY
                                },
                                SettingDescriptor {
                                    label: "Frequency",
                                    description: Some("Frequency / radius of the sharpening, relative to the tape speed's cutoff frequency."),
                                    kind: SettingKind::FloatRange { range: 0.5..=4.0, logarithmic: false, default_value: default_settings.vhs_settings.settings.sharpen.settings.frequency },
                                    id: SettingID::VHS_SHARPEN_FREQUENCY
                                }
                            ], default_value: true },
                            id: SettingID::VHS_SHARPEN_ENABLED
                        },
                        SettingDescriptor {
                            label: "Edge wave",
                            description: Some("Horizontal waving of the image."),
                            kind: SettingKind::Group {
                                children: vec![
                                    SettingDescriptor {
                                        label: "Intensity",
                                        description: Some("Horizontal waving of the image, in pixels."),
                                        kind: SettingKind::FloatRange { range: 0.0..=20.0, logarithmic: false, default_value: default_settings.vhs_settings.settings.edge_wave.settings.intensity },
                                        id: SettingID::VHS_EDGE_WAVE_INTENSITY
                                    },
                                    SettingDescriptor {
                                        label: "Speed",
                                        description: Some("Speed at which the horizontal waving occurs."),
                                        kind: SettingKind::FloatRange { range: 0.0..=10.0, logarithmic: false, default_value: default_settings.vhs_settings.settings.edge_wave.settings.speed },
                                        id: SettingID::VHS_EDGE_WAVE_SPEED
                                    },
                                    SettingDescriptor {
                                        label: "Frequency",
                                        description: Some("Base wavelength for the horizontal waving."),
                                        kind: SettingKind::FloatRange { range: 0.0..=0.5, logarithmic: false, default_value: default_settings.vhs_settings.settings.edge_wave.settings.frequency },
                                        id: SettingID::VHS_EDGE_WAVE_FREQUENCY
                                    },
                                    SettingDescriptor {
                                        label: "Detail",
                                        description: Some("Octaves of noise for the waves."),
                                        kind: SettingKind::IntRange { range: 1..=5, default_value: default_settings.vhs_settings.settings.edge_wave.settings.detail },
                                        id: SettingID::VHS_EDGE_WAVE_DETAIL
                                    },
                                ],
                                default_value: true
                            },
                            id: SettingID::VHS_EDGE_WAVE_ENABLED
                        }
                    ],
                    default_value: true,
                },
                id: SettingID::VHS_SETTINGS,
            },
            SettingDescriptor {
                label: "Vertically blend chroma",
                description: Some("Vertically blend each scanline's chrominance with the scanline above it."),
                kind: SettingKind::Boolean { default_value: default_settings.chroma_vert_blend },
                id: SettingID::CHROMA_VERT_BLEND
            },
            SettingDescriptor {
                label: "Chroma low-pass out",
                description: Some("Apply a low-pass filter to the output chroma signal."),
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "Full",
                            description: Some("Full-intensity low-pass filter."),
                            index: ChromaLowpass::Full.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "Light",
                            description: Some("Less intense low-pass filter."),
                            index: ChromaLowpass::Light.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "None",
                            description: Some("No low-pass filter."),
                            index: ChromaLowpass::None.to_u32().unwrap(),
                        },
                    ],
                    default_value: default_settings.chroma_lowpass_out.to_u32().unwrap(),
                },
                id: SettingID::CHROMA_LOWPASS_OUT,
            },
        ];

        let mut by_id = Vec::new();
        Self::construct_id_map(&v, &mut by_id, &[]);

        SettingsList {
            settings: v.into_boxed_slice(),
            by_id: by_id.into_boxed_slice(),
        }
    }

    fn settings_to_json(
        dst: &mut HashMap<String, JsonValue>,
        descriptors: &[SettingDescriptor],
        settings: &NtscEffectFullSettings,
    ) {
        for descriptor in descriptors {
            let value = match &descriptor.kind {
                SettingKind::Enumeration { .. } => {
                    JsonValue::Number(descriptor.id.get_field_enum(settings).unwrap() as f64)
                }
                SettingKind::Percentage { .. } | SettingKind::FloatRange { .. } => {
                    JsonValue::Number(*descriptor.id.get_field_ref::<f32>(settings).unwrap() as f64)
                }
                SettingKind::IntRange { .. } => JsonValue::Number(
                    if let Some(n) = descriptor.id.get_field_ref::<u32>(settings) {
                        *n as f64
                    } else if let Some(n) = descriptor.id.get_field_ref::<i32>(settings) {
                        *n as f64
                    } else {
                        panic!("int setting descriptor is not an i32 or u32")
                    },
                ),
                SettingKind::Boolean { .. } => {
                    JsonValue::Boolean(*descriptor.id.get_field_ref::<bool>(settings).unwrap())
                }
                SettingKind::Group { children, .. } => {
                    Self::settings_to_json(dst, children, settings);
                    JsonValue::Boolean(*descriptor.id.get_field_ref::<bool>(settings).unwrap())
                }
            };

            dst.insert(descriptor.id.name().to_string(), value);
        }
    }

    pub fn to_json(&self, settings: &NtscEffectFullSettings) -> JsonValue {
        let mut dst_map = HashMap::<String, JsonValue>::new();
        Self::settings_to_json(&mut dst_map, &self.settings, settings);

        dst_map.insert("version".to_string(), JsonValue::Number(1.0));

        JsonValue::Object(dst_map)
    }

    fn settings_from_json(
        json: &HashMap<String, JsonValue>,
        descriptors: &[SettingDescriptor],
        settings: &mut NtscEffectFullSettings,
    ) -> Result<(), ParseSettingsError> {
        for descriptor in descriptors {
            let key = descriptor.id.name();
            match &descriptor.kind {
                SettingKind::Enumeration { .. } => {
                    json.get_and_expect_number(key)?
                        .map(|n| descriptor.id.set_field_enum(settings, n as u32))
                        .transpose()?;
                }
                SettingKind::Percentage { .. } | SettingKind::FloatRange { .. } => {
                    json.get_and_expect_number(key)?.map(|n| {
                        *descriptor.id.get_field_mut::<f32>(settings).unwrap() = n as f32;
                    });
                }
                SettingKind::IntRange { .. } => {
                    json.get_and_expect_number(key)?.map(|n| {
                        if let Some(field) = descriptor.id.get_field_mut::<u32>(settings) {
                            *field = n as u32;
                        } else if let Some(field) = descriptor.id.get_field_mut::<i32>(settings) {
                            *field = n as i32;
                        }
                    });
                }
                SettingKind::Boolean { .. } => {
                    json.get_and_expect_bool(key)?.map(|b| {
                        *descriptor.id.get_field_mut::<bool>(settings).unwrap() = b;
                    });
                }
                SettingKind::Group { children, .. } => {
                    json.get_and_expect_bool(key)?.map(|b| {
                        *descriptor.id.get_field_mut::<bool>(settings).unwrap() = b;
                    });
                    Self::settings_from_json(json, children, settings)?;
                }
            }
        }

        Ok(())
    }

    pub fn from_json(&self, json: &str) -> Result<NtscEffectFullSettings, ParseSettingsError> {
        let parsed = json.parse::<JsonValue>()?;

        let parsed_map = parsed.get::<HashMap<_, _>>().ok_or_else(|| {
            ParseSettingsError::InvalidSettingType {
                key: "<root>".to_string(),
                expected: "object",
            }
        })?;

        let version = parsed_map
            .get_and_expect_number("version")?
            .ok_or_else(|| ParseSettingsError::MissingField { field: "version" })?;
        if version != 1.0 {
            return Err(ParseSettingsError::UnsupportedVersion { version });
        }

        let mut dst_settings = NtscEffectFullSettings::default();
        Self::settings_from_json(parsed_map, &self.settings, &mut dst_settings)?;

        Ok(dst_settings)
    }
}
