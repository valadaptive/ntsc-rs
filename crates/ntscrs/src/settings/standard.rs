use std::collections::HashMap;

use crate::{impl_settings_for, settings::SettingsBlock, yiq_fielding::YiqField};
use macros::FullSettings;
use tinyjson::JsonValue;

use super::{
    GetAndExpect, MenuItem, ParseSettingsError, SettingDescriptor, SettingKind, SettingsList,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromPrimitive)]
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

    pub fn interlaced_output_allowed(&self) -> bool {
        matches!(
            self,
            UseField::InterleavedUpper | UseField::InterleavedLower
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromPrimitive)]
pub enum FilterType {
    ConstantK = 0,
    Butterworth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromPrimitive)]
pub enum LumaLowpass {
    None,
    Box,
    Notch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromPrimitive)]
pub enum PhaseShift {
    Degrees0,
    Degrees90,
    Degrees180,
    Degrees270,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromPrimitive)]
pub enum VHSTapeSpeed {
    NONE,
    SP,
    LP,
    EP,
}

pub(crate) struct VHSTapeParams {
    pub luma_cut: f32,
    pub chroma_cut: f32,
    pub chroma_delay: usize,
}

impl VHSTapeSpeed {
    pub(crate) fn filter_params(&self) -> Option<VHSTapeParams> {
        match self {
            Self::NONE => None,
            Self::SP => Some(VHSTapeParams {
                luma_cut: 2400000.0,
                chroma_cut: 320000.0,
                chroma_delay: 4,
            }),
            Self::LP => Some(VHSTapeParams {
                luma_cut: 1900000.0,
                chroma_cut: 300000.0,
                chroma_delay: 5,
            }),
            Self::EP => Some(VHSTapeParams {
                luma_cut: 1400000.0,
                chroma_cut: 280000.0,
                chroma_delay: 6,
            }),
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
            intensity: 0.25,
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
            intensity: 0.5,
            speed: 4.0,
            frequency: 0.05,
            detail: 2,
        }
    }
}

#[derive(FullSettings, Debug, Clone, PartialEq)]
pub struct VHSSettings {
    pub tape_speed: VHSTapeSpeed,
    pub chroma_loss: f32,
    #[settings_block]
    pub sharpen: Option<VHSSharpenSettings>,
    #[settings_block]
    pub edge_wave: Option<VHSEdgeWaveSettings>,
}

impl Default for VHSSettings {
    fn default() -> Self {
        Self {
            tape_speed: VHSTapeSpeed::LP,
            chroma_loss: 0.000025,
            sharpen: Some(VHSSharpenSettings::default()),
            edge_wave: Some(VHSEdgeWaveSettings::default()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromPrimitive)]
pub enum ChromaLowpass {
    None,
    Light,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromPrimitive)]
pub enum ChromaDemodulationFilter {
    Box,
    Notch,
    OneLineComb,
    TwoLineComb,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HeadSwitchingMidLineSettings {
    pub position: f32,
    pub jitter: f32,
}

impl Default for HeadSwitchingMidLineSettings {
    fn default() -> Self {
        Self {
            position: 0.95,
            jitter: 0.03,
        }
    }
}

#[derive(FullSettings, Debug, Clone, PartialEq)]
pub struct HeadSwitchingSettings {
    pub height: u32,
    pub offset: u32,
    pub horiz_shift: f32,
    #[settings_block]
    pub mid_line: Option<HeadSwitchingMidLineSettings>,
}

impl Default for HeadSwitchingSettings {
    fn default() -> Self {
        Self {
            height: 8,
            offset: 3,
            horiz_shift: 72.0,
            mid_line: Some(HeadSwitchingMidLineSettings::default()),
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
            height: 12,
            wave_intensity: 15.0,
            snow_intensity: 0.025,
            snow_anisotropy: 0.25,
            noise_intensity: 0.25,
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

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FbmNoiseSettings {
    pub frequency: f32,
    pub intensity: f32,
    pub detail: u32,
}

#[rustfmt::skip]
pub mod setting_id {
    use crate::settings::SettingID;
    use super::NtscEffectFullSettings;
    type NtscSettingID = SettingID<NtscEffectFullSettings>;

    pub const CHROMA_LOWPASS_IN: NtscSettingID = SettingID::new(0, "chroma_lowpass_in");
    pub const COMPOSITE_SHARPENING: NtscSettingID = SettingID::new(1, "composite_preemphasis");
    pub const VIDEO_SCANLINE_PHASE_SHIFT: NtscSettingID = SettingID::new(2, "video_scanline_phase_shift");
    pub const VIDEO_SCANLINE_PHASE_SHIFT_OFFSET: NtscSettingID = SettingID::new(3, "video_scanline_phase_shift_offset");
    pub const COMPOSITE_NOISE_INTENSITY: NtscSettingID = SettingID::new(4, "composite_noise_intensity");
    pub const CHROMA_NOISE_INTENSITY: NtscSettingID = SettingID::new(5, "chroma_noise_intensity");
    pub const SNOW_INTENSITY: NtscSettingID = SettingID::new(6, "snow_intensity");
    pub const CHROMA_PHASE_NOISE_INTENSITY: NtscSettingID = SettingID::new(7, "chroma_phase_noise_intensity");
    pub const CHROMA_DELAY_HORIZONTAL: NtscSettingID = SettingID::new(8, "chroma_delay_horizontal");
    pub const CHROMA_DELAY_VERTICAL: NtscSettingID = SettingID::new(9, "chroma_delay_vertical");
    pub const CHROMA_LOWPASS_OUT: NtscSettingID = SettingID::new(10, "chroma_lowpass_out");
    pub const HEAD_SWITCHING: NtscSettingID = SettingID::new(11, "head_switching");
    pub const HEAD_SWITCHING_HEIGHT: NtscSettingID = SettingID::new(12, "head_switching_height");
    pub const HEAD_SWITCHING_OFFSET: NtscSettingID = SettingID::new(13, "head_switching_offset");
    pub const HEAD_SWITCHING_HORIZONTAL_SHIFT: NtscSettingID = SettingID::new(14, "head_switching_horizontal_shift");
    pub const TRACKING_NOISE: NtscSettingID = SettingID::new(15, "tracking_noise");
    pub const TRACKING_NOISE_HEIGHT: NtscSettingID = SettingID::new(16, "tracking_noise_height");
    pub const TRACKING_NOISE_WAVE_INTENSITY: NtscSettingID = SettingID::new(17, "tracking_noise_wave_intensity");
    pub const TRACKING_NOISE_SNOW_INTENSITY: NtscSettingID = SettingID::new(18, "tracking_noise_snow_intensity");
    pub const RINGING: NtscSettingID = SettingID::new(19, "ringing");
    pub const RINGING_FREQUENCY: NtscSettingID = SettingID::new(20, "ringing_frequency");
    pub const RINGING_POWER: NtscSettingID = SettingID::new(21, "ringing_power");
    pub const RINGING_SCALE: NtscSettingID = SettingID::new(22, "ringing_scale");
    pub const VHS_SETTINGS: NtscSettingID = SettingID::new(23, "vhs_settings");
    pub const VHS_TAPE_SPEED: NtscSettingID = SettingID::new(24, "vhs_tape_speed");
    pub const CHROMA_VERT_BLEND: NtscSettingID = SettingID::new(25, "vhs_chroma_vert_blend");
    pub const VHS_CHROMA_LOSS: NtscSettingID = SettingID::new(26, "vhs_chroma_loss");
    pub const VHS_SHARPEN_INTENSITY: NtscSettingID = SettingID::new(27, "vhs_sharpen");
    pub const VHS_EDGE_WAVE_INTENSITY: NtscSettingID = SettingID::new(28, "vhs_edge_wave");
    pub const VHS_EDGE_WAVE_SPEED: NtscSettingID = SettingID::new(29, "vhs_edge_wave_speed");
    pub const USE_FIELD: NtscSettingID = SettingID::new(30, "use_field");
    pub const TRACKING_NOISE_NOISE_INTENSITY: NtscSettingID = SettingID::new(31, "tracking_noise_noise_intensity");
    pub const BANDWIDTH_SCALE: NtscSettingID = SettingID::new(32, "bandwidth_scale");
    pub const CHROMA_DEMODULATION: NtscSettingID = SettingID::new(33, "chroma_demodulation");
    pub const SNOW_ANISOTROPY: NtscSettingID = SettingID::new(34, "snow_anisotropy");
    pub const TRACKING_NOISE_SNOW_ANISOTROPY: NtscSettingID = SettingID::new(35, "tracking_noise_snow_anisotropy");
    pub const RANDOM_SEED: NtscSettingID = SettingID::new(36, "random_seed");
    pub const CHROMA_PHASE_ERROR: NtscSettingID = SettingID::new(37, "chroma_phase_error");
    pub const INPUT_LUMA_FILTER: NtscSettingID = SettingID::new(38, "input_luma_filter");
    pub const VHS_EDGE_WAVE_ENABLED: NtscSettingID = SettingID::new(39, "vhs_edge_wave_enabled");
    pub const VHS_EDGE_WAVE_FREQUENCY: NtscSettingID = SettingID::new(40, "vhs_edge_wave_frequency");
    pub const VHS_EDGE_WAVE_DETAIL: NtscSettingID = SettingID::new(41, "vhs_edge_wave_detail");
    pub const CHROMA_NOISE: NtscSettingID = SettingID::new(42, "chroma_noise");
    pub const CHROMA_NOISE_FREQUENCY: NtscSettingID = SettingID::new(43, "chroma_noise_frequency");
    pub const CHROMA_NOISE_DETAIL: NtscSettingID = SettingID::new(44, "chroma_noise_detail");
    pub const LUMA_SMEAR: NtscSettingID = SettingID::new(45, "luma_smear");
    pub const FILTER_TYPE: NtscSettingID = SettingID::new(46, "filter_type");
    pub const VHS_SHARPEN_ENABLED: NtscSettingID = SettingID::new(47, "vhs_sharpen_enabled");
    pub const VHS_SHARPEN_FREQUENCY: NtscSettingID = SettingID::new(48, "vhs_sharpen_frequency");
    pub const HEAD_SWITCHING_START_MID_LINE: NtscSettingID = SettingID::new(49, "head_switching_start_mid_line");
    pub const HEAD_SWITCHING_MID_LINE_POSITION: NtscSettingID = SettingID::new(50, "head_switching_mid_line_position");
    pub const HEAD_SWITCHING_MID_LINE_JITTER: NtscSettingID = SettingID::new(51, "head_switching_mid_line_jitter");
    pub const COMPOSITE_NOISE: NtscSettingID = SettingID::new(52, "composite_noise");
    pub const COMPOSITE_NOISE_FREQUENCY: NtscSettingID = SettingID::new(53, "composite_noise_frequency");
    pub const COMPOSITE_NOISE_DETAIL: NtscSettingID = SettingID::new(54, "composite_noise_detail");
    pub const LUMA_NOISE: NtscSettingID = SettingID::new(55, "luma_noise");
    pub const LUMA_NOISE_FREQUENCY: NtscSettingID = SettingID::new(56, "luma_noise_frequency");
    pub const LUMA_NOISE_INTENSITY: NtscSettingID = SettingID::new(57, "luma_noise_intensity");
    pub const LUMA_NOISE_DETAIL: NtscSettingID = SettingID::new(58, "luma_noise_detail");
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
    pub composite_sharpening: f32,
    pub video_scanline_phase_shift: PhaseShift,
    pub video_scanline_phase_shift_offset: i32,
    #[settings_block(nested)]
    pub head_switching: Option<HeadSwitchingSettings>,
    #[settings_block]
    pub tracking_noise: Option<TrackingNoiseSettings>,
    #[settings_block]
    pub composite_noise: Option<FbmNoiseSettings>,
    #[settings_block]
    pub ringing: Option<RingingSettings>,
    #[settings_block]
    pub luma_noise: Option<FbmNoiseSettings>,
    #[settings_block]
    pub chroma_noise: Option<FbmNoiseSettings>,
    pub snow_intensity: f32,
    pub snow_anisotropy: f32,
    pub chroma_phase_noise_intensity: f32,
    pub chroma_phase_error: f32,
    pub chroma_delay_horizontal: f32,
    pub chroma_delay_vertical: i32,
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
            use_field: UseField::InterleavedUpper,
            filter_type: FilterType::Butterworth,
            input_luma_filter: LumaLowpass::Notch,
            chroma_lowpass_in: ChromaLowpass::Full,
            chroma_demodulation: ChromaDemodulationFilter::Notch,
            luma_smear: 0.5,
            chroma_lowpass_out: ChromaLowpass::Full,
            composite_sharpening: 1.0,
            video_scanline_phase_shift: PhaseShift::Degrees180,
            video_scanline_phase_shift_offset: 0,
            head_switching: Some(HeadSwitchingSettings::default()),
            tracking_noise: Some(TrackingNoiseSettings::default()),
            ringing: Some(RingingSettings::default()),
            snow_intensity: 0.00025,
            snow_anisotropy: 0.5,
            composite_noise: Some(FbmNoiseSettings {
                frequency: 0.5,
                intensity: 0.05,
                detail: 1,
            }),
            luma_noise: Some(FbmNoiseSettings {
                frequency: 0.5,
                intensity: 0.01,
                detail: 1,
            }),
            chroma_noise: Some(FbmNoiseSettings {
                frequency: 0.05,
                intensity: 0.1,
                detail: 2,
            }),
            chroma_phase_noise_intensity: 0.001,
            chroma_phase_error: 0.0,
            chroma_delay_horizontal: 0.0,
            chroma_delay_vertical: 0,
            vhs_settings: Some(VHSSettings::default()),
            chroma_vert_blend: true,
            bandwidth_scale: 1.0,
        }
    }
}

impl_settings_for!(
    NtscEffectFullSettings,
    (setting_id::CHROMA_LOWPASS_IN, chroma_lowpass_in, IS_AN_ENUM),
    (setting_id::COMPOSITE_SHARPENING, composite_sharpening),
    (
        setting_id::VIDEO_SCANLINE_PHASE_SHIFT,
        video_scanline_phase_shift,
        IS_AN_ENUM
    ),
    (
        setting_id::VIDEO_SCANLINE_PHASE_SHIFT_OFFSET,
        video_scanline_phase_shift_offset
    ),
    (
        setting_id::COMPOSITE_NOISE_INTENSITY,
        composite_noise.settings.intensity
    ),
    (
        setting_id::CHROMA_NOISE_INTENSITY,
        chroma_noise.settings.intensity
    ),
    (setting_id::SNOW_INTENSITY, snow_intensity),
    (
        setting_id::CHROMA_PHASE_NOISE_INTENSITY,
        chroma_phase_noise_intensity
    ),
    (setting_id::CHROMA_DELAY_HORIZONTAL, chroma_delay_horizontal),
    (setting_id::CHROMA_DELAY_VERTICAL, chroma_delay_vertical),
    (
        setting_id::CHROMA_LOWPASS_OUT,
        chroma_lowpass_out,
        IS_AN_ENUM
    ),
    (setting_id::HEAD_SWITCHING, head_switching.enabled),
    (
        setting_id::HEAD_SWITCHING_HEIGHT,
        head_switching.settings.height
    ),
    (
        setting_id::HEAD_SWITCHING_OFFSET,
        head_switching.settings.offset
    ),
    (
        setting_id::HEAD_SWITCHING_HORIZONTAL_SHIFT,
        head_switching.settings.horiz_shift
    ),
    (setting_id::TRACKING_NOISE, tracking_noise.enabled),
    (
        setting_id::TRACKING_NOISE_HEIGHT,
        tracking_noise.settings.height
    ),
    (
        setting_id::TRACKING_NOISE_WAVE_INTENSITY,
        tracking_noise.settings.wave_intensity
    ),
    (
        setting_id::TRACKING_NOISE_SNOW_INTENSITY,
        tracking_noise.settings.snow_intensity
    ),
    (setting_id::RINGING, ringing.enabled),
    (setting_id::RINGING_FREQUENCY, ringing.settings.frequency),
    (setting_id::RINGING_POWER, ringing.settings.power),
    (setting_id::RINGING_SCALE, ringing.settings.intensity),
    (setting_id::VHS_SETTINGS, vhs_settings.enabled),
    (
        setting_id::VHS_TAPE_SPEED,
        vhs_settings.settings.tape_speed,
        IS_AN_ENUM
    ),
    (setting_id::CHROMA_VERT_BLEND, chroma_vert_blend),
    (
        setting_id::VHS_CHROMA_LOSS,
        vhs_settings.settings.chroma_loss
    ),
    (
        setting_id::VHS_SHARPEN_INTENSITY,
        vhs_settings.settings.sharpen.settings.intensity
    ),
    (
        setting_id::VHS_EDGE_WAVE_INTENSITY,
        vhs_settings.settings.edge_wave.settings.intensity
    ),
    (
        setting_id::VHS_EDGE_WAVE_SPEED,
        vhs_settings.settings.edge_wave.settings.speed
    ),
    (setting_id::USE_FIELD, use_field, IS_AN_ENUM),
    (
        setting_id::TRACKING_NOISE_NOISE_INTENSITY,
        tracking_noise.settings.noise_intensity
    ),
    (setting_id::BANDWIDTH_SCALE, bandwidth_scale),
    (
        setting_id::CHROMA_DEMODULATION,
        chroma_demodulation,
        IS_AN_ENUM
    ),
    (setting_id::SNOW_ANISOTROPY, snow_anisotropy),
    (
        setting_id::TRACKING_NOISE_SNOW_ANISOTROPY,
        tracking_noise.settings.snow_anisotropy
    ),
    (setting_id::RANDOM_SEED, random_seed),
    (setting_id::CHROMA_PHASE_ERROR, chroma_phase_error),
    (setting_id::INPUT_LUMA_FILTER, input_luma_filter, IS_AN_ENUM),
    (
        setting_id::VHS_EDGE_WAVE_ENABLED,
        vhs_settings.settings.edge_wave.enabled
    ),
    (
        setting_id::VHS_EDGE_WAVE_FREQUENCY,
        vhs_settings.settings.edge_wave.settings.frequency
    ),
    (
        setting_id::VHS_EDGE_WAVE_DETAIL,
        vhs_settings.settings.edge_wave.settings.detail
    ),
    (setting_id::CHROMA_NOISE, chroma_noise.enabled),
    (
        setting_id::CHROMA_NOISE_FREQUENCY,
        chroma_noise.settings.frequency
    ),
    (
        setting_id::CHROMA_NOISE_DETAIL,
        chroma_noise.settings.detail
    ),
    (setting_id::LUMA_SMEAR, luma_smear),
    (setting_id::FILTER_TYPE, filter_type, IS_AN_ENUM),
    (
        setting_id::VHS_SHARPEN_ENABLED,
        vhs_settings.settings.sharpen.enabled
    ),
    (
        setting_id::VHS_SHARPEN_FREQUENCY,
        vhs_settings.settings.sharpen.settings.frequency
    ),
    (
        setting_id::HEAD_SWITCHING_START_MID_LINE,
        head_switching.settings.mid_line.enabled
    ),
    (
        setting_id::HEAD_SWITCHING_MID_LINE_POSITION,
        head_switching.settings.mid_line.settings.position
    ),
    (
        setting_id::HEAD_SWITCHING_MID_LINE_JITTER,
        head_switching.settings.mid_line.settings.jitter
    ),
    (setting_id::COMPOSITE_NOISE, composite_noise.enabled),
    (
        setting_id::COMPOSITE_NOISE_FREQUENCY,
        composite_noise.settings.frequency
    ),
    (
        setting_id::COMPOSITE_NOISE_DETAIL,
        composite_noise.settings.detail
    ),
    (setting_id::LUMA_NOISE, luma_noise.enabled),
    (
        setting_id::LUMA_NOISE_FREQUENCY,
        luma_noise.settings.frequency
    ),
    (
        setting_id::LUMA_NOISE_INTENSITY,
        luma_noise.settings.intensity
    ),
    (setting_id::LUMA_NOISE_DETAIL, luma_noise.settings.detail),
);

impl SettingsList<NtscEffectFullSettings> {
    /// Construct a list of all the effect settings. This isn't meant to be mutated--you should just create one instance
    /// of this to use for your entire application/plugin.
    pub fn new() -> Self {
        let default_settings = NtscEffectFullSettings::default();

        let v = vec![
            SettingDescriptor {
                label: "Random seed",
                description: None,
                kind: SettingKind::IntRange { range: i32::MIN..=i32::MAX, default_value: default_settings.random_seed },
                id: setting_id::RANDOM_SEED,
            },
            SettingDescriptor {
                label: "Use field",
                description: Some("Choose which rows (\"fields\" in NTSC parlance) of the source image will be used."),
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "Alternating",
                            description: Some("Skip every other row, alternating between skipping even and odd rows."),
                            index: UseField::Alternating as u32,
                        },
                        MenuItem {
                            label: "Upper only",
                            description: Some("Skip every lower row, keeping the upper ones."),
                            index: UseField::Upper as u32,
                        },
                        MenuItem {
                            label: "Lower only",
                            description: Some("Skip every upper row, keeping the lower ones."),
                            index: UseField::Lower as u32,
                        },
                        MenuItem {
                            label: "Interleaved (upper first)",
                            description: Some("Treat the video as interlaced, with the upper field as the earlier frame."),
                            index: UseField::InterleavedUpper as u32,
                        },
                        MenuItem {
                            label: "Interleaved (lower first)",
                            description: Some("Treat the video as interlaced, with the lower field as the earlier frame."),
                            index: UseField::InterleavedLower as u32,
                        },
                        MenuItem {
                            label: "Both",
                            description: Some("Use all rows; don't skip any."),
                            index: UseField::Both as u32,
                        },
                    ],
                    default_value: default_settings.use_field as u32,
                },
                id: setting_id::USE_FIELD,
            },
            SettingDescriptor {
                label: "Lowpass filter type",
                description: Some("The low-pass filter to use throughout the effect."),
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "Constant K (blurry)",
                            description: Some("Simple constant-k filter. Produces longer, blurry results."),
                            index: FilterType::ConstantK as u32,
                        },
                        MenuItem {
                            label: "Butterworth (sharper)",
                            description: Some("Filter with a sharper falloff. Produces sharpened, less blurry results."),
                            index: FilterType::Butterworth as u32,
                        },
                    ],
                    default_value: default_settings.filter_type as u32,
                },
                id: setting_id::FILTER_TYPE,
            },
            SettingDescriptor {
                label: "Input luma filter",
                description: Some("Filter the input luminance to decrease rainbow artifacts."),
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "Notch",
                            description: Some("Apply a notch filter to the input luminance signal. Sharp, but has ringing artifacts."),
                            index: LumaLowpass::Notch as u32,
                        },
                        MenuItem {
                            label: "Box",
                            description: Some("Apply a simple box filter to the input luminance signal."),
                            index: LumaLowpass::Box as u32,
                        },
                        MenuItem {
                            label: "None",
                            description: Some("Do not filter the luminance signal. Adds rainbow artifacts."),
                            index: LumaLowpass::None as u32,
                        },
                    ],
                    default_value: default_settings.input_luma_filter as u32,
                },
                id: setting_id::INPUT_LUMA_FILTER,
            },
            SettingDescriptor {
                label: "Chroma low-pass in",
                description: Some("Apply a low-pass filter to the input chrominance (color) signal."),
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "Full",
                            description: Some("Full-intensity low-pass filter."),
                            index: ChromaLowpass::Full as u32,
                        },
                        MenuItem {
                            label: "Light",
                            description: Some("Less intense low-pass filter."),
                            index: ChromaLowpass::Light as u32,
                        },
                        MenuItem {
                            label: "None",
                            description: Some("No low-pass filter."),
                            index: ChromaLowpass::None as u32,
                        },
                    ],
                    default_value: default_settings.chroma_lowpass_in as u32,
                },
                id: setting_id::CHROMA_LOWPASS_IN,
            },
            SettingDescriptor {
                label: "Composite signal sharpening",
                description: Some("Boost high frequencies in the NTSC signal, sharpening the image and intensifying colors."),
                kind: SettingKind::FloatRange {
                    range: -1.0..=2.0,
                    logarithmic: false,
                    default_value: default_settings.composite_sharpening,
                },
                id: setting_id::COMPOSITE_SHARPENING,
            },

            SettingDescriptor {
                label: "Composite signal noise",
                description: Some("Noise applied to the composite NTSC signal."),
                kind: SettingKind::Group {
                    children: vec![
                        SettingDescriptor {
                            label: "Intensity",
                            description: Some("Intensity of the noise."),
                            kind: SettingKind::Percentage { logarithmic: true, default_value: default_settings.composite_noise.settings.intensity },
                            id: setting_id::COMPOSITE_NOISE_INTENSITY
                        },
                        SettingDescriptor {
                            label: "Frequency",
                            description: Some("Base wavelength, in pixels, of the noise."),
                            kind: SettingKind::FloatRange { range: 0.0..=1.0, logarithmic: false, default_value: default_settings.composite_noise.settings.frequency },
                            id: setting_id::COMPOSITE_NOISE_FREQUENCY
                        },
                        SettingDescriptor {
                            label: "Detail",
                            description: Some("Octaves of noise."),
                            kind: SettingKind::IntRange { range: 1..=5, default_value: default_settings.composite_noise.settings.detail as i32 },
                            id: setting_id::COMPOSITE_NOISE_DETAIL
                        },
                    ],
                    default_value: true,
                },
                id: setting_id::COMPOSITE_NOISE,
            },
            SettingDescriptor {
                label: "Snow",
                description: Some("Frequency of random speckles in the image."),
                kind: SettingKind::FloatRange {
                    range: 0.0..=100.0,
                    logarithmic: true,
                    default_value: default_settings.snow_intensity,
                },
                id: setting_id::SNOW_INTENSITY,
            },
            SettingDescriptor {
                label: "Snow anisotropy",
                description: Some("Determines whether the speckles are placed truly randomly or concentrated in certain rows."),
                kind: SettingKind::Percentage {
                    logarithmic: false,
                    default_value: default_settings.snow_anisotropy,
                },
                id: setting_id::SNOW_ANISOTROPY,
            },
            SettingDescriptor {
                label: "Scanline phase shift",
                description: Some("Phase shift of the chrominance (color) signal each scanline. Usually 180 degrees."),
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "0 degrees",
                            description: None,
                            index: PhaseShift::Degrees0 as u32,
                        },
                        MenuItem {
                            label: "90 degrees",
                            description: None,
                            index: PhaseShift::Degrees90 as u32,
                        },
                        MenuItem {
                            label: "180 degrees",
                            description: None,
                            index: PhaseShift::Degrees180 as u32,
                        },
                        MenuItem {
                            label: "270 degrees",
                            description: None,
                            index: PhaseShift::Degrees270 as u32,
                        },
                    ],
                    default_value: default_settings.video_scanline_phase_shift as u32,
                },
                id: setting_id::VIDEO_SCANLINE_PHASE_SHIFT,
            },
            SettingDescriptor {
                label: "Scanline phase shift offset",
                description: None,
                kind: SettingKind::IntRange {
                    range: 0..=3,
                    default_value: default_settings.video_scanline_phase_shift_offset,
                },
                id: setting_id::VIDEO_SCANLINE_PHASE_SHIFT_OFFSET,
            },
            SettingDescriptor {
                label: "Chroma demodulation filter",
                description: Some("Filter used to modulate the chrominance (color) data out of the composite NTSC signal."),
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "Box",
                            description: Some("Simple horizontal box blur."),
                            index: ChromaDemodulationFilter::Box as u32
                        },
                        MenuItem {
                            label: "Notch",
                            description: Some("Notch filter. Sharper than a box blur, but with ringing artifacts."),
                            index: ChromaDemodulationFilter::Notch as u32
                        },
                        MenuItem {
                            label: "1-line comb",
                            description: Some("Average the current row with the previous one, phase-cancelling the chrominance (color) signals. Only works if the scanline phase shift is 180 degrees."),
                            index: ChromaDemodulationFilter::OneLineComb as u32
                        },
                        MenuItem {
                            label: "2-line comb",
                            description: Some("Average the current row with the previous and next ones, phase-cancelling the chrominance (color) signals. Only works if the scanline phase shift is 180 degrees."),
                            index: ChromaDemodulationFilter::TwoLineComb as u32
                        }
                    ],
                    default_value: default_settings.chroma_demodulation as u32,
                },
                id: setting_id::CHROMA_DEMODULATION,
            },
            SettingDescriptor {
                label: "Luma smear",
                description: None,
                kind: SettingKind::FloatRange { range: 0.0..=1.0, logarithmic: false, default_value: default_settings.luma_smear },
                id: setting_id::LUMA_SMEAR
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
                            id: setting_id::HEAD_SWITCHING_HEIGHT
                        },
                        SettingDescriptor {
                            label: "Offset",
                            description: Some("How much of the head-switching artifact is off-screen."),
                            kind: SettingKind::IntRange { range: 0..=24, default_value: default_settings.head_switching.settings.offset as i32 },
                            id: setting_id::HEAD_SWITCHING_OFFSET
                        },
                        SettingDescriptor {
                            label: "Horizontal shift",
                            description: Some("How much the head-switching artifact shifts rows horizontally."),
                            kind: SettingKind::FloatRange { range: -100.0..=100.0, logarithmic: false, default_value: default_settings.head_switching.settings.horiz_shift },
                            id: setting_id::HEAD_SWITCHING_HORIZONTAL_SHIFT
                        },
                        SettingDescriptor {
                            label: "Start mid-line",
                            description: Some("Start the head-switching artifact mid-scanline, with some static where it begins."),
                            kind: SettingKind::Group { children: vec![
                                SettingDescriptor {
                                    label: "Position",
                                    description: Some("Horizontal position at which the head-switching starts."),
                                    kind: SettingKind::Percentage { logarithmic: false, default_value: default_settings.head_switching.settings.mid_line.settings.position },
                                    id: setting_id::HEAD_SWITCHING_MID_LINE_POSITION
                                },
                                SettingDescriptor {
                                    label: "Jitter",
                                    description: Some("How much the head-switching artifact \"jitters\" horizontally."),
                                    kind: SettingKind::Percentage { logarithmic: true, default_value: default_settings.head_switching.settings.mid_line.settings.jitter },
                                    id: setting_id::HEAD_SWITCHING_MID_LINE_JITTER
                                }
                            ], default_value: true },
                            id: setting_id::HEAD_SWITCHING_START_MID_LINE
                        }
                    ],
                    default_value: true,
                },
                id: setting_id::HEAD_SWITCHING,
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
                            id: setting_id::TRACKING_NOISE_HEIGHT
                        },
                        SettingDescriptor {
                            label: "Wave intensity",
                            description: Some("How much the affected scanlines \"wave\" back and forth."),
                            kind: SettingKind::FloatRange { range: -50.0..=50.0, logarithmic: false, default_value: default_settings.tracking_noise.settings.wave_intensity },
                            id: setting_id::TRACKING_NOISE_WAVE_INTENSITY
                        },
                        SettingDescriptor {
                            label: "Snow intensity",
                            description: Some("Frequency of speckle-type noise in the artifacts."),
                            kind: SettingKind::FloatRange { range: 0.0..=1.0, logarithmic: true, default_value: default_settings.tracking_noise.settings.snow_intensity },
                            id: setting_id::TRACKING_NOISE_SNOW_INTENSITY
                        },
                        SettingDescriptor {
                            label: "Snow anisotropy",
                            description: Some("How much the speckles are clustered by scanline."),
                            kind: SettingKind::Percentage { logarithmic: false, default_value: default_settings.tracking_noise.settings.snow_intensity },
                            id: setting_id::TRACKING_NOISE_SNOW_ANISOTROPY
                        },
                        SettingDescriptor {
                            label: "Noise intensity",
                            description: Some("Intensity of non-speckle noise."),
                            kind: SettingKind::Percentage { logarithmic: true, default_value: default_settings.tracking_noise.settings.noise_intensity },
                            id: setting_id::TRACKING_NOISE_NOISE_INTENSITY
                        },
                    ],
                    default_value: true,
                },
                id: setting_id::TRACKING_NOISE,
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
                            id: setting_id::RINGING_FREQUENCY
                        },
                        SettingDescriptor {
                            label: "Power",
                            description: Some("The power of the notch filter / how far out the ringing extends."),
                            kind: SettingKind::FloatRange { range: 1.0..=10.0, logarithmic: false, default_value: default_settings.ringing.settings.power },
                            id: setting_id::RINGING_POWER
                        },
                        SettingDescriptor {
                            label: "Scale",
                            description: Some("Intensity of the ringing."),
                            kind: SettingKind::FloatRange { range: 0.0..=10.0, logarithmic: false, default_value: default_settings.ringing.settings.intensity },
                            id: setting_id::RINGING_SCALE
                        },
                    ],
                    default_value: true,
                },
                id: setting_id::RINGING,
            },
            SettingDescriptor {
                label: "Luma noise",
                description: Some("Noise applied to the luminance signal. Useful for higher-frequency noise than the \"Composite noise\" setting can provide."),
                kind: SettingKind::Group {
                    children: vec![
                        SettingDescriptor {
                            label: "Intensity",
                            description: Some("Intensity of the noise."),
                            kind: SettingKind::Percentage { logarithmic: true, default_value: default_settings.luma_noise.settings.intensity },
                            id: setting_id::LUMA_NOISE_INTENSITY
                        },
                        SettingDescriptor {
                            label: "Frequency",
                            description: Some("Base wavelength, in pixels, of the noise."),
                            kind: SettingKind::FloatRange { range: 0.0..=1.0, logarithmic: false, default_value: default_settings.luma_noise.settings.frequency },
                            id: setting_id::LUMA_NOISE_FREQUENCY
                        },
                        SettingDescriptor {
                            label: "Detail",
                            description: Some("Octaves of noise."),
                            kind: SettingKind::IntRange { range: 1..=5, default_value: default_settings.luma_noise.settings.detail as i32 },
                            id: setting_id::LUMA_NOISE_DETAIL
                        },
                    ],
                    default_value: true,
                },
                id: setting_id::LUMA_NOISE,
            },
            SettingDescriptor {
                label: "Chroma noise",
                description: Some("Noise applied to the chrominance (color) signal."),
                kind: SettingKind::Group {
                    children: vec![
                        SettingDescriptor {
                            label: "Intensity",
                            description: Some("Intensity of the noise."),
                            kind: SettingKind::Percentage { logarithmic: true, default_value: default_settings.chroma_noise.settings.intensity },
                            id: setting_id::CHROMA_NOISE_INTENSITY
                        },
                        SettingDescriptor {
                            label: "Frequency",
                            description: Some("Base wavelength, in pixels, of the noise."),
                            kind: SettingKind::FloatRange { range: 0.0..=0.5, logarithmic: false, default_value: default_settings.chroma_noise.settings.frequency },
                            id: setting_id::CHROMA_NOISE_FREQUENCY
                        },
                        SettingDescriptor {
                            label: "Detail",
                            description: Some("Octaves of noise."),
                            kind: SettingKind::IntRange { range: 1..=5, default_value: default_settings.chroma_noise.settings.detail as i32 },
                            id: setting_id::CHROMA_NOISE_DETAIL
                        },
                    ],
                    default_value: true,
                },
                id: setting_id::CHROMA_NOISE,
            },
            SettingDescriptor {
                label: "Chroma phase error",
                description: Some("Phase error for the chrominance (color) signal."),
                kind: SettingKind::Percentage {
                    logarithmic: false,
                    default_value: default_settings.chroma_phase_error,
                },
                id: setting_id::CHROMA_PHASE_ERROR,
            },
            SettingDescriptor {
                label: "Chroma phase noise",
                description: Some("Noise applied per-scanline to the phase of the chrominance (color) signal."),
                kind: SettingKind::Percentage {
                    logarithmic: true,
                    default_value: default_settings.chroma_phase_noise_intensity,
                },
                id: setting_id::CHROMA_PHASE_NOISE_INTENSITY,
            },
            SettingDescriptor {
                label: "Chroma delay (horizontal)",
                description: Some("Horizontal offset of the chrominance (color) signal."),
                kind: SettingKind::FloatRange {
                    range: -40.0..=40.0,
                    logarithmic: false,
                    default_value: default_settings.chroma_delay_horizontal,
                },
                id: setting_id::CHROMA_DELAY_HORIZONTAL,
            },
            SettingDescriptor {
                label: "Chroma delay (vertical)",
                description: Some("Vertical offset of the chrominance (color) signal. Usually increases with VHS generation loss."),
                kind: SettingKind::IntRange {
                    range: -20..=20,
                    default_value: default_settings.chroma_delay_vertical,
                },
                id: setting_id::CHROMA_DELAY_VERTICAL,
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
                                        index: VHSTapeSpeed::SP as u32,
                                    },
                                    MenuItem {
                                        label: "LP (Long Play)",
                                        description: None,
                                        index: VHSTapeSpeed::LP as u32,
                                    },
                                    MenuItem {
                                        label: "EP (Extended Play)",
                                        description: None,
                                        index: VHSTapeSpeed::EP as u32,
                                    },
                                    MenuItem {
                                        label: "None",
                                        description: None,
                                        index: 0,
                                    },
                                ],
                                default_value: default_settings.vhs_settings.settings.tape_speed as u32,
                            },
                            id: setting_id::VHS_TAPE_SPEED
                        },
                        SettingDescriptor {
                            label: "Chroma loss",
                            description: Some("Chance that the chrominance (color) signal is completely lost in each scanline."),
                            kind: SettingKind::Percentage { logarithmic: true, default_value: default_settings.vhs_settings.settings.chroma_loss },
                            id: setting_id::VHS_CHROMA_LOSS
                        },
                        SettingDescriptor {
                            label: "Sharpen",
                            description: Some("Sharpening of the image, as done by some VHS decks."),
                            kind: SettingKind::Group { children: vec![
                                SettingDescriptor {
                                    label: "Intensity",
                                    description: Some("Amount of sharpening to apply."),
                                    kind: SettingKind::FloatRange { range: 0.0..=5.0, logarithmic: false, default_value: default_settings.vhs_settings.settings.sharpen.settings.intensity },
                                    id: setting_id::VHS_SHARPEN_INTENSITY
                                },
                                SettingDescriptor {
                                    label: "Frequency",
                                    description: Some("Frequency / radius of the sharpening, relative to the tape speed's cutoff frequency."),
                                    kind: SettingKind::FloatRange { range: 0.5..=4.0, logarithmic: false, default_value: default_settings.vhs_settings.settings.sharpen.settings.frequency },
                                    id: setting_id::VHS_SHARPEN_FREQUENCY
                                }
                            ], default_value: true },
                            id: setting_id::VHS_SHARPEN_ENABLED
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
                                        id: setting_id::VHS_EDGE_WAVE_INTENSITY
                                    },
                                    SettingDescriptor {
                                        label: "Speed",
                                        description: Some("Speed at which the horizontal waving occurs."),
                                        kind: SettingKind::FloatRange { range: 0.0..=10.0, logarithmic: false, default_value: default_settings.vhs_settings.settings.edge_wave.settings.speed },
                                        id: setting_id::VHS_EDGE_WAVE_SPEED
                                    },
                                    SettingDescriptor {
                                        label: "Frequency",
                                        description: Some("Base wavelength for the horizontal waving."),
                                        kind: SettingKind::FloatRange { range: 0.0..=0.5, logarithmic: false, default_value: default_settings.vhs_settings.settings.edge_wave.settings.frequency },
                                        id: setting_id::VHS_EDGE_WAVE_FREQUENCY
                                    },
                                    SettingDescriptor {
                                        label: "Detail",
                                        description: Some("Octaves of noise for the waves."),
                                        kind: SettingKind::IntRange { range: 1..=5, default_value: default_settings.vhs_settings.settings.edge_wave.settings.detail },
                                        id: setting_id::VHS_EDGE_WAVE_DETAIL
                                    },
                                ],
                                default_value: true
                            },
                            id: setting_id::VHS_EDGE_WAVE_ENABLED
                        }
                    ],
                    default_value: true,
                },
                id: setting_id::VHS_SETTINGS,
            },
            SettingDescriptor {
                label: "Vertically blend chroma",
                description: Some("Vertically blend each scanline's chrominance with the scanline above it."),
                kind: SettingKind::Boolean { default_value: default_settings.chroma_vert_blend },
                id: setting_id::CHROMA_VERT_BLEND
            },
            SettingDescriptor {
                label: "Chroma low-pass out",
                description: Some("Apply a low-pass filter to the output chroma signal."),
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "Full",
                            description: Some("Full-intensity low-pass filter."),
                            index: ChromaLowpass::Full as u32,
                        },
                        MenuItem {
                            label: "Light",
                            description: Some("Less intense low-pass filter."),
                            index: ChromaLowpass::Light as u32,
                        },
                        MenuItem {
                            label: "None",
                            description: Some("No low-pass filter."),
                            index: ChromaLowpass::None as u32,
                        },
                    ],
                    default_value: default_settings.chroma_lowpass_out as u32,
                },
                id: setting_id::CHROMA_LOWPASS_OUT,
            },
            SettingDescriptor {
                label: "Bandwidth scale",
                description: Some("Horizontally scale the effect by this amount. For 480p video, leave this at 1.0 for the most physically-accurate result."),
                kind: SettingKind::FloatRange { range: 0.125..=8.0, logarithmic: false, default_value: default_settings.bandwidth_scale },
                id: setting_id::BANDWIDTH_SCALE,
            },
        ];

        SettingsList {
            settings: v.into_boxed_slice(),
        }
    }

    pub fn from_json(&self, json: &str) -> Result<NtscEffectFullSettings, ParseSettingsError> {
        let parsed = json.parse::<JsonValue>()?;

        let parsed_map = parsed.get::<HashMap<_, _>>().ok_or_else(|| {
            ParseSettingsError::InvalidSettingType {
                key: "<root>".to_string(),
                expected: "object",
            }
        })?;

        if parsed_map.contains_key("_composite_preemphasis") {
            return Self::from_ntscqt_json(parsed_map);
        }

        let version = parsed_map
            .get_and_expect::<f64>("version")?
            .ok_or(ParseSettingsError::MissingField { field: "version" })?;
        if version != 1.0 {
            return Err(ParseSettingsError::UnsupportedVersion { version });
        }

        let mut dst_settings = Default::default();
        Self::settings_from_json(parsed_map, &self.settings, &mut dst_settings)?;

        Ok(dst_settings)
    }

    pub fn from_ntscqt_json(
        json: &HashMap<String, JsonValue>,
    ) -> Result<NtscEffectFullSettings, ParseSettingsError> {
        let mut settings = NtscEffectFullSettings::default();
        settings.use_field = UseField::Upper;
        settings.filter_type = FilterType::ConstantK;
        settings.input_luma_filter = LumaLowpass::Box;
        settings.chroma_lowpass_in = {
            let in_lowpass = json
                .get_and_expect::<bool>("_composite_in_chroma_lowpass")?
                .unwrap_or_default();
            if in_lowpass {
                ChromaLowpass::Full
            } else {
                ChromaLowpass::None
            }
        };
        settings.chroma_demodulation = ChromaDemodulationFilter::Box;
        settings.luma_smear = 0.0;
        settings.composite_sharpening = json
            .get_and_expect::<f64>("_composite_preemphasis")?
            .unwrap_or_default() as f32;
        settings.video_scanline_phase_shift = match json
            .get_and_expect::<f64>("_video_scanline_phase_shift")?
            .unwrap_or_default()
        {
            90.0 => PhaseShift::Degrees90,
            180.0 => PhaseShift::Degrees180,
            270.0 => PhaseShift::Degrees270,
            _ => PhaseShift::Degrees0,
        };
        settings.video_scanline_phase_shift_offset = json
            .get_and_expect::<f64>("_video_scanline_phase_shift_offset")?
            .unwrap_or_default() as i32;
        settings.head_switching = SettingsBlock {
            enabled: json
                .get_and_expect::<bool>("_vhs_head_switching")?
                .unwrap_or_default(),
            settings: HeadSwitchingSettingsFullSettings {
                height: 6,
                offset: 0,
                horiz_shift: 6.0,
                mid_line: SettingsBlock {
                    enabled: false,
                    settings: Default::default(),
                },
            },
        };
        settings.tracking_noise.enabled = false;
        settings.composite_noise = SettingsBlock {
            enabled: true,
            settings: FbmNoiseSettings {
                frequency: 0.1,
                intensity: json
                    .get_and_expect::<f64>("_video_noise")?
                    .unwrap_or_default() as f32
                    / 50000.0,
                detail: 3,
            },
        };
        settings.ringing = SettingsBlock {
            enabled: true,
            settings: {
                let ringing2 = json
                    .get_and_expect::<bool>("_enable_ringing2")?
                    .unwrap_or_default();
                if ringing2 {
                    let power = json
                        .get_and_expect::<f64>("_ringing_power")?
                        .unwrap_or_default() as f32;
                    let shift = json
                        .get_and_expect::<f64>("_ringing_shift")?
                        .unwrap_or_default() as f32;

                    let frequency = ((shift * 0.75) + 0.25).clamp(0.0, 1.0);

                    RingingSettings {
                        frequency,
                        power: 6.0,
                        intensity: power * 0.1,
                    }
                } else {
                    RingingSettings {
                        frequency: json.get_and_expect::<f64>("_ringing")?.unwrap_or_default()
                            as f32
                            / 3.0,
                        power: 5.0,
                        intensity: 2.0,
                    }
                }
            },
        };
        settings.luma_noise.enabled = false;
        settings.chroma_noise = SettingsBlock {
            enabled: true,
            settings: FbmNoiseSettings {
                frequency: 0.2,
                intensity: (json
                    .get_and_expect::<f64>("_video_chroma_noise")?
                    .unwrap_or_default()
                    * (0.4 / 16384.0)) as f32,
                detail: 1,
            },
        };
        settings.snow_intensity = 0.0;
        settings.chroma_phase_noise_intensity = (json
            .get_and_expect::<f64>("_video_chroma_phase_noise")?
            .unwrap_or_default()
            * 0.005) as f32;
        settings.chroma_phase_error = 0.0;
        settings.chroma_delay_horizontal = json
            .get_and_expect::<f64>("_color_bleed_horiz")?
            .unwrap_or_default() as f32;
        settings.chroma_delay_vertical = json
            .get_and_expect::<f64>("_color_bleed_vert")?
            .unwrap_or_default() as i32;
        settings.vhs_settings = SettingsBlock {
            enabled: json
                .get_and_expect::<bool>("_emulating_vhs")?
                .unwrap_or_default(),
            settings: VHSSettingsFullSettings {
                tape_speed: match json.get_and_expect::<f64>("_output_vhs_tape_speed")? {
                    Some(1.0) => VHSTapeSpeed::LP,
                    Some(2.0) => VHSTapeSpeed::EP,
                    _ => VHSTapeSpeed::SP,
                },
                chroma_loss: (json
                    .get_and_expect::<f64>("_video_chroma_loss")?
                    .unwrap_or_default()
                    / 100000.0) as f32,
                sharpen: SettingsBlock {
                    enabled: true,
                    settings: VHSSharpenSettings {
                        intensity: json
                            .get_and_expect::<f64>("_vhs_out_sharpen")?
                            .unwrap_or_default() as f32,
                        frequency: 1.0,
                    },
                },
                edge_wave: SettingsBlock {
                    enabled: true,
                    settings: VHSEdgeWaveSettings {
                        intensity: (json
                            .get_and_expect::<f64>("_vhs_edge_wave")?
                            .unwrap_or_default()
                            * 0.5) as f32,
                        speed: 10.0,
                        frequency: 0.125,
                        detail: 3,
                    },
                },
            },
        };
        settings.chroma_vert_blend = json
            .get_and_expect::<bool>("_vhs_chroma_vert_blend")?
            .unwrap_or_default();
        settings.chroma_lowpass_out = {
            let chroma_lowpass_out = json
                .get_and_expect::<bool>("_composite_out_chroma_lowpass")?
                .unwrap_or_default();
            let chroma_lowpass_out_lite = json
                .get_and_expect::<bool>("_composite_out_chroma_lowpass_lite")?
                .unwrap_or_default();
            match (chroma_lowpass_out, chroma_lowpass_out_lite) {
                (true, false) => ChromaLowpass::Full,
                (true, true) => ChromaLowpass::Light,
                _ => ChromaLowpass::None,
            }
        };
        settings.bandwidth_scale = 1.0;

        Ok(settings)
    }
}
