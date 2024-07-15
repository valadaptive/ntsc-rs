use macros::FullSettings;
use num_traits::ToPrimitive;

use crate::{
    impl_settings_for,
    ntsc::{NtscEffectFullSettings, VHSEdgeWaveSettings},
};

use super::{
    standard::{
        ChromaDemodulationFilter, ChromaLowpass, FbmNoiseSettings, FilterType,
        HeadSwitchingMidLineSettings, HeadSwitchingSettingsFullSettings, LumaLowpass, PhaseShift,
        RingingSettings, TrackingNoiseSettings, UseField, VHSSettingsFullSettings,
        VHSSharpenSettings, VHSTapeSpeed,
    },
    MenuItem, SettingDescriptor, SettingKind, SettingsBlock, SettingsList,
};

#[derive(Clone, Debug, PartialEq)]
pub struct EzTrackingNoiseSettings {
    height: u32,
    intensity: f32,
}

impl Default for EzTrackingNoiseSettings {
    fn default() -> Self {
        Self {
            height: 12,
            intensity: 0.5,
        }
    }
}

#[derive(FullSettings, Clone, Debug, PartialEq)]
pub struct EzVHSSettings {
    tape_speed: VHSTapeSpeed,
    chroma_loss: f32,
    sharpen: f32,
    edge_wave: f32,
    head_switching: f32,
    #[settings_block]
    tracking_noise: Option<EzTrackingNoiseSettings>,
}

impl Default for EzVHSSettings {
    fn default() -> Self {
        Self {
            tape_speed: VHSTapeSpeed::SP,
            chroma_loss: 0.0,
            sharpen: 0.0,
            edge_wave: 0.5,
            head_switching: 6.0,
            tracking_noise: Some(EzTrackingNoiseSettings::default()),
        }
    }
}

#[derive(FullSettings, Clone, Debug, PartialEq)]
pub struct EasyMode {
    random_seed: i32,
    use_field: UseField,
    filter_type: FilterType,
    saturation: f32,
    snow: f32,
    chroma_demodulation_filter: ChromaDemodulationFilter,
    luma_smear: f32,
    ringing: f32,
    #[settings_block]
    luma_noise: Option<FbmNoiseSettings>,
    #[settings_block]
    chroma_noise: Option<FbmNoiseSettings>,
    chroma_phase_noise: f32,
    #[settings_block(nested)]
    vhs_settings: Option<EzVHSSettings>,
}

impl Default for EasyMode {
    fn default() -> Self {
        Self {
            random_seed: 0,
            use_field: UseField::InterleavedUpper,
            filter_type: FilterType::Butterworth,
            saturation: 0.25,
            snow: 0.0005,
            chroma_demodulation_filter: ChromaDemodulationFilter::Notch,
            luma_smear: 0.2,
            ringing: 0.5,
            luma_noise: Some(FbmNoiseSettings {
                frequency: 0.1,
                intensity: 0.01,
                detail: 2,
            }),
            chroma_noise: Some(FbmNoiseSettings {
                frequency: 0.05,
                intensity: 0.1,
                detail: 2,
            }),
            chroma_phase_noise: 0.01,
            vhs_settings: Some(EzVHSSettings::default()),
        }
    }
}

#[rustfmt::skip]
pub mod setting_id {
    use crate::settings::SettingID;
    use super::EasyModeFullSettings;
    type EasySettingID = SettingID<EasyModeFullSettings>;

    pub const RANDOM_SEED: EasySettingID = SettingID::new(4096 | 0, "ez_random_seed");
    pub const USE_FIELD: EasySettingID = SettingID::new(4096 | 1, "ez_use_field");
    pub const FILTER_TYPE: EasySettingID = SettingID::new(4096 | 2, "ez_filter_type");
    pub const SATURATION: EasySettingID = SettingID::new(4096 | 3, "ez_saturation");
    pub const SNOW: EasySettingID = SettingID::new(4096 | 4, "ez_snow");
    pub const CHROMA_DEMODULATION_FILTER: EasySettingID = SettingID::new(4096 | 5, "ez_chroma_demodulation_filter");
    pub const LUMA_SMEAR: EasySettingID = SettingID::new(4096 | 6, "ez_luma_smear");
    pub const RINGING: EasySettingID = SettingID::new(4096 | 7, "ez_ringing");
    pub const VHS_SETTINGS: EasySettingID = SettingID::new(4096 | 8, "ez_vhs_settings");
    pub const VHS_HEAD_SWITCHING: EasySettingID = SettingID::new(4096 | 9, "ez_vhs_head_switching");
    pub const TRACKING_NOISE_ENABLED: EasySettingID = SettingID::new(4096 | 10, "ez_tracking_noise_enabled");
    pub const TRACKING_NOISE_HEIGHT: EasySettingID = SettingID::new(4096 | 11, "ez_tracking_noise_height");
    pub const TRACKING_NOISE_INTENSITY: EasySettingID = SettingID::new(4096 | 12, "ez_tracking_noise_intensity");

    pub const LUMA_NOISE_ENABLED: EasySettingID = SettingID::new(4096 | 13, "ez_luma_noise_enabled");
    pub const LUMA_NOISE_FREQUENCY: EasySettingID = SettingID::new(4096 | 14, "ez_luma_noise_frequency");
    pub const LUMA_NOISE_INTENSITY: EasySettingID = SettingID::new(4096 | 15, "ez_luma_noise_intensity");
    pub const LUMA_NOISE_DETAIL: EasySettingID = SettingID::new(4096 | 16, "ez_luma_noise_detail");

    pub const CHROMA_NOISE_ENABLED: EasySettingID = SettingID::new(4096 | 17, "ez_chroma_noise_enabled");
    pub const CHROMA_NOISE_FREQUENCY: EasySettingID = SettingID::new(4096 | 18, "ez_chroma_noise_frequency");
    pub const CHROMA_NOISE_INTENSITY: EasySettingID = SettingID::new(4096 | 19, "ez_chroma_noise_intensity");
    pub const CHROMA_NOISE_DETAIL: EasySettingID = SettingID::new(4096 | 20, "ez_chroma_noise_detail");

    pub const CHROMA_PHASE_NOISE: EasySettingID = SettingID::new(4096 | 21, "ez_chroma_phase_noise");
    pub const VHS_TAPE_SPEED: EasySettingID = SettingID::new(4096 | 22, "ez_vhs_tape_speed");
    pub const VHS_CHROMA_LOSS: EasySettingID = SettingID::new(4096 | 22, "ez_vhs_chroma_loss");
    pub const VHS_SHARPEN: EasySettingID = SettingID::new(4096 | 23, "ez_vhs_sharpen");
    pub const VHS_EDGE_WAVE: EasySettingID = SettingID::new(4096 | 24, "ez_vhs_edge_wave");
}

impl_settings_for!(
    EasyModeFullSettings,
    (setting_id::RANDOM_SEED, random_seed),
    (setting_id::USE_FIELD, use_field, IS_AN_ENUM),
    (setting_id::FILTER_TYPE, filter_type, IS_AN_ENUM),
    (setting_id::SATURATION, saturation),
    (setting_id::SNOW, snow),
    (
        setting_id::CHROMA_DEMODULATION_FILTER,
        chroma_demodulation_filter,
        IS_AN_ENUM
    ),
    (setting_id::LUMA_SMEAR, luma_smear),
    (setting_id::RINGING, ringing),
    (setting_id::VHS_SETTINGS, vhs_settings.enabled),
    (
        setting_id::VHS_HEAD_SWITCHING,
        vhs_settings.settings.head_switching
    ),
    (
        setting_id::TRACKING_NOISE_ENABLED,
        vhs_settings.settings.tracking_noise.enabled
    ),
    (
        setting_id::TRACKING_NOISE_HEIGHT,
        vhs_settings.settings.tracking_noise.settings.height
    ),
    (
        setting_id::TRACKING_NOISE_INTENSITY,
        vhs_settings.settings.tracking_noise.settings.intensity
    ),
    (setting_id::LUMA_NOISE_ENABLED, luma_noise.enabled),
    (
        setting_id::LUMA_NOISE_FREQUENCY,
        luma_noise.settings.frequency
    ),
    (
        setting_id::LUMA_NOISE_INTENSITY,
        luma_noise.settings.intensity
    ),
    (setting_id::LUMA_NOISE_DETAIL, luma_noise.settings.detail),
    (setting_id::CHROMA_NOISE_ENABLED, chroma_noise.enabled),
    (
        setting_id::CHROMA_NOISE_FREQUENCY,
        chroma_noise.settings.frequency
    ),
    (
        setting_id::CHROMA_NOISE_INTENSITY,
        chroma_noise.settings.intensity
    ),
    (
        setting_id::CHROMA_NOISE_DETAIL,
        chroma_noise.settings.detail
    ),
    (setting_id::CHROMA_PHASE_NOISE, chroma_phase_noise),
    (
        setting_id::VHS_TAPE_SPEED,
        vhs_settings.settings.tape_speed,
        IS_AN_ENUM
    ),
    (
        setting_id::VHS_CHROMA_LOSS,
        vhs_settings.settings.chroma_loss
    ),
    (setting_id::VHS_SHARPEN, vhs_settings.settings.sharpen),
    (setting_id::VHS_EDGE_WAVE, vhs_settings.settings.edge_wave)
);

impl SettingsList<EasyModeFullSettings> {
    /// Construct a list of all the effect settings. This isn't meant to be mutated--you should just create one instance
    /// of this to use for your entire application/plugin.
    pub fn new() -> Self {
        let default_settings = EasyModeFullSettings::default();

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
                id: setting_id::FILTER_TYPE,
            },
            SettingDescriptor {
                label: "Saturation",
                description: Some("Boost high frequencies in the NTSC signal, sharpening the image and intensifying colors."),
                kind: SettingKind::FloatRange {
                    range: -1.0..=2.0,
                    logarithmic: false,
                    default_value: default_settings.saturation,
                },
                id: setting_id::SATURATION,
            },
            SettingDescriptor {
                label: "Snow",
                description: Some("Frequency of random speckles in the image."),
                kind: SettingKind::FloatRange {
                    range: 0.0..=100.0,
                    logarithmic: true,
                    default_value: default_settings.snow,
                },
                id: setting_id::SNOW,
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
                    default_value: default_settings.chroma_demodulation_filter.to_u32().unwrap(),
                },
                id: setting_id::CHROMA_DEMODULATION_FILTER,
            },
            SettingDescriptor {
                label: "Luma smear",
                description: None,
                kind: SettingKind::FloatRange { range: 0.0..=1.0, logarithmic: false, default_value: default_settings.luma_smear },
                id: setting_id::LUMA_SMEAR
            },
            SettingDescriptor {
                label: "Ringing",
                description: Some("Additional ringing artifacts, simulated with a notch filter."),
                kind: SettingKind::Percentage { logarithmic: false, default_value: default_settings.ringing },
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
                id: setting_id::LUMA_NOISE_ENABLED,
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
                id: setting_id::CHROMA_NOISE_ENABLED,
            },
            SettingDescriptor {
                label: "Chroma phase noise",
                description: Some("Noise applied per-scanline to the phase of the chrominance signal."),
                kind: SettingKind::Percentage {
                    logarithmic: false,
                    default_value: default_settings.chroma_phase_noise,
                },
                id: setting_id::CHROMA_PHASE_NOISE,
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
                                default_value: default_settings.vhs_settings.settings.tape_speed.to_u32().unwrap(),
                            },
                            id: setting_id::VHS_TAPE_SPEED
                        },
                        SettingDescriptor {
                            label: "Head switching",
                            description: Some("Emulate VHS head-switching artifacts at the bottom of the image."),
                            kind: SettingKind::FloatRange { range: 0.0..=24.0, default_value: default_settings.vhs_settings.settings.head_switching, logarithmic: false },
                            id: setting_id::VHS_HEAD_SWITCHING,
                        },
                        SettingDescriptor {
                            label: "Tracking noise",
                            description: Some("Emulate noise from VHS tracking error."),
                            kind: SettingKind::Group {
                                children: vec![
                                    SettingDescriptor {
                                        label: "Height",
                                        description: None,
                                        kind: SettingKind::IntRange { range: 0..=120, default_value: default_settings.vhs_settings.settings.tracking_noise.settings.height as i32 },
                                        id: setting_id::TRACKING_NOISE_HEIGHT
                                    },
                                    SettingDescriptor {
                                        label: "Intensity",
                                        description: None,
                                        kind: SettingKind::Percentage { logarithmic: false, default_value: default_settings.vhs_settings.settings.tracking_noise.settings.intensity },
                                        id: setting_id::TRACKING_NOISE_INTENSITY
                                    },
                                ],
                                default_value: true,
                            },
                            id: setting_id::TRACKING_NOISE_ENABLED,
                        },
                        SettingDescriptor {
                            label: "Chroma loss",
                            description: Some("Chance that the chrominance signal is completely lost in each scanline."),
                            kind: SettingKind::Percentage { logarithmic: true, default_value: default_settings.vhs_settings.settings.chroma_loss },
                            id: setting_id::VHS_CHROMA_LOSS
                        },
                        SettingDescriptor {
                            label: "Sharpen",
                            description: Some("Sharpening of the image, as done by some VHS decks."),
                            kind: SettingKind::Percentage { logarithmic: false, default_value: default_settings.vhs_settings.settings.sharpen },
                            id: setting_id::VHS_SHARPEN
                        },
                        SettingDescriptor {
                            label: "Edge wave",
                            description: Some("Horizontal waving of the image."),
                            kind: SettingKind::FloatRange { range: 0.0..=20.0, logarithmic: false, default_value: default_settings.vhs_settings.settings.edge_wave },
                            id: setting_id::VHS_EDGE_WAVE
                        }
                    ],
                    default_value: true,
                },
                id: setting_id::VHS_SETTINGS,
            },
        ];

        SettingsList {
            settings: v.into_boxed_slice(),
        }
    }
}

impl From<&EasyModeFullSettings> for NtscEffectFullSettings {
    fn from(easy_settings: &EasyModeFullSettings) -> Self {
        Self {
            random_seed: easy_settings.random_seed,
            use_field: easy_settings.use_field,
            filter_type: easy_settings.filter_type,
            input_luma_filter: LumaLowpass::Notch,
            chroma_lowpass_in: ChromaLowpass::None,
            chroma_demodulation: easy_settings.chroma_demodulation_filter,
            luma_smear: easy_settings.luma_smear,
            composite_preemphasis: easy_settings.saturation,
            video_scanline_phase_shift: PhaseShift::Degrees180,
            video_scanline_phase_shift_offset: 0,
            head_switching: SettingsBlock {
                enabled: easy_settings.vhs_settings.enabled && easy_settings.vhs_settings.settings.head_switching > 0.0,
                settings: HeadSwitchingSettingsFullSettings {
                    height: (easy_settings.vhs_settings.settings.head_switching * 1.25).round() as u32,
                    offset: (easy_settings.vhs_settings.settings.head_switching * 0.25).round() as u32,
                    horiz_shift: easy_settings.vhs_settings.settings.head_switching * 8.0,
                    mid_line: SettingsBlock {
                        enabled: true,
                        settings: HeadSwitchingMidLineSettings {
                            position: 0.8,
                            jitter: 0.1,
                        },
                    },
                },
            },
            tracking_noise: SettingsBlock {
                enabled: easy_settings.vhs_settings.enabled && easy_settings.vhs_settings.settings.tracking_noise.enabled,
                settings: TrackingNoiseSettings {
                    height: easy_settings
                        .vhs_settings
                        .settings
                        .tracking_noise
                        .settings
                        .height,
                    wave_intensity: easy_settings
                        .vhs_settings
                        .settings
                        .tracking_noise
                        .settings
                        .intensity * 5.0,
                    snow_intensity: easy_settings
                        .vhs_settings
                        .settings
                        .tracking_noise
                        .settings
                        .intensity * 0.5,
                    snow_anisotropy: 0.25,
                    noise_intensity: easy_settings
                        .vhs_settings
                        .settings
                        .tracking_noise
                        .settings
                        .intensity * 0.5,
                },
            },
            composite_noise: SettingsBlock {
                enabled: false,
                settings: NtscEffectFullSettings::default().composite_noise.settings,
            },
            ringing: SettingsBlock {
                enabled: easy_settings.ringing > 0.0,
                settings: RingingSettings {
                    frequency: 0.4,
                    power: 2.0,
                    intensity: easy_settings.ringing,
                },
            },
            luma_noise: easy_settings.luma_noise.clone(),
            chroma_noise: easy_settings.chroma_noise.clone(),
            snow_intensity: easy_settings.snow,
            snow_anisotropy: 0.5,
            chroma_phase_noise_intensity: easy_settings.chroma_phase_noise * 0.5,
            chroma_phase_error: 0.0,
            chroma_delay_horizontal: 0.0,
            chroma_delay_vertical: 0,
            vhs_settings: SettingsBlock {
                enabled: easy_settings.vhs_settings.enabled,
                settings: VHSSettingsFullSettings {
                    tape_speed: easy_settings.vhs_settings.settings.tape_speed,
                    chroma_loss: easy_settings.vhs_settings.settings.chroma_loss,
                    sharpen: SettingsBlock {
                        enabled: easy_settings.vhs_settings.settings.sharpen > 0.0,
                        settings: VHSSharpenSettings {
                            intensity: easy_settings.vhs_settings.settings.sharpen,
                            frequency: 0.5,
                        },
                    },
                    edge_wave: SettingsBlock {
                        enabled: easy_settings.vhs_settings.settings.edge_wave > 0.0,
                        settings: VHSEdgeWaveSettings {
                            intensity: easy_settings.vhs_settings.settings.edge_wave,
                            speed: 7.0,
                            frequency: 0.1,
                            detail: 2,
                        },
                    },
                },
            },
            chroma_vert_blend: !easy_settings.vhs_settings.enabled,
            chroma_lowpass_out: ChromaLowpass::Full,
            bandwidth_scale: 1.0,
        }
    }
}
