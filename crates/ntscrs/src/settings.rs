use std::{any::Any, ops::RangeInclusive};

use macros::FullSettings;
use crate::{FromPrimitive, ToPrimitive};

/// This is used to dynamically inform API consumers of the settings that can be passed to ntsc-rs. This lets various
/// UIs and effect plugins to query this set of settings and display them in their preferred format without having to
/// duplicate a bunch of code.

// TODO: replace with a bunch of metaprogramming macro magic?

#[derive(Clone, Copy, PartialEq, Eq, FromPrimitive, ToPrimitive)]
pub enum UseField {
    Alternating = 0,
    Upper,
    Lower,
    Both,
}

#[derive(Clone, Copy, PartialEq, Eq, FromPrimitive, ToPrimitive)]
pub enum PhaseShift {
    Degrees0,
    Degrees90,
    Degrees180,
    Degrees270,
}

#[derive(Clone, Copy, PartialEq, Eq, FromPrimitive, ToPrimitive)]
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

#[derive(Clone, PartialEq)]
pub struct VHSSettings {
    pub tape_speed: Option<VHSTapeSpeed>,
    pub chroma_vert_blend: bool,
    pub chroma_loss: f32,
    pub sharpen: f32,
    pub edge_wave: f32,
    pub edge_wave_speed: f32,
}

impl Default for VHSSettings {
    fn default() -> Self {
        Self {
            tape_speed: Some(VHSTapeSpeed::LP),
            chroma_vert_blend: true,
            chroma_loss: 0.0,
            sharpen: 1.0,
            edge_wave: 1.0,
            edge_wave_speed: 4.0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, FromPrimitive, ToPrimitive)]
pub enum ChromaLowpass {
    None,
    Light,
    Full,
}

#[derive(Clone, PartialEq)]
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

#[derive(Clone, PartialEq)]
pub struct TrackingNoiseSettings {
    pub height: u32,
    pub wave_intensity: f32,
    pub snow_intensity: f32,
    pub noise_intensity: f32,
}

impl Default for TrackingNoiseSettings {
    fn default() -> Self {
        Self {
            height: 24,
            wave_intensity: 5.0,
            snow_intensity: 0.005,
            noise_intensity: 0.005,
        }
    }
}

#[derive(Clone, PartialEq)]
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
            settings: opt.unwrap_or_else(T::default),
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

#[derive(FullSettings)]
pub struct NtscEffect {
    pub use_field: UseField,
    pub chroma_lowpass_in: ChromaLowpass,
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
    pub chroma_noise_intensity: f32,
    pub snow_intensity: f32,
    pub chroma_phase_noise_intensity: f32,
    pub chroma_delay: (f32, i32),
    #[settings_block]
    pub vhs_settings: Option<VHSSettings>,
    pub chroma_lowpass_out: ChromaLowpass,
}

impl Default for NtscEffect {
    fn default() -> Self {
        Self {
            use_field: UseField::Alternating,
            chroma_lowpass_in: ChromaLowpass::Full,
            chroma_lowpass_out: ChromaLowpass::Full,
            composite_preemphasis: 1.0,
            video_scanline_phase_shift: PhaseShift::Degrees180,
            video_scanline_phase_shift_offset: 0,
            head_switching: Some(HeadSwitchingSettings::default()),
            tracking_noise: Some(TrackingNoiseSettings::default()),
            ringing: Some(RingingSettings::default()),
            snow_intensity: 0.00001,
            composite_noise_intensity: 0.01,
            chroma_noise_intensity: 0.1,
            chroma_phase_noise_intensity: 0.001,
            chroma_delay: (0.0, 0),
            vhs_settings: Some(VHSSettings::default()),
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

#[allow(non_camel_case_types)]
#[derive(Debug, FromPrimitive, ToPrimitive, Clone, Copy)]
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
    VHS_CHROMA_VERT_BLEND,
    VHS_CHROMA_LOSS,
    VHS_SHARPEN,
    VHS_EDGE_WAVE,
    VHS_EDGE_WAVE_SPEED,

    USE_FIELD,
    TRACKING_NOISE_NOISE_INTENSITY,
}

impl SettingID {
    pub fn set_field_enum(&self, settings: &mut NtscEffectFullSettings, value: u32) {
        // We have to handle each enum manually since FromPrimitive isn't object-safe
        match self {
            SettingID::CHROMA_LOWPASS_IN => {
                settings.chroma_lowpass_in = ChromaLowpass::from_u32(value).unwrap()
            }
            SettingID::VIDEO_SCANLINE_PHASE_SHIFT => {
                settings.video_scanline_phase_shift = PhaseShift::from_u32(value).unwrap()
            }
            SettingID::CHROMA_LOWPASS_OUT => {
                settings.chroma_lowpass_out = ChromaLowpass::from_u32(value).unwrap()
            }
            SettingID::VHS_TAPE_SPEED => {
                settings.vhs_settings.settings.tape_speed = VHSTapeSpeed::from_u32(value)
            }
            SettingID::USE_FIELD => {
                settings.use_field = UseField::from_u32(value).unwrap()
            }
            _ => {}
        }
    }

    pub fn get_field_enum(&self, settings: &NtscEffectFullSettings) -> Option<u32> {
        // We have to handle each enum manually since FromPrimitive isn't object-safe
        match self {
            SettingID::CHROMA_LOWPASS_IN => {
                Some(settings.chroma_lowpass_in.to_u32().unwrap())
            }
            SettingID::VIDEO_SCANLINE_PHASE_SHIFT => {
                Some(settings.video_scanline_phase_shift.to_u32().unwrap())
            }
            SettingID::CHROMA_LOWPASS_OUT => {
                Some(settings.chroma_lowpass_out.to_u32().unwrap())
            }
            SettingID::VHS_TAPE_SPEED => {
                Some(match settings.vhs_settings.settings.tape_speed {
                    Some(tape_speed) => tape_speed.to_u32().unwrap(),
                    None => 0
                })
            }
            SettingID::USE_FIELD => {
                Some(settings.use_field.to_u32().unwrap())
            }
            _ => None
        }
    }

    pub fn get_field_ref<'a, T: 'static>(&self, settings: &'a mut NtscEffectFullSettings) -> Option<&'a mut T> {
        let field_ref: &mut dyn Any = match self {
            SettingID::USE_FIELD => &mut settings.use_field,
            SettingID::CHROMA_LOWPASS_IN => &mut settings.chroma_lowpass_in,
            SettingID::COMPOSITE_PREEMPHASIS => &mut settings.composite_preemphasis,
            SettingID::VIDEO_SCANLINE_PHASE_SHIFT => &mut settings.video_scanline_phase_shift,
            SettingID::VIDEO_SCANLINE_PHASE_SHIFT_OFFSET => {
                &mut settings.video_scanline_phase_shift_offset
            }
            SettingID::COMPOSITE_NOISE_INTENSITY => &mut settings.composite_noise_intensity,
            SettingID::CHROMA_NOISE_INTENSITY => &mut settings.chroma_noise_intensity,
            SettingID::SNOW_INTENSITY => &mut settings.snow_intensity,
            SettingID::CHROMA_PHASE_NOISE_INTENSITY => &mut settings.chroma_phase_noise_intensity,
            SettingID::CHROMA_DELAY_HORIZONTAL => &mut settings.chroma_delay.0,
            SettingID::CHROMA_DELAY_VERTICAL => &mut settings.chroma_delay.1,
            SettingID::CHROMA_LOWPASS_OUT => &mut settings.chroma_lowpass_out,

            SettingID::HEAD_SWITCHING => &mut settings.head_switching.enabled,
            SettingID::HEAD_SWITCHING_HEIGHT => &mut settings.head_switching.settings.height,
            SettingID::HEAD_SWITCHING_OFFSET => &mut settings.head_switching.settings.offset,
            SettingID::HEAD_SWITCHING_HORIZONTAL_SHIFT => &mut settings.head_switching.settings.horiz_shift,

            SettingID::TRACKING_NOISE => &mut settings.tracking_noise.enabled,
            SettingID::TRACKING_NOISE_HEIGHT => &mut settings.tracking_noise.settings.height,
            SettingID::TRACKING_NOISE_WAVE_INTENSITY => &mut settings.tracking_noise.settings.wave_intensity,
            SettingID::TRACKING_NOISE_SNOW_INTENSITY => &mut settings.tracking_noise.settings.snow_intensity,
            SettingID::TRACKING_NOISE_NOISE_INTENSITY => &mut settings.tracking_noise.settings.noise_intensity,

            SettingID::RINGING => &mut settings.ringing.enabled,
            SettingID::RINGING_FREQUENCY => &mut settings.ringing.settings.frequency,
            SettingID::RINGING_POWER => &mut settings.ringing.settings.power,
            SettingID::RINGING_SCALE => &mut settings.ringing.settings.intensity,

            SettingID::VHS_SETTINGS => &mut settings.vhs_settings.enabled,
            SettingID::VHS_TAPE_SPEED => &mut settings.vhs_settings.settings.tape_speed,
            SettingID::VHS_CHROMA_VERT_BLEND => &mut settings.vhs_settings.settings.chroma_vert_blend,
            SettingID::VHS_CHROMA_LOSS => &mut settings.vhs_settings.settings.chroma_loss,
            SettingID::VHS_SHARPEN => &mut settings.vhs_settings.settings.sharpen,
            SettingID::VHS_EDGE_WAVE => &mut settings.vhs_settings.settings.edge_wave,
            SettingID::VHS_EDGE_WAVE_SPEED => &mut settings.vhs_settings.settings.edge_wave_speed,
        };

        field_ref.downcast_mut::<T>()
    }
}

pub struct SettingsList {
    pub settings: Box<[SettingDescriptor]>,
    pub by_id: Box<[Option<Box<[usize]>>]>
}

impl SettingsList {
    fn construct_id_map(settings: &[SettingDescriptor], map: &mut Vec<Option<Box<[usize]>>>, parent_path: &Vec<usize>) {
        for (index, descriptor) in settings.iter().enumerate() {
            let id = descriptor.id as usize;
            if id >= map.len() {
                map.resize(id + 1, None);
            }

            let mut path = parent_path.clone();
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
                label: "Use field",
                description: None,
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "Alternating",
                            description: None,
                            index: UseField::Alternating.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "Upper",
                            description: None,
                            index: UseField::Upper.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "Lower",
                            description: None,
                            index: UseField::Lower.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "Both",
                            description: None,
                            index: UseField::Both.to_u32().unwrap(),
                        },
                    ],
                    default_value: default_settings.use_field.to_u32().unwrap(),
                },
                id: SettingID::USE_FIELD,
            },
            SettingDescriptor {
                label: "Chroma low-pass in",
                description: None,
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "Full",
                            description: None,
                            index: ChromaLowpass::Full.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "Light",
                            description: None,
                            index: ChromaLowpass::Light.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "None",
                            description: None,
                            index: ChromaLowpass::None.to_u32().unwrap(),
                        },
                    ],
                    default_value: default_settings.chroma_lowpass_in.to_u32().unwrap(),
                },
                id: SettingID::CHROMA_LOWPASS_IN,
            },
            SettingDescriptor {
                label: "Composite preemphasis",
                description: None,
                kind: SettingKind::FloatRange {
                    range: 0.0..=2.0,
                    logarithmic: false,
                    default_value: default_settings.composite_preemphasis,
                },
                id: SettingID::COMPOSITE_PREEMPHASIS,
            },
            SettingDescriptor {
                label: "Composite noise",
                description: None,
                kind: SettingKind::Percentage {
                    logarithmic: true,
                    default_value: default_settings.composite_noise_intensity,
                },
                id: SettingID::COMPOSITE_NOISE_INTENSITY,
            },
            SettingDescriptor {
                label: "Snow",
                description: None,
                kind: SettingKind::FloatRange {
                    range: 0.0..=100.0,
                    logarithmic: true,
                    default_value: default_settings.snow_intensity,
                },
                id: SettingID::SNOW_INTENSITY,
            },
            SettingDescriptor {
                label: "Scanline phase shift",
                description: None,
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
                label: "Head switching",
                description: None,
                kind: SettingKind::Group {
                    children: vec![
                        SettingDescriptor {
                            label: "Height",
                            description: None,
                            kind: SettingKind::IntRange { range: 0..=24, default_value: default_settings.head_switching.settings.height as i32 },
                            id: SettingID::HEAD_SWITCHING_HEIGHT
                        },
                        SettingDescriptor {
                            label: "Offset",
                            description: None,
                            kind: SettingKind::IntRange { range: 0..=24, default_value: default_settings.head_switching.settings.offset as i32 },
                            id: SettingID::HEAD_SWITCHING_OFFSET
                        },
                        SettingDescriptor {
                            label: "Horizontal shift",
                            description: None,
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
                description: None,
                kind: SettingKind::Group {
                    children: vec![
                        SettingDescriptor {
                            label: "Height",
                            description: None,
                            kind: SettingKind::IntRange { range: 0..=120, default_value: default_settings.tracking_noise.settings.height as i32 },
                            id: SettingID::TRACKING_NOISE_HEIGHT
                        },
                        SettingDescriptor {
                            label: "Wave intensity",
                            description: None,
                            kind: SettingKind::FloatRange { range: -50.0..=50.0, logarithmic: false, default_value: default_settings.tracking_noise.settings.wave_intensity },
                            id: SettingID::TRACKING_NOISE_WAVE_INTENSITY
                        },
                        SettingDescriptor {
                            label: "Snow intensity",
                            description: None,
                            kind: SettingKind::FloatRange { range: 0.0..=0.25, logarithmic: true, default_value: default_settings.tracking_noise.settings.snow_intensity },
                            id: SettingID::TRACKING_NOISE_SNOW_INTENSITY
                        },
                        SettingDescriptor {
                            label: "Noise intensity",
                            description: None,
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
                description: None,
                kind: SettingKind::Group {
                    children: vec![
                        SettingDescriptor {
                            label: "Frequency",
                            description: None,
                            kind: SettingKind::Percentage { logarithmic: false, default_value: default_settings.ringing.settings.frequency },
                            id: SettingID::RINGING_FREQUENCY
                        },
                        SettingDescriptor {
                            label: "Power",
                            description: None,
                            kind: SettingKind::FloatRange { range: 0.0..=10.0, logarithmic: false, default_value: default_settings.ringing.settings.power },
                            id: SettingID::RINGING_POWER
                        },
                        SettingDescriptor {
                            label: "Scale",
                            description: None,
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
                description: None,
                kind: SettingKind::FloatRange {
                    range: 0.0..=2.0,
                    logarithmic: false,
                    default_value: default_settings.chroma_noise_intensity,
                },
                id: SettingID::CHROMA_NOISE_INTENSITY,
            },
            SettingDescriptor {
                label: "Chroma phase noise",
                description: None,
                kind: SettingKind::Percentage {
                    logarithmic: true,
                    default_value: default_settings.chroma_phase_noise_intensity,
                },
                id: SettingID::CHROMA_PHASE_NOISE_INTENSITY,
            },
            SettingDescriptor {
                label: "Chroma delay (horizontal)",
                description: None,
                kind: SettingKind::FloatRange {
                    range: -20.0..=20.0,
                    logarithmic: false,
                    default_value: default_settings.chroma_delay.0,
                },
                id: SettingID::CHROMA_DELAY_HORIZONTAL,
            },
            SettingDescriptor {
                label: "Chroma delay (vertical)",
                description: None,
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
                            description: None,
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
                            label: "Vertically blend chroma",
                            description: None,
                            kind: SettingKind::Boolean { default_value: default_settings.vhs_settings.settings.chroma_vert_blend },
                            id: SettingID::VHS_CHROMA_VERT_BLEND
                        },
                        SettingDescriptor {
                            label: "Chroma loss",
                            description: None,
                            kind: SettingKind::Percentage { logarithmic: true, default_value: default_settings.vhs_settings.settings.chroma_loss },
                            id: SettingID::VHS_CHROMA_LOSS
                        },
                        SettingDescriptor {
                            label: "Sharpen",
                            description: None,
                            kind: SettingKind::FloatRange { range: 0.0..=5.0, logarithmic: false, default_value: default_settings.vhs_settings.settings.sharpen },
                            id: SettingID::VHS_SHARPEN
                        },
                        SettingDescriptor {
                            label: "Edge wave intensity",
                            description: None,
                            kind: SettingKind::FloatRange { range: 0.0..=10.0, logarithmic: false, default_value: default_settings.vhs_settings.settings.edge_wave },
                            id: SettingID::VHS_EDGE_WAVE
                        },
                        SettingDescriptor {
                            label: "Edge wave speed",
                            description: None,
                            kind: SettingKind::FloatRange { range: 0.0..=10.0, logarithmic: false, default_value: default_settings.vhs_settings.settings.edge_wave_speed },
                            id: SettingID::VHS_EDGE_WAVE_SPEED
                        },
                    ],
                    default_value: true,
                },
                id: SettingID::VHS_SETTINGS,
            },
            SettingDescriptor {
                label: "Chroma low-pass out",
                description: None,
                kind: SettingKind::Enumeration {
                    options: vec![
                        MenuItem {
                            label: "Full",
                            description: None,
                            index: ChromaLowpass::Full.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "Light",
                            description: None,
                            index: ChromaLowpass::Light.to_u32().unwrap(),
                        },
                        MenuItem {
                            label: "None",
                            description: None,
                            index: ChromaLowpass::None.to_u32().unwrap(),
                        },
                    ],
                    default_value: default_settings.chroma_lowpass_out.to_u32().unwrap(),
                },
                id: SettingID::CHROMA_LOWPASS_OUT,
            },
        ];

        let mut by_id = Vec::new();
        Self::construct_id_map(&v, &mut by_id, &vec![]);

        SettingsList { settings: v.into_boxed_slice(), by_id: by_id.into_boxed_slice() }
    }
}
