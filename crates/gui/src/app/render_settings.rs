use std::path::PathBuf;

use gstreamer::{ClockTime, Fraction};
use ntscrs::ntsc::{NtscEffect, NtscEffectFullSettings, UseField};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct H264Settings {
    /// Quality (the inverse of the constant rate factor). Although libx264 says it ranges from 0-51, its actual range
    /// appears to be 0-50 inclusive. It's flipped from CRF so 50 is lossless and 0 is worst.
    pub quality: u8,
    /// 0-8 for libx264 presets veryslow-ultrafast
    pub encode_speed: u8,
    /// Enable 10-bit color
    pub ten_bit: bool,
    /// Subsample chroma to 4:2:0
    pub chroma_subsampling: bool,
}

impl Default for H264Settings {
    fn default() -> Self {
        Self {
            quality: 27,
            encode_speed: 5,
            ten_bit: false,
            chroma_subsampling: true,
        }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ffv1BitDepth {
    #[default]
    Bits8,
    Bits10,
    Bits12,
}

impl Ffv1BitDepth {
    pub fn label(&self) -> &'static str {
        match self {
            Ffv1BitDepth::Bits8 => "8-bit",
            Ffv1BitDepth::Bits10 => "10-bit",
            Ffv1BitDepth::Bits12 => "12-bit",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Ffv1Settings {
    pub bit_depth: Ffv1BitDepth,
    // Subsample chroma to 4:2:0
    pub chroma_subsampling: bool,
}

#[derive(Debug, Clone)]
pub struct PngSettings {
    pub seek_to: ClockTime,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputCodec {
    #[default]
    H264,
    Ffv1,
}

impl OutputCodec {
    pub fn label(&self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::Ffv1 => "FFV1 (Lossless)",
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::H264 => "mp4",
            Self::Ffv1 => "mkv",
        }
    }
}

#[derive(Debug, Clone)]
pub enum RenderPipelineCodec {
    H264(H264Settings),
    Ffv1(Ffv1Settings),
    Png(PngSettings),
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderInterlaceMode {
    #[default]
    Progressive,
    TopFieldFirst,
    BottomFieldFirst,
}

impl RenderInterlaceMode {
    pub fn from_use_field(use_field: UseField, enable_interlacing: bool) -> Self {
        match (
            use_field.interlaced_output_allowed() && enable_interlacing,
            use_field,
        ) {
            (true, UseField::InterleavedUpper) => RenderInterlaceMode::TopFieldFirst,
            (true, UseField::InterleavedLower) => RenderInterlaceMode::BottomFieldFirst,
            _ => RenderInterlaceMode::Progressive,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StillImageSettings {
    pub framerate: Fraction,
    pub duration: ClockTime,
}

#[derive(Debug, Clone)]
pub struct RenderPipelineSettings {
    pub codec_settings: RenderPipelineCodec,
    pub output_path: PathBuf,
    pub interlacing: RenderInterlaceMode,
    pub effect_settings: NtscEffect,
}

impl RenderPipelineSettings {
    pub fn from_gui_settings(
        effect_settings: &NtscEffectFullSettings,
        render_settings: &RenderSettings,
    ) -> Self {
        Self {
            codec_settings: render_settings.into(),
            output_path: render_settings.output_path.clone(),
            interlacing: RenderInterlaceMode::from_use_field(
                effect_settings.use_field,
                render_settings.interlaced,
            ),
            effect_settings: effect_settings.into(),
        }
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct RenderSettings {
    pub output_codec: OutputCodec,
    // we want to keep these around even if the user changes their mind and selects ffv1, so they don't lose the
    // settings if they change back
    pub h264_settings: H264Settings,
    pub ffv1_settings: Ffv1Settings,
    #[serde(skip)]
    pub output_path: PathBuf,
    pub duration: ClockTime,
    pub interlaced: bool,
}

impl From<&RenderSettings> for RenderPipelineCodec {
    fn from(value: &RenderSettings) -> Self {
        match value.output_codec {
            OutputCodec::H264 => RenderPipelineCodec::H264(value.h264_settings.clone()),
            OutputCodec::Ffv1 => RenderPipelineCodec::Ffv1(value.ffv1_settings.clone()),
        }
    }
}
