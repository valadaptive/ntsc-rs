use std::path::PathBuf;

use gstreamer::ClockTime;
use ntscrs::ntsc::NtscEffect;

#[derive(Debug, Clone)]
pub struct H264Settings {
    // Quality / constant rate factor (0-51)
    pub crf: u8,
    // 0-8 for libx264 presets veryslow-ultrafast
    pub encode_speed: u8,
    // Enable 10-bit color
    pub ten_bit: bool,
    // Subsample chroma to 4:2:0
    pub chroma_subsampling: bool,
}

impl Default for H264Settings {
    fn default() -> Self {
        Self {
            crf: 23,
            encode_speed: 5,
            ten_bit: false,
            chroma_subsampling: true,
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone, Default)]
pub struct Ffv1Settings {
    pub bit_depth: Ffv1BitDepth,
    // Subsample chroma to 4:2:0
    pub chroma_subsampling: bool,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
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
    Png,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderInterlaceMode {
    Progressive,
    TopFieldFirst,
    BottomFieldFirst,
}

#[derive(Debug, Clone)]
pub struct RenderPipelineSettings {
    pub codec_settings: RenderPipelineCodec,
    pub output_path: PathBuf,
    pub duration: ClockTime,
    pub interlacing: RenderInterlaceMode,
    pub effect_settings: NtscEffect,
}

#[derive(Default, Debug, Clone)]
pub struct RenderSettings {
    pub output_codec: OutputCodec,
    // we want to keep these around even if the user changes their mind and selects ffv1, so they don't lose the
    // settings if they change back
    pub h264_settings: H264Settings,
    pub ffv1_settings: Ffv1Settings,
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
