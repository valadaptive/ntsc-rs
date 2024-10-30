use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use eframe::egui;
use gstreamer::ClockTime;
use gstreamer_video::VideoInterlaceMode;

use crate::gst_utils::{
    gstreamer_error::GstreamerError,
    ntsc_pipeline::{NtscPipeline, PipelineError},
};

#[derive(Debug, Default)]
pub struct PipelineMetadata {
    pub is_still_image: Option<bool>,
    pub has_audio: Option<bool>,
    pub framerate: Option<gstreamer::Fraction>,
    pub interlace_mode: Option<VideoInterlaceMode>,
    pub resolution: Option<(usize, usize)>,
}

#[derive(Debug)]
pub enum PipelineStatus {
    Loading,
    Loaded,
    Error(PipelineError),
}

pub struct PipelineInfo {
    pub pipeline: NtscPipeline,
    pub state: Arc<Mutex<PipelineStatus>>,
    pub path: PathBuf,
    pub egui_sink: gstreamer::Element,
    pub last_seek_pos: ClockTime,
    pub preview: egui::TextureHandle,
    pub at_eos: Arc<Mutex<bool>>,
    pub metadata: Arc<Mutex<PipelineMetadata>>,
}

impl PipelineInfo {
    pub fn toggle_playing(&self) -> Result<(), GstreamerError> {
        match self.pipeline.current_state() {
            gstreamer::State::Paused | gstreamer::State::Ready => {
                // Restart from the beginning if "play" is pressed at the end of the video
                let (position, duration) = (
                    self.pipeline.query_position::<ClockTime>(),
                    self.pipeline.query_duration::<ClockTime>(),
                );
                if let (Some(position), Some(duration)) = (position, duration) {
                    if position == duration {
                        self.pipeline.seek_simple(
                            gstreamer::SeekFlags::FLUSH | gstreamer::SeekFlags::ACCURATE,
                            ClockTime::ZERO,
                        )?;
                    }
                }

                self.pipeline.set_state(gstreamer::State::Playing)?;
            }
            gstreamer::State::Playing => {
                self.pipeline.set_state(gstreamer::State::Paused)?;
            }
            _ => {}
        }

        Ok(())
    }
}

impl Drop for PipelineInfo {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gstreamer::State::Null);
    }
}
