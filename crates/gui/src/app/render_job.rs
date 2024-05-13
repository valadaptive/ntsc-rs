use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use gstreamer::prelude::ElementExt;

use crate::{
    app::render_settings::RenderPipelineSettings, gst_utils::gstreamer_error::GstreamerError,
};

#[derive(Debug, Clone)]
pub enum RenderJobState {
    Waiting,
    Rendering,
    Paused,
    Complete { end_time: f64 },
    Error(Arc<GstreamerError>),
}

#[derive(Debug)]
pub struct RenderJob {
    pub settings: RenderPipelineSettings,
    pub pipeline: gstreamer::Pipeline,
    pub state: Arc<Mutex<RenderJobState>>,
    pub last_progress: f64,
    /// Used for estimating time remaining. A queue that holds (progress, timestamp) pairs.
    progress_samples: VecDeque<(f64, f64)>,
    pub start_time: Option<f64>,
    pub estimated_completion_time: Option<f64>,
}

impl RenderJob {
    pub fn new(
        settings: RenderPipelineSettings,
        pipeline: gstreamer::Pipeline,
        state: Arc<Mutex<RenderJobState>>,
    ) -> Self {
        Self {
            settings,
            pipeline,
            state,
            last_progress: 0.0,
            progress_samples: VecDeque::new(),
            start_time: None,
            estimated_completion_time: None,
        }
    }

    pub fn update_estimated_completion_time(&mut self, progress: f64, current_time: f64) {
        const NUM_PROGRESS_SAMPLES: usize = 5;
        const PROGRESS_SAMPLE_TIME_DELTA: f64 = 1.0;

        let most_recent_sample = self.progress_samples.back().copied();
        let should_update_estimate = if let Some((_, sample_time)) = most_recent_sample {
            current_time - sample_time > PROGRESS_SAMPLE_TIME_DELTA
        } else {
            true
        };
        if should_update_estimate {
            if self.start_time.is_none() {
                self.start_time = Some(current_time);
            }
            let new_sample = (progress, current_time);
            let oldest_sample = if self.progress_samples.len() >= NUM_PROGRESS_SAMPLES {
                self.progress_samples.pop_front()
            } else {
                self.progress_samples.front().copied()
            };
            self.progress_samples.push_back(new_sample);
            if let Some((old_progress, old_sample_time)) = oldest_sample {
                let time_estimate = (current_time - old_sample_time) / (progress - old_progress)
                    + self.start_time.unwrap();
                if time_estimate.is_finite() {
                    self.estimated_completion_time = Some(time_estimate);
                }
            }
        }
    }
}

impl Drop for RenderJob {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gstreamer::State::Null);
    }
}
