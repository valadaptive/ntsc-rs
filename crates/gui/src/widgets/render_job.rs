use std::borrow::Cow;

use eframe::egui::{self, InnerResponse, Response};
use gstreamer::{prelude::*, ClockTime};
use snafu::ResultExt;

use crate::{
    app::{
        error::{ApplicationError, RenderJobPipelineSnafu},
        render_job::{RenderJob, RenderJobState},
    },
    gst_utils::gstreamer_error::GstreamerError,
};

pub struct RenderJobWidget<'a> {
    render_job: &'a mut RenderJob,
}

impl<'a> RenderJobWidget<'a> {
    pub fn new(render_job: &'a mut RenderJob) -> Self {
        Self { render_job }
    }
}

pub struct RenderJobResponse {
    pub response: Response,
    pub closed: bool,
    pub error: Option<ApplicationError>,
}

impl RenderJobWidget<'_> {
    pub fn show(self, ui: &mut egui::Ui) -> RenderJobResponse {
        let job = self.render_job;
        let InnerResponse {
            inner: (closed, error),
            response,
        } = ui.with_layout(egui::Layout::top_down_justified(egui::Align::Min), |ui| {
            let mut closed = false;
            let mut error = None;

            let fill = ui.style().visuals.faint_bg_color;
            egui::Frame::none()
                .fill(fill)
                .stroke(ui.style().visuals.window_stroke)
                .rounding(ui.style().noninteractive().rounding)
                .inner_margin(ui.style().spacing.window_margin)
                .show(ui, |ui| {
                    let job_state = job.state.lock().unwrap().clone();

                    let (progress, job_position, job_duration) = match job_state {
                        RenderJobState::Waiting => (0.0, None, None),
                        RenderJobState::Paused
                        | RenderJobState::Rendering
                        | RenderJobState::Error(_) => {
                            let job_position = job.pipeline.query_position::<ClockTime>();
                            let job_duration = job.pipeline.query_duration::<ClockTime>();

                            (
                                if let (Some(job_position), Some(job_duration)) =
                                    (job_position, job_duration)
                                {
                                    job_position.nseconds() as f64 / job_duration.nseconds() as f64
                                } else {
                                    job.last_progress
                                },
                                job_position,
                                job_duration,
                            )
                        }
                        RenderJobState::Complete { .. } => (1.0, None, None),
                    };

                    if matches!(
                        job_state,
                        RenderJobState::Rendering | RenderJobState::Waiting
                    ) {
                        let current_time = ui.ctx().input(|input| input.time);
                        job.update_estimated_time_remaining(progress, current_time);
                    }

                    ui.horizontal(|ui| {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            closed = ui.button("🗙").clicked();

                            match &job_state {
                                RenderJobState::Rendering => {
                                    if ui.button("⏸").clicked() {
                                        let current_time = ui.ctx().input(|input| input.time);
                                        job.set_pause_time(current_time);
                                        error = job
                                            .pipeline
                                            .set_state(gstreamer::State::Paused)
                                            .map_err(GstreamerError::from)
                                            .context(RenderJobPipelineSnafu)
                                            .err();
                                    }
                                }
                                RenderJobState::Paused => {
                                    if ui.button("▶").clicked() {
                                        let current_time = ui.ctx().input(|input| input.time);
                                        job.resume_at_time(current_time);
                                        error = job
                                            .pipeline
                                            .set_state(gstreamer::State::Playing)
                                            .map_err(GstreamerError::from)
                                            .context(RenderJobPipelineSnafu)
                                            .err();
                                    }
                                }
                                _ => {}
                            }

                            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                ui.add(
                                    egui::Label::new(job.settings.output_path.to_string_lossy())
                                        .truncate(),
                                );
                            })
                        });
                    });

                    ui.separator();

                    ui.add(egui::ProgressBar::new(progress as f32).show_percentage());
                    if let RenderJobState::Rendering = job_state {
                        ui.ctx().request_repaint();
                    }

                    ui.label(match &job_state {
                        RenderJobState::Waiting => Cow::Borrowed("Waiting..."),
                        RenderJobState::Rendering => {
                            if let (Some(position), Some(duration)) = (job_position, job_duration) {
                                Cow::Owned(format!(
                                    "Rendering... ({:.2} / {:.2})",
                                    position, duration
                                ))
                            } else {
                                Cow::Borrowed("Rendering...")
                            }
                        }
                        RenderJobState::Paused => Cow::Borrowed("Paused"),
                        // if the job's start_time is missing, it's probably because it never got a chance to update--in that case, just say it took 0 seconds
                        RenderJobState::Complete { end_time } => Cow::Owned(format!(
                            "Completed in {:.2}",
                            ClockTime::from_mseconds(
                                ((*end_time - job.start_time.unwrap_or(*end_time)) * 1000.0) as u64
                            )
                        )),
                        RenderJobState::Error(err) => Cow::Owned(format!("Error: {err}")),
                    });

                    if matches!(
                        job_state,
                        RenderJobState::Rendering | RenderJobState::Paused
                    ) {
                        if let Some(time_remaining) = job.estimated_time_remaining {
                            let time_remaining = time_remaining.ceil();
                            ui.label(format!("Time remaining: {time_remaining:.0} seconds"));
                        }
                    }

                    job.last_progress = progress;
                });

            (closed, error)
        });

        RenderJobResponse {
            response,
            closed,
            error,
        }
    }
}
