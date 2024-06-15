use std::sync::{atomic::AtomicBool, Arc};

use eframe::egui::util::undoer::Undoer;
use ntscrs::ntsc::{NtscEffectFullSettings, SettingsList};

pub mod app_state;
pub mod error;
pub mod executor;
pub mod main;
pub mod pipeline_info;
pub mod render_job;
pub mod render_settings;

pub type AppFn = Box<dyn FnOnce(&mut NtscApp) -> Result<(), error::ApplicationError> + Send>;

pub struct NtscApp {
    pub gstreamer_initialized: Arc<AtomicBool>,
    pub settings_list: SettingsList,
    pub executor: executor::AppExecutor,
    pub pipeline: Option<pipeline_info::PipelineInfo>,
    pub undoer: Undoer<NtscEffectFullSettings>,
    pub video_zoom: app_state::VideoZoom,
    pub video_scale: app_state::VideoScale,
    pub audio_volume: app_state::AudioVolume,
    pub effect_preview: app_state::EffectPreviewSettings,
    pub left_panel_state: app_state::LeftPanelState,
    pub effect_settings: NtscEffectFullSettings,
    pub render_settings: render_settings::RenderSettings,
    pub render_jobs: Vec<render_job::RenderJob>,
    pub settings_json_paste: String,
    pub last_error: Option<String>,
    pub color_theme: app_state::ColorTheme,
    pub credits_dialog_open: bool,
    pub licenses_dialog_open: bool,
}
