use std::cell::RefCell;

use app_state::GstreamerInitState;
use eframe::egui::util::undoer::Undoer;
use ntscrs::ntsc::{NtscEffectFullSettings, SettingsList};
use presets::PresetsState;

pub mod app_state;
pub mod error;
pub mod executor;
pub mod layout_helper;
pub mod license_dialog;
pub mod main;
pub mod pipeline_info;
pub mod presets;
pub mod render_job;
pub mod render_settings;

pub type AppFn = Box<dyn FnOnce(&mut NtscApp) -> Result<(), error::ApplicationError> + Send>;

pub struct NtscApp {
    pub gstreamer_init: GstreamerInitState,
    pub settings_list: SettingsList<NtscEffectFullSettings>,
    pub executor: executor::AppExecutor,
    pub pipeline: Option<pipeline_info::PipelineInfo>,
    pub undoer: Undoer<NtscEffectFullSettings>,
    pub video_zoom: app_state::VideoZoom,
    pub video_scale: app_state::VideoScale,
    pub audio_volume: app_state::AudioVolume,
    pub effect_preview: app_state::EffectPreviewSettings,
    pub left_panel_state: app_state::LeftPanelState,
    pub effect_settings: NtscEffectFullSettings,
    pub presets_state: PresetsState,
    pub render_settings: render_settings::RenderSettings,
    pub render_jobs: Vec<render_job::RenderJob>,
    pub settings_json_paste: String,
    pub last_error: RefCell<Option<String>>,
    pub color_theme: app_state::ColorTheme,
    pub credits_dialog_open: bool,
    pub third_party_licenses_dialog_open: bool,
    pub license_dialog_open: bool,
}
