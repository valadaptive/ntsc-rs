use std::cell::RefCell;

use app_state::GstreamerInitState;
use eframe::egui::util::undoer::Undoer;
use ntscrs::{
    NtscEffectFullSettings,
    settings::{SettingsList, easy::EasyModeFullSettings},
};
use presets::PresetsState;

pub mod app_state;
pub mod dnd_overlay;
pub mod error;
pub mod executor;
pub mod format_eta;
pub mod layout_helper;
pub mod license_dialog;
pub mod main;
pub mod pipeline_info;
pub mod presets;
pub mod render_job;
pub mod render_settings;
pub mod system_fonts;
pub mod third_party_licenses_dialog;
pub mod ui_context;

pub type AppFn = Box<dyn FnOnce(&mut NtscApp) -> Result<(), error::ApplicationError> + Send>;
pub type ApplessFn = Box<dyn FnOnce() -> Result<(), error::ApplicationError> + Send>;

pub struct NtscApp {
    pub gstreamer_init: GstreamerInitState,
    pub settings_list: SettingsList<NtscEffectFullSettings>,
    pub settings_list_easy: SettingsList<EasyModeFullSettings>,
    pub executor: executor::AppExecutor,
    pub pipeline: Option<pipeline_info::PipelineInfo>,
    pub undoer: Undoer<NtscEffectFullSettings>,
    pub video_zoom: app_state::VideoZoom,
    pub video_scale: app_state::VideoScaleState,
    pub audio_volume: app_state::AudioVolume,
    pub effect_preview: app_state::EffectPreviewSettings,
    pub left_panel_state: app_state::LeftPanelState,
    pub easy_mode_enabled: bool,
    pub effect_settings: NtscEffectFullSettings,
    pub easy_mode_settings: EasyModeFullSettings,
    pub presets_state: PresetsState,
    pub render_settings: render_settings::RenderSettings,
    pub render_jobs: Vec<render_job::RenderJob>,
    pub settings_json_paste: String,
    pub last_error: RefCell<Option<String>>,
    pub credits_dialog_open: bool,
    pub third_party_licenses_dialog_open: bool,
    pub license_dialog_open: bool,
    pub image_sequence_dialog_queued_render_job: Option<
        Box<dyn FnOnce(&mut Self) -> Result<render_job::RenderJob, error::ApplicationError>>,
    >,
}
