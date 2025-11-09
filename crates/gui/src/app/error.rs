use ntscrs::settings::{ParseSettingsError, sval_json};
use snafu::Snafu;

use crate::gst_utils::{gstreamer_error::GstreamerError, ntsc_pipeline::PipelineError};

#[cfg(not(target_os = "android"))]
type TrashError = trash::Error;
#[cfg(target_os = "android")]
type TrashError = std::io::Error;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum ApplicationError {
    #[snafu(display("Error initializing GStreamer: {source}"))]
    GstreamerInit { source: GstreamerError },

    #[snafu(display("Error loading video: {source}"))]
    LoadVideo { source: GstreamerError },

    #[snafu(display("Error creating pipeline: {source}"))]
    CreatePipeline { source: PipelineError },

    #[snafu(display("Error creating render job: {source}"))]
    CreateRenderJob { source: GstreamerError },

    #[snafu(display("Error during render job: {source}"))]
    RenderJobPipeline { source: GstreamerError },

    #[snafu(display("Error reading JSON: {source}"))]
    JSONRead { source: std::io::Error },

    #[snafu(display("Error parsing JSON: {source}"))]
    JSONParse {
        #[snafu(source(from(ParseSettingsError, Box::new)))]
        source: Box<ParseSettingsError>,
    },

    #[snafu(display("Error creating presets directory: {source}"))]
    CreatePresetsDirectory { source: std::io::Error },

    #[snafu(display("Error creating preset: {source}"))]
    CreatePresetFile { source: std::io::Error },

    #[snafu(display("Error creating preset: {source}"))]
    CreatePresetJSON { source: sval_json::Error },

    #[snafu(display("Error deleting preset: {source}"))]
    DeletePreset {
        #[snafu(source(from(TrashError, Box::new)))]
        source: Box<TrashError>,
    },

    #[snafu(display("Error renaming preset: {source}"))]
    RenamePreset { source: std::io::Error },

    #[snafu(display("Error installing preset: {source}"))]
    InstallPreset { source: std::io::Error },

    #[snafu(display("Filesystem error: {source}"))]
    Fs { source: std::io::Error },

    #[snafu(display("Only one file at a time can be dropped here"))]
    DroppedMultipleFiles,
}
