use ntscrs::ntsc::ParseSettingsError;
use snafu::Snafu;

use crate::gst_utils::{gstreamer_error::GstreamerError, pipeline_utils::PipelineError};

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum ApplicationError {
    #[snafu(display("Error loading video: {source}"))]
    LoadVideo { source: GstreamerError },

    #[snafu(display("Error creating pipeline: {source}"))]
    CreatePipeline { source: PipelineError },

    #[snafu(display("Error creating render job: {source}"))]
    CreateRenderJob { source: GstreamerError },

    #[snafu(display("Error reading JSON: {source}"))]
    JSONRead { source: std::io::Error },

    #[snafu(display("Error parsing JSON: {source}"))]
    JSONParse { source: ParseSettingsError },

    #[snafu(display("Error saving JSON: {source}"))]
    JSONSave { source: std::io::Error },
}
