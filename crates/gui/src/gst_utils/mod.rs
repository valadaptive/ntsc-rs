use gstreamer::glib;
use gstreamer::prelude::*;

pub mod egui_sink;
pub mod clock_format;
pub mod ntscrs_filter;

glib::wrapper! {
    pub struct EguiSink(ObjectSubclass<egui_sink::EguiSink>) @extends gstreamer_video::VideoSink, gstreamer_base::BaseSink, gstreamer::Element, gstreamer::Object;
}

glib::wrapper! {
    pub struct NtscFilter(ObjectSubclass<ntscrs_filter::NtscFilter>) @extends gstreamer_base::BaseTransform, gstreamer::Element, gstreamer::Object;
}

pub fn register(plugin: &gstreamer::Plugin) -> Result<(), glib::BoolError> {
    gstreamer::Element::register(
        Some(plugin),
        "eguisink",
        gstreamer::Rank::None,
        EguiSink::static_type(),
    )
}
