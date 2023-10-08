pub mod egui_sink;
pub mod clock_format;
pub mod gstreamer_error;
pub mod ntscrs_filter;
pub mod pipeline_utils;
pub mod video_pad_filter;

pub mod elements {
    use gstreamer::glib;
    use super::{egui_sink, ntscrs_filter, video_pad_filter};
    glib::wrapper! {
        pub struct EguiSink(ObjectSubclass<egui_sink::EguiSink>) @extends gstreamer_video::VideoSink, gstreamer_base::BaseSink, gstreamer::Element, gstreamer::Object;
    }

    glib::wrapper! {
        pub struct NtscFilter(ObjectSubclass<ntscrs_filter::NtscFilter>) @extends gstreamer_base::BaseTransform, gstreamer::Element, gstreamer::Object;
    }

    glib::wrapper! {
        pub struct VideoPadFilter(ObjectSubclass<video_pad_filter::VideoPadFilter>) @extends gstreamer_base::BaseTransform, gstreamer::Element, gstreamer::Object;
    }
}

pub fn scale_from_caps(caps: &gstreamer::Caps, scanlines: usize) -> Option<(i32, i32)> {
    let caps_structure = caps.structure(0)?;
    let src_width = caps_structure.get::<i32>("width").ok()?;
    let src_height = caps_structure.get::<i32>("height").ok()?;

    let scale_factor = scanlines as f32 / src_height as f32;
    let dst_width = (src_width as f32 * scale_factor).round() as i32;

    Some((dst_width, scanlines as i32))
}