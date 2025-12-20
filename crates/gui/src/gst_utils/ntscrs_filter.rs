use std::sync::{OnceLock, RwLock};

use gstreamer::glib;
use gstreamer::prelude::{GstParamSpecBuilderExt, ParamSpecBuilderExt, ToValue};
use gstreamer_video::subclass::prelude::*;
use gstreamer_video::{VideoFormat, VideoFrameExt};

use ntscrs::NtscEffect;
use ntscrs::yiq_fielding::{Bgrx, Rgbx, Xbgr, Xrgb};

use super::process_gst_frame::process_gst_frame;

#[derive(Clone, glib::Boxed, Default)]
#[boxed_type(name = "NtscFilterSettings")]
pub struct NtscFilterSettings(pub NtscEffect);

#[derive(Default)]
pub struct NtscFilter {
    info: RwLock<Option<gstreamer_video::VideoInfo>>,
    settings: RwLock<NtscFilterSettings>,
}

impl NtscFilter {}

#[glib::object_subclass]
impl ObjectSubclass for NtscFilter {
    const NAME: &'static str = "ntscrs";
    type Type = super::elements::NtscFilter;
    type ParentType = gstreamer_video::VideoFilter;
}

impl ObjectImpl for NtscFilter {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: OnceLock<Vec<glib::ParamSpec>> = OnceLock::new();

        PROPERTIES.get_or_init(|| {
            vec![
                glib::ParamSpecBoxed::builder::<NtscFilterSettings>("settings")
                    .nick("Settings")
                    .blurb("ntsc-rs settings block")
                    .mutable_playing()
                    .controllable()
                    .build(),
            ]
        })
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        if pspec.name() != "settings" {
            panic!("Incorrect param spec name {}", pspec.name());
        }

        let mut settings = self.settings.write().unwrap();
        let new_settings = value.get().unwrap();
        *settings = new_settings;
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        if pspec.name() != "settings" {
            panic!("Incorrect param spec name {}", pspec.name());
        }

        let settings = self.settings.read().unwrap();
        settings.to_value()
    }
}

impl GstObjectImpl for NtscFilter {}

impl ElementImpl for NtscFilter {
    fn metadata() -> Option<&'static gstreamer::subclass::ElementMetadata> {
        static PROPERTIES: OnceLock<gstreamer::subclass::ElementMetadata> = OnceLock::new();
        Some(PROPERTIES.get_or_init(|| {
            gstreamer::subclass::ElementMetadata::new(
                "NTSC-rs Filter",
                "Filter/Effect/Converter/Video",
                "Applies an NTSC/VHS effect to video",
                "valadaptive",
            )
        }))
    }

    fn pad_templates() -> &'static [gstreamer::PadTemplate] {
        static PAD_TEMPLATES: OnceLock<Vec<gstreamer::PadTemplate>> = OnceLock::new();
        PAD_TEMPLATES.get_or_init(|| {
            let caps = gstreamer_video::VideoCapsBuilder::new()
                .format_list([
                    VideoFormat::Rgbx,
                    VideoFormat::Rgba,
                    VideoFormat::Bgrx,
                    VideoFormat::Bgra,
                    VideoFormat::Xrgb,
                    VideoFormat::Xbgr,
                    VideoFormat::Argb64,
                ])
                .build();

            let src_pad_template = gstreamer::PadTemplate::builder(
                "src",
                gstreamer::PadDirection::Src,
                gstreamer::PadPresence::Always,
                &caps,
            )
            .build()
            .unwrap();

            let sink_pad_template = gstreamer::PadTemplate::builder(
                "sink",
                gstreamer::PadDirection::Sink,
                gstreamer::PadPresence::Always,
                &caps,
            )
            .build()
            .unwrap();

            vec![src_pad_template, sink_pad_template]
        })
    }
}

impl BaseTransformImpl for NtscFilter {
    const MODE: gstreamer_base::subclass::BaseTransformMode =
        gstreamer_base::subclass::BaseTransformMode::NeverInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;
}

impl VideoFilterImpl for NtscFilter {
    fn set_info(
        &self,
        incaps: &gstreamer::Caps,
        in_info: &gstreamer_video::VideoInfo,
        outcaps: &gstreamer::Caps,
        out_info: &gstreamer_video::VideoInfo,
    ) -> Result<(), gstreamer::LoggableError> {
        let mut info = self.info.write().unwrap();
        *info = Some(in_info.clone());
        self.parent_set_info(incaps, in_info, outcaps, out_info)
    }

    fn transform_frame(
        &self,
        in_frame: &gstreamer_video::VideoFrameRef<&gstreamer::BufferRef>,
        out_frame: &mut gstreamer_video::VideoFrameRef<&mut gstreamer::BufferRef>,
    ) -> Result<gstreamer::FlowSuccess, gstreamer::FlowError> {
        let settings = self
            .settings
            .read()
            .or(Err(gstreamer::FlowError::Error))?
            .clone()
            .0;

        let out_stride = out_frame.plane_stride()[0] as usize;
        let out_format = out_frame.format();
        let out_data = out_frame
            .plane_data_mut(0)
            .or(Err(gstreamer::FlowError::Error))?;

        match out_format {
            VideoFormat::Rgbx | VideoFormat::Rgba => {
                process_gst_frame::<Rgbx, u8>(in_frame, out_data, out_stride, None, &settings)?;
            }
            VideoFormat::Bgrx | VideoFormat::Bgra => {
                process_gst_frame::<Bgrx, u8>(in_frame, out_data, out_stride, None, &settings)?;
            }
            VideoFormat::Xrgb | VideoFormat::Argb => {
                process_gst_frame::<Xrgb, u8>(in_frame, out_data, out_stride, None, &settings)?;
            }
            VideoFormat::Xbgr | VideoFormat::Abgr => {
                process_gst_frame::<Xbgr, u8>(in_frame, out_data, out_stride, None, &settings)?;
            }
            VideoFormat::Argb64 => {
                let data_16 = unsafe { out_data.align_to_mut::<u16>() }.1;
                process_gst_frame::<Xrgb, u16>(in_frame, data_16, out_stride, None, &settings)?;
            }
            _ => Err(gstreamer::FlowError::NotSupported)?,
        };

        Ok(gstreamer::FlowSuccess::Ok)
    }
}
