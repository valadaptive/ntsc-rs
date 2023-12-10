use std::sync::RwLock;

use gstreamer::glib;
use gstreamer::glib::once_cell::sync::Lazy;
use gstreamer::prelude::{GstParamSpecBuilderExt, ParamSpecBuilderExt, ToValue};
use gstreamer_video::subclass::prelude::*;
use gstreamer_video::VideoFormat;

use ntscrs::ntsc::NtscEffect;
use ntscrs::settings::UseField;
use ntscrs::yiq_fielding::{Bgrx8, Rgbx8, Xbgr8, Xrgb16, Xrgb8, YiqField, YiqOwned, YiqView};

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
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecBoxed::builder::<NtscFilterSettings>("settings")
                    .nick("Settings")
                    .blurb("ntsc-rs settings block")
                    .mutable_playing()
                    .controllable()
                    .build(),
            ]
        });

        PROPERTIES.as_ref()
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
        static ELEMENT_METADATA: Lazy<gstreamer::subclass::ElementMetadata> = Lazy::new(|| {
            gstreamer::subclass::ElementMetadata::new(
                "NTSC-rs Filter",
                "Filter/Effect/Converter/Video",
                "Applies an NTSC/VHS effect to video",
                "valadaptive",
            )
        });

        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gstreamer::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gstreamer::PadTemplate>> = Lazy::new(|| {
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
        });

        PAD_TEMPLATES.as_ref()
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
        let info = self.info.read().or(Err(gstreamer::FlowError::Error))?;
        let info = info.as_ref().ok_or(gstreamer::FlowError::Error)?;

        let width = in_frame.width() as usize;
        let height = in_frame.height() as usize;
        let in_stride = in_frame.plane_stride()[0] as usize;
        let in_data = in_frame
            .plane_data(0)
            .or(Err(gstreamer::FlowError::Error))?;
        let in_format = in_frame.format();
        let out_stride = out_frame.plane_stride()[0] as usize;
        let out_format = out_frame.format();
        let out_data = out_frame
            .plane_data_mut(0)
            .or(Err(gstreamer::FlowError::Error))?;

        let timestamp = in_frame
            .buffer()
            .pts()
            .ok_or(gstreamer::FlowError::Error)?
            .nseconds();
        let frame = (info.fps().numer() as u128 * (timestamp + 100) as u128
            / info.fps().denom() as u128) as u64
            / gstreamer::ClockTime::SECOND.nseconds();

        let settings = self
            .settings
            .read()
            .or(Err(gstreamer::FlowError::Error))?
            .clone()
            .0;

        let field = match settings.use_field {
            UseField::Alternating => {
                if frame & 1 == 0 {
                    YiqField::Lower
                } else {
                    YiqField::Upper
                }
            }
            UseField::Upper => YiqField::Upper,
            UseField::Lower => YiqField::Lower,
            UseField::Both => YiqField::Both,
        };

        let mut yiq = match in_format {
            VideoFormat::Rgbx | VideoFormat::Rgba => {
                YiqOwned::from_strided_buffer::<Rgbx8>(in_data, in_stride, width, height, field)
            }
            VideoFormat::Bgrx | VideoFormat::Bgra => {
                YiqOwned::from_strided_buffer::<Bgrx8>(in_data, in_stride, width, height, field)
            }
            VideoFormat::Xrgb | VideoFormat::Argb => {
                YiqOwned::from_strided_buffer::<Xrgb8>(in_data, in_stride, width, height, field)
            }
            VideoFormat::Xbgr | VideoFormat::Abgr => {
                YiqOwned::from_strided_buffer::<Xbgr8>(in_data, in_stride, width, height, field)
            }

            VideoFormat::Argb64 => {
                let data_16 = unsafe { in_data.align_to::<u16>() }.1;
                YiqOwned::from_strided_buffer::<Xrgb16>(data_16, in_stride, width, height, field)
            }
            _ => Err(gstreamer::FlowError::NotSupported)?,
        };
        let mut view = YiqView::from(&mut yiq);

        settings.apply_effect_to_yiq(&mut view, frame as usize);

        match out_format {
            VideoFormat::Rgbx | VideoFormat::Rgba => {
                view.write_to_strided_buffer::<Rgbx8>(out_data, out_stride)
            }
            VideoFormat::Bgrx | VideoFormat::Bgra => {
                view.write_to_strided_buffer::<Bgrx8>(out_data, out_stride)
            }
            VideoFormat::Xrgb | VideoFormat::Argb => {
                view.write_to_strided_buffer::<Xrgb8>(out_data, out_stride)
            }
            VideoFormat::Xbgr | VideoFormat::Abgr => {
                view.write_to_strided_buffer::<Xbgr8>(out_data, out_stride)
            }
            VideoFormat::Argb64 => {
                let data_16 = unsafe { out_data.align_to_mut::<u16>() }.1;
                view.write_to_strided_buffer::<Xrgb16>(data_16, out_stride)
            }
            _ => Err(gstreamer::FlowError::NotSupported)?,
        };

        Ok(gstreamer::FlowSuccess::Ok)
    }
}
