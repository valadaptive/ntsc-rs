use eframe::egui::Context;
use eframe::egui::TextureOptions;
use eframe::epaint::{Color32, ColorImage, TextureHandle};
use gstreamer::glib::once_cell::sync::Lazy;
use gstreamer::prelude::*;
use gstreamer::{glib, PadTemplate};
use gstreamer_video::subclass::prelude::*;
use ntscrs::settings::UseField;
use ntscrs::yiq_fielding::{Rgbx8, YiqField, YiqOwned, YiqView};
use std::fmt::Debug;
use std::sync::Mutex;

use super::ntscrs_filter::NtscFilterSettings;

#[derive(Clone, glib::Boxed, Default)]
#[boxed_type(name = "SinkTexture")]
pub struct SinkTexture(pub Option<TextureHandle>);

impl Debug for SinkTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut t = f.debug_tuple("SinkTexture");

        match &self.0 {
            Some(_) => {
                t.field(&"TextureHandle");
            }
            None => {
                t.field(&"None");
            }
        }

        t.finish()
    }
}

#[derive(glib::Properties, Default)]
#[properties(wrapper_type = super::elements::EguiSink)]
pub struct EguiSink {
    #[property(get, set)]
    texture: Mutex<SinkTexture>,
    #[property(get, set)]
    ctx: Mutex<EguiCtx>,
    #[property(get, set = Self::set_settings)]
    settings: Mutex<NtscFilterSettings>,

    video_info: Mutex<Option<gstreamer_video::VideoInfo>>,
    last_frame: Mutex<
        Option<(
            gstreamer_video::VideoFrame<gstreamer_video::video_frame::Readable>,
            u64,
        )>,
    >,
}

#[derive(Debug, Clone, glib::Boxed, Default)]
#[boxed_type(name = "EguiCtx")]
pub struct EguiCtx(pub Option<Context>);

impl EguiSink {
    fn set_settings(&self, value: NtscFilterSettings) {
        *self.settings.lock().unwrap() = value;
        let _ = self.update_texture();
    }

    pub fn update_texture(&self) -> Result<(), gstreamer::FlowError> {
        let mut tex = self.texture.lock().unwrap();
        let vframe = self.last_frame.lock().unwrap();
        let (vframe, frame_num) = vframe.as_ref().ok_or(gstreamer::FlowError::Error)?;

        let width = vframe.width() as usize;
        let height = vframe.height() as usize;
        let mut image = ColorImage::new([width, height], Color32::BLACK);

        let stride = vframe.plane_stride()[0] as usize;

        let settings = self.settings.lock().unwrap();
        let field = match settings.0.use_field {
            UseField::Alternating => {
                if frame_num & 1 == 0 {
                    YiqField::Lower
                } else {
                    YiqField::Upper
                }
            }
            UseField::Upper => YiqField::Upper,
            UseField::Lower => YiqField::Lower,
            UseField::Both => YiqField::Both,
        };

        let mut yiq = YiqOwned::from_strided_buffer::<Rgbx8>(
            vframe.plane_data(0).or(Err(gstreamer::FlowError::Error))?,
            stride,
            width,
            height,
            field,
        );
        let mut view = YiqView::from(&mut yiq);
        settings
            .0
            .apply_effect_to_yiq(&mut view, *frame_num as usize);
        view.write_to_strided_buffer::<Rgbx8>(image.as_raw_mut(), width * 4);

        tex.0
            .as_mut()
            .ok_or(gstreamer::FlowError::Error)?
            .set(image, TextureOptions::LINEAR);
        if let Some(ctx) = &self.ctx.lock().unwrap().0 {
            ctx.request_repaint();
        }

        Ok(())
    }
}

#[glib::object_subclass]
impl ObjectSubclass for EguiSink {
    const NAME: &'static str = "EguiSink";
    type Type = super::elements::EguiSink;
    type ParentType = gstreamer_video::VideoSink;
}

#[glib::derived_properties]
impl ObjectImpl for EguiSink {}

impl GstObjectImpl for EguiSink {}

impl ElementImpl for EguiSink {
    fn metadata() -> Option<&'static gstreamer::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gstreamer::subclass::ElementMetadata> = Lazy::new(|| {
            gstreamer::subclass::ElementMetadata::new(
                "egui sink",
                "Sink/Video",
                "Video sink for egui texture",
                "valadaptive",
            )
        });

        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gstreamer::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<PadTemplate>> = Lazy::new(|| {
            let caps = gstreamer_video::VideoCapsBuilder::new()
                .format(gstreamer_video::VideoFormat::Rgbx)
                .build();
            let pad_template = gstreamer::PadTemplate::builder(
                "sink",
                gstreamer::PadDirection::Sink,
                gstreamer::PadPresence::Always,
                &caps,
            )
            .build()
            .unwrap();

            vec![pad_template]
        });

        PAD_TEMPLATES.as_ref()
    }
}

impl BaseSinkImpl for EguiSink {
    fn set_caps(&self, caps: &gstreamer::Caps) -> Result<(), gstreamer::LoggableError> {
        let mut video_info = self.video_info.lock().unwrap();
        *video_info = Some(gstreamer_video::VideoInfo::from_caps(caps)?);
        Ok(())
    }
}

impl VideoSinkImpl for EguiSink {
    fn show_frame(
        &self,
        buffer: &gstreamer::Buffer,
    ) -> Result<gstreamer::FlowSuccess, gstreamer::FlowError> {
        let video_info = self.video_info.lock().unwrap();
        let video_info = video_info.as_ref().ok_or(gstreamer::FlowError::Error)?;

        let timestamp = buffer.pts().ok_or(gstreamer::FlowError::Error)?.nseconds();
        let frame_num = (video_info.fps().numer() as u128 * (timestamp + 100) as u128
            / video_info.fps().denom() as u128) as u64
            / gstreamer::ClockTime::SECOND.nseconds();

        let mut last_frame = self.last_frame.lock().unwrap();
        let should_rerender = match last_frame.as_ref() {
            Some((last, last_frame_num)) => {
                last.buffer() != buffer.as_ref() || *last_frame_num != frame_num
            }
            None => true,
        };

        if should_rerender {
            let owned_frame =
                gstreamer_video::VideoFrame::from_buffer_readable(buffer.copy(), video_info)
                    .unwrap();
            *last_frame = Some((owned_frame, frame_num));
            drop(last_frame);
            self.update_texture()?;
        }

        Ok(gstreamer::FlowSuccess::Ok)
    }
}
