use eframe::egui::{ColorImage, Context, Rect, TextureFilter, TextureOptions};
use eframe::epaint::{Color32, TextureHandle};
use gstreamer::{Fraction, prelude::*};
use gstreamer::{PadTemplate, glib};
use gstreamer_video::subclass::prelude::*;
use gstreamer_video::video_frame::Readable;
use gstreamer_video::{VideoFrame, VideoFrameExt};
use ntscrs::yiq_fielding::{self, Rgbx8};
use std::fmt::Debug;
use std::sync::{Mutex, OnceLock};

use super::ntscrs_filter::NtscFilterSettings;
use super::process_gst_frame::process_gst_frame;

#[derive(Clone, glib::Boxed, Default)]
#[boxed_type(name = "SinkTexture")]
pub struct SinkTexture {
    pub handle: Option<TextureHandle>,
    pub pixel_aspect_ratio: Option<Fraction>,
}

impl SinkTexture {
    pub fn new() -> Self {
        Self {
            handle: None,
            pixel_aspect_ratio: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, glib::Boxed, Default)]
#[boxed_type(name = "VideoPreviewSetting")]
pub enum EffectPreviewSetting {
    #[default]
    Enabled,
    Disabled,
    SplitScreen(Rect),
}

impl Debug for SinkTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SinkTexture")
            .field("pixel_aspect_ratio", &self.pixel_aspect_ratio)
            .finish()
    }
}

#[derive(Debug, Clone, glib::Boxed, Default)]
#[boxed_type(name = "EguiCtx")]
pub struct EguiCtx(pub Option<Context>);

#[derive(glib::Properties, Default)]
#[properties(wrapper_type = super::elements::EguiSink)]
pub struct EguiSink {
    #[property(get, set)]
    texture: Mutex<SinkTexture>,
    #[property(get, set)]
    ctx: Mutex<EguiCtx>,
    #[property(get, set = Self::set_settings)]
    settings: Mutex<NtscFilterSettings>,
    #[property(get, set = Self::set_video_preview_mode)]
    preview_mode: Mutex<EffectPreviewSetting>,

    video_info: Mutex<Option<gstreamer_video::VideoInfo>>,
    last_frame: Mutex<
        Option<(
            gstreamer_video::VideoFrame<gstreamer_video::video_frame::Readable>,
            u64,
        )>,
    >,
}

impl EguiSink {
    fn set_settings(&self, value: NtscFilterSettings) {
        *self.settings.lock().unwrap() = value;
        let _ = self.update_texture();
    }

    fn set_video_preview_mode(&self, value: EffectPreviewSetting) {
        *self.preview_mode.lock().unwrap() = value;
        let _ = self.update_texture();
    }

    fn apply_effect(
        &self,
        vframe: &VideoFrame<Readable>,
        image: &mut ColorImage,
        rect: Option<yiq_fielding::Rect>,
    ) -> Result<(), gstreamer::FlowError> {
        let out_stride = image.width() * 4;
        process_gst_frame::<Rgbx8>(
            &vframe.as_video_frame_ref(),
            image.as_raw_mut(),
            out_stride,
            rect,
            &self.settings.lock().unwrap().0,
        )?;

        Ok(())
    }

    pub fn get_image(&self) -> Result<ColorImage, gstreamer::FlowError> {
        let vframe = self.last_frame.lock().unwrap();
        let (vframe, ..) = vframe.as_ref().ok_or(gstreamer::FlowError::Error)?;

        let width = vframe.width() as usize;
        let height = vframe.height() as usize;
        let mut image = ColorImage::filled([width, height], Color32::BLACK);
        self.apply_effect(vframe, &mut image, None)?;
        Ok(image)
    }

    pub fn update_texture(&self) -> Result<(), gstreamer::FlowError> {
        let mut tex = self.texture.lock().unwrap();
        let vframe = self.last_frame.lock().unwrap();
        let (vframe, ..) = vframe.as_ref().ok_or(gstreamer::FlowError::Error)?;
        tex.pixel_aspect_ratio = Some(vframe.info().par());

        let width = vframe.width() as usize;
        let height = vframe.height() as usize;
        let mut image = ColorImage::filled([width, height], Color32::BLACK);

        match *self.preview_mode.lock().unwrap() {
            EffectPreviewSetting::Enabled => {
                self.apply_effect(vframe, &mut image, None)?;
            }
            EffectPreviewSetting::Disabled => {
                // Copy directly to egui image when effect is disabled
                let src_buf = vframe.plane_data(0).or(Err(gstreamer::FlowError::Error))?;
                image.as_raw_mut().copy_from_slice(src_buf);
            }
            EffectPreviewSetting::SplitScreen(split) => {
                let src_buf = vframe.plane_data(0).or(Err(gstreamer::FlowError::Error))?;
                image.as_raw_mut().copy_from_slice(src_buf);

                let rect_to_blit_coord = |coord: f32, dim: usize| {
                    (coord * dim as f32).round().clamp(0.0, dim as f32) as usize
                };

                let rect = yiq_fielding::Rect::new(
                    rect_to_blit_coord(split.top(), height),
                    rect_to_blit_coord(split.left(), width),
                    rect_to_blit_coord(split.bottom(), height),
                    rect_to_blit_coord(split.right(), width),
                );

                self.apply_effect(vframe, &mut image, Some(rect))?;
            }
        }

        let Some(ctx) = &self.ctx.lock().unwrap().0 else {
            return Err(gstreamer::FlowError::Error);
        };

        let options = TextureOptions {
            magnification: TextureFilter::Nearest,
            minification: TextureFilter::Linear,
            ..Default::default()
        };
        match &mut tex.handle {
            Some(handle) => {
                handle.set(image, options);
            }
            None => {
                tex.handle = Some(ctx.load_texture("preview", image, options));
            }
        }
        ctx.request_repaint();

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
        static ELEMENT_METADATA: OnceLock<gstreamer::subclass::ElementMetadata> = OnceLock::new();
        Some(ELEMENT_METADATA.get_or_init(|| {
            gstreamer::subclass::ElementMetadata::new(
                "egui sink",
                "Sink/Video",
                "Video sink for egui texture",
                "valadaptive",
            )
        }))
    }

    fn pad_templates() -> &'static [gstreamer::PadTemplate] {
        static PAD_TEMPLATES: OnceLock<Vec<PadTemplate>> = OnceLock::new();
        PAD_TEMPLATES.get_or_init(|| {
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
        })
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
                *last_frame_num != frame_num || last.buffer() != buffer.as_ref()
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
