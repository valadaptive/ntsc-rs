use eframe::egui::Context;
use eframe::egui::{TextureFilter, TextureOptions};
use eframe::epaint::{Color32, ColorImage, TextureHandle};
use gstreamer::glib::once_cell::sync::Lazy;
use gstreamer::prelude::*;
use gstreamer::{glib, PadTemplate};
use gstreamer_video::subclass::prelude::*;
use gstreamer_video::video_frame::Readable;
use gstreamer_video::VideoFrame;
use ntscrs::yiq_fielding::Rgbx8;
use std::fmt::Debug;
use std::sync::Mutex;

use super::ntscrs_filter::NtscFilterSettings;
use super::process_gst_frame::process_gst_frame;

#[derive(Clone, glib::Boxed, Default)]
#[boxed_type(name = "SinkTexture")]
pub struct SinkTexture(pub Option<TextureHandle>);

#[derive(Debug, Clone, Copy, PartialEq, glib::Boxed, Default)]
#[boxed_type(name = "VideoPreviewSetting")]
pub enum EffectPreviewSetting {
    #[default]
    Enabled,
    Disabled,
    SplitScreen(f64),
}

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

#[derive(Debug, Clone, glib::Boxed, Default)]
#[boxed_type(name = "EguiCtx")]
pub struct EguiCtx(pub Option<Context>);

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
    ) -> Result<(), gstreamer::FlowError> {
        let out_stride = image.width() * 4;
        process_gst_frame::<u8, Rgbx8>(
            &vframe.as_video_frame_ref(),
            image.as_raw_mut(),
            out_stride,
            &self.settings.lock().unwrap().0,
        )?;

        Ok(())
    }

    pub fn update_texture(&self) -> Result<(), gstreamer::FlowError> {
        let mut tex = self.texture.lock().unwrap();
        let vframe = self.last_frame.lock().unwrap();
        let (vframe, ..) = vframe.as_ref().ok_or(gstreamer::FlowError::Error)?;

        let width = vframe.width() as usize;
        let height = vframe.height() as usize;
        let mut image = ColorImage::new([width, height], Color32::BLACK);

        match *self.preview_mode.lock().unwrap() {
            EffectPreviewSetting::Enabled => {
                self.apply_effect(vframe, &mut image)?;
            }
            #[allow(illegal_floating_point_literal_pattern)]
            EffectPreviewSetting::Disabled | EffectPreviewSetting::SplitScreen(0f64) => {
                // Copy directly to egui image when effect is disabled
                let src_buf = vframe.plane_data(0).or(Err(gstreamer::FlowError::Error))?;
                image.as_raw_mut().copy_from_slice(src_buf);
            }
            EffectPreviewSetting::SplitScreen(split) => {
                let buf = vframe.plane_data(0).or(Err(gstreamer::FlowError::Error))?;
                self.apply_effect(vframe, &mut image)?;

                let split_boundary =
                    (split * width as f64).round().clamp(0.0, width as f64) as usize;
                let image_data = image.as_raw_mut();
                image_data
                    .chunks_exact_mut(width * 4)
                    .zip(buf.chunks_exact(width * 4))
                    .for_each(|(img_row, vid_row)| {
                        img_row[split_boundary * 4..]
                            .copy_from_slice(&vid_row[split_boundary * 4..]);
                    });
            }
        }

        tex.0.as_mut().ok_or(gstreamer::FlowError::Error)?.set(
            image,
            TextureOptions {
                magnification: TextureFilter::Nearest,
                minification: TextureFilter::Linear,
                ..Default::default()
            },
        );
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
