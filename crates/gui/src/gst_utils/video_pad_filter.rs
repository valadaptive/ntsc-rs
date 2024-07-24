use std::sync::RwLock;

use gstreamer::glib;
use gstreamer_video::subclass::prelude::*;
use gstreamer_video::{VideoFormat, VideoFrameExt};
use once_cell::sync::Lazy;

#[derive(Default)]
pub struct VideoPadFilter {
    info: RwLock<Option<gstreamer_video::VideoInfo>>,
}

impl VideoPadFilter {}

#[glib::object_subclass]
impl ObjectSubclass for VideoPadFilter {
    const NAME: &'static str = "ntscrs_video_pad";
    type Type = super::elements::VideoPadFilter;
    type ParentType = gstreamer_video::VideoFilter;
}

impl ObjectImpl for VideoPadFilter {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(Vec::new);

        PROPERTIES.as_ref()
    }
}

impl GstObjectImpl for VideoPadFilter {}

impl ElementImpl for VideoPadFilter {
    fn metadata() -> Option<&'static gstreamer::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gstreamer::subclass::ElementMetadata> = Lazy::new(|| {
            gstreamer::subclass::ElementMetadata::new(
                "Video Pad (for YUV)",
                "Filter/Effect/Converter/Video",
                "Applies padding to extend a video to even dimensions",
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

impl BaseTransformImpl for VideoPadFilter {
    const MODE: gstreamer_base::subclass::BaseTransformMode =
        gstreamer_base::subclass::BaseTransformMode::NeverInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn transform_caps(
        &self,
        direction: gstreamer::PadDirection,
        caps: &gstreamer::Caps,
        filter: Option<&gstreamer::Caps>,
    ) -> Option<gstreamer::Caps> {
        let other_caps = match direction {
            gstreamer::PadDirection::Unknown => None,
            gstreamer::PadDirection::Src => Some({
                let mut caps = caps.clone();
                for s in caps.make_mut().iter_mut() {
                    if let Ok(width) = s.value("width").ok()?.get::<i32>() {
                        if width % 2 == 0 {
                            s.set_value(
                                "width",
                                (&gstreamer::IntRange::<i32>::new(width - 1, width)).into(),
                            );
                        }
                    }

                    if let Ok(height) = s.value("height").ok()?.get::<i32>() {
                        if height % 2 == 0 {
                            s.set_value(
                                "height",
                                (&gstreamer::IntRange::<i32>::new(height - 1, height)).into(),
                            );
                        }
                    }
                }
                caps
            }),
            gstreamer::PadDirection::Sink => Some({
                let mut out_caps = gstreamer::Caps::new_empty();

                {
                    let out_caps = out_caps.get_mut().unwrap();

                    for (idx, s) in caps.iter().enumerate() {
                        let mut s_out = s.to_owned();
                        if let Ok(mut width) = s_out.value("width").ok()?.get::<i32>() {
                            width += width % 2;
                            s_out.set_value("width", (&width).into());
                        }

                        if let Ok(mut height) = s_out.value("height").ok()?.get::<i32>() {
                            height += height % 2;
                            s_out.set_value("height", (&height).into());
                        }

                        out_caps.append_structure(s_out);
                        out_caps.set_features(idx, caps.features(idx).map(|f| f.to_owned()));
                    }
                }

                out_caps
            }),
        }?;

        match filter {
            Some(filter) => {
                Some(filter.intersect_with_mode(&other_caps, gstreamer::CapsIntersectMode::First))
            }
            None => Some(other_caps),
        }
    }
}

impl VideoFilterImpl for VideoPadFilter {
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
        let in_width = in_frame.width() as usize;
        let in_height = in_frame.height() as usize;
        let in_stride = in_frame.plane_stride()[0] as usize;
        let in_format = in_frame.format();
        let in_data = in_frame
            .plane_data(0)
            .or(Err(gstreamer::FlowError::Error))?;

        let out_width = out_frame.width() as usize;
        let out_height = out_frame.height() as usize;
        let out_stride = out_frame.plane_stride()[0] as usize;
        let out_format = out_frame.format();
        let out_data = out_frame
            .plane_data_mut(0)
            .or(Err(gstreamer::FlowError::Error))?;

        if in_format != out_format || out_width < in_width || out_height < in_height {
            return Err(gstreamer::FlowError::NotSupported);
        }

        let pixel_stride = in_frame.comp_pstride(0) as usize;

        out_data
            .chunks_exact_mut(out_stride)
            .enumerate()
            .for_each(|(row_idx, chunk)| {
                let dst_row = &mut chunk[0..(out_width * pixel_stride)];
                let src_idx = row_idx.min(in_height - 1);
                let src_row =
                    &in_data[in_stride * src_idx..in_stride * src_idx + (in_width * pixel_stride)];
                dst_row[0..(in_width * pixel_stride)].copy_from_slice(src_row);
                dst_row[(in_width * pixel_stride)..].fill(*src_row.last().unwrap());
            });

        Ok(gstreamer::FlowSuccess::Ok)
    }
}
