use std::convert::identity;

use gstreamer::{BufferRef, ClockTime, FlowError};
use gstreamer_video::{VideoFormat, VideoFrameExt, VideoFrameRef, VideoInterlaceMode};
use ntscrs::{
    settings::standard::NtscEffect,
    yiq_fielding::{
        Bgrx8, BlitInfo, DeinterlaceMode, PixelFormat, Rect, Rgbx8, Xbgr8, Xrgb16, Xrgb8, YiqField,
        YiqOwned, YiqView,
    },
};

fn frame_to_yiq(
    in_frame: &VideoFrameRef<&BufferRef>,
    field: YiqField,
) -> Result<YiqOwned, FlowError> {
    let width = in_frame.width() as usize;
    let height = in_frame.height() as usize;
    let in_stride = in_frame.plane_stride()[0] as usize;
    let in_data = in_frame.plane_data(0).or(Err(FlowError::Error))?;
    let in_format = in_frame.format();
    Ok(match in_format {
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
        _ => Err(FlowError::NotSupported)?,
    })
}

pub fn process_gst_frame<S: PixelFormat>(
    in_frame: &VideoFrameRef<&BufferRef>,
    out_frame: &mut [S::DataFormat],
    out_stride: usize,
    out_rect: Option<Rect>,
    settings: &NtscEffect,
) -> Result<(), FlowError> {
    let info = in_frame.info();

    let timestamp = in_frame.buffer().pts().ok_or(FlowError::Error)?.nseconds();
    let frame = (info.fps().numer() as u128 * (timestamp + 100) as u128
        / info.fps().denom() as u128) as u64
        / ClockTime::SECOND.nseconds();

    let blit_info = out_rect
        .map(|rect| {
            BlitInfo::new(
                rect,
                (rect.left, rect.top),
                out_stride,
                in_frame.height() as usize,
                false,
            )
        })
        .unwrap_or_else(|| {
            BlitInfo::from_full_frame(
                in_frame.width() as usize,
                in_frame.height() as usize,
                out_stride,
            )
        });

    match in_frame.info().interlace_mode() {
        VideoInterlaceMode::Progressive => {
            let field = settings.use_field.to_yiq_field(frame as usize);
            let mut yiq = frame_to_yiq(in_frame, field)?;
            let mut view = YiqView::from(&mut yiq);
            settings.apply_effect_to_yiq(&mut view, frame as usize, [1.0, 1.0]);
            view.write_to_strided_buffer::<S, _>(
                out_frame,
                blit_info,
                DeinterlaceMode::Bob,
                identity,
            );
        }
        VideoInterlaceMode::Interleaved | VideoInterlaceMode::Mixed => {
            let field = match (in_frame.is_tff(), in_frame.is_onefield()) {
                (true, true) => YiqField::Upper,
                (false, true) => YiqField::Lower,
                (true, false) => YiqField::InterleavedUpper,
                (false, false) => YiqField::InterleavedLower,
            };

            let mut yiq = frame_to_yiq(in_frame, field)?;
            let mut view = YiqView::from(&mut yiq);
            settings.apply_effect_to_yiq(
                &mut view,
                if in_frame.is_onefield() {
                    frame as usize * 2
                } else {
                    frame as usize
                },
                [1.0, 1.0],
            );
            view.write_to_strided_buffer::<S, _>(
                out_frame,
                blit_info,
                DeinterlaceMode::Skip,
                identity,
            );
        }
        _ => Err(FlowError::NotSupported)?,
    }

    Ok(())
}
