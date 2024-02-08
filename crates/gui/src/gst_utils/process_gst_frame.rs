use gstreamer::{BufferRef, ClockTime, FlowError};
use gstreamer_video::{VideoFormat, VideoFrameRef, VideoInterlaceMode};
use ntscrs::{
    settings::NtscEffect,
    yiq_fielding::{
        Bgrx8, DeinterlaceMode, PixelFormat, Rgbx8, Xbgr8, Xrgb16, Xrgb8, YiqField, YiqOwned,
        YiqView,
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

pub fn process_gst_frame<T, OutFormat: PixelFormat<DataFormat = T>>(
    in_frame: &VideoFrameRef<&BufferRef>,
    out_frame: &mut [T],
    out_stride: usize,
    settings: &NtscEffect,
) -> Result<(), FlowError> {
    let info = in_frame.info();

    let timestamp = in_frame.buffer().pts().ok_or(FlowError::Error)?.nseconds();
    let frame = (info.fps().numer() as u128 * (timestamp + 100) as u128
        / info.fps().denom() as u128) as u64
        / ClockTime::SECOND.nseconds();

    match in_frame.info().interlace_mode() {
        VideoInterlaceMode::Progressive => {
            let field = settings.use_field.to_yiq_field(frame as usize);
            let mut yiq = frame_to_yiq(in_frame, field)?;
            let mut view = YiqView::from(&mut yiq);
            settings.apply_effect_to_yiq(&mut view, frame as usize);
            view.write_to_strided_buffer::<OutFormat>(out_frame, out_stride, DeinterlaceMode::Bob);
        }
        VideoInterlaceMode::Interleaved | VideoInterlaceMode::Mixed => {
            let top_field_first = in_frame.is_tff();
            let (first_field, second_field) = if top_field_first {
                (YiqField::Upper, YiqField::Lower)
            } else {
                (YiqField::Lower, YiqField::Upper)
            };

            let mut yiq_first = frame_to_yiq(in_frame, first_field)?;
            let mut view_first = YiqView::from(&mut yiq_first);
            settings.apply_effect_to_yiq(&mut view_first, frame as usize * 2);
            view_first.write_to_strided_buffer::<OutFormat>(
                out_frame,
                out_stride,
                DeinterlaceMode::Skip,
            );

            if !in_frame.is_onefield() {
                let mut yiq_second = frame_to_yiq(in_frame, second_field)?;
                let mut view_second = YiqView::from(&mut yiq_second);
                settings.apply_effect_to_yiq(&mut view_second, frame as usize * 2 + 1);
                view_second.write_to_strided_buffer::<OutFormat>(
                    out_frame,
                    out_stride,
                    DeinterlaceMode::Skip,
                );
            }
        }
        _ => Err(FlowError::NotSupported)?,
    }

    Ok(())
}
