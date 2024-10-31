use gstreamer::prelude::*;

use super::{elements, gstreamer_error::GstreamerError};

pub fn initialize_gstreamer() -> Result<(), GstreamerError> {
    gstreamer::init()?;

    gstreamer::Element::register(
        None,
        "eguisink",
        gstreamer::Rank::NONE,
        elements::EguiSink::static_type(),
    )?;

    gstreamer::Element::register(
        None,
        "ntscfilter",
        gstreamer::Rank::NONE,
        elements::NtscFilter::static_type(),
    )?;

    gstreamer::Element::register(
        None,
        "videopadfilter",
        gstreamer::Rank::NONE,
        elements::VideoPadFilter::static_type(),
    )?;

    // PulseAudio has a severe bug that will greatly delay initial playback to the point of unusability:
    // https://gitlab.freedesktop.org/pulseaudio/pulseaudio/-/issues/1383
    // A fix was merged a *year* ago, but the Pulse devs, in their infinite wisdom, won't give it to us until their
    // next major release, the first RC of which will apparently arrive "soon":
    // https://gitlab.freedesktop.org/pulseaudio/pulseaudio/-/issues/3757#note_2038416
    // Until then, disable it and pray that someone writes a PipeWire sink so we don't have to deal with any more
    // bugs like this
    if let Some(sink) = gstreamer::ElementFactory::find("pulsesink") {
        sink.set_rank(gstreamer::Rank::NONE);
    }

    Ok(())
}
