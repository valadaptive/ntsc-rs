use std::{sync::{atomic::AtomicBool, Arc, Mutex}, error::Error, fmt::Display};
use gstreamer::{element_error, element_warning, glib, prelude::*};
use super::{scale_from_caps, gstreamer_error::GstreamerError};
use log::debug;

#[derive(Clone, Debug, glib::Boxed)]
#[boxed_type(name = "ErrorValue")]
struct ErrorValue(Arc<Mutex<Option<GstreamerError>>>);

#[derive(Debug, Clone)]
pub enum PipelineError {
    GlibError(glib::Error),
    NoVideoError
}

impl Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GlibError(e) => e.fmt(f),
            Self::NoVideoError => f.write_str("Pipeline source has no video")
        }
    }
}

impl Error for PipelineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::GlibError(e) => Some(e),
            Self::NoVideoError => None
        }
    }
}

impl From<glib::Error> for PipelineError {
    fn from(value: glib::Error) -> Self {
        Self::GlibError(value)
    }
}

pub fn create_pipeline<
    AudioElemCallback: FnOnce(&gstreamer::Pipeline) -> Result<Option<gstreamer::Element>, GstreamerError>
        + Send
        + Sync
        + 'static,
    VideoElemCallback: FnOnce(&gstreamer::Pipeline) -> Result<gstreamer::Element, GstreamerError>
        + Send
        + Sync
        + 'static,
    BusHandler: Fn(&gstreamer::Bus, &gstreamer::Message) -> gstreamer::BusSyncReply + Send + Sync + 'static,
    PipelineCallback: FnOnce(Result<gstreamer::Pipeline, PipelineError>) + Send + Sync + 'static,
>(
    src_pad: gstreamer::Element,
    audio_sink: AudioElemCallback,
    video_sink: VideoElemCallback,
    bus_handler: BusHandler,
    duration: Option<gstreamer::ClockTime>,
    initial_scale: Option<usize>,
    initial_still_image_framerate: gstreamer::Fraction,
    callback: Option<PipelineCallback>,
) -> Result<gstreamer::Pipeline, GstreamerError> {
    let pipeline = gstreamer::Pipeline::default();
    let decodebin = gstreamer::ElementFactory::make("decodebin").build()?;
    pipeline.add_many([&src_pad, &decodebin])?;
    gstreamer::Element::link_many([&src_pad, &decodebin])?;

    let has_audio = Mutex::new(false);
    let has_video = Arc::new(Mutex::new(false));
    let has_video_for_bus_handler = Arc::clone(&has_video);

    let handler_id: Arc<Mutex<Option<glib::SignalHandlerId>>> = Arc::new(Mutex::new(None));
    let handler_id_for_handler = Arc::clone(&handler_id);

    let audio_sink = Mutex::new(Some(audio_sink));
    let video_sink = Mutex::new(Some(video_sink));

    let pipeline_weak = gstreamer::prelude::ObjectExt::downgrade(&pipeline);
    let pad_added_handler = decodebin.connect_pad_added(move |dbin, src_pad| {
        let pipeline = &pipeline_weak;
        let handler_id = &handler_id_for_handler;
        // Try to detect whether the raw stream decodebin provided us with
        // just now is either audio or video (or none of both, e.g. subtitles).
        let (is_audio, is_video) = {
            let media_type = src_pad.current_caps().and_then(|caps| {
                debug!("{:?}", &caps);
                caps.structure(0).map(|s| {
                    let name = s.name();
                    (name.starts_with("audio/"), name.starts_with("video/"))
                })
            });

            match media_type {
                None => {
                    element_warning!(
                        dbin,
                        gstreamer::CoreError::Negotiation,
                        ("Failed to get media type from pad {}", src_pad.name())
                    );

                    return;
                }
                Some(media_type) => media_type,
            }
        };

        let insert_sink = |is_audio, is_video| -> Result<(), GstreamerError> {
            let mut has_audio = has_audio.lock().unwrap();
            let mut has_video = has_video.lock().unwrap();
            if is_audio && !*has_audio {
                debug!("connected audio");

                if let Some(pipeline) = pipeline.upgrade() {
                    let audio_sink = audio_sink.lock().unwrap().take();
                    if let Some(sink) = audio_sink.and_then(|sink| Some(sink(&pipeline))) {
                        if let Some(sink) = sink? {
                            let audio_queue = gstreamer::ElementFactory::make("queue").build()?;
                            let audio_convert =
                                gstreamer::ElementFactory::make("audioconvert").build()?;
                            let audio_resample =
                                gstreamer::ElementFactory::make("audioresample").build()?;

                            let audio_elements = &[&audio_queue, &audio_convert, &audio_resample];
                            pipeline.add_many(audio_elements)?;
                            gstreamer::Element::link_many(audio_elements)?;

                            audio_resample.link(&sink)?;
                            sink.sync_state_with_parent()?;

                            for e in audio_elements {
                                e.sync_state_with_parent()?
                            }

                            // Get the queue element's sink pad and link the decodebin's newly created
                            // src pad for the audio stream to it.
                            let sink_pad = audio_queue
                                .static_pad("sink")
                                .expect("queue has no sinkpad");
                            src_pad.link(&sink_pad)?;
                        }
                    }

                    *has_audio = true;
                }

                // has_audio
            } else if is_video && !*has_video {
                debug!("connected video");

                if let Some(pipeline) = pipeline.upgrade() {
                    let video_sink = video_sink.lock().unwrap().take();
                    if let Some(video_sink) = video_sink {
                        let video_queue = gstreamer::ElementFactory::make("queue")
                            .name("video_queue")
                            .build()?;
                        let video_convert =
                            gstreamer::ElementFactory::make("videoconvert").build()?;
                        let video_rate = gstreamer::ElementFactory::make("videorate")
                            .name("video_rate")
                            .build()?;
                        let video_scale = gstreamer::ElementFactory::make("videoscale")
                            .name("video_scale")
                            .build()?;
                        let caps_filter = gstreamer::ElementFactory::make("capsfilter")
                            .name("caps_filter")
                            .build()?;
                        let framerate_caps_filter = gstreamer::ElementFactory::make("capsfilter")
                            .name("framerate_caps_filter")
                            .build()?;

                        let video_elements = &[
                            &video_queue,
                            &video_convert,
                            &video_rate,
                            &video_scale,
                            &caps_filter,
                            &framerate_caps_filter,
                        ];
                        pipeline.add_many(video_elements)?;
                        gstreamer::Element::link_many(video_elements)?;

                        let video_sink = video_sink(&pipeline)?;
                        framerate_caps_filter.link(&video_sink)?;

                        for e in video_elements {
                            e.sync_state_with_parent()?;
                        }
                        video_sink.sync_state_with_parent()?;

                        // Get the queue element's sink pad and link the decodebin's newly created
                        // src pad for the video stream to it.
                        let sink_pad = video_queue
                            .static_pad("sink")
                            .expect("queue has no sinkpad");

                        let caps = src_pad.current_caps();

                        let framerate = caps.as_ref().and_then(|caps| {
                            Some(
                                caps.structure(0)?
                                    .get::<gstreamer::Fraction>("framerate")
                                    .ok()?,
                            )
                        });

                        let is_still_image = match framerate {
                            Some(framerate) => framerate.numer() == 0,
                            None => false,
                        };

                        if caps.is_some() && initial_scale.is_some() {
                            if let Some((width, height)) =
                                scale_from_caps(&caps.unwrap(), initial_scale.unwrap())
                            {
                                caps_filter.set_property(
                                    "caps",
                                    gstreamer_video::VideoCapsBuilder::default()
                                        .width(width)
                                        .height(height)
                                        .build(),
                                );
                            }
                        }

                        if is_still_image {
                            let image_freeze = gstreamer::ElementFactory::make("imagefreeze")
                                .name("still_image_freeze")
                                .build()?;
                            let video_caps = gstreamer_video::VideoCapsBuilder::new()
                                .framerate(initial_still_image_framerate)
                                .build();
                            framerate_caps_filter.set_property("caps", video_caps);

                            pipeline.add(&image_freeze)?;
                            src_pad.link(&image_freeze.static_pad("sink").unwrap())?;
                            gstreamer::Element::link_many([&image_freeze, &video_queue])?;
                            image_freeze.sync_state_with_parent()?;

                            // We cannot move this functionality into create_render_job. If we do this seek outside of
                            // this callback, the output video will be truncated. No idea why.
                            if let Some(duration) = duration {
                                image_freeze.seek(
                                    1.0,
                                    gstreamer::SeekFlags::FLUSH | gstreamer::SeekFlags::ACCURATE,
                                    gstreamer::SeekType::Set,
                                    gstreamer::ClockTime::ZERO,
                                    gstreamer::SeekType::Set,
                                    duration,
                                )?;
                            }
                        } else {
                            src_pad.link(&sink_pad)?;
                        }
                    }
                }

                *has_video = true;
            }

            // We have both streams. No need to search for more.
            if *has_audio && *has_video {
                let mut id = handler_id.lock().unwrap();
                let id = id.take();
                if let Some(id) = id {
                    dbin.disconnect(id);
                }
            }

            Ok(())
        };

        // When adding and linking new elements in a callback fails, error information is often sparse.
        // GStreamer's built-in debugging can be hard to link back to the exact position within the code
        // that failed. Since callbacks are called from random threads within the pipeline, it can get hard
        // to get good error information. The macros used in the following can solve that. With the use
        // of those, one can send arbitrary rust types (using the pipeline's bus) into the mainloop.
        // What we send here is unpacked down below, in the iteration-code over sent bus-messages.
        if let Err(err) = insert_sink(is_audio, is_video) {
            debug!("got error: {:?}", &err);
            element_error!(
                dbin,
                gstreamer::LibraryError::Failed,
                ("Failed to insert sink"),
                details: gstreamer::Structure::builder("error-details")
                            .field("error",
                                   &ErrorValue(Arc::new(Mutex::new(Some(err)))))
                            .build()
            );
        }
    });

    handler_id.lock().unwrap().replace(pad_added_handler);

    let bus = pipeline
        .bus()
        .expect("Pipeline without bus. Shouldn't happen!");

    let finished_loading = AtomicBool::new(false);
    let handler_callback = Mutex::new(callback);
    bus.set_sync_handler(move |bus, msg| {
        if !finished_loading.load(std::sync::atomic::Ordering::SeqCst) {
            match msg.view() {
                gstreamer::MessageView::AsyncDone(a) => {
                    if let Some(pipeline) = a
                        .src()
                        .and_then(|a| a.downcast_ref::<gstreamer::Pipeline>())
                    {
                        debug!("pipeline state change done");

                        let mut id = handler_id.lock().unwrap();
                        let id = id.take();
                        if let Some(id) = id {
                            decodebin.disconnect(id);
                        }
                        if let Some(callback) = handler_callback.lock().unwrap().take() {
                            if *has_video_for_bus_handler.lock().unwrap() {
                                callback(Ok(pipeline.clone()));
                            } else {
                                callback(Err(PipelineError::NoVideoError));
                            }
                        }
                        finished_loading.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                },
                gstreamer::MessageView::Error(e) => {
                    if let Some(callback) = handler_callback.lock().unwrap().take() {
                        callback(Err(PipelineError::GlibError(e.error())));
                    }
                    finished_loading.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                _ => {}
            }
        }

        bus_handler(bus, msg)
    });

    Ok(pipeline)
}
