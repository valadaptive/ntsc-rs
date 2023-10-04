#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::borrow::Cow;
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::error::Error;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fmt::Display;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::marker::PhantomData;
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::sync::Mutex;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;
use std::thread::JoinHandle;

use async_executor::Executor;
use async_executor::LocalExecutor;
use async_task::Task;
use eframe::egui;
use eframe::epaint::vec2;
use eframe::App;
use futures_lite::future;
use futures_lite::future::PollOnce;
use futures_lite::pin;
use futures_lite::Future;
use futures_lite::FutureExt;
use gstreamer::bus::BusWatchGuard;
use gstreamer::element_error;
use gstreamer::element_warning;
use gstreamer::glib;
use gstreamer::glib::clone::Downgrade;
use gstreamer::glib::subclass::prelude::*;
use gstreamer::glib::PropertyGet;
use gstreamer::glib::SignalHandlerId;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::ObjectSubclass;
use gstreamer::PadLinkError;
use gstreamer_controller::prelude::*;
use gstreamer_video::subclass::prelude::*;
use gstreamer_video::VideoCapsBuilder;
use gstreamer_video::VideoFormat;
use gui::expression_parser::eval_expression_string;
use gui::gst_utils::clock_format::clock_time_format;
use gui::gst_utils::clock_format::clock_time_formatter;
use gui::gst_utils::clock_format::clock_time_parser;
use gui::gst_utils::egui_sink::EguiCtx;
use gui::gst_utils::ntscrs_filter;
use gui::gst_utils::ntscrs_filter::NtscFilterSettings;
use gui::gst_utils::NtscFilter;
use image::ImageError;
use ntscrs::settings::NtscEffectFullSettings;
use ntscrs::settings::ParseSettingsError;
use ntscrs::settings::SettingDescriptor;
use ntscrs::settings::SettingsList;
use rfd::FileHandle;
use snafu::prelude::*;

use gui::gst_utils::{egui_sink::SinkTexture, EguiSink};
use gui::timeline::Timeline;

#[derive(Debug, Clone)]
enum GstreamerError {
    GlibError(glib::Error),
    BoolError(glib::BoolError),
    PadLinkError(gstreamer::PadLinkError),
    StateChangeError(gstreamer::StateChangeError),
}

impl From<glib::Error> for GstreamerError {
    fn from(value: glib::Error) -> Self {
        GstreamerError::GlibError(value)
    }
}

impl From<glib::BoolError> for GstreamerError {
    fn from(value: glib::BoolError) -> Self {
        GstreamerError::BoolError(value)
    }
}

impl From<gstreamer::PadLinkError> for GstreamerError {
    fn from(value: gstreamer::PadLinkError) -> Self {
        GstreamerError::PadLinkError(value)
    }
}

impl From<gstreamer::StateChangeError> for GstreamerError {
    fn from(value: gstreamer::StateChangeError) -> Self {
        GstreamerError::StateChangeError(value)
    }
}

impl Display for GstreamerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GstreamerError::GlibError(e) => e.fmt(f),
            GstreamerError::BoolError(e) => e.fmt(f),
            GstreamerError::PadLinkError(e) => e.fmt(f),
            GstreamerError::StateChangeError(e) => e.fmt(f),
        }
    }
}

impl Error for GstreamerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            GstreamerError::GlibError(e) => Some(e),
            GstreamerError::BoolError(e) => Some(e),
            GstreamerError::PadLinkError(e) => Some(e),
            GstreamerError::StateChangeError(e) => Some(e),
        }
    }
}

#[derive(Debug, Snafu)]
enum ApplicationError {
    #[snafu(display("Error loading video: {source}"))]
    LoadVideo { source: GstreamerError },

    #[snafu(display("Error creating pipeline: {source}"))]
    CreatePipeline { source: GstreamerError },

    #[snafu(display("Error reading JSON: {source}"))]
    JSONRead { source: std::io::Error },

    #[snafu(display("Error parsing JSON: {source}"))]
    JSONParse { source: ParseSettingsError },

    #[snafu(display("Error saving JSON: {source}"))]
    JSONSave { source: std::io::Error },
}

fn initialize_gstreamer() -> Result<(), GstreamerError> {
    gstreamer::init()?;

    gstreamer::Element::register(
        None,
        "eguisink",
        gstreamer::Rank::None,
        EguiSink::static_type(),
    )?;

    gstreamer::Element::register(
        None,
        "ntscfilter",
        gstreamer::Rank::None,
        NtscFilter::static_type(),
    )?;

    // PulseAudio has a severe bug that will greatly delay initial playback to the point of unusability:
    // https://gitlab.freedesktop.org/pulseaudio/pulseaudio/-/issues/1383
    // A fix was merged a *year* ago, but the Pulse devs, in their infinite wisdom, won't give it to us until their
    // next major release, the first RC of which will apparently arrive "soon":
    // https://gitlab.freedesktop.org/pulseaudio/pulseaudio/-/issues/3757#note_2038416
    // Until then, disable it and pray that someone writes a PipeWire sink so we don't have to deal with any more
    // bugs like this
    if let Some(sink) = gstreamer::ElementFactory::find("pulsesink") {
        sink.set_rank(gstreamer::Rank::None);
    }

    Ok(())
}

fn format_percentage(n: f64, prec: RangeInclusive<usize>) -> String {
    format!("{:.*}%", prec.start().max(&2) - 2, n * 100.0)
}

fn main() -> Result<(), Box<dyn Error>> {
    initialize_gstreamer()?;

    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(1200.0, 720.0)),
        ..Default::default()
    };
    Ok(eframe::run_native(
        "ntsc-rs",
        options,
        Box::new(|cc| {
            //let mut app = Box::<NtscApp>::default();
            let ctx = &cc.egui_ctx;
            let app = Box::new(NtscApp::new(ctx.clone()));
            if let Some(storage) = cc.storage {
                let path = storage.get_string("image_path");
                if let Some(path) = path {
                    if path != "" {
                        // TODO
                    }
                }
            }
            app
        }),
    )?)
}

#[derive(Debug)]
enum PipelineInfoState {
    Loading,
    Loaded,
    Error(GstreamerError),
}

struct PipelineInfo {
    pipeline: gstreamer::Pipeline,
    state: Arc<Mutex<PipelineInfoState>>,
    path: PathBuf,
    file_src: gstreamer::Element,
    egui_sink: gstreamer::Element,
    //filter: gstreamer::Element,
    last_seek_pos: gstreamer::ClockTime,
    preview: egui::TextureHandle,
    at_eos: Arc<Mutex<bool>>,
    is_still_image: Arc<Mutex<bool>>,
    framerate: Arc<Mutex<Option<gstreamer::Fraction>>>,
}

impl PipelineInfo {
    fn toggle_playing(&self) -> Result<(), gstreamer::StateChangeError> {
        match self.pipeline.current_state() {
            gstreamer::State::Paused | gstreamer::State::Ready => {
                self.pipeline.set_state(gstreamer::State::Playing)?;
            }
            gstreamer::State::Playing => {
                self.pipeline.set_state(gstreamer::State::Paused)?;
            }
            _ => {}
        }

        Ok(())
    }
}

#[derive(Debug)]
struct VideoZoom {
    scale: f64,
    fit: bool,
}

#[derive(Debug)]
struct VideoScale {
    scale: usize,
    enabled: bool,
}

#[derive(Debug)]
enum RenderJobState {
    Waiting,
    Rendering,
    Paused,
    Complete,
    Error(GstreamerError),
}

#[derive(Debug)]
struct RenderJob {
    settings: RenderSettings,
    pipeline: gstreamer::Pipeline,
    state: Arc<Mutex<RenderJobState>>,
    last_progress: f64,
    /// Used for estimating time remaining. A queue that holds (progress, timestamp) pairs.
    progress_samples: VecDeque<(f64, f64)>,
    start_time: Option<f64>,
    estimated_time_remaining: Option<f64>,
}

const NUM_PROGRESS_SAMPLES: usize = 5;
const PROGRESS_SAMPLE_TIME_DELTA: f64 = 1.0;

impl Drop for RenderJob {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gstreamer::State::Null);
    }
}

#[derive(Debug, Clone)]
struct H264Settings {
    // Quality / constant rate factor (0-51)
    crf: u8,
    // 0-8 for libx264 presets veryslow-ultrafast
    encode_speed: u8,
    // Enable 10-bit color
    ten_bit: bool,
    // Subsample chroma to 4:2:0
    chroma_subsampling: bool,
}

impl Default for H264Settings {
    fn default() -> Self {
        Self {
            crf: 23,
            encode_speed: 5,
            ten_bit: false,
            chroma_subsampling: true,
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
enum Ffv1BitDepth {
    #[default]
    Bits8,
    Bits10,
    Bits12,
}

impl Ffv1BitDepth {
    fn label(&self) -> &'static str {
        match self {
            Ffv1BitDepth::Bits8 => "8-bit",
            Ffv1BitDepth::Bits10 => "10-bit",
            Ffv1BitDepth::Bits12 => "12-bit",
        }
    }
}

#[derive(Debug, Clone)]
struct Ffv1Settings {
    bit_depth: Ffv1BitDepth,
    // Subsample chroma to 4:2:0
    chroma_subsampling: bool,
}

impl Default for Ffv1Settings {
    fn default() -> Self {
        Self {
            bit_depth: Ffv1BitDepth::default(),
            chroma_subsampling: false,
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
enum OutputCodec {
    #[default]
    H264,
    Ffv1,
}

impl OutputCodec {
    fn label(&self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::Ffv1 => "FFV1 (Lossless)",
        }
    }
}

#[derive(Default, Debug, Clone)]
struct RenderSettings {
    output_codec: OutputCodec,
    // we want to keep these around even if the user changes their mind and selects ffv1, so they don't lose the
    // settings if they change back
    h264_settings: H264Settings,
    ffv1_settings: Ffv1Settings,
    output_path: PathBuf,
    duration: gstreamer::ClockTime,
}

#[derive(Default, PartialEq, Eq)]
enum LeftPanelState {
    #[default]
    EffectSettings,
    RenderSettings,
}

type AppFn = Box<dyn FnOnce(&mut NtscApp) -> Result<(), ApplicationError> + Send>;

struct CallbackFutureInner<T: Unpin> {
    waker: Option<Waker>,
    resolved: Option<T>,
}

struct CallbackFuture<T: Unpin>(Arc<Mutex<CallbackFutureInner<T>>>);

impl<T: Unpin> CallbackFuture<T> {
    fn new() -> Self {
        CallbackFuture(Arc::new(Mutex::new(CallbackFutureInner {
            waker: None,
            resolved: None,
        })))
    }

    fn resolver(&self) -> impl FnOnce(T) {
        let weak = Arc::downgrade(&self.0);
        move |value| {
            if let Some(inner) = weak.upgrade() {
                let mut inner = inner.lock().unwrap();
                inner.resolved = Some(value);
                if let Some(waker) = inner.waker.take() {
                    waker.wake();
                }
            }
        }
    }
}

impl<T: Unpin> Future for CallbackFuture<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut inner = self.0.lock().unwrap();
        match inner.resolved.take() {
            None => {
                inner.waker = Some(cx.waker().clone());
                Poll::Pending
            }
            Some(resolved) => Poll::Ready(resolved),
        }
    }
}

struct AppExecutor {
    waker: Waker,
    ctx: egui::Context,
    run_tasks: Vec<Pin<Box<dyn Future<Output = Option<AppFn>> + Send>>>,
    queued_instant: VecDeque<AppFn>,
}

impl AppExecutor {
    fn new(ctx: egui::Context) -> Self {
        let waker_ctx = ctx.clone();
        AppExecutor {
            waker: waker_fn::waker_fn(move || waker_ctx.request_repaint()),
            ctx,
            run_tasks: Vec::new(),
            queued_instant: VecDeque::new(),
        }
    }
    fn tick(&mut self, app: &mut NtscApp) {
        let mut i = 0usize;
        while i < self.run_tasks.len() {
            let task = &mut self.run_tasks[i];
            match task.poll(&mut Context::from_waker(&self.waker)) {
                Poll::Pending => {}
                Poll::Ready(f) => {
                    if let Some(f) = f {
                        let res = f(app);
                        app.handle_result(res);
                    }
                    self.run_tasks.swap_remove(i);
                }
            }

            i += 1;
        }

        for queued_fn in self.queued_instant.drain(..) {
            let res = queued_fn(app);
            app.handle_result(res);
        }
    }

    fn spawn(
        &mut self,
        future: impl Future<Output = Option<AppFn>> + 'static + Send,
        next_frame: bool,
    ) {
        let mut boxed = Box::pin(future);
        if next_frame {
            self.run_tasks.push(boxed);
            self.ctx.request_repaint();
        } else {
            match boxed.poll(&mut Context::from_waker(&self.waker)) {
                Poll::Ready(f) => {
                    if let Some(f) = f {
                        self.queued_instant.push_back(f);
                    }
                }
                Poll::Pending => {
                    self.run_tasks.push(boxed);
                }
            }
        }
    }
}

struct NtscApp {
    settings_list: SettingsList,
    executor: Arc<Mutex<AppExecutor>>,
    pipeline: Option<PipelineInfo>,
    video_zoom: VideoZoom,
    video_scale: VideoScale,
    left_panel_state: LeftPanelState,
    effect_settings: NtscEffectFullSettings,
    render_settings: RenderSettings,
    render_jobs: Vec<RenderJob>,
    settings_json_paste: String,
    last_error: Option<String>,
}

impl NtscApp {
    fn new(ctx: egui::Context) -> Self {
        Self {
            settings_list: SettingsList::new(),
            pipeline: None,
            executor: Arc::new(Mutex::new(AppExecutor::new(ctx.clone()))),
            video_zoom: VideoZoom {
                scale: 1.0,
                fit: true,
            },
            video_scale: VideoScale {
                scale: 480,
                enabled: false,
            },
            left_panel_state: LeftPanelState::default(),
            effect_settings: NtscEffectFullSettings::default(),
            render_settings: RenderSettings::default(),
            render_jobs: Vec::new(),
            settings_json_paste: String::new(),
            last_error: None,
        }
    }
}

#[derive(Debug, Snafu)]
enum LoadImageError {
    #[snafu()]
    IO { source: std::io::Error },
    #[snafu()]
    Image { source: ImageError },
}

#[derive(Clone, Debug, glib::Boxed)]
#[boxed_type(name = "ErrorValue")]
struct ErrorValue(Arc<Mutex<Option<GstreamerError>>>);

impl NtscApp {
    fn spawn(&mut self, future: impl Future<Output = Option<AppFn>> + 'static + Send) {
        self.executor.lock().unwrap().spawn(future, false);
    }

    fn execute_fn<T: Future<Output = Option<AppFn>> + 'static + Send>(&self) -> impl Fn(T) + Send {
        let weak_exec = self.executor.downgrade();

        move |future: T| {
            if let Some(exec) = weak_exec.upgrade() {
                exec.lock().unwrap().spawn(future, false);
            }
        }
    }

    fn execute_fn_next_frame<T: Future<Output = Option<AppFn>> + 'static + Send>(
        &self,
    ) -> impl Fn(T) + Send {
        let weak_exec = self.executor.downgrade();

        move |future: T| {
            if let Some(exec) = weak_exec.upgrade() {
                exec.lock().unwrap().spawn(future, true);
            }
        }
    }

    fn load_video(&mut self, ctx: &egui::Context, path: PathBuf) -> Result<(), ApplicationError> {
        self.remove_pipeline().context(LoadVideoSnafu)?;
        self.pipeline = Some(
            self.create_preview_pipeline(ctx, path)
                .context(LoadVideoSnafu)?,
        );
        println!("new pipeline");

        Ok(())
    }

    fn create_pipeline<
        F1: FnOnce(gstreamer::Pipeline) -> Result<gstreamer::Element, GstreamerError>
            + Send
            + Sync
            + 'static,
        F2: FnOnce(gstreamer::Pipeline) -> Result<gstreamer::Element, GstreamerError>
            + Send
            + Sync
            + 'static,
        CB: FnOnce() + Send + Sync + 'static,
        G: Fn(&gstreamer::Bus, &gstreamer::Message) -> gstreamer::BusSyncReply + Send + Sync + 'static,
    >(
        src_pad: gstreamer::Element,
        audio_sink: F1,
        video_sink: F2,
        bus_handler: G,
        duration: Option<gstreamer::ClockTime>,
        initial_scale: Option<usize>,
        callback: Option<CB>,
    ) -> Result<gstreamer::Pipeline, GstreamerError> {
        let pipeline = gstreamer::Pipeline::default();
        let decodebin = gstreamer::ElementFactory::make("decodebin").build()?;
        pipeline.add_many([&src_pad, &decodebin])?;
        gstreamer::Element::link_many([&src_pad, &decodebin])?;

        let video_queue = gstreamer::ElementFactory::make("queue")
            .name("video_queue")
            .build()?;
        let video_convert = gstreamer::ElementFactory::make("videoconvert").build()?;
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
            &video_sink(pipeline.clone())?,
        ];
        pipeline.add_many(video_elements)?;
        gstreamer::Element::link_many(video_elements)?;

        for e in video_elements {
            e.sync_state_with_parent()?;
        }

        let has_audio = Arc::new(Mutex::new(false));
        let has_video = Arc::new(Mutex::new(false));

        let handler_id: Arc<Mutex<Option<SignalHandlerId>>> = Arc::new(Mutex::new(None));
        let handler_id_for_handler = Arc::clone(&handler_id);

        let audio_sink = Mutex::new(Some(audio_sink));

        let pipeline_weak = gstreamer::prelude::ObjectExt::downgrade(&pipeline);
        let pad_added_handler = decodebin.connect_pad_added(move |dbin, src_pad| {
            let pipeline = &pipeline_weak;
            let handler_id = &handler_id_for_handler;
            // Try to detect whether the raw stream decodebin provided us with
            // just now is either audio or video (or none of both, e.g. subtitles).
            let (is_audio, is_video) = {
                let media_type = src_pad.current_caps().and_then(|caps| {
                    dbg!(&caps);
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
                    dbg!("connected audio");

                    let audio_sink = audio_sink.lock().unwrap().take();
                    if let Some(audio_sink) = audio_sink {
                        if let Some(pipeline) = pipeline.upgrade() {
                            let audio_queue = gstreamer::ElementFactory::make("queue").build()?;
                            let audio_convert =
                                gstreamer::ElementFactory::make("audioconvert").build()?;
                            let audio_resample =
                                gstreamer::ElementFactory::make("audioresample").build()?;

                            let audio_elements = &[&audio_queue, &audio_convert, &audio_resample];
                            dbg!("adding many audio");
                            pipeline.add_many(audio_elements)?;
                            dbg!("added many audio");
                            gstreamer::Element::link_many(audio_elements)?;

                            let sink = audio_sink(pipeline.clone())?;
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

                        *has_audio = true;
                    }

                    // has_audio
                } else if is_video && !*has_video {
                    dbg!("connected video");

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
                            Self::scale_from_caps(&caps.unwrap(), initial_scale.unwrap())
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
                            .framerate(gstreamer::Fraction::from(30))
                            .build();
                        framerate_caps_filter.set_property("caps", video_caps);

                        if let Some(pipeline) = pipeline.upgrade() {
                            pipeline.add(&image_freeze)?;
                            src_pad.link(&image_freeze.static_pad("sink").unwrap())?;
                            gstreamer::Element::link_many([
                                &image_freeze,
                                &video_queue,
                            ])?;
                            image_freeze.sync_state_with_parent()?;

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
                        }
                    } else {
                        src_pad.link(&sink_pad)?;
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
                dbg!(&err);
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

        let pipeline_weak = gstreamer::prelude::ObjectExt::downgrade(&pipeline);
        let finished_loading = AtomicBool::new(false);
        let handler_callback = Mutex::new(callback);
        bus.set_sync_handler(move |bus, msg| {
            if !finished_loading.load(std::sync::atomic::Ordering::SeqCst) {
                if let gstreamer::MessageView::AsyncDone(a) = msg.view() {
                    if let Some(pipeline) = pipeline_weak.upgrade() {
                        if let Some(src_pipeline) = a
                            .src()
                            .and_then(|a| a.downcast_ref::<gstreamer::Pipeline>())
                        {
                            let pipeline_state_change_done = *src_pipeline == pipeline;

                            if pipeline_state_change_done {
                                let mut id = handler_id.lock().unwrap();
                                let id = id.take();
                                if let Some(id) = id {
                                    decodebin.disconnect(id);

                                    if let Some(callback) = handler_callback.lock().unwrap().take()
                                    {
                                        callback();
                                    }
                                }
                                finished_loading.store(true, std::sync::atomic::Ordering::SeqCst);
                            }
                        }
                    }
                }
            }

            bus_handler(bus, msg)
        });

        Ok(pipeline)
    }

    fn scale_from_caps(caps: &gstreamer::Caps, scanlines: usize) -> Option<(i32, i32)> {
        let caps_structure = caps.structure(0)?;
        let src_width = caps_structure.get::<i32>("width").ok()?;
        let src_height = caps_structure.get::<i32>("height").ok()?;

        let scale_factor = scanlines as f32 / src_height as f32;
        let dst_width = (src_width as f32 * scale_factor).round() as i32;

        Some((dst_width, scanlines as i32))
    }

    fn rescale_video(
        pipeline: &gstreamer::Pipeline,
        scanlines: Option<usize>,
    ) -> Result<(), GstreamerError> {
        let caps_filter = pipeline.by_name("caps_filter").unwrap();

        if let Some(scanlines) = scanlines {
            let scale_caps = pipeline
                .by_name("video_scale")
                .and_then(|elem| elem.static_pad("src"))
                .and_then(|pad| pad.current_caps());
            let scale_caps = match scale_caps {
                Some(caps) => caps,
                None => return Ok(()),
            };

            if let Some((dst_width, dst_height)) = Self::scale_from_caps(&scale_caps, scanlines) {
                dbg!(dst_width);
                dbg!(scanlines);

                caps_filter.set_property(
                    "caps",
                    gstreamer_video::VideoCapsBuilder::default()
                        .width(dst_width)
                        .height(dst_height)
                        .build(),
                );
            }
        } else {
            caps_filter.set_property("caps", gstreamer_video::VideoCapsBuilder::default().build());
        }

        pipeline.seek_simple(
            gstreamer::SeekFlags::FLUSH | gstreamer::SeekFlags::ACCURATE,
            pipeline.query_position::<gstreamer::ClockTime>().unwrap(),
        )?;

        Ok(())
    }

    fn set_still_image_framerate(info: &PipelineInfo, framerate: gstreamer::Fraction) -> Result<(), GstreamerError> {
        let caps_filter = info.pipeline.by_name("framerate_caps_filter");
        if let Some(caps_filter) = caps_filter {
            caps_filter.set_property("caps", VideoCapsBuilder::default().framerate(framerate).build());
            // This seek is necessary to prevent caps negotiation from failing due to race conditions, for some reason.
            // It seems like in some cases, there would be "tearing" in the caps between different elements, where some
            // elements' caps would use the old framerate and some would use the new framerate. This would cause caps
            // negotiation to fail, even though the caps filter sends a "reconfigure" event. This in turn woulc make the
            // entire pipeline error out.
            info.pipeline.seek_simple(
                gstreamer::SeekFlags::FLUSH | gstreamer::SeekFlags::ACCURATE,
                info.pipeline.query_position::<gstreamer::ClockTime>().unwrap(),
            )?;
            *info.framerate.lock().unwrap() = Some(framerate);
        }

        Ok(())
    }

    fn create_preview_pipeline(
        &mut self,
        ctx: &egui::Context,
        path: PathBuf,
    ) -> Result<PipelineInfo, GstreamerError> {
        let src = gstreamer::ElementFactory::make("filesrc")
            .property("location", path.as_path())
            .build()?;

        let audio_sink = gstreamer::ElementFactory::make("autoaudiosink").build()?;

        let tex = ctx.load_texture(
            "preview",
            egui::ColorImage::from_rgb([1, 1], &[0, 0, 0]),
            egui::TextureOptions::LINEAR,
        );
        let tex_sink = SinkTexture(Some(tex.clone()));
        let egui_ctx = EguiCtx(Some(ctx.clone()));
        let video_sink = gstreamer::ElementFactory::make("eguisink")
            .property("texture", tex_sink)
            .property("ctx", egui_ctx)
            .build()?;

        let pipeline_info_state = Arc::new(Mutex::new(PipelineInfoState::Loading));
        let pipeline_info_state_for_handler = Arc::clone(&pipeline_info_state);
        let at_eos = Arc::new(Mutex::new(false));
        let at_eos_for_handler = Arc::clone(&at_eos);
        let is_still_image = Arc::new(Mutex::new(false));
        let is_still_image_for_handler = Arc::clone(&is_still_image);
        let framerate = Arc::new(Mutex::new(None));
        let framerate_for_handler = Arc::clone(&framerate);
        let ctx_for_handler = ctx.clone();

        let audio_sink_for_closure = audio_sink.clone();
        let video_sink_for_closure = video_sink.clone();

        let pipeline = NtscApp::create_pipeline(
            src.clone(),
            |pipeline| {
                pipeline.add(&audio_sink_for_closure)?;
                Ok(audio_sink_for_closure)
            },
            |_| Ok(video_sink_for_closure),
            move |bus, msg| {
                dbg!(msg);
                let at_eos = &at_eos_for_handler;
                let ctx = &ctx_for_handler;
                let pipeline_info_state = &pipeline_info_state_for_handler;
                let is_still_image = &is_still_image_for_handler;
                let framerate = &framerate_for_handler;

                let handle_msg = move |_bus, msg: &gstreamer::Message| -> Option<()> {
                    // Make sure we're listening to a pipeline event
                    let src = msg.src()?;

                    if let gstreamer::MessageView::Error(err_msg) = msg.view() {
                        dbg!("handling error message");
                        dbg!(&msg);
                        let mut pipeline_state = pipeline_info_state.lock().unwrap();
                        if !matches!(&*pipeline_state, PipelineInfoState::Error(_)) {
                            *pipeline_state = PipelineInfoState::Error(err_msg.error().into());
                            ctx.request_repaint();
                        }
                    }

                    if msg.type_() == gstreamer::MessageType::Error {}

                    if let Some(pipeline) = src.downcast_ref::<gstreamer::Pipeline>() {
                        // We want to pause the pipeline at EOS, but setting an element's state inside the bus handler doesn't
                        // work. Instead, wait for the next egui event loop then pause.
                        if let gstreamer::MessageView::Eos(_) = msg.view() {
                            *at_eos.lock().unwrap() = true;
                            ctx.request_repaint();
                        }

                        if let gstreamer::MessageView::StateChanged(state_changed) = msg.view() {
                            if state_changed.old() == gstreamer::State::Ready
                                && matches!(
                                    state_changed.current(),
                                    gstreamer::State::Paused | gstreamer::State::Playing
                                )
                            {
                                // Changed from READY to PAUSED/PLAYING.
                                *pipeline_info_state.lock().unwrap() = PipelineInfoState::Loaded;
                                let is_still_image_inner = pipeline.by_name("still_image_freeze").is_some();
                                *is_still_image.lock().unwrap() = is_still_image_inner;

                                let video_rate = pipeline.by_name("video_rate").unwrap();
                                let caps =
                                    video_rate.static_pad("src").and_then(|pad| pad.caps());

                                *framerate.lock().unwrap() = if let Some(caps) = caps {
                                    let structure = caps.structure(0);
                                    structure.and_then(|structure| {
                                        structure.get::<gstreamer::Fraction>("framerate").ok()
                                    })
                                } else {
                                    None
                                };
                            }
                        }
                    }

                    Some(())
                };

                handle_msg(bus, msg);

                gstreamer::BusSyncReply::Drop
            },
            None,
            if self.video_scale.enabled {
                Some(self.video_scale.scale)
            } else {
                None
            },
            Some(|| {
                dbg!("got pipeline!");
            }),
        )?;

        dbg!(pipeline.set_state(gstreamer::State::Paused))?;

        Ok(PipelineInfo {
            pipeline,
            state: pipeline_info_state,
            path,
            file_src: src,
            egui_sink: video_sink,
            at_eos,
            last_seek_pos: gstreamer::ClockTime::ZERO,
            preview: tex,
            is_still_image,
            framerate,
        })
    }

    fn pixel_formats_for(bit_depth: usize, chroma_subsampling: bool) -> &'static [VideoFormat] {
        match (bit_depth, chroma_subsampling) {
            (8, false) => &[
                VideoFormat::Y444,
                VideoFormat::V308,
                VideoFormat::Iyu2,
                VideoFormat::Nv24,
            ],
            (8, true) => &[
                VideoFormat::I420,
                VideoFormat::Yv12,
                VideoFormat::Nv12,
                VideoFormat::Nv21,
            ],
            (10, false) => &[VideoFormat::Y44410be, VideoFormat::Y44410le],
            (10, true) => &[VideoFormat::I42010be, VideoFormat::I42010le],
            (12, false) => &[VideoFormat::Y44412be, VideoFormat::Y44412le],
            (12, true) => &[VideoFormat::I42012be, VideoFormat::I42012le],
            _ => panic!("No pixel format for bit depth {bit_depth}"),
        }
    }

    fn create_render_job(
        &mut self,
        ctx: &egui::Context,
        src_path: PathBuf,
        settings: RenderSettings,
    ) -> Result<RenderJob, GstreamerError> {
        let src = gstreamer::ElementFactory::make("filesrc")
            .property("location", src_path.as_path())
            .build()?;

        let (audio_enc, video_enc, video_mux, pixel_formats) = match settings.output_codec {
            OutputCodec::H264 => {
                // Load the x264enc plugin so the enum classes exist. Nothing seems to work except actually instantiating an Element.
                let _ = gstreamer::ElementFactory::make("x264enc").build().unwrap();
                #[allow(non_snake_case)]
                let GstX264EncPass = gstreamer::glib::EnumClass::with_type(
                    gstreamer::glib::Type::from_name("GstX264EncPass").unwrap(),
                )
                .unwrap();
                #[allow(non_snake_case)]
                let GstX264EncPreset = gstreamer::glib::EnumClass::with_type(
                    gstreamer::glib::Type::from_name("GstX264EncPreset").unwrap(),
                )
                .unwrap();

                let audio_enc = gstreamer::ElementFactory::make("avenc_aac").build()?;

                let video_enc = gstreamer::ElementFactory::make("x264enc")
                    // CRF mode
                    .property("pass", GstX264EncPass.to_value_by_nick("quant").unwrap())
                    // invert CRF (so that low numbers = low quality)
                    .property("quantizer", 51 - settings.h264_settings.crf as u32)
                    .property(
                        "speed-preset",
                        GstX264EncPreset
                            .to_value(9 - settings.h264_settings.encode_speed as i32)
                            .unwrap(),
                    )
                    .build()?;

                let video_mux = gstreamer::ElementFactory::make("mp4mux").build()?;

                let pixel_formats = Self::pixel_formats_for(
                    if settings.h264_settings.ten_bit {
                        10
                    } else {
                        8
                    },
                    settings.h264_settings.chroma_subsampling,
                );

                (audio_enc, video_enc, video_mux, pixel_formats)
            }
            OutputCodec::Ffv1 => {
                let audio_enc = gstreamer::ElementFactory::make("flacenc").build()?;
                let video_enc = gstreamer::ElementFactory::make("avenc_ffv1").build()?;
                let video_mux = gstreamer::ElementFactory::make("matroskamux").build()?;

                let pixel_formats = Self::pixel_formats_for(
                    match settings.ffv1_settings.bit_depth {
                        Ffv1BitDepth::Bits8 => 8,
                        Ffv1BitDepth::Bits10 => 10,
                        Ffv1BitDepth::Bits12 => 12,
                    },
                    settings.ffv1_settings.chroma_subsampling,
                );

                (audio_enc, video_enc, video_mux, pixel_formats)
            }
        };

        let video_ntsc = gstreamer::ElementFactory::make("ntscfilter")
            .property(
                "settings",
                NtscFilterSettings(self.effect_settings.clone().into()),
            )
            .build()?;
        let ntsc_caps_filter = gstreamer::ElementFactory::make("capsfilter")
            .property(
                "caps",
                gstreamer_video::VideoCapsBuilder::new()
                    .format(gstreamer_video::VideoFormat::Argb64)
                    .build(),
            )
            .build()?;
        let video_convert = gstreamer::ElementFactory::make("videoconvert").build()?;

        let job_state = Arc::new(Mutex::new(RenderJobState::Waiting));
        let job_state_for_handler = Arc::clone(&job_state);
        let exec = self.execute_fn_next_frame();
        let ctx_for_handler = ctx.clone();

        let audio_enc_for_closure = audio_enc.clone();
        let video_mux_for_closure = video_mux.clone();
        let video_ntsc_for_closure = video_ntsc.clone();

        let pipeline = Self::create_pipeline(
            src,
            move |pipeline| {
                pipeline.add(&audio_enc_for_closure)?;
                audio_enc_for_closure.link(&video_mux_for_closure)?;
                Ok(audio_enc_for_closure)
            },
            |_| Ok(video_ntsc_for_closure),
            move |bus, msg| {
                let job_state = &job_state_for_handler;
                let exec = &exec;
                let ctx = &ctx_for_handler;

                let handle_msg = move |_bus, msg: &gstreamer::Message| -> Option<()> {
                    //dbg!(msg);
                    let src = msg.src()?;

                    if let gstreamer::MessageView::Error(err) = msg.view() {
                        let mut job_state = job_state.lock().unwrap();
                        if !matches!(*job_state, RenderJobState::Error(_)) {
                            *job_state = RenderJobState::Error(err.error().into());
                            ctx.request_repaint();
                        }
                    }

                    // Make sure we're listening to a pipeline event
                    if let Some(pipeline) = src.downcast_ref::<gstreamer::Pipeline>() {
                        let pipeline_for_handler = pipeline.clone();
                        if let gstreamer::MessageView::Eos(_) = msg.view() {
                            let job_state_inner = Arc::clone(&job_state);
                            exec(async move {
                                let _ = pipeline_for_handler.set_state(gstreamer::State::Null);
                                *job_state_inner.lock().unwrap() = RenderJobState::Complete;
                                None
                            })
                        }

                        if let gstreamer::MessageView::StateChanged(state_changed) = msg.view() {
                            if state_changed.pending() == gstreamer::State::Null {
                                *job_state.lock().unwrap() = RenderJobState::Complete;
                            } else {
                                *job_state.lock().unwrap() = match state_changed.current() {
                                    gstreamer::State::Paused => RenderJobState::Paused,
                                    gstreamer::State::Playing => RenderJobState::Rendering,
                                    gstreamer::State::Ready => RenderJobState::Waiting,
                                    gstreamer::State::Null => RenderJobState::Complete,
                                    gstreamer::State::VoidPending => {
                                        unreachable!("current state should never be VOID_PENDING")
                                    }
                                };
                            }
                            ctx.request_repaint();
                        }
                    }

                    Some(())
                };

                handle_msg(bus, msg);

                gstreamer::BusSyncReply::Drop
            },
            Some(self.render_settings.duration),
            if self.video_scale.enabled {
                Some(self.video_scale.scale)
            } else {
                None
            },
            None::<fn()>,
        )?;

        let video_caps = gstreamer_video::VideoCapsBuilder::new()
            .format_list(pixel_formats.iter().copied())
            .build();
        let caps_filter = gstreamer::ElementFactory::make("capsfilter")
            .property("caps", &video_caps)
            .build()?;
        dbg!(&video_caps);

        let dst = gstreamer::ElementFactory::make("filesink")
            .property("location", settings.output_path.as_path())
            .build()?;
        dbg!("adding many render");
        pipeline.add_many([
            &ntsc_caps_filter,
            &video_convert,
            &caps_filter,
            &video_enc,
            &video_mux,
            &dst,
        ])?;
        dbg!("added many render");

        gstreamer::Element::link_many([
            &video_ntsc,
            &ntsc_caps_filter,
            &video_convert,
            &caps_filter,
            &video_enc,
            &video_mux,
        ])?;

        gstreamer::Element::link(&video_mux, &dst)?;

        video_mux.sync_state_with_parent()?;
        dst.sync_state_with_parent()?;

        dbg!(pipeline.set_state(gstreamer::State::Playing))?;

        Ok(RenderJob {
            settings,
            pipeline,
            state: job_state,
            last_progress: 0.0,
            progress_samples: VecDeque::new(),
            start_time: None,
            estimated_time_remaining: None,
        })
    }

    fn remove_pipeline(&mut self) -> Result<(), GstreamerError> {
        if let Some(PipelineInfo { pipeline, .. }) = &mut self.pipeline {
            pipeline.set_state(gstreamer::State::Null)?;
            self.pipeline = None;
        }

        Ok(())
    }

    fn update_effect(&self) {
        if let Some(PipelineInfo { egui_sink, .. }) = &self.pipeline {
            egui_sink.set_property(
                "settings",
                NtscFilterSettings((&self.effect_settings).into()),
            );
        }
    }

    fn settings_from_descriptors(
        effect_settings: &mut NtscEffectFullSettings,
        ui: &mut egui::Ui,
        descriptors: &[SettingDescriptor],
    ) -> bool {
        let parser = |input: &str| eval_expression_string(input).ok();
        let mut changed = false;
        for descriptor in descriptors {
            let response = match &descriptor.kind {
                ntscrs::settings::SettingKind::Enumeration {
                    options,
                    default_value: _,
                } => {
                    let selected_index = descriptor.id.get_field_enum(effect_settings).unwrap();
                    let selected_item = options
                        .iter()
                        .find(|option| option.index == selected_index)
                        .unwrap();
                    egui::ComboBox::from_label(descriptor.label)
                        .selected_text(selected_item.label)
                        .show_ui(ui, |ui| {
                            for item in options {
                                let mut label =
                                    ui.selectable_label(selected_index == item.index, item.label);

                                if let Some(desc) = item.description {
                                    label = label.on_hover_text(desc);
                                }

                                if label.clicked() {
                                    let _ =
                                        descriptor.id.set_field_enum(effect_settings, item.index);
                                    // a selectable_label being clicked doesn't set response.changed
                                    changed = true;
                                };
                            }
                        })
                        .response
                }
                ntscrs::settings::SettingKind::Percentage {
                    logarithmic,
                    default_value: _,
                } => ui.add(
                    egui::Slider::new(
                        descriptor.id.get_field_mut::<f32>(effect_settings).unwrap(),
                        0.0..=1.0,
                    )
                    .custom_parser(parser)
                    .custom_formatter(format_percentage)
                    .logarithmic(*logarithmic)
                    .text(descriptor.label),
                ),
                ntscrs::settings::SettingKind::IntRange {
                    range,
                    default_value: _,
                } => {
                    let mut value = 0i32;
                    if let Some(v) = descriptor.id.get_field_mut::<i32>(effect_settings) {
                        value = *v;
                    } else if let Some(v) = descriptor.id.get_field_mut::<u32>(effect_settings) {
                        value = *v as i32;
                    }

                    let slider = ui.add(
                        egui::Slider::new(&mut value, range.clone())
                            .custom_parser(parser)
                            .text(descriptor.label),
                    );

                    if slider.changed() {
                        if let Some(v) = descriptor.id.get_field_mut::<i32>(effect_settings) {
                            *v = value;
                        } else if let Some(v) = descriptor.id.get_field_mut::<u32>(effect_settings)
                        {
                            *v = value as u32;
                        }
                    }

                    slider
                }
                ntscrs::settings::SettingKind::FloatRange {
                    range,
                    logarithmic,
                    default_value: _,
                } => ui.add(
                    egui::Slider::new(
                        descriptor.id.get_field_mut::<f32>(effect_settings).unwrap(),
                        range.clone(),
                    )
                    .custom_parser(parser)
                    .logarithmic(*logarithmic)
                    .text(descriptor.label),
                ),
                ntscrs::settings::SettingKind::Boolean { default_value: _ } => {
                    let checkbox = ui.checkbox(
                        descriptor
                            .id
                            .get_field_mut::<bool>(effect_settings)
                            .unwrap(),
                        descriptor.label,
                    );

                    checkbox
                }
                ntscrs::settings::SettingKind::Group {
                    children,
                    default_value: _,
                } => {
                    ui.group(|ui| {
                        let checkbox = ui.checkbox(
                            descriptor
                                .id
                                .get_field_mut::<bool>(effect_settings)
                                .unwrap(),
                            descriptor.label,
                        );

                        ui.set_enabled(
                            *descriptor
                                .id
                                .get_field_mut::<bool>(effect_settings)
                                .unwrap(),
                        );

                        changed |= Self::settings_from_descriptors(effect_settings, ui, &children);

                        checkbox
                    })
                    .inner
                }
            };

            changed |= response.changed();

            if let Some(desc) = descriptor.description {
                response.on_hover_text(desc);
            }
        }

        changed
    }

    fn show_effect_settings(&mut self, ui: &mut egui::Ui) {
        egui::TopBottomPanel::bottom("effect_load_save")
            .exact_height(ui.spacing().interact_size.y * 2.0)
            .show_inside(ui, |ui| {
                ui.horizontal_centered(|ui| {
                    if ui.button("Save").clicked() {
                        let json = self.settings_list.to_json(&mut self.effect_settings);
                        let handle = rfd::AsyncFileDialog::new()
                            .set_file_name("settings.json")
                            .save_file();
                        self.spawn(async move {
                            let handle = handle.await;
                            let handle = match handle {
                                Some(h) => h,
                                None => return None,
                            };

                            Some(Box::new(move |_: &mut NtscApp| {
                                let mut file =
                                    File::create(handle.path()).context(JSONSaveSnafu)?;
                                json.write_to(&mut file).context(JSONSaveSnafu)?;
                                Ok(())
                            }) as _)
                        });
                    }

                    if ui.button("Load").clicked() {
                        let handle = rfd::AsyncFileDialog::new()
                            .add_filter("JSON", &["json"])
                            .pick_file();
                        self.spawn(async move {
                            let handle = handle.await;

                            Some(Box::new(
                                move |app: &mut NtscApp| -> Result<(), ApplicationError> {
                                    let handle = match handle {
                                        Some(h) => h,
                                        // user cancelled the operation
                                        None => return Ok(()),
                                    };

                                    let mut file =
                                        File::open(handle.path()).context(JSONReadSnafu)?;

                                    let mut buf = String::new();
                                    file.read_to_string(&mut buf).context(JSONReadSnafu)?;

                                    let settings = app
                                        .settings_list
                                        .from_json(&buf)
                                        .context(JSONParseSnafu)?;

                                    app.effect_settings = settings;

                                    Ok(())
                                },
                            ) as _)
                        });
                    }

                    if ui.button("📋 Copy").clicked() {
                        ui.output_mut(|output| {
                            output.copied_text = self
                                .settings_list
                                .to_json(&self.effect_settings)
                                .stringify()
                                .unwrap()
                        });
                    }

                    let pasted_text = ui.input(|input| {
                        input.events.iter().find_map(|event| match event {
                            egui::Event::Paste(data) => Some(data.clone()),
                            _ => None,
                        })
                    });
                    if let Some(p) = pasted_text {
                        dbg!(p);
                    }

                    let btn = ui.button("📄 Paste");

                    let paste_popup_id = ui.make_persistent_id("paste_popup_open");

                    if btn.clicked() {
                        ui.ctx().data_mut(|map| {
                            let old_value =
                                map.get_temp_mut_or_insert_with(paste_popup_id, || false);
                            *old_value = !*old_value;
                        });
                    }

                    if ui
                        .ctx()
                        .data(|map| map.get_temp(paste_popup_id).unwrap_or(false))
                    {
                        let mut is_open = true;
                        //ui.visuals_mut().clip_rect_margin = 3.0;
                        egui::Window::new("Paste JSON")
                            .default_pos(btn.rect.center_top())
                            .open(&mut is_open)
                            .show(ui.ctx(), |ui| {
                                ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                                    if ui.button("Load").clicked() {
                                        if let Ok(settings) =
                                            self.settings_list.from_json(&self.settings_json_paste)
                                        {
                                            self.effect_settings = settings;
                                            self.update_effect();
                                        }
                                    }
                                    ui.with_layout(
                                        egui::Layout::top_down(egui::Align::Min),
                                        |ui| {
                                            egui::ScrollArea::new([false, true])
                                                .auto_shrink([true, false])
                                                .show(ui, |ui| {
                                                    ui.add_sized(
                                                        ui.available_size(),
                                                        egui::TextEdit::multiline(
                                                            &mut self.settings_json_paste,
                                                        ),
                                                    );
                                                });
                                        },
                                    );
                                });
                            });

                        if !is_open {
                            ui.ctx()
                                .data_mut(|map| map.insert_temp(paste_popup_id, false));
                        }
                    }

                    if ui.button("Reset").clicked() {
                        self.effect_settings = NtscEffectFullSettings::default();
                    }
                });
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    ui.style_mut().spacing.slider_width = 200.0;
                    let Self {
                        settings_list,
                        effect_settings,
                        ..
                    } = self;
                    let settings_changed = Self::settings_from_descriptors(
                        effect_settings,
                        ui,
                        &settings_list.settings,
                    );
                    if settings_changed {
                        self.update_effect();
                    }
                });
        });
    }

    fn show_render_settings(&mut self, ui: &mut egui::Ui) {
        egui::Frame::central_panel(ui.style()).show(ui, |ui| {
            egui::ComboBox::from_label("Codec")
                .selected_text(self.render_settings.output_codec.label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.render_settings.output_codec,
                        OutputCodec::H264,
                        OutputCodec::H264.label(),
                    );
                    ui.selectable_value(
                        &mut self.render_settings.output_codec,
                        OutputCodec::Ffv1,
                        OutputCodec::Ffv1.label(),
                    );
                });

            match self.render_settings.output_codec {
                OutputCodec::H264 => {
                    ui.add(
                        egui::Slider::new(&mut self.render_settings.h264_settings.crf, 0..=51)
                            .text("Quality"),
                    );
                    ui.add(
                        egui::Slider::new(
                            &mut self.render_settings.h264_settings.encode_speed,
                            0..=8,
                        )
                        .text("Encoding speed"),
                    );
                    ui.checkbox(
                        &mut self.render_settings.h264_settings.ten_bit,
                        "10-bit color",
                    );
                    ui.checkbox(
                        &mut self.render_settings.h264_settings.chroma_subsampling,
                        "4:2:0 chroma subsampling",
                    );
                }

                OutputCodec::Ffv1 => {
                    egui::ComboBox::from_label("Bit depth")
                        .selected_text(self.render_settings.ffv1_settings.bit_depth.label())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.render_settings.ffv1_settings.bit_depth,
                                Ffv1BitDepth::Bits8,
                                Ffv1BitDepth::Bits8.label(),
                            );
                            ui.selectable_value(
                                &mut self.render_settings.ffv1_settings.bit_depth,
                                Ffv1BitDepth::Bits10,
                                Ffv1BitDepth::Bits10.label(),
                            );
                            ui.selectable_value(
                                &mut self.render_settings.ffv1_settings.bit_depth,
                                Ffv1BitDepth::Bits12,
                                Ffv1BitDepth::Bits12.label(),
                            );
                        });

                    ui.checkbox(
                        &mut self.render_settings.ffv1_settings.chroma_subsampling,
                        "4:2:0 chroma subsampling",
                    );
                }
            }

            ui.separator();

            ui.horizontal(|ui| {
                let path = &self.render_settings.output_path;
                let mut path = path.to_string_lossy();
                ui.label("Destination file:");
                if ui.text_edit_singleline(&mut path).changed() {
                    self.render_settings.output_path = PathBuf::from(OsStr::new(path.as_ref()));
                }
                if ui.button("📁").clicked() {
                    let mut dialog_path = &self.render_settings.output_path;
                    if dialog_path.components().next().is_none() {
                        if let Some(PipelineInfo { path, .. }) = &self.pipeline {
                            dialog_path = path;
                        }
                    }
                    let mut file_dialog = rfd::AsyncFileDialog::new();

                    if dialog_path.components().next().is_some() {
                        if let Some(parent) = dialog_path.parent() {
                            file_dialog = file_dialog.set_directory(parent);
                        }
                        if let Some(file_name) = dialog_path.file_stem() {
                            let extension = match self.render_settings.output_codec {
                                OutputCodec::H264 => ".mp4",
                                OutputCodec::Ffv1 => ".mkv",
                            };
                            file_dialog = file_dialog.set_file_name(format!(
                                "{}_ntsc{}",
                                file_name.to_string_lossy(),
                                extension
                            ));
                        }
                    }

                    let file_dialog = file_dialog.save_file();
                    let _ = self.spawn(async move {
                        let handle = file_dialog.await;
                        Some(Box::new(|app: &mut NtscApp| {
                            if let Some(handle) = handle {
                                app.render_settings.output_path = handle.into();
                            }

                            Ok(())
                        }) as _)
                    });
                }
            });

            let src_path = self.pipeline.as_ref().and_then(|info| Some(&info.path));

            let mut duration = self.render_settings.duration.mseconds();
            if self
                .pipeline
                .as_ref()
                .map_or(false, |info| *info.is_still_image.lock().unwrap())
            {
                ui.horizontal(|ui| {
                    ui.label("Duration:");
                    if ui
                        .add(
                            egui::DragValue::new(&mut duration)
                                .custom_formatter(|value, _| {
                                    clock_time_format(
                                        (value * gstreamer::ClockTime::MSECOND.nseconds() as f64)
                                            as u64,
                                    )
                                })
                                .custom_parser(clock_time_parser)
                                .speed(100.0),
                        )
                        .changed()
                    {
                        self.render_settings.duration =
                            gstreamer::ClockTime::from_mseconds(duration);
                        dbg!(self.render_settings.duration);
                    }
                });
            }

            if ui
                .add_enabled(
                    self.render_settings.output_path.as_os_str().len() > 0 && src_path.is_some(),
                    egui::Button::new("Render"),
                )
                .clicked()
            {
                let render_job = self.create_render_job(
                    ui.ctx(),
                    src_path.unwrap().clone(),
                    self.render_settings.clone(),
                );
                match render_job {
                    Ok(render_job) => {
                        self.render_jobs.push(render_job);
                    }
                    Err(err) => {
                        self.handle_error(&err);
                    }
                }
            }

            ui.separator();

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let mut removed_job_idx = None;
                    for (idx, job) in self.render_jobs.iter_mut().enumerate() {
                        ui.with_layout(egui::Layout::top_down_justified(egui::Align::Min), |ui| {
                            let fill = ui.style().visuals.faint_bg_color;
                            egui::Frame::none()
                                .fill(fill)
                                .stroke(ui.style().visuals.window_stroke)
                                .rounding(ui.style().noninteractive().rounding)
                                .inner_margin(ui.style().spacing.window_margin)
                                .show(ui, |ui| {
                                    let job_state = &*job.state.lock().unwrap();

                                    let (progress, job_position, job_duration) = match job_state {
                                        RenderJobState::Waiting => (0.0, None, None),
                                        RenderJobState::Paused
                                        | RenderJobState::Rendering
                                        | RenderJobState::Error(_) => {
                                            let job_position = job
                                                .pipeline
                                                .query_position::<gstreamer::ClockTime>();
                                            let job_duration = job
                                                .pipeline
                                                .query_duration::<gstreamer::ClockTime>();

                                            (
                                                if job_position.is_some() && job_duration.is_some()
                                                {
                                                    job_position.unwrap().nseconds() as f64
                                                        / job_duration.unwrap().nseconds() as f64
                                                } else {
                                                    job.last_progress
                                                },
                                                job_position,
                                                job_duration,
                                            )
                                        }
                                        RenderJobState::Complete => (1.0, None, None),
                                    };

                                    if matches!(
                                        job_state,
                                        RenderJobState::Rendering | RenderJobState::Waiting
                                    ) {
                                        let current_time = ui.ctx().input(|input| input.time);
                                        let most_recent_sample = job
                                            .progress_samples
                                            .back()
                                            .and_then(|sample| Some(sample.clone()));
                                        let should_update_estimate = if let Some((_, sample_time)) =
                                            most_recent_sample
                                        {
                                            current_time - sample_time > PROGRESS_SAMPLE_TIME_DELTA
                                        } else {
                                            true
                                        };
                                        if should_update_estimate {
                                            if job.start_time.is_none() {
                                                job.start_time = Some(current_time);
                                            }
                                            let new_sample = (progress, current_time);
                                            let oldest_sample = if job.progress_samples.len()
                                                >= NUM_PROGRESS_SAMPLES
                                            {
                                                job.progress_samples.pop_front()
                                            } else {
                                                job.progress_samples
                                                    .front()
                                                    .and_then(|sample| Some(sample.clone()))
                                            };
                                            job.progress_samples.push_back(new_sample);
                                            if let Some((old_progress, old_sample_time)) =
                                                oldest_sample
                                            {
                                                let time_estimate = (current_time
                                                    - old_sample_time)
                                                    / (progress - old_progress);
                                                if time_estimate.is_finite() {
                                                    let elapsed_time =
                                                        current_time - job.start_time.unwrap();
                                                    let remaining_time =
                                                        (time_estimate - elapsed_time).max(0.0);
                                                    job.estimated_time_remaining =
                                                        Some(remaining_time);
                                                }
                                            }
                                        }
                                    }

                                    ui.horizontal(|ui| {
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                if ui.button("🗙").clicked() {
                                                    removed_job_idx = Some(idx);
                                                }
                                                ui.with_layout(
                                                    egui::Layout::left_to_right(
                                                        egui::Align::Center,
                                                    ),
                                                    |ui| {
                                                        ui.add(
                                                            egui::Label::new(
                                                                job.settings
                                                                    .output_path
                                                                    .to_string_lossy(),
                                                            )
                                                            .truncate(true),
                                                        );
                                                    },
                                                )
                                            },
                                        );
                                    });

                                    ui.separator();

                                    ui.add(
                                        egui::ProgressBar::new(progress as f32).show_percentage(),
                                    );
                                    if let RenderJobState::Rendering = job_state {
                                        ui.ctx().request_repaint();
                                    }

                                    ui.label(match job_state {
                                        RenderJobState::Waiting => Cow::Borrowed("Waiting..."),
                                        RenderJobState::Rendering => {
                                            if job_position.is_some() && job_duration.is_some() {
                                                Cow::Owned(format!(
                                                    "Rendering... ({:.2} / {:.2})",
                                                    job_position.unwrap(),
                                                    job_duration.unwrap()
                                                ))
                                            } else {
                                                Cow::Borrowed("Rendering...")
                                            }
                                        }
                                        RenderJobState::Paused => Cow::Borrowed("Paused"),
                                        RenderJobState::Complete => Cow::Borrowed("Complete"),
                                        RenderJobState::Error(err) => {
                                            Cow::Owned(format!("Error: {err}"))
                                        }
                                    });

                                    if matches!(
                                        job_state,
                                        RenderJobState::Rendering | RenderJobState::Paused
                                    ) {
                                        if let Some(time_remaining) = job.estimated_time_remaining {
                                            ui.label(format!(
                                                "Time remaining: {time_remaining:.0} seconds"
                                            ));
                                        }
                                    }

                                    job.last_progress = progress;
                                });
                        });
                    }

                    if let Some(remove_idx) = removed_job_idx {
                        self.render_jobs.remove(remove_idx);
                    }
                });
        });
    }

    fn handle_error(&mut self, err: &dyn Error) {
        self.last_error = Some(format!("{}", err));
    }

    fn handle_result<T, E: Error>(&mut self, result: Result<T, E>) {
        if let Err(err) = result {
            self.handle_error(&err);
        }
    }

    fn handle_result_with<T, E: Error, F: FnOnce(&mut Self) -> Result<T, E>>(&mut self, cb: F) {
        let result = cb(self);
        self.handle_result(result);
    }
}

impl eframe::App for NtscApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        {
            let exec = Arc::clone(&self.executor);
            let mut exec = exec.lock().unwrap();
            exec.tick(self);
        }

        let mut pipeline_error = None::<GstreamerError>;
        if let Some(pipeline) = &self.pipeline {
            let state = pipeline.state.lock().unwrap();
            let state = &*state;
            match state {
                PipelineInfoState::Loading => {}
                PipelineInfoState::Loaded => {
                    let pipeline = self.pipeline.as_ref().unwrap();
                    let mut at_eos = pipeline.at_eos.lock().unwrap();
                    if *at_eos {
                        let _ = pipeline.pipeline.set_state(gstreamer::State::Paused);
                        *at_eos = false;
                    }
                }
                PipelineInfoState::Error(err) => {
                    pipeline_error = Some(err.clone());
                }
            };
        }

        if let Some(err) = pipeline_error {
            let _ = self.remove_pipeline();
            self.handle_error(&err);
        }

        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.heading("ntsc-rs");
                ui.menu_button("File", |ui| {
                    if ui.button("Open").clicked() {
                        let file_dialog = rfd::AsyncFileDialog::new().pick_file();
                        let ctx = ctx.clone();
                        self.spawn(async move {
                            let handle = file_dialog.await;

                            Some(Box::new(move |app: &mut NtscApp| {
                                if let Some(handle) = handle {
                                    app.load_video(&ctx, handle.into())
                                } else {
                                    Ok(())
                                }
                            }) as _)
                        });

                        ui.close_menu();
                    }
                    if ui.button("Quit").clicked() {
                        frame.close();
                        ui.close_menu();
                    }
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let mut close_error = false;
                    if let Some(error) = self.last_error.as_ref() {
                        egui::Frame::none()
                            .rounding(3.0)
                            .stroke(ui.style().noninteractive().fg_stroke)
                            .inner_margin(ui.style().spacing.button_padding)
                            .show(ui, |ui| {
                                if ui.button("OK").clicked() {
                                    close_error = true;
                                }
                                ui.label(error);
                                ui.colored_label(egui::Color32::YELLOW, "⚠");
                            });
                    }
                    if close_error {
                        self.last_error = None;
                    }
                });
            });
        });

        egui::SidePanel::left("controls")
            .frame(egui::Frame::side_top_panel(&ctx.style()).inner_margin(0.0))
            .resizable(true)
            .default_width(400.0)
            .width_range(200.0..=800.0)
            .show(ctx, |ui| {
                ui.visuals_mut().clip_rect_margin = 0.0;
                egui::TopBottomPanel::top("left_tabs").show_inside(ui, |ui| {
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        ui.selectable_value(
                            &mut self.left_panel_state,
                            LeftPanelState::EffectSettings,
                            "Effect",
                        );
                        ui.selectable_value(
                            &mut self.left_panel_state,
                            LeftPanelState::RenderSettings,
                            "Render",
                        );
                    });
                });

                egui::CentralPanel::default()
                    .frame(egui::Frame::central_panel(&ctx.style()).inner_margin(0.0))
                    .show_inside(ui, |ui| match self.left_panel_state {
                        LeftPanelState::EffectSettings => {
                            self.show_effect_settings(ui);
                        }
                        LeftPanelState::RenderSettings => {
                            self.show_render_settings(ui);
                        }
                    });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::side_top_panel(&ctx.style()).inner_margin(0.0))
            .show(ctx, |ui| {
                ui.visuals_mut().clip_rect_margin = 0.0;
                let last_seek_pos = if let Some(info) = &mut self.pipeline {
                    // While seeking, GStreamer sometimes doesn't return a timecode. In that case, use the last timecode it
                    // did respond with.
                    let queried_pos = info.pipeline.query_position::<gstreamer::ClockTime>();
                    if let Some(position) = queried_pos {
                        info.last_seek_pos = position;
                    }
                    info.last_seek_pos
                } else {
                    gstreamer::ClockTime::ZERO
                };

                let framerate = (|| {
                    let caps = self
                        .pipeline
                        .as_ref()?
                        .pipeline
                        .by_name("video_queue")?
                        .static_pad("sink")?
                        .current_caps()?;
                    let framerate = caps
                        .structure(0)?
                        .get::<gstreamer::Fraction>("framerate")
                        .ok()?;
                    Some(framerate)
                })();

                egui::TopBottomPanel::top("video_info").show_inside(ui, |ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let mut remove_pipeline = false;
                        let mut res = None;
                        if let Some(info) = &mut self.pipeline {
                            let framerate = *info.framerate.lock().unwrap();
                            if ui.button("🗙").clicked() {
                                remove_pipeline = true;
                            }

                            if let Some(framerate) = framerate {
                                ui.separator();
                                if *info.is_still_image.lock().unwrap() {
                                    let mut new_framerate = framerate.numer() as f64 / framerate.denom() as f64;
                                    ui.label("fps");
                                    if ui.add(egui::DragValue::new(&mut new_framerate).clamp_range(0.0..=240.0)).changed() {
                                        let framerate_fraction = gstreamer::Fraction::approximate_f64(new_framerate);
                                        if let Some(f) = framerate_fraction {
                                            res = Some(Self::set_still_image_framerate(info, f));
                                        }
                                    }
                                } else {
                                    ui.label(format!("{:.2} fps", framerate.numer() as f64 / framerate.denom() as f64));
                                }
                            }

                            ui.with_layout(
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    ui.add(
                                        egui::Label::new(info.path.to_string_lossy())
                                            .truncate(true),
                                    );
                                },
                            );
                        }

                        if let Some(res) = res {
                            self.handle_result(res);
                        }

                        if remove_pipeline {
                            self.handle_result_with(|app| app.remove_pipeline());
                        }
                    });
                });

                egui::TopBottomPanel::bottom("video_controls")
                    .exact_height(ui.spacing().interact_size.y * 2.0)
                    .show_inside(ui, |ui| {
                        ui.set_enabled(self.pipeline.is_some());
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                            let btn_widget = egui::Button::new(match &self.pipeline {
                                Some(PipelineInfo { pipeline, .. }) => {
                                    let state = pipeline.current_state();
                                    match state {
                                        gstreamer::State::Paused | gstreamer::State::Ready => "▶",
                                        gstreamer::State::Playing => "⏸",
                                        _ => "▶",
                                    }
                                }
                                None => "▶",
                            });
                            let mut btn = ui.add_sized(
                                vec2(
                                    ui.spacing().interact_size.y * 1.5,
                                    ui.spacing().interact_size.y * 1.5,
                                ),
                                btn_widget,
                            );

                            if !ctx.wants_keyboard_input() && ctx.input(|i| {
                                i.events.iter().any(|event| {
                                    if let egui::Event::Key {
                                        key,
                                        pressed,
                                        repeat,
                                        modifiers,
                                    } = event
                                    {
                                        *key == egui::Key::Space
                                            && *pressed
                                            && !repeat
                                            && modifiers.is_none()
                                    } else {
                                        false
                                    }
                                })
                            }) {
                                let res = self.pipeline.as_mut().and_then(|p| Some(p.toggle_playing()));
                                if let Some(res) = res {
                                    self.handle_result(res);
                                }
                            }

                            if btn.clicked() {
                                let res = self.pipeline.as_mut().and_then(|p| Some(p.toggle_playing()));
                                if let Some(res) = res {
                                    self.handle_result(res);
                                }
                            }

                            let duration = if let Some(info) = &self.pipeline {
                                info.pipeline.query_duration::<gstreamer::ClockTime>()
                            } else {
                                None
                            };

                            let mut timecode_ms = last_seek_pos.nseconds() as f64
                                / gstreamer::ClockTime::MSECOND.nseconds() as f64;
                            let frame_pace = if let Some(framerate) = framerate {
                                framerate.denom() as f64 / framerate.numer() as f64
                            } else {
                                1f64 / 30f64
                            };

                            let mut drag_value = egui::DragValue::new(&mut timecode_ms)
                                .custom_formatter(|value, _| {
                                    clock_time_format(
                                        (value * gstreamer::ClockTime::MSECOND.nseconds() as f64)
                                            as u64,
                                    )
                                })
                                .custom_parser(clock_time_parser)
                                .speed(frame_pace * 1000.0 * 0.5);

                            if let Some(duration) = duration {
                                drag_value = drag_value.clamp_range(0..=duration.mseconds());
                            }

                            if ui.add(drag_value).changed() {
                                if let Some(info) = &self.pipeline {
                                    // don't use KEY_UNIT here; it causes seeking to often be very inaccurate (almost a second of deviation)
                                    let _ = info.pipeline.seek_simple(
                                        gstreamer::SeekFlags::FLUSH
                                            | gstreamer::SeekFlags::ACCURATE,
                                        gstreamer::ClockTime::from_nseconds(
                                            (timecode_ms
                                                * gstreamer::ClockTime::MSECOND.nseconds() as f64)
                                                as u64,
                                        ),
                                    );
                                }
                            }

                            ui.separator();

                            ui.label("🔎");
                            ui.add_enabled(
                                !self.video_zoom.fit,
                                egui::DragValue::new(&mut self.video_zoom.scale)
                                    .clamp_range(0.0..=8.0)
                                    .speed(0.01)
                                    .custom_formatter(format_percentage)
                                    .custom_parser(|input: &str| {
                                        let mut expr = eval_expression_string(input).ok()?;
                                        // greater than 800% zoom? the user probably meant to input a raw percentage and
                                        // not a decimal in 0..1
                                        if expr >= 8.0 {
                                            expr /= 100.0;
                                        }
                                        Some(expr)
                                    }),
                            );
                            ui.checkbox(&mut self.video_zoom.fit, "Fit");

                            ui.separator();

                            let scale_checkbox =
                                ui.checkbox(&mut self.video_scale.enabled, "Scale to");
                            ui.set_enabled(self.video_scale.enabled);
                            let drag_resp = ui.add(
                                egui::DragValue::new(&mut self.video_scale.scale)
                                    .clamp_range(1..=usize::MAX),
                            );
                            if drag_resp.changed() || scale_checkbox.changed() {
                                if let Some(pipeline) = &self.pipeline {
                                    let res = Self::rescale_video(
                                        &pipeline.pipeline,
                                        if self.video_scale.enabled {
                                            Some(self.video_scale.scale)
                                        } else {
                                            None
                                        },
                                    );
                                    self.handle_result(res);
                                }
                            }
                            ui.label("scanlines");
                        });
                    });

                egui::CentralPanel::default()
                    .frame(egui::Frame::side_top_panel(&ctx.style()).inner_margin(0.0))
                    .show_inside(ui, |ui| {
                        ui.visuals_mut().clip_rect_margin = 0.0;
                        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                            if let Some(info) = &mut self.pipeline {
                                let mut timecode = info.last_seek_pos.nseconds();

                                let duration =
                                    info.pipeline.query_duration::<gstreamer::ClockTime>();

                                if let Some(duration) = duration {
                                    if ui
                                        .add(Timeline::new(
                                            &mut timecode,
                                            0..=duration.nseconds(),
                                            framerate,
                                        ))
                                        .changed()
                                    {
                                        let _ = info.pipeline.seek_simple(
                                            gstreamer::SeekFlags::FLUSH
                                                | gstreamer::SeekFlags::ACCURATE,
                                            gstreamer::ClockTime::from_nseconds(timecode),
                                        );
                                    }
                                }
                            }
                            egui::ScrollArea::both()
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    ui.with_layout(
                                        egui::Layout::centered_and_justified(
                                            egui::Direction::TopDown,
                                        ),
                                        |ui| {
                                            if let Some(PipelineInfo { preview, .. }) =
                                                &mut self.pipeline
                                            {
                                                let texture_size = if self.video_scale.enabled {
                                                    let texture_actual_size = preview.size_vec2();
                                                    let scale_factor = self.video_scale.scale
                                                        as f32
                                                        / texture_actual_size.y;
                                                    vec2(
                                                        (texture_actual_size.x * scale_factor)
                                                            .round(),
                                                        self.video_scale.scale as f32,
                                                    )
                                                } else {
                                                    preview.size_vec2()
                                                };
                                                let scale_factor = if self.video_zoom.fit {
                                                    // Due to floating-point error, a scrollbar may appear even if we scale down. To
                                                    // avoid the scrollbar popping in and out of existence, subtract a constant value
                                                    // from available_size.
                                                    ((ui.available_size() - vec2(1.0, 1.0))
                                                        / texture_size)
                                                        .min_elem()
                                                        .min(1.0)
                                                } else {
                                                    self.video_zoom.scale as f32
                                                };
                                                ui.image((
                                                    preview.id(),
                                                    texture_size * scale_factor,
                                                ));
                                            } else {
                                                ui.heading("No media loaded");
                                            }
                                        },
                                    );
                                });
                        });
                    });
            });
    }
}

impl Drop for NtscApp {
    fn drop(&mut self) {
        let _ = self.remove_pipeline();
        dbg!("dropping app");
    }
}
