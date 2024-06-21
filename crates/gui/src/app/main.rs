use std::{
    borrow::Cow,
    cell::RefCell,
    error::Error,
    ffi::OsStr,
    fs::File,
    io::Read,
    ops::RangeInclusive,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
};

use eframe::egui::{self, util::undoer::Undoer, vec2, ColorImage, Response};
use futures_lite::Future;
use gstreamer::{glib::subclass::types::ObjectSubclassExt, prelude::*, ClockTime};
use gstreamer_video::{VideoCapsBuilder, VideoInterlaceMode};

use crate::{
    expression_parser::eval_expression_string,
    gst_utils::{
        clock_format::{clock_time_format, clock_time_parser},
        egui_sink::{EffectPreviewSetting, EguiCtx, EguiSink, SinkTexture},
        elements,
        gstreamer_error::GstreamerError,
        ntscrs_filter::NtscFilterSettings,
        pipeline_utils::{create_pipeline, PipelineError, VideoElemMetadata},
        scale_from_caps,
    },
    third_party_licenses::get_third_party_licenses,
    widgets::{render_job::RenderJobWidget, splitscreen::SplitScreen, timeline::Timeline},
};

use ntscrs::settings::{
    NtscEffectFullSettings, SettingDescriptor, SettingID, SettingKind, SettingsList, UseField,
};
use snafu::ResultExt;

use log::debug;

use super::{
    app_state::{
        AudioVolume, ColorTheme, EffectPreviewMode, EffectPreviewSettings, LeftPanelState,
        VideoScale, VideoZoom,
    },
    error::{ApplicationError, JSONParseSnafu, JSONReadSnafu, JSONSaveSnafu, LoadVideoSnafu},
    executor::AppExecutor,
    layout_helper::LayoutHelper,
    pipeline_info::{PipelineInfo, PipelineMetadata, PipelineStatus},
    presets::PresetsState,
    render_job::RenderJob,
    render_settings::{
        Ffv1BitDepth, OutputCodec, PngSettings, RenderInterlaceMode, RenderPipelineCodec,
        RenderPipelineSettings, RenderSettings, StillImageSettings,
    },
    AppFn, NtscApp,
};

fn initialize_gstreamer() -> Result<(), GstreamerError> {
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

fn format_percentage(n: f64, prec: RangeInclusive<usize>) -> String {
    format!("{:.*}%", prec.start().max(&2) - 2, n * 100.0)
}

/// Parse a textbox input as either a decimal or percentage, depending on whether it's greater than a certain threshold.
/// Returns a decimal.
///
/// # Arguments
/// - `input` - The text input from the user.
/// - `threshold` - The number above which the input will be treated as a percentage rather than a decimal.
fn parse_decimal_or_percentage(input: &str, threshold: f64) -> Option<f64> {
    let mut expr = eval_expression_string(input).ok()?;
    if expr >= threshold {
        // The user probably meant to input a raw percentage and not a decimal in 0..1
        expr /= 100.0;
    }
    Some(expr)
}

static ICON: &[u8] = include_bytes!("../../../../assets/icon.png");

pub fn run() -> Result<(), Box<dyn Error>> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1300.0, 720.0])
            .with_icon(eframe::icon_data::from_png_bytes(ICON)?)
            .with_app_id(NtscApp::APP_ID),
        ..Default::default()
    };
    Ok(eframe::run_native(
        NtscApp::APP_ID,
        options,
        Box::new(|cc| {
            // GStreamer can be slow to initialize (on the order of minutes). Do it off-thread so we can display a
            // loading screen in the meantime. Thanks for being thread-safe, unlike GTK!
            let gstreamer_initialized = Arc::new(AtomicBool::new(false));
            let gstreamer_initialized_for_thread = Arc::clone(&gstreamer_initialized);
            thread::spawn(move || {
                initialize_gstreamer().unwrap();
                gstreamer_initialized_for_thread.store(true, Ordering::Release);
            });

            let settings_list = SettingsList::new();
            let (settings, theme) = if let Some(storage) = cc.storage {
                // Load previous effect settings from storage
                let settings = storage
                    .get_string("effect_settings")
                    .and_then(|saved_settings| settings_list.from_json(&saved_settings).ok())
                    .unwrap_or_default();

                let theme = storage
                    .get_string("color_theme")
                    .and_then(|color_theme| ColorTheme::try_from(color_theme.as_str()).ok())
                    .unwrap_or_default();

                (settings, theme)
            } else {
                (NtscEffectFullSettings::default(), ColorTheme::default())
            };

            let ctx = cc.egui_ctx.clone();
            ctx.set_visuals(theme.visuals(&cc.integration_info));
            ctx.style_mut(|style| style.interaction.tooltip_delay = 0.5);
            Box::new(NtscApp::new(
                ctx,
                settings_list,
                settings,
                theme,
                gstreamer_initialized,
            ))
        }),
    )?)
}

impl NtscApp {
    pub const APP_ID: &'static str = "ntsc-rs";

    fn new(
        ctx: egui::Context,
        settings_list: SettingsList,
        effect_settings: NtscEffectFullSettings,
        color_theme: ColorTheme,
        gstreamer_initialized: Arc<AtomicBool>,
    ) -> Self {
        Self {
            gstreamer_initialized,
            settings_list,
            pipeline: None,
            undoer: Undoer::default(),
            executor: AppExecutor::new(ctx.clone()),
            video_zoom: VideoZoom {
                scale: 1.0,
                fit: true,
            },
            video_scale: VideoScale {
                scale: 480,
                enabled: false,
            },
            audio_volume: AudioVolume::default(),
            effect_preview: EffectPreviewSettings::default(),
            left_panel_state: LeftPanelState::default(),
            effect_settings,
            presets_state: PresetsState::default(),
            render_settings: RenderSettings::default(),
            render_jobs: Vec::new(),
            settings_json_paste: String::new(),
            last_error: RefCell::new(None),
            color_theme,
            credits_dialog_open: false,
            third_party_licenses_dialog_open: false,
            license_dialog_open: false,
        }
    }

    pub fn spawn(&self, future: impl Future<Output = Option<AppFn>> + 'static + Send) {
        self.executor.spawn(future);
    }

    fn tick(&mut self) {
        loop {
            // Get the functions to be executed at the end of the completed futures.
            let app_fns = self.executor.tick();

            // If there are none, we're done. If there are, loop--executing them may spawn more futures.
            if app_fns.is_empty() {
                break;
            }

            // Execute functions outside the executor--if they call `spawn`, we don't want to recursively lock the
            // executor's mutex.
            for f in app_fns {
                self.handle_result_with(f);
            }
        }
    }

    fn load_video(&mut self, ctx: &egui::Context, path: PathBuf) -> Result<(), ApplicationError> {
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(
            path.file_name()
                .map(|file_name| format!("{} — {}", Self::APP_ID, file_name.to_string_lossy()))
                .unwrap_or_else(|| Self::APP_ID.to_string()),
        ));

        self.pipeline = Some(
            self.create_preview_pipeline(ctx, path)
                .context(LoadVideoSnafu)?,
        );

        Ok(())
    }

    fn close_video(&mut self, ctx: &egui::Context) {
        self.pipeline = None;
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(Self::APP_ID.to_string()));
    }

    fn rescale_video(
        pipeline: &gstreamer::Pipeline,
        seek_pos: ClockTime,
        scanlines: Option<usize>,
    ) -> Result<(), GstreamerError> {
        let caps_filter = pipeline.by_name("caps_filter").unwrap();

        if let Some(scanlines) = scanlines {
            let scale_caps = pipeline
                .by_name("video_scale")
                .and_then(|elem| elem.static_pad("sink"))
                .and_then(|pad| pad.current_caps());
            let scale_caps = match scale_caps {
                Some(caps) => caps,
                None => return Ok(()),
            };

            if let Some((dst_width, dst_height)) = scale_from_caps(&scale_caps, scanlines) {
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
            pipeline.query_position::<ClockTime>().unwrap_or(seek_pos),
        )?;

        Ok(())
    }

    fn set_still_image_framerate(
        pipeline: &gstreamer::Pipeline,
        framerate: gstreamer::Fraction,
    ) -> Result<Option<gstreamer::Fraction>, GstreamerError> {
        let Some(caps_filter) = pipeline.by_name("framerate_caps_filter") else {
            return Ok(None);
        };

        caps_filter.set_property(
            "caps",
            VideoCapsBuilder::default().framerate(framerate).build(),
        );
        // This seek is necessary to prevent caps negotiation from failing due to race conditions, for some reason.
        // It seems like in some cases, there would be "tearing" in the caps between different elements, where some
        // elements' caps would use the old framerate and some would use the new framerate. This would cause caps
        // negotiation to fail, even though the caps filter sends a "reconfigure" event. This in turn woulc make the
        // entire pipeline error out.
        if let Some(seek_pos) = pipeline.query_position::<ClockTime>() {
            pipeline.seek_simple(
                gstreamer::SeekFlags::FLUSH | gstreamer::SeekFlags::ACCURATE,
                seek_pos,
            )?;
            Ok(Some(framerate))
        } else {
            Ok(None)
        }
    }

    fn set_pipeline_volume(pipeline: &gstreamer::Pipeline, volume: f64, mute: bool) {
        let Some(audio_volume) = pipeline.by_name("audio_volume") else {
            return;
        };

        audio_volume.set_property("volume", volume);
        audio_volume.set_property("mute", mute);
    }

    fn sink_preview_mode(preview_settings: &EffectPreviewSettings) -> EffectPreviewSetting {
        match preview_settings.mode {
            EffectPreviewMode::Enabled => EffectPreviewSetting::Enabled,
            EffectPreviewMode::Disabled => EffectPreviewSetting::Disabled,
            EffectPreviewMode::SplitScreen => {
                EffectPreviewSetting::SplitScreen(preview_settings.preview_rect)
            }
        }
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
            egui::ColorImage::from_rgb([0, 0], &[]),
            egui::TextureOptions::LINEAR,
        );
        let tex_sink = SinkTexture(Some(tex.clone()));
        let egui_ctx = EguiCtx(Some(ctx.clone()));
        let video_sink = gstreamer::ElementFactory::make("eguisink")
            .property("texture", tex_sink)
            .property("ctx", egui_ctx)
            .property(
                "settings",
                NtscFilterSettings((&self.effect_settings).into()),
            )
            .property(
                "preview-mode",
                Self::sink_preview_mode(&self.effect_preview),
            )
            .build()?;

        let pipeline_info_state = Arc::new(Mutex::new(PipelineStatus::Loading));
        let pipeline_info_state_for_handler = Arc::clone(&pipeline_info_state);
        let pipeline_info_state_for_callback = Arc::clone(&pipeline_info_state);
        let at_eos = Arc::new(Mutex::new(false));
        let at_eos_for_handler = Arc::clone(&at_eos);
        let ctx_for_handler = ctx.clone();
        let ctx_for_callback = ctx.clone();

        let metadata = Arc::new(Mutex::new(PipelineMetadata::default()));
        let metadata_for_audio_handler = metadata.clone();
        let metadata_for_bus_handler = metadata.clone();

        let audio_sink_for_closure = audio_sink.clone();
        let video_sink_for_closure = video_sink.clone();

        let pipeline = create_pipeline(
            src.clone(),
            move |pipeline| {
                pipeline.add(&audio_sink_for_closure)?;
                metadata_for_audio_handler.lock().unwrap().has_audio = Some(true);
                Ok(Some(audio_sink_for_closure))
            },
            move |pipeline, VideoElemMetadata { .. }| {
                pipeline.add(&video_sink_for_closure)?;
                Ok(video_sink_for_closure)
            },
            move |bus, msg| {
                debug!("{:?}", msg);
                let at_eos = &at_eos_for_handler;
                let ctx = &ctx_for_handler;
                let pipeline_info_state = &pipeline_info_state_for_handler;
                let metadata = &metadata_for_bus_handler;

                let handle_msg = move |_bus, msg: &gstreamer::Message| -> Option<()> {
                    // Make sure we're listening to a pipeline event
                    let src = msg.src()?;

                    if let gstreamer::MessageView::Error(err_msg) = msg.view() {
                        debug!("handling error message: {:?}", msg);
                        let mut pipeline_state = pipeline_info_state.lock().unwrap();
                        if !matches!(&*pipeline_state, PipelineStatus::Error(_)) {
                            *pipeline_state = PipelineStatus::Error(err_msg.error().into());
                            ctx.request_repaint();
                        }
                    }

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
                                *pipeline_info_state.lock().unwrap() = PipelineStatus::Loaded;

                                let mut metadata = metadata.lock().unwrap();

                                let is_still_image =
                                    pipeline.by_name("still_image_freeze").is_some();
                                metadata.is_still_image = Some(is_still_image);

                                let video_rate = pipeline.by_name("video_rate");
                                let caps = video_rate.and_then(|video_rate| {
                                    video_rate
                                        .static_pad("src")
                                        .and_then(|pad| pad.current_caps())
                                });

                                if let Some(caps) = caps {
                                    let structure = caps.structure(0);

                                    metadata.framerate = structure.and_then(|structure| {
                                        structure.get::<gstreamer::Fraction>("framerate").ok()
                                    });

                                    metadata.interlace_mode = structure.and_then(|structure| {
                                        Some(VideoInterlaceMode::from_string(
                                            structure.get("interlace-mode").ok()?,
                                        ))
                                    });

                                    metadata.resolution = structure.and_then(|structure| {
                                        Some((
                                            structure.get::<i32>("width").ok()? as usize,
                                            structure.get::<i32>("height").ok()? as usize,
                                        ))
                                    });
                                } else {
                                    metadata.framerate = None;
                                    metadata.interlace_mode = None;
                                    metadata.resolution = None;
                                }
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
            gstreamer::Fraction::from(30),
            Some(move |p: Result<gstreamer::Pipeline, PipelineError>| {
                if let Err(e) = p {
                    *pipeline_info_state_for_callback.lock().unwrap() = PipelineStatus::Error(e);
                    ctx_for_callback.request_repaint();
                }
            }),
        )?;

        pipeline.set_state(gstreamer::State::Paused)?;

        Ok(PipelineInfo {
            pipeline,
            state: pipeline_info_state,
            path,
            egui_sink: video_sink,
            at_eos,
            last_seek_pos: ClockTime::ZERO,
            preview: tex,
            metadata,
        })
    }

    fn interlaced_output_allowed(&self) -> bool {
        matches!(
            self.effect_settings.use_field,
            UseField::InterleavedUpper | UseField::InterleavedLower
        )
    }

    fn create_render_job(
        &mut self,
        ctx: &egui::Context,
        src_path: &Path,
        settings: RenderPipelineSettings,
    ) -> Result<RenderJob, GstreamerError> {
        let scale = if self.video_scale.enabled {
            Some(self.video_scale.scale)
        } else {
            None
        };

        let still_image_settings = {
            let framerate = self
                .pipeline
                .as_ref()
                .map(|info| info.metadata.lock().unwrap())
                .and_then(|metadata: std::sync::MutexGuard<PipelineMetadata>| metadata.framerate)
                .unwrap_or(gstreamer::Fraction::from(30));

            StillImageSettings {
                framerate,
                duration: self.render_settings.duration,
            }
        };

        RenderJob::create(
            self.executor.make_spawner(),
            ctx,
            src_path,
            settings,
            &still_image_settings,
            scale,
        )
    }

    fn update_effect(&self) {
        if let Some(PipelineInfo { egui_sink, .. }) = &self.pipeline {
            egui_sink.set_property(
                "settings",
                NtscFilterSettings((&self.effect_settings).into()),
            );
        }
    }

    pub fn set_effect_settings(&mut self, effect_settings: NtscEffectFullSettings) {
        self.effect_settings = effect_settings;
        self.update_effect();
    }

    pub fn handle_error(&self, err: &dyn Error) {
        *self.last_error.borrow_mut() = Some(format!("{}", err));
    }

    pub fn handle_result<T, E: Error>(&self, result: Result<T, E>) {
        if let Err(err) = result {
            self.handle_error(&err);
        }
    }

    pub fn handle_result_with<T, E: Error, F: FnOnce(&mut Self) -> Result<T, E>>(&mut self, cb: F) {
        let result = cb(self);
        self.handle_result(result);
    }

    fn undo(&mut self) {
        if let Some(new_state) = self.undoer.undo(&self.effect_settings).cloned() {
            self.set_effect_settings(new_state);
        }
    }

    fn redo(&mut self) {
        if let Some(new_state) = self.undoer.redo(&self.effect_settings).cloned() {
            self.set_effect_settings(new_state);
        }
    }
}

fn parse_expression_string(input: &str) -> Option<f64> {
    eval_expression_string(input).ok()
}

impl NtscApp {
    fn setting_from_descriptor(
        ui: &mut egui::Ui,
        effect_settings: &mut NtscEffectFullSettings,
        descriptor: &SettingDescriptor,
        interlace_mode: VideoInterlaceMode,
    ) -> (Response, bool) {
        let mut changed = false;
        let resp = match &descriptor {
            SettingDescriptor {
                id: SettingID::RANDOM_SEED,
                ..
            } => {
                ui.horizontal(|ui| {
                    let rand_btn_width = ui.spacing().interact_size.y + 4.0;
                    let resp = ui.add_sized(
                        egui::vec2(
                            ui.spacing().slider_width + ui.spacing().interact_size.x
                                - rand_btn_width,
                            ui.spacing().interact_size.y,
                        ),
                        egui::DragValue::new(&mut effect_settings.random_seed)
                            .clamp_range(i32::MIN..=i32::MAX),
                    );

                    if ui
                        .add_sized(
                            egui::vec2(rand_btn_width, ui.spacing().interact_size.y),
                            egui::Button::new("🎲"),
                        )
                        .on_hover_text("Randomize seed")
                        .clicked()
                    {
                        effect_settings.random_seed = rand::random::<i32>();
                        changed = true;
                    }

                    let label = ui.add(egui::Label::new(descriptor.label).truncate(true));
                    if let Some(description) = descriptor.description {
                        label.on_hover_text(description);
                    }

                    // Return the DragValue response because that's what we want to add the tooltip to
                    resp
                })
                .response
            }
            SettingDescriptor {
                kind: SettingKind::Enumeration { options, .. },
                ..
            } => {
                let selected_index = descriptor.id.get_field_enum(effect_settings).unwrap();
                let selected_item = options
                    .iter()
                    .find(|option| option.index == selected_index)
                    .unwrap();
                egui::ComboBox::new(descriptor.id, descriptor.label)
                    .selected_text(selected_item.label)
                    .show_ui(ui, |ui| {
                        for item in options {
                            let mut label =
                                ui.selectable_label(selected_index == item.index, item.label);

                            if let Some(desc) = item.description {
                                label = label.on_hover_text(desc);
                            }

                            if label.clicked() {
                                let _ = descriptor.id.set_field_enum(effect_settings, item.index);
                                // a selectable_label being clicked doesn't set response.changed
                                changed = true;
                            };
                        }
                    })
                    .response
            }
            SettingDescriptor {
                kind: SettingKind::Percentage { logarithmic, .. },
                ..
            } => ui.add(
                egui::Slider::new(
                    descriptor.id.get_field_mut::<f32>(effect_settings).unwrap(),
                    0.0..=1.0,
                )
                .text(descriptor.label)
                .custom_parser(parse_expression_string)
                .custom_formatter(format_percentage)
                .logarithmic(*logarithmic),
            ),
            SettingDescriptor {
                kind: SettingKind::IntRange { range, .. },
                ..
            } => {
                let mut value = 0i32;
                if let Some(v) = descriptor.id.get_field_mut::<i32>(effect_settings) {
                    value = *v;
                } else if let Some(v) = descriptor.id.get_field_mut::<u32>(effect_settings) {
                    value = *v as i32;
                }

                let slider = ui.add(
                    egui::Slider::new(&mut value, range.clone())
                        .text(descriptor.label)
                        .custom_parser(parse_expression_string),
                );

                if slider.changed() {
                    if let Some(v) = descriptor.id.get_field_mut::<i32>(effect_settings) {
                        *v = value;
                    } else if let Some(v) = descriptor.id.get_field_mut::<u32>(effect_settings) {
                        *v = value as u32;
                    }
                }

                slider
            }
            SettingDescriptor {
                kind:
                    SettingKind::FloatRange {
                        range, logarithmic, ..
                    },
                ..
            } => ui.add(
                egui::Slider::new(
                    descriptor.id.get_field_mut::<f32>(effect_settings).unwrap(),
                    range.clone(),
                )
                .text(descriptor.label)
                .custom_parser(parse_expression_string)
                .logarithmic(*logarithmic),
            ),
            SettingDescriptor {
                kind: SettingKind::Boolean { .. },
                ..
            } => {
                let checkbox = ui.checkbox(
                    descriptor
                        .id
                        .get_field_mut::<bool>(effect_settings)
                        .unwrap(),
                    descriptor.label,
                );

                checkbox
            }
            SettingDescriptor {
                kind: SettingKind::Group { children, .. },
                id,
                ..
            } => {
                ui.add_space(2.0);
                let resp = ui
                    .group(|ui| {
                        ui.set_width(ui.max_rect().width());
                        let checked = descriptor
                            .id
                            .get_field_mut::<bool>(effect_settings)
                            .unwrap();
                        let was_checked = *checked;

                        let id = ui.make_persistent_id(id);
                        let mut state =
                            egui::collapsing_header::CollapsingState::load_with_default_open(
                                ui.ctx(),
                                id,
                                *checked,
                            );

                        let checkbox = ui
                            .horizontal(|ui| {
                                let checkbox = ui.checkbox(checked, descriptor.label);

                                // Show twirly arrow at the rightmost position
                                let rect = ui.max_rect();
                                let rect = rect.with_min_x(rect.max.x - rect.height());
                                let response = ui.allocate_rect(rect, egui::Sense::click());
                                egui::collapsing_header::paint_default_icon(
                                    ui,
                                    state.openness(ui.ctx()),
                                    // The default arrow size is big and distracting, but we want to keep the large
                                    // hitbox, so only shrink the response for painting.
                                    &response.clone().with_new_rect(response.rect.shrink(2.0)),
                                );

                                if response.clicked() {
                                    state.toggle(ui);
                                }

                                checkbox
                            })
                            .inner;

                        ui.set_enabled(*checked);

                        // When a settings group is re-enabled, expand it automatically.
                        if *checked && !was_checked {
                            state.set_open(true);
                        }

                        let child_response = state.show_body_unindented(ui, |ui| {
                            Self::settings_from_descriptors(
                                effect_settings,
                                ui,
                                children,
                                interlace_mode,
                            )
                        });

                        if let Some(egui::InnerResponse { inner, .. }) = child_response {
                            changed |= inner;
                        }

                        checkbox
                    })
                    .inner;
                ui.add_space(2.0);
                resp
            }
        };

        (resp, changed)
    }

    fn settings_from_descriptors(
        effect_settings: &mut NtscEffectFullSettings,
        ui: &mut egui::Ui,
        descriptors: &[SettingDescriptor],
        interlace_mode: VideoInterlaceMode,
    ) -> bool {
        let mut changed = false;
        for descriptor in descriptors {
            // The "Use field" setting has no effect on interlaced video.
            let (response, setting_changed) = if descriptor.id == SettingID::USE_FIELD
                && interlace_mode != VideoInterlaceMode::Progressive
            {
                let resp = ui.add_enabled_ui(false, |ui| {
                    Self::setting_from_descriptor(
                        ui,
                        effect_settings,
                        descriptor,
                        VideoInterlaceMode::Progressive,
                    )
                });

                resp.inner
            } else {
                Self::setting_from_descriptor(ui, effect_settings, descriptor, interlace_mode)
            };

            changed |= response.changed() || setting_changed;

            if let Some(desc) = descriptor.description {
                response.on_hover_text(desc);
            }
        }

        changed
    }

    fn show_effect_settings(&mut self, ui: &mut egui::Ui) {
        egui::TopBottomPanel::bottom("preset_copy_paste")
            .exact_height(ui.spacing().interact_size.y * 2.0)
            .show_inside(ui, |ui| {
                ui.horizontal_centered(|ui| {
                    if ui.button("Save to...").clicked() {
                        let json = self.settings_list.to_json(&self.effect_settings);
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

                    if ui.button("Load from...").clicked() {
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

                                    app.set_effect_settings(settings);

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
                        egui::Window::new("Paste JSON")
                            .default_pos(btn.rect.center_top())
                            .open(&mut is_open)
                            .show(ui.ctx(), |ui| {
                                ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                                    if ui.button("Load").clicked() {
                                        match self
                                            .settings_list
                                            .from_json(&self.settings_json_paste)
                                        {
                                            Ok(settings) => {
                                                self.set_effect_settings(settings);
                                                // Close the popup if the JSON was successfully loaded
                                                ui.ctx().data_mut(|map| {
                                                    map.insert_temp(paste_popup_id, false)
                                                });
                                            }
                                            Err(e) => {
                                                self.handle_error(&e);
                                            }
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
                        self.set_effect_settings(NtscEffectFullSettings::default());
                    }
                });
            });

        let id = ui.make_persistent_id("presets_manager_open");
        let collapse_state =
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
        egui::TopBottomPanel::bottom("preset_manager")
            .resizable(collapse_state.is_open())
            // Determined experimentall to prevent the presets manager from becoming too tall to drag back down by its
            // top edge
            .max_height(600.0f32.min(ui.available_height() - 100.0))
            .min_height(egui::lerp(
                ui.spacing().interact_size.y..=200.0,
                collapse_state.openness(ui.ctx()),
            ))
            .show_inside(ui, |ui| {
                // Prevent buttons in the preset manager from having their outlines cut off
                ui.visuals_mut().clip_rect_margin = 2.0;
                let collapse_state = collapse_state.show_header(ui, |ui| {
                    // In order to properly resize the panel when we open the "Presets" header, we need to create the
                    // CollapsingState outside this UI. That means we can't just use a regular CollapsingHeader and must
                    // draw it ourselves. Somehow, this `interact` causes the header to be clickable throughout.
                    let resp =
                        ui.interact(ui.available_rect_before_wrap(), id, egui::Sense::click());
                    let style = ui.style().interact(&resp);
                    ui.add(
                        egui::Label::new(egui::RichText::new("Presets").color(style.text_color()))
                            .selectable(false),
                    );
                });

                collapse_state.body_unindented(|ui| {
                    self.show_presets_pane(ui);
                });
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.visuals_mut().clip_rect_margin = 4.0;
            egui::ScrollArea::vertical()
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    Self::setup_control_rows(ui);

                    let Self {
                        settings_list,
                        effect_settings,
                        pipeline,
                        ..
                    } = self;
                    let interlace_mode = pipeline
                        .as_ref()
                        .and_then(|pipeline| pipeline.metadata.lock().unwrap().interlace_mode)
                        .unwrap_or(VideoInterlaceMode::Progressive);
                    let settings_changed = Self::settings_from_descriptors(
                        effect_settings,
                        ui,
                        &settings_list.settings,
                        interlace_mode,
                    );
                    if settings_changed {
                        self.update_effect();
                    }
                });
        });
    }

    fn setup_control_rows(ui: &mut egui::Ui) {
        const LABEL_WIDTH: f32 = 180.0;

        let remaining_width = ui.max_rect().width() - LABEL_WIDTH;

        let spacing = ui.spacing_mut();
        spacing.slider_width = remaining_width - 48.0;
        spacing.interact_size.x = 48.0;
        spacing.combo_width =
            spacing.slider_width + spacing.interact_size.x + spacing.item_spacing.x;
    }

    fn show_render_settings(&mut self, ui: &mut egui::Ui) {
        egui::Frame::central_panel(ui.style()).show(ui, |ui| {
            Self::setup_control_rows(ui);
            let mut codec_changed = false;
            egui::ComboBox::from_label("Codec")
                .selected_text(self.render_settings.output_codec.label())
                .show_ui(ui, |ui| {
                    codec_changed |= ui.selectable_value(
                        &mut self.render_settings.output_codec,
                        OutputCodec::H264,
                        OutputCodec::H264.label(),
                    ).changed();
                    codec_changed |= ui.selectable_value(
                        &mut self.render_settings.output_codec,
                        OutputCodec::Ffv1,
                        OutputCodec::Ffv1.label(),
                    ).changed();
                });

            if codec_changed {
                self.render_settings.output_path.set_extension(self.render_settings.output_codec.extension());
            }

            match self.render_settings.output_codec {
                OutputCodec::H264 => {
                    ui.add(
                        egui::Slider::new(&mut self.render_settings.h264_settings.crf, 0..=50)
                            .text("Quality"),
                    ).on_hover_text("Video quality factor, where 0 is the worst quality and 50 is the best. Higher quality videos take up more space.");
                    ui.add(
                        egui::Slider::new(
                            &mut self.render_settings.h264_settings.encode_speed,
                            0..=8,
                        )
                        .text("Encoding speed"),
                    ).on_hover_text("Encoding speed preset. Higher encoding speeds provide a worse compression ratio, resulting in larger videos at a given quality.");
                    // Disabled for now until I can find a way to query for 10-bit support
                    /*ui.checkbox(
                        &mut self.render_settings.h264_settings.ten_bit,
                        "10-bit color",
                    );*/
                    ui.checkbox(
                        &mut self.render_settings.h264_settings.chroma_subsampling,
                        "4:2:0 chroma subsampling",
                    ).on_hover_text("Subsample the chrominance planes to half the resolution of the luminance plane. Increases playback compatibility.");
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
                    ).on_hover_text("Subsample the chrominance planes to half the resolution of the luminance plane. Results in smaller files.");
                }
            }

            ui.separator();

            ui.rtl(|ui| {
                let save_file = ui.button("📁").on_hover_text("Browse for a path").clicked();

                ui.ltr(|ui| {
                    ui.label("Destination file:");
                    let mut path = self.render_settings.output_path.to_string_lossy();
                    if ui.add_sized(ui.available_size(), egui::TextEdit::singleline(&mut path)).changed() {
                        self.render_settings.output_path = PathBuf::from(OsStr::new(path.as_ref()));
                    }
                });

                if save_file {
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
                            file_dialog = file_dialog.set_file_name(format!(
                                "{}_ntsc.{}",
                                file_name.to_string_lossy(),
                                self.render_settings.output_codec.extension()
                            ));
                        }
                    }

                    let file_dialog = file_dialog.save_file();
                    self.spawn(async move {
                        let handle = file_dialog.await;
                        Some(Box::new(|app: &mut NtscApp| {
                            if let Some(handle) = handle {
                                let mut output_path: PathBuf = handle.into();
                                if output_path.extension().is_none() {
                                    output_path.set_extension(app.render_settings.output_codec.extension());
                                }
                                app.render_settings.output_path = output_path;
                            }

                            Ok(())
                        }) as _)
                    });
                }
            });

            let src_path = self.pipeline.as_ref().map(|info| &info.path);

            let mut duration = self.render_settings.duration.mseconds();
            if self
                .pipeline
                .as_ref()
                .map(|info| info.metadata.lock().unwrap())
                .and_then(|metadata| metadata.is_still_image)
                .unwrap_or(false)
            {
                ui.horizontal(|ui| {
                    ui.label("Duration:");
                    if ui
                        .add(
                            egui::DragValue::new(&mut duration)
                                .custom_formatter(|value, _| {
                                    clock_time_format(
                                        (value * ClockTime::MSECOND.nseconds() as f64)
                                            as u64,
                                    )
                                })
                                .custom_parser(clock_time_parser)
                                .speed(100.0),
                        )
                        .changed()
                    {
                        self.render_settings.duration =
                            ClockTime::from_mseconds(duration);
                    }
                });
            }

            ui
                .add_enabled(
                    self.interlaced_output_allowed(),
                    egui::Checkbox::new(&mut self.render_settings.interlaced, "Interlaced output")
                )
                .on_disabled_hover_text("To enable interlaced output, set the \"Use field\" setting to \"Interleaved\".");


            if ui
                .add_enabled(
                    !self.render_settings.output_path.as_os_str().is_empty() && src_path.is_some(),
                    egui::Button::new("Render"),
                )
                .clicked()
            {
                let render_job = self.create_render_job(
                    ui.ctx(),
                    &src_path.unwrap().clone(),
                    RenderPipelineSettings {
                        codec_settings: (&self.render_settings).into(),
                        output_path: self.render_settings.output_path.clone(),
                        interlacing: match (
                            self.interlaced_output_allowed() && self.render_settings.interlaced,
                            self.effect_settings.use_field
                        ) {
                            (true, UseField::InterleavedUpper) => RenderInterlaceMode::TopFieldFirst,
                            (true, UseField::InterleavedLower) => RenderInterlaceMode::BottomFieldFirst,
                            _ => RenderInterlaceMode::Progressive,
                        },
                        effect_settings: (&self.effect_settings).into(),
                    },
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
                    self.render_jobs.retain_mut(|job| {
                        let mut closed = false;
                        ui.add(RenderJobWidget::new(job, &mut closed));
                        !closed
                    })
                });
        });
    }

    fn show_video_pane(&mut self, ui: &mut egui::Ui) {
        let last_seek_pos = if let Some(info) = &mut self.pipeline {
            // While seeking, GStreamer sometimes doesn't return a timecode. In that case, use the last timecode it
            // did respond with.
            let queried_pos = info.pipeline.query_position::<ClockTime>();
            if let Some(position) = queried_pos {
                info.last_seek_pos = position;
            }
            info.last_seek_pos
        } else {
            ClockTime::ZERO
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
                let mut change_framerate_res = None;
                let mut save_image_to: Option<(PathBuf, PathBuf)> = None;
                let mut copy_image_res: Option<Result<ColorImage, GstreamerError>> = None;
                if let Some(info) = &mut self.pipeline {
                    let mut metadata = info.metadata.lock().unwrap();
                    if ui.button("🗙").clicked() {
                        remove_pipeline = true;
                    }

                    ui.separator();

                    if ui.button("Save frame").clicked() {
                        let src_path = info.path.clone();

                        let dst_path = src_path.with_extension("");
                        save_image_to = Some((src_path, dst_path));
                    }

                    if ui.button("Copy frame").clicked() {
                        let egui_sink =
                            info.egui_sink.downcast_ref::<elements::EguiSink>().unwrap();

                        let egui_sink = EguiSink::from_obj(egui_sink);
                        copy_image_res = Some(egui_sink.get_image().map_err(|e| e.into()));
                    }

                    if let Some(current_framerate) = metadata.framerate {
                        ui.separator();
                        match metadata.is_still_image {
                            Some(true) => {
                                let mut new_framerate = current_framerate.numer() as f64
                                    / current_framerate.denom() as f64;
                                ui.label("fps");
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut new_framerate)
                                            .clamp_range(0.0..=240.0),
                                    )
                                    .changed()
                                {
                                    let framerate_fraction =
                                        gstreamer::Fraction::approximate_f64(new_framerate);
                                    if let Some(f) = framerate_fraction {
                                        let changed_framerate =
                                            Self::set_still_image_framerate(&info.pipeline, f);
                                        if let Ok(Some(new_framerate)) = changed_framerate {
                                            metadata.framerate = Some(new_framerate);
                                        }

                                        change_framerate_res = Some(changed_framerate);
                                    }
                                }
                            }
                            Some(false) => {
                                let mut fps_display = format!(
                                    "{:.2} fps",
                                    current_framerate.numer() as f64
                                        / current_framerate.denom() as f64
                                );
                                if let Some(interlace_mode) = metadata.interlace_mode {
                                    fps_display.push_str(match interlace_mode {
                                        VideoInterlaceMode::Progressive => " (progressive)",
                                        VideoInterlaceMode::Interleaved => " (interlaced)",
                                        VideoInterlaceMode::Mixed => " (telecined)",
                                        _ => "",
                                    });
                                }
                                ui.label(fps_display);
                            }
                            None => {}
                        }
                    }

                    if let Some((width, height)) = metadata.resolution {
                        ui.separator();
                        ui.label(format!("{}x{}", width, height));
                    }

                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        ui.add(egui::Label::new(info.path.to_string_lossy()).truncate(true));
                    });
                }

                if let Some(res) = change_framerate_res {
                    self.handle_result(res);
                }

                if let Some(res) = copy_image_res {
                    match res {
                        Ok(image) => {
                            let res = arboard::Clipboard::new().and_then(|mut cb| {
                                let data = arboard::ImageData {
                                    width: image.width(),
                                    height: image.height(),
                                    bytes: Cow::from(image.as_raw()),
                                };
                                cb.set_image(data)?;
                                Ok(())
                            });
                            self.handle_result(res);
                        }
                        Err(e) => {
                            self.handle_error(&e);
                        }
                    }
                }

                if remove_pipeline {
                    self.close_video(ui.ctx())
                }

                if let Some((src_path, dst_path)) = save_image_to {
                    let ctx = ui.ctx().clone();
                    self.spawn(async move {
                        let handle = rfd::AsyncFileDialog::new()
                            .set_directory(dst_path.parent().unwrap_or(Path::new("/")))
                            .set_file_name(format!(
                                "{}_ntsc.png",
                                dst_path.file_name().to_owned().unwrap().to_string_lossy()
                            ))
                            .save_file()
                            .await;

                        handle.map(|handle| {
                            Box::new(move |app: &mut NtscApp| {
                                let current_time = app
                                    .pipeline
                                    .as_ref()
                                    .and_then(|info| info.pipeline.query_position::<ClockTime>())
                                    .unwrap_or(ClockTime::ZERO);

                                let res = app.create_render_job(
                                    &ctx,
                                    &src_path.clone(),
                                    RenderPipelineSettings {
                                        codec_settings: RenderPipelineCodec::Png(PngSettings {
                                            seek_to: current_time,
                                        }),
                                        output_path: handle.into(),
                                        interlacing: RenderInterlaceMode::Progressive,
                                        effect_settings: (&app.effect_settings).into(),
                                    },
                                );
                                if let Ok(job) = res {
                                    app.render_jobs.push(job);
                                } else {
                                    app.handle_result(res);
                                }
                                Ok(())
                            }) as _
                        })
                    });
                }
            });
        });

        egui::TopBottomPanel::bottom("video_controls")
            .exact_height(ui.spacing().interact_size.y * 2.0)
            .show_inside(ui, |ui| {
                ui.set_enabled(self.pipeline.is_some());
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
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
                    let btn = ui.add_sized(
                        vec2(
                            ui.spacing().interact_size.y * 1.5,
                            ui.spacing().interact_size.y * 1.5,
                        ),
                        btn_widget,
                    );

                    let ctx = ui.ctx();
                    if !ctx.wants_keyboard_input()
                        && ctx.input(|i| {
                            i.events.iter().any(|event| {
                                if let egui::Event::Key {
                                    key,
                                    pressed,
                                    repeat,
                                    modifiers,
                                    ..
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
                        })
                    {
                        let res = self.pipeline.as_mut().map(|p| p.toggle_playing());
                        if let Some(res) = res {
                            self.handle_result(res);
                        }
                    }

                    if btn.clicked() {
                        let res = self.pipeline.as_mut().map(|p| p.toggle_playing());
                        if let Some(res) = res {
                            self.handle_result(res);
                        }
                    }

                    let duration = if let Some(info) = &self.pipeline {
                        info.pipeline.query_duration::<ClockTime>()
                    } else {
                        None
                    };

                    let mut timecode_ms =
                        last_seek_pos.nseconds() as f64 / ClockTime::MSECOND.nseconds() as f64;
                    let frame_pace = if let Some(framerate) = framerate {
                        framerate.denom() as f64 / framerate.numer() as f64
                    } else {
                        1f64 / 30f64
                    };

                    let mut drag_value = egui::DragValue::new(&mut timecode_ms)
                        .custom_formatter(|value, _| {
                            clock_time_format((value * ClockTime::MSECOND.nseconds() as f64) as u64)
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
                                gstreamer::SeekFlags::FLUSH | gstreamer::SeekFlags::ACCURATE,
                                ClockTime::from_nseconds(
                                    (timecode_ms * ClockTime::MSECOND.nseconds() as f64) as u64,
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
                            // Treat as a percentage above 8x zoom
                            .custom_parser(|input| parse_decimal_or_percentage(input, 8.0)),
                    );
                    ui.checkbox(&mut self.video_zoom.fit, "Fit");

                    ui.separator();

                    let scale_checkbox = ui.checkbox(&mut self.video_scale.enabled, "Scale to");
                    ui.add_enabled_ui(self.video_scale.enabled, |ui| {
                        let drag_resp = ui.add(
                            egui::DragValue::new(&mut self.video_scale.scale)
                                .clamp_range(1..=usize::MAX),
                        );
                        if drag_resp.changed() || scale_checkbox.changed() {
                            if let Some(pipeline) = &self.pipeline {
                                let res = Self::rescale_video(
                                    &pipeline.pipeline,
                                    pipeline.last_seek_pos,
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

                    ui.separator();

                    let has_audio = self
                        .pipeline
                        .as_ref()
                        .map(|info| info.metadata.lock().unwrap())
                        .and_then(|metadata| metadata.has_audio)
                        .unwrap_or(false);

                    ui.add_enabled_ui(has_audio, |ui| {
                        let mut update_volume = false;

                        if ui
                            .button(if self.audio_volume.mute {
                                "🔇"
                            } else {
                                match self.audio_volume.gain {
                                    0.0 => "🔇",
                                    0.0..=0.33 => "🔈",
                                    0.0..=0.67 => "🔉",
                                    _ => "🔊",
                                }
                            })
                            .on_hover_text(if self.audio_volume.mute {
                                "Unmute"
                            } else {
                                "Mute"
                            })
                            .clicked()
                        {
                            self.audio_volume.mute = !self.audio_volume.mute;
                            // "<= 0.0" to handle negative zero (not sure if it'll ever happen; better safe than sorry)
                            if !self.audio_volume.mute && self.audio_volume.gain <= 0.0 {
                                // Restore the previous gain after the user mutes by dragging the slider to 0 then unmutes
                                self.audio_volume.gain = self.audio_volume.gain_pre_mute;
                            }
                            update_volume = true;
                        }

                        let resp = ui.add_enabled(
                            !self.audio_volume.mute,
                            egui::Slider::new(&mut self.audio_volume.gain, 0.0..=1.25)
                                // Treat as a percentage above 125% volume
                                .custom_parser(|input| parse_decimal_or_percentage(input, 1.25))
                                .custom_formatter(format_percentage),
                        );

                        if resp.drag_stopped() {
                            if self.audio_volume.gain > 0.0 {
                                // Set the gain to restore after dragging the slider to 0
                                self.audio_volume.gain_pre_mute = self.audio_volume.gain;
                            } else {
                                // Wait for drag release to mute because it disables the slider
                                self.audio_volume.mute = true;
                            }
                        }

                        if resp.changed() || resp.drag_stopped() {
                            update_volume = true;
                        }

                        if update_volume {
                            if let Some(pipeline_info) = &self.pipeline {
                                NtscApp::set_pipeline_volume(
                                    &pipeline_info.pipeline,
                                    // Unlogarithmify volume (at least to my ears, this gives more control at the low end
                                    // of the slider)
                                    10f64.powf(self.audio_volume.gain - 1.0).max(0.0),
                                    self.audio_volume.mute || self.audio_volume.gain == 0.0,
                                );
                            }
                        }
                    });

                    ui.separator();

                    let mut update_effect_preview = false;
                    ui.label("✨").on_hover_text("Effect preview");
                    update_effect_preview |= ui
                        .selectable_value(
                            &mut self.effect_preview.mode,
                            EffectPreviewMode::Enabled,
                            "Enable",
                        )
                        .changed();
                    update_effect_preview |= ui
                        .selectable_value(
                            &mut self.effect_preview.mode,
                            EffectPreviewMode::Disabled,
                            "Disable",
                        )
                        .changed();
                    update_effect_preview |= ui
                        .selectable_value(
                            &mut self.effect_preview.mode,
                            EffectPreviewMode::SplitScreen,
                            "Split",
                        )
                        .changed();

                    if update_effect_preview {
                        if let Some(PipelineInfo { egui_sink, .. }) = &self.pipeline {
                            egui_sink.set_property(
                                "preview-mode",
                                Self::sink_preview_mode(&self.effect_preview),
                            );
                        }
                    }
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::side_top_panel(&ui.ctx().style()).inner_margin(0.0))
            .show_inside(ui, |ui| {
                ui.visuals_mut().clip_rect_margin = 0.0;
                ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                    if let Some(info) = &mut self.pipeline {
                        let mut timecode = info.last_seek_pos.nseconds();

                        let duration = info.pipeline.query_duration::<ClockTime>();

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
                                    gstreamer::SeekFlags::FLUSH | gstreamer::SeekFlags::ACCURATE,
                                    ClockTime::from_nseconds(timecode),
                                );
                            }
                        }
                    }
                    egui::ScrollArea::both()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.with_layout(
                                egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
                                |ui| {
                                    let Some(PipelineInfo {
                                        preview, egui_sink, ..
                                    }) = &mut self.pipeline
                                    else {
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new("No media loaded").heading(),
                                            )
                                            .selectable(false),
                                        );
                                        return;
                                    };

                                    if preview.size().iter().any(|dim| *dim == 0) {
                                        ui.add(egui::Spinner::new());
                                        return;
                                    }

                                    let texture_size = if self.video_scale.enabled {
                                        let texture_actual_size = preview.size_vec2();
                                        let scale_factor =
                                            self.video_scale.scale as f32 / texture_actual_size.y;
                                        vec2(
                                            (texture_actual_size.x * scale_factor).round(),
                                            self.video_scale.scale as f32,
                                        )
                                    } else {
                                        preview.size_vec2()
                                    };
                                    let scale_factor = if self.video_zoom.fit {
                                        // Due to floating-point error, a scrollbar may appear even if we scale down. To
                                        // avoid the scrollbar popping in and out of existence, subtract a constant value
                                        // from available_size.
                                        ((ui.available_size() - vec2(1.0, 1.0)) / texture_size)
                                            .min_elem()
                                            .min(1.0)
                                    } else {
                                        self.video_zoom.scale as f32
                                    };

                                    // We need to render the splitscreen bar in the same area as the image. The
                                    // Response returned from ui.image() fills the entire scroll area, so we need
                                    // to do the layout ourselves.
                                    let image = egui::Image::from_texture((
                                        preview.id(),
                                        texture_size * scale_factor,
                                    ));
                                    let (rect, _) = ui.allocate_exact_size(
                                        texture_size * scale_factor,
                                        egui::Sense::hover(),
                                    );
                                    // Avoid texture sampling at non-integer coordinates (causes jaggies)
                                    let rect = egui::Rect::from_points(&[
                                        rect.min.floor(),
                                        rect.max.floor(),
                                    ]);
                                    ui.put(rect, image);

                                    if self.effect_preview.mode == EffectPreviewMode::SplitScreen
                                        && ui
                                            .put(
                                                rect,
                                                SplitScreen::new(
                                                    &mut self.effect_preview.preview_rect,
                                                ),
                                            )
                                            .changed()
                                    {
                                        egui_sink.set_property(
                                            "preview_mode",
                                            Self::sink_preview_mode(&self.effect_preview),
                                        )
                                    }
                                },
                            );
                        });
                });
            });
    }

    fn show_credits_dialog(&mut self, ctx: &egui::Context) {
        egui::Window::new("About + Credits")
            .open(&mut self.credits_dialog_open)
            .default_width(400.0)
            .show(ctx, |ui| {
                const VERSION: &str = env!("CARGO_PKG_VERSION");
                ui.heading(format!("{} v{VERSION}", Self::APP_ID));

                ui.separator();

                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.label("by ");
                    ui.add(egui::Hyperlink::from_label_and_url(
                        "valadaptive",
                        "https://github.com/valadaptive/",
                    ));
                });

                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.label("...loosely based on ");
                    ui.add(egui::Hyperlink::from_label_and_url(
                        "JargeZ/ntscqt",
                        "https://github.com/JargeZ/ntscqt/",
                    ));
                });

                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.label("...which is a GUI for ");
                    ui.add(egui::Hyperlink::from_label_and_url(
                        "zhuker/ntsc",
                        "https://github.com/zhuker/ntsc/",
                    ));
                });

                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.label("...which is a port of ");
                    ui.add(egui::Hyperlink::from_label_and_url(
                        "joncampbell123/composite-video-simulator",
                        "https://github.com/joncampbell123/composite-video-simulator/",
                    ));
                });
            });
    }

    fn show_third_party_licenses_dialog(&mut self, ctx: &egui::Context) {
        egui::Window::new("Third-Party Licenses")
            .open(&mut self.third_party_licenses_dialog_open)
            .default_width(400.0)
            .default_height(400.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for (i, license) in get_third_party_licenses().iter().enumerate() {
                            if i != 0 {
                                ui.separator();
                            }
                            egui::CollapsingHeader::new(&license.name)
                                .id_source(i)
                                .show(ui, |ui| {
                                    ui.label(&license.text);
                                });
                            ui.indent(i, |ui| {
                                ui.label("Used by:");
                                for used_by in license.used_by.iter() {
                                    ui.add(egui::Hyperlink::from_label_and_url(
                                        format!("{} {}", used_by.name, used_by.version),
                                        &used_by.url,
                                    ));
                                }
                            });
                        }
                    });
            });
    }

    fn show_app(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open").clicked() {
                        let file_dialog = rfd::AsyncFileDialog::new().pick_file();
                        let ctx = ctx.clone();
                        self.spawn(async move {
                            let handle = file_dialog.await;

                            Some(Box::new(move |app: &mut NtscApp| match handle {
                                Some(handle) => app.load_video(&ctx, handle.into()),
                                None => Ok(()),
                            }) as _)
                        });

                        ui.close_menu();
                    }
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        ui.close_menu();
                    }
                });

                ui.menu_button("Edit", |ui| {
                    if ui
                        .add_enabled(
                            self.undoer.has_undo(&mut self.effect_settings),
                            egui::Button::new("Undo"),
                        )
                        .clicked()
                    {
                        self.undo();
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(
                            self.undoer.has_redo(&mut self.effect_settings),
                            egui::Button::new("Redo"),
                        )
                        .clicked()
                    {
                        self.redo();
                        ui.close_menu();
                    }
                });

                ui.menu_button("View", |ui| {
                    ui.menu_button("Theme", |ui| {
                        let mut color_theme_changed = false;
                        color_theme_changed |= ui
                            .selectable_value(&mut self.color_theme, ColorTheme::System, "System")
                            .on_hover_text("Follow system color theme")
                            .changed();
                        color_theme_changed |= ui
                            .selectable_value(&mut self.color_theme, ColorTheme::Light, "Light")
                            .on_hover_text("Use light mode")
                            .changed();
                        color_theme_changed |= ui
                            .selectable_value(&mut self.color_theme, ColorTheme::Dark, "Dark")
                            .on_hover_text("Use dark mode")
                            .changed();

                        if color_theme_changed {
                            // Results in a bit of "theme tearing" since every widget rendered after this will use a
                            // different color scheme than those rendered before it. Not really noticeable in practice.
                            ui.ctx().set_visuals(self.color_theme.visuals(frame.info()));
                            ui.close_menu();
                        }
                    });

                    ui.menu_button("Zoom", |ui| {
                        let mut zoom = ui.ctx().zoom_factor();
                        let mut changed = false;
                        const ZOOM_FACTORS: &[f32] = &[0.75, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0];
                        for item_zoom_factor in ZOOM_FACTORS {
                            changed |= ui
                                .selectable_value(
                                    &mut zoom,
                                    *item_zoom_factor,
                                    format!("{}%", item_zoom_factor * 100.0),
                                )
                                .changed();
                        }
                        if changed {
                            ui.ctx().set_zoom_factor(zoom);
                            ui.close_menu()
                        }
                    })
                });

                ui.menu_button("Help", |ui| {
                    if ui.button("Online Documentation ⤴").clicked() {
                        ui.ctx().open_url(egui::OpenUrl::new_tab(
                            "https://ntsc.rs/docs/standalone-application/",
                        ));
                        ui.close_menu();
                    }

                    if ui.button("License").clicked() {
                        self.license_dialog_open = true;
                        ui.close_menu();
                    }

                    if ui.button("Third-Party Licenses").clicked() {
                        self.third_party_licenses_dialog_open = true;
                        ui.close_menu();
                    }

                    if ui.button("About + Credits").clicked() {
                        self.credits_dialog_open = true;
                        ui.close_menu();
                    }
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    const VERSION: &str = env!("CARGO_PKG_VERSION");
                    ui.label(format!("{} v{VERSION}", Self::APP_ID));

                    let mut close_error = false;
                    if let Some(error) = self.last_error.borrow().as_ref() {
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
                        *self.last_error.borrow_mut() = None;
                    }
                });
            });
        });

        egui::SidePanel::left("controls")
            .frame(egui::Frame::side_top_panel(&ctx.style()).inner_margin(0.0))
            .resizable(true)
            .default_width(425.0)
            .width_range(300.0..=800.0)
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
                self.show_video_pane(ui);
            });

        if self.credits_dialog_open {
            self.show_credits_dialog(ctx);
        }

        if self.third_party_licenses_dialog_open {
            self.show_third_party_licenses_dialog(ctx);
        }

        if self.license_dialog_open {
            self.show_license_dialog(ctx);
        }
    }

    fn show_loading_screen(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.centered_and_justified(|ui| {
                ui.add(egui::Spinner::new().size(128.0));
            });
        });
    }

    fn handle_keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        // Seems to deadlock if we call undo() / redo() inside the ctx.input callback, probably due to Undoer accessing
        // context state from behind a mutex.
        let (should_undo, should_redo) = ctx.input(|input| {
            (
                // Note that we match command/ctrl *only*; otherwise Ctrl+Shift+Z would count as undo since Ctrl+Z is a subset of Ctrl+Shift+Z
                input.modifiers.command_only() && input.key_pressed(egui::Key::Z),
                (input.modifiers.command_only() && input.key_pressed(egui::Key::Y))
                    || (input
                        .modifiers
                        .matches_exact(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT)
                        && input.key_pressed(egui::Key::Z)),
            )
        });
        if should_undo {
            self.undo();
        } else if should_redo {
            self.redo();
        }
    }
}

impl eframe::App for NtscApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.tick();

        if !self.gstreamer_initialized.load(Ordering::Acquire) {
            self.show_loading_screen(ctx);
            return;
        }

        let mut pipeline_error = None::<PipelineError>;
        if let Some(pipeline) = &self.pipeline {
            let state = pipeline.state.lock().unwrap();
            let state = &*state;
            match state {
                PipelineStatus::Loading => {}
                PipelineStatus::Loaded => {
                    let pipeline = self.pipeline.as_ref().unwrap();
                    let mut at_eos = pipeline.at_eos.lock().unwrap();
                    if *at_eos {
                        let _ = pipeline.pipeline.set_state(gstreamer::State::Paused);
                        *at_eos = false;
                    }
                }
                PipelineStatus::Error(err) => {
                    pipeline_error = Some(err.clone());
                }
            };
        }

        if let Some(err) = pipeline_error {
            self.close_video(ctx);
            self.handle_error(&err);
        }

        self.handle_keyboard_shortcuts(ctx);

        self.show_app(ctx, frame);

        self.undoer
            .feed_state(ctx.input(|input| input.time), &self.effect_settings);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Ok(settings_json) = self
            .settings_list
            .to_json(&self.effect_settings)
            .stringify()
        {
            storage.set_string("effect_settings", settings_json);
        }

        storage.set_string(
            "color_theme",
            <&ColorTheme as Into<&str>>::into(&self.color_theme).to_owned(),
        );
    }
}
