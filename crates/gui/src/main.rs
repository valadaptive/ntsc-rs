#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::{
    path::{Path, PathBuf},
    time::SystemTime,
};

use eframe::egui;
use image::{io::Reader as ImageReader, ImageError, RgbImage};
use ntscrs::{
    ntsc::{ChromaLowpass, NtscEffect, NtscEffectFullSettings, VHSTapeSpeed},
    settings::{SettingDescriptor, SettingsList},
};
use snafu::prelude::*;

fn main() -> Result<(), eframe::Error> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(1200.0, 720.0)),
        ..Default::default()
    };
    eframe::run_native(
        "ntsc-rs",
        options,
        Box::new(|cc| {
            let mut app = Box::<NtscApp>::default();
            if let Some(storage) = cc.storage {
                let path = storage.get_string("image_path");
                if let Some(path) = path {
                    if path != "" {
                        let _ = app.load_image(&cc.egui_ctx, &PathBuf::from(path));
                    }
                }
            }
            app
        }),
    )
}

struct PlayInfo {
    start_frame: usize,
    play_start: SystemTime,
}

struct NtscApp {
    image_path: Option<String>,
    image: Option<RgbImage>,
    preview: Option<egui::TextureHandle>,
    seed: u64,
    frame: usize,
    play_info: Option<PlayInfo>,
    settings: NtscEffectFullSettings,
}

impl Default for NtscApp {
    fn default() -> Self {
        Self {
            image_path: None,
            image: None,
            preview: None,
            seed: 0,
            frame: 0,
            play_info: None,
            settings: NtscEffectFullSettings::default(),
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

impl NtscApp {
    fn load_image(&mut self, ctx: &egui::Context, path: &Path) -> Result<(), LoadImageError> {
        let image = ImageReader::open(path)
            .context(IOSnafu)?
            .decode()
            .context(ImageSnafu)?;
        let image = image.into_rgb8();
        self.image = Some(image);
        self.image_path = path.as_os_str().to_str().map(String::from);
        self.update_effect(ctx);
        Ok(())
    }

    fn update_effect(&mut self, ctx: &egui::Context) {
        if let Some(image) = &self.image {
            let result =
                NtscEffect::from(&self.settings).apply_effect(image, self.frame, self.seed);
            let egui_image = egui::ColorImage::from_rgb(
                [result.width() as usize, result.height() as usize],
                result.as_raw(),
            );
            self.preview =
                Some(ctx.load_texture("preview", egui_image, egui::TextureOptions::LINEAR));
        }
    }

    fn update_frame(&mut self, ctx: &egui::Context) {
        if let Some(info) = &self.play_info {
            if let Ok(elapsed) = SystemTime::now().duration_since(info.play_start) {
                let frame = (elapsed.as_secs_f64() * 60.0) as usize + info.start_frame;
                if frame != self.frame {
                    self.frame = frame;
                    self.update_effect(ctx);
                }
            }
            ctx.request_repaint();
        }
    }

    fn settings_from_descriptors(&mut self, ui: &mut egui::Ui, descriptors: &[SettingDescriptor]) {
        for descriptor in descriptors {
            match &descriptor.kind {
                ntscrs::settings::SettingKind::Enumeration {
                    options,
                    default_value: _,
                } => {
                    let mut selected_index = descriptor.id.get_field_enum(&self.settings).unwrap();
                    let selected_item = options.iter().find(|option| {option.index == selected_index}).unwrap();
                    egui::ComboBox::from_label(descriptor.label)
                        .selected_text(selected_item.label)
                        .show_ui(ui, |ui| {
                            for item in options {
                                if ui
                                    .selectable_value(
                                        &mut selected_index,
                                        item.index,
                                        item.label,
                                    )
                                    .changed()
                                {
                                    descriptor.id.set_field_enum(&mut self.settings, selected_index);
                                    self.update_effect(ui.ctx());
                                };
                            }
                        });
                },
                ntscrs::settings::SettingKind::Percentage {
                    logarithmic,
                    default_value: _,
                } => {
                    if ui
                        .add(
                            egui::Slider::new(
                                descriptor.id.get_field_ref::<f32>(&mut self.settings).unwrap(),
                                0.0..=1.0,
                            )
                            .logarithmic(*logarithmic)
                            .text(descriptor.label),
                        )
                        .changed()
                    {
                        self.update_effect(ui.ctx());
                    }
                },
                ntscrs::settings::SettingKind::IntRange {
                    range,
                    default_value: _,
                } => {
                    let mut value = 0i32;
                    if let Some(v) = descriptor.id.get_field_ref::<i32>(&mut self.settings) {
                        value = *v;
                    } else if let Some(v) = descriptor.id.get_field_ref::<u32>(&mut self.settings) {
                        value = *v as i32;
                    }
                    if ui
                        .add(
                            egui::Slider::new(
                                &mut value,
                                range.clone(),
                            )
                            .text(descriptor.label),
                        )
                        .changed()
                    {
                        if let Some(v) = descriptor.id.get_field_ref::<i32>(&mut self.settings) {
                            *v = value;
                        } else if let Some(v) = descriptor.id.get_field_ref::<u32>(&mut self.settings) {
                            *v = value as u32;
                        }
                        self.update_effect(ui.ctx());
                    }
                }
                ntscrs::settings::SettingKind::FloatRange {
                    range,
                    logarithmic,
                    default_value: _,
                } => {
                    if ui
                        .add(
                            egui::Slider::new(
                                descriptor.id.get_field_ref::<f32>(&mut self.settings).unwrap(),
                                range.clone(),
                            )
                            .logarithmic(*logarithmic)
                            .text(descriptor.label),
                        )
                        .changed()
                    {
                        self.update_effect(ui.ctx());
                    }
                }
                ntscrs::settings::SettingKind::Boolean { default_value: _ } => {
                    if ui
                        .checkbox(
                            descriptor.id.get_field_ref::<bool>(&mut self.settings).unwrap(),
                            descriptor.label,
                        )
                        .changed()
                    {
                        self.update_effect(ui.ctx());
                    }
                },
                ntscrs::settings::SettingKind::Group {
                    children,
                    default_value: _,
                } => {
                    ui.group(|ui| {
                        if ui
                            .checkbox(
                                descriptor.id.get_field_ref::<bool>(&mut self.settings).unwrap(),
                                descriptor.label,
                            )
                            .changed()
                        {
                            self.update_effect(ui.ctx());
                        }

                        ui.set_enabled(*descriptor.id.get_field_ref::<bool>(&mut self.settings).unwrap());

                        self.settings_from_descriptors(ui, &children);
                    });
                },
            }
        }
    }
}

impl eframe::App for NtscApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.update_frame(ctx);

        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.heading("ntsc-rs");
                ui.menu_button("File", |ui| {
                    if ui.button("Open").clicked() {
                        if let Some(path) = rfd::FileDialog::new().pick_file() {
                            ui.close_menu();

                            let _ = self.load_image(ctx, &path);
                        }
                    }
                    if ui.button("Quit").clicked() {
                        frame.close();
                        ui.close_menu();
                    }
                });
            });
        });

        egui::SidePanel::left("controls")
            .resizable(true)
            .default_width(400.0)
            .width_range(200.0..=800.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        ui.heading("Controls");
                        ui.style_mut().spacing.slider_width = 200.0;

                        let settings = SettingsList::new();

                        self.settings_from_descriptors(ui, &settings.settings);

                        // ui.checkbox(&mut self.settings.chroma_lowpass_in, "Chroma low-pass in");

                        /*egui::ComboBox::from_label("Chroma low-pass in")
                            .selected_text(match &self.settings.chroma_lowpass_in {
                                ChromaLowpass::Full => "Full",
                                ChromaLowpass::Light => "Light",
                                ChromaLowpass::None => "None",
                            })
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_value(
                                        &mut self.settings.chroma_lowpass_in,
                                        ChromaLowpass::Full,
                                        "Full",
                                    )
                                    .changed()
                                {
                                    self.update_effect(ctx);
                                };
                                if ui
                                    .selectable_value(
                                        &mut self.settings.chroma_lowpass_in,
                                        ChromaLowpass::Light,
                                        "Light",
                                    )
                                    .changed()
                                {
                                    self.update_effect(ctx);
                                };
                                if ui
                                    .selectable_value(
                                        &mut self.settings.chroma_lowpass_in,
                                        ChromaLowpass::None,
                                        "None",
                                    )
                                    .changed()
                                {
                                    self.update_effect(ctx);
                                };
                            });

                        if ui
                            .add(
                                egui::Slider::new(
                                    &mut self.settings.composite_preemphasis,
                                    0f32..=2f32,
                                )
                                .text("Composite preemphasis"),
                            )
                            .changed()
                        {
                            self.update_effect(ctx);
                        }

                        if ui
                            .add(
                                egui::Slider::new(
                                    &mut self.settings.composite_noise_intensity,
                                    0f32..=0.1f32,
                                )
                                .text("Composite noise"),
                            )
                            .changed()
                        {
                            self.update_effect(ctx);
                        }

                        if ui
                            .add(
                                egui::Slider::new(&mut self.settings.snow_intensity, 0.0..=1.0)
                                    .logarithmic(true)
                                    .text("Snow"),
                            )
                            .changed()
                        {
                            self.update_effect(ctx);
                        }

                        ui.group(|ui| {
                            if ui
                                .checkbox(
                                    &mut self.settings.tracking_noise.enabled,
                                    "Tracking noise",
                                )
                                .changed()
                            {
                                self.update_effect(ctx);
                            }

                            ui.set_enabled(self.settings.tracking_noise.enabled);

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.settings.tracking_noise.settings.height,
                                        1..=120,
                                    )
                                    .text("Height"),
                                )
                                .changed()
                            {
                                self.update_effect(ctx)
                            }

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self
                                            .settings
                                            .tracking_noise
                                            .settings
                                            .wave_intensity,
                                        0.0..=50.0,
                                    )
                                    .text("Wave intensity"),
                                )
                                .changed()
                            {
                                self.update_effect(ctx)
                            }

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self
                                            .settings
                                            .tracking_noise
                                            .settings
                                            .snow_intensity,
                                        0.0..=1.0,
                                    )
                                    .logarithmic(true)
                                    .text("Snow intensity"),
                                )
                                .changed()
                            {
                                self.update_effect(ctx)
                            }

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self
                                            .settings
                                            .tracking_noise
                                            .settings
                                            .noise_intensity,
                                        0.0..=1.0,
                                    )
                                    .logarithmic(true)
                                    .text("Noise intensity"),
                                )
                                .changed()
                            {
                                self.update_effect(ctx)
                            }
                        });

                        ui.group(|ui| {
                            if ui
                                .checkbox(
                                    &mut self.settings.head_switching.enabled,
                                    "Head switching",
                                )
                                .changed()
                            {
                                self.update_effect(ctx);
                            }

                            ui.set_enabled(self.settings.head_switching.enabled);

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.settings.head_switching.settings.height,
                                        1..=24,
                                    )
                                    .text("Height"),
                                )
                                .changed()
                            {
                                self.update_effect(ctx)
                            }

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.settings.head_switching.settings.offset,
                                        0..=self.settings.head_switching.settings.height,
                                    )
                                    .text("Offset"),
                                )
                                .changed()
                            {
                                self.update_effect(ctx)
                            }

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.settings.head_switching.settings.horiz_shift,
                                        -100.0..=100.0,
                                    )
                                    .text("Horizontal shift"),
                                )
                                .changed()
                            {
                                self.update_effect(ctx)
                            }
                        });

                        ui.group(|ui| {
                            if ui
                                .checkbox(&mut self.settings.ringing.enabled, "Ringing")
                                .changed()
                            {
                                self.update_effect(ctx);
                            }

                            ui.set_enabled(self.settings.ringing.enabled);

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.settings.ringing.settings.frequency,
                                        0.0..=1.0,
                                    )
                                    .text("Frequency"),
                                )
                                .changed()
                            {
                                self.update_effect(ctx)
                            }

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.settings.ringing.settings.power,
                                        1.0..=10.0,
                                    )
                                    .text("Power"),
                                )
                                .changed()
                            {
                                self.update_effect(ctx)
                            }

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.settings.ringing.settings.intensity,
                                        0.0..=10.0,
                                    )
                                    .text("Scale"),
                                )
                                .changed()
                            {
                                self.update_effect(ctx)
                            }
                        });

                        if ui
                            .add(
                                egui::Slider::new(
                                    &mut self.settings.chroma_noise_intensity,
                                    0.0..=2.0,
                                )
                                .logarithmic(true)
                                .text("Chroma noise"),
                            )
                            .changed()
                        {
                            self.update_effect(ctx);
                        }

                        if ui
                            .add(
                                egui::Slider::new(
                                    &mut self.settings.chroma_phase_noise_intensity,
                                    0.0..=1.0,
                                )
                                .logarithmic(true)
                                .text("Chroma phase noise"),
                            )
                            .changed()
                        {
                            self.update_effect(ctx);
                        }

                        if ui
                            .add(
                                egui::Slider::new(
                                    &mut self.settings.chroma_delay.0,
                                    -10.0..=10.0,
                                )
                                .text("Chroma delay (horizontal)"),
                            )
                            .changed()
                        {
                            self.update_effect(ctx);
                        }

                        if ui
                            .add(
                                egui::Slider::new(
                                    &mut self.settings.chroma_delay.1,
                                    -10..=10,
                                )
                                .text("Chroma delay (vertical)"),
                            )
                            .changed()
                        {
                            self.update_effect(ctx);
                        }

                        ui.group(|ui| {
                            if ui
                                .checkbox(&mut self.settings.vhs_settings.enabled, "Emulate VHS")
                                .changed()
                            {
                                self.update_effect(ctx);
                            }

                            ui.set_enabled(self.settings.vhs_settings.enabled);

                            egui::ComboBox::from_label("Tape speed")
                                .selected_text(
                                    match &self.settings.vhs_settings.settings.tape_speed {
                                        Some(VHSTapeSpeed::SP) => "SP (Standard Play)",
                                        Some(VHSTapeSpeed::LP) => "LP (Long Play)",
                                        Some(VHSTapeSpeed::EP) => "EP (Extended Play)",
                                        None => "Off",
                                    },
                                )
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_value(
                                            &mut self.settings.vhs_settings.settings.tape_speed,
                                            Some(VHSTapeSpeed::SP),
                                            "SP (Standard Play)",
                                        )
                                        .changed()
                                    {
                                        self.update_effect(ctx);
                                    };
                                    if ui
                                        .selectable_value(
                                            &mut self.settings.vhs_settings.settings.tape_speed,
                                            Some(VHSTapeSpeed::LP),
                                            "LP (Long Play)",
                                        )
                                        .changed()
                                    {
                                        self.update_effect(ctx);
                                    };
                                    if ui
                                        .selectable_value(
                                            &mut self.settings.vhs_settings.settings.tape_speed,
                                            Some(VHSTapeSpeed::EP),
                                            "EP (Extended Play)",
                                        )
                                        .changed()
                                    {
                                        self.update_effect(ctx);
                                    };
                                    if ui
                                        .selectable_value(
                                            &mut self.settings.vhs_settings.settings.tape_speed,
                                            None,
                                            "Off",
                                        )
                                        .changed()
                                    {
                                        self.update_effect(ctx);
                                    };
                                });

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.settings.vhs_settings.settings.chroma_loss,
                                        0.0..=1.0,
                                    )
                                    .logarithmic(true)
                                    .text("Chroma loss"),
                                )
                                .changed()
                            {
                                self.update_effect(ctx)
                            }

                            if ui
                                .checkbox(
                                    &mut self.settings.vhs_settings.settings.chroma_vert_blend,
                                    "Chroma vertical blend",
                                )
                                .changed()
                            {
                                self.update_effect(ctx);
                            }

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.settings.vhs_settings.settings.sharpen,
                                        0.0..=5.0,
                                    )
                                    .text("Sharpen"),
                                )
                                .changed()
                            {
                                self.update_effect(ctx)
                            }

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.settings.vhs_settings.settings.edge_wave,
                                        0.0..=10.0,
                                    )
                                    .text("Edge wave intensity"),
                                )
                                .changed()
                            {
                                self.update_effect(ctx)
                            }

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.settings.vhs_settings.settings.edge_wave_speed,
                                        0.0..=10.0,
                                    )
                                    .text("Edge wave speed"),
                                )
                                .changed()
                            {
                                self.update_effect(ctx)
                            }
                        });

                        egui::ComboBox::from_label("Chroma low-pass out")
                            .selected_text(match &self.settings.chroma_lowpass_out {
                                ChromaLowpass::Full => "Full",
                                ChromaLowpass::Light => "Light",
                                ChromaLowpass::None => "None",
                            })
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_value(
                                        &mut self.settings.chroma_lowpass_out,
                                        ChromaLowpass::Full,
                                        "Full",
                                    )
                                    .changed()
                                {
                                    self.update_effect(ctx);
                                };
                                if ui
                                    .selectable_value(
                                        &mut self.settings.chroma_lowpass_out,
                                        ChromaLowpass::Light,
                                        "Light",
                                    )
                                    .changed()
                                {
                                    self.update_effect(ctx);
                                };
                                if ui
                                    .selectable_value(
                                        &mut self.settings.chroma_lowpass_out,
                                        ChromaLowpass::None,
                                        "None",
                                    )
                                    .changed()
                                {
                                    self.update_effect(ctx);
                                };
                            });*/
                    });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Preview");

            if let Some(preview_texture) = &self.preview {
                ui.image(preview_texture, preview_texture.size_vec2());
            }

            ui.horizontal(|ui| {
                ui.label("Seed:");
                if ui.add(egui::DragValue::new(&mut self.seed)).changed() {
                    self.update_effect(ctx);
                }
                ui.separator();
                ui.label("Frame:");
                if ui.add(egui::DragValue::new(&mut self.frame)).changed() {
                    self.update_effect(ctx);
                }
                ui.separator();

                if ui
                    .button(if let Some(_) = self.play_info {
                        "⏸"
                    } else {
                        "▶"
                    })
                    .clicked()
                {
                    if let Some(_) = self.play_info {
                        self.play_info = None;
                    } else {
                        self.play_info = Some(PlayInfo {
                            start_frame: self.frame,
                            play_start: SystemTime::now(),
                        });
                    }
                }
            })
        });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        storage.set_string(
            "image_path",
            match &self.image_path {
                Some(path) => path.clone(),
                None => String::from(""),
            },
        )
    }
}
