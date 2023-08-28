#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::{
    path::{Path, PathBuf},
    time::SystemTime,
};

use eframe::egui;
use image::{io::Reader as ImageReader, ImageError, RgbImage};
use ntscrs::{
    ntsc::{NtscEffect, NtscEffectFullSettings},
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
