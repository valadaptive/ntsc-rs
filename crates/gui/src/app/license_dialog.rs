use eframe::egui;

use super::NtscApp;

const LICENSE_TEXT: &'static str = include_str!("../../LICENSE");

impl NtscApp {
    pub fn show_license_dialog(&mut self, ctx: &egui::Context) {
        egui::Window::new("License")
            .open(&mut self.license_dialog_open)
            .default_width(500.0)
            .default_height(400.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.monospace(LICENSE_TEXT);
                    });
            });
    }
}
