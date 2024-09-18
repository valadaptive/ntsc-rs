use eframe::egui;

use crate::third_party_licenses::get_third_party_licenses;

use super::NtscApp;

impl NtscApp {
    pub fn show_third_party_licenses_dialog(&mut self, ctx: &egui::Context) {
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
                                .id_salt(i)
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
}
