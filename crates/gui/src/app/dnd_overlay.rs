use eframe::egui::{self, DroppedFile};

pub trait UiDndExt {
    fn show_dnd_overlay(&mut self, text: impl Into<egui::RichText>) -> Option<Vec<DroppedFile>>;
}

impl UiDndExt for egui::Ui {
    fn show_dnd_overlay(&mut self, text: impl Into<egui::RichText>) -> Option<Vec<DroppedFile>> {
        let max_rect = self.max_rect();
        let dropped_files = self.ctx().input_mut(|input| {
            input
                .pointer
                .latest_pos()
                .is_some_and(|p| max_rect.contains(p))
                .then(|| std::mem::take(&mut input.raw.dropped_files))
                .and_then(|files| if files.is_empty() { None } else { Some(files) })
        });
        if dropped_files.is_some() {
            return dropped_files;
        }

        let dragging_files = self
            .ctx()
            .input(|input| !input.raw.hovered_files.is_empty());
        if !dragging_files {
            return None;
        }

        let overlay_size = max_rect.size();
        let area = egui::Area::new(self.auto_id_with("dnd_overlay"))
            .fixed_pos(max_rect.left_top())
            .default_size(overlay_size);
        area.show(self.ctx(), |ui| {
            ui.allocate_ui(overlay_size, |ui| {
                egui::Frame::none()
                    .fill(ui.visuals().extreme_bg_color.gamma_multiply(0.8))
                    .stroke(ui.style().visuals.widgets.noninteractive.bg_stroke)
                    .show(ui, |ui| {
                        ui.centered_and_justified(|ui| {
                            ui.heading(text);
                        });
                    });
            });
        });

        None
    }
}
