use std::f32;

use eframe::{
    egui::{self, Sense, Widget},
    emath::remap_clamp,
};

pub struct SplitScreen<'a> {
    value: &'a mut f64,
}

impl<'a> SplitScreen<'a> {
    pub fn new(value: &'a mut f64) -> Self {
        Self {
            value,
        }
    }
}

impl<'a> Widget for SplitScreen<'a> {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let desired_size = ui.available_size();

        let grab_radius = ui.style().interaction.resize_grab_radius_side;
        let (id, rect) = ui.allocate_space(desired_size);

        let x = rect.lerp_inside(egui::vec2(*self.value as f32, 0.0)).x;
        let interact_rect = egui::Rect::from_min_max(egui::pos2(x - grab_radius, rect.top()), egui::pos2(x + grab_radius, rect.bottom()));

        let mut response = ui.interact(interact_rect, id, Sense::drag());
        if response.hovered() || response.dragged() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        }

        if let Some(pointer_position_2d) = response.interact_pointer_pos() {
            let position = pointer_position_2d.x;
            let normalized = remap_clamp(
                position,
                rect.x_range(),
                0.0..=1.0,
            );
            *self.value = normalized as f64;
            response.mark_changed();
        }

        if ui.is_rect_visible(rect) {
            let visuals = ui.style().interact(&response);
            let painter = ui.painter();


            painter.line_segment([egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())], visuals.fg_stroke);

            // Fill with red rectangle in debug mode
            #[cfg(debug_assertions)]
            if ui.ctx().debug_on_hover() && ui.interact(rect, id, Sense::hover()).hovered() {
                painter.rect_filled(rect, egui::Rounding::ZERO, egui::Color32::from_rgba_unmultiplied(255, 0, 0, 64));
            }
        }

        response
    }
}
