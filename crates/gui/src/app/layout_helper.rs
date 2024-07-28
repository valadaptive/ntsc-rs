use eframe::egui::{self, vec2};

pub trait LayoutHelper {
    fn ltr<R>(&mut self, add_contents: impl FnOnce(&mut Self) -> R) -> egui::InnerResponse<R>;
    fn rtl<R>(&mut self, add_contents: impl FnOnce(&mut Self) -> R) -> egui::InnerResponse<R>;
}

fn ui_with_layout<'c, R>(
    ui: &mut egui::Ui,
    layout: egui::Layout,
    add_contents: Box<dyn FnOnce(&mut egui::Ui) -> R + 'c>,
) -> egui::InnerResponse<R> {
    let initial_size = vec2(
        ui.available_size_before_wrap().x,
        ui.spacing().interact_size.y,
    );

    ui.allocate_ui_with_layout(initial_size, layout, |ui| add_contents(ui))
}

impl LayoutHelper for egui::Ui {
    fn ltr<R>(&mut self, add_contents: impl FnOnce(&mut egui::Ui) -> R) -> egui::InnerResponse<R> {
        ui_with_layout(
            self,
            egui::Layout::left_to_right(egui::Align::Center),
            Box::new(add_contents),
        )
    }

    fn rtl<R>(&mut self, add_contents: impl FnOnce(&mut egui::Ui) -> R) -> egui::InnerResponse<R> {
        ui_with_layout(
            self,
            egui::Layout::right_to_left(egui::Align::Center),
            Box::new(add_contents),
        )
    }
}

pub trait TopBottomPanelExt {
    fn interact_height(self, ctx: &egui::Context) -> Self;
    fn interact_height_tall(self, ctx: &egui::Context) -> Self;
}

impl TopBottomPanelExt for egui::TopBottomPanel {
    fn interact_height(self, ctx: &egui::Context) -> Self {
        let mut frame = egui::Frame::side_top_panel(&ctx.style());
        let expected_margin = frame.inner_margin;
        frame.inner_margin.top = 0.0;
        frame.inner_margin.bottom = 0.0;
        // TODO: there needs to be a fudge factor of 2 here, to prevent the contents from being off-center.
        self.exact_height(
            ctx.style().spacing.interact_size.y
                + expected_margin.top
                + expected_margin.bottom
                + 2.0,
        )
        .frame(frame)
    }

    fn interact_height_tall(self, ctx: &egui::Context) -> Self {
        let mut frame = egui::Frame::side_top_panel(&ctx.style());
        // Kludge to restore centering of buttons along the bottom
        frame.inner_margin.bottom = 0.0;
        self.exact_height(
            ctx.style().spacing.interact_size.y * 2.0,
        )
        .frame(frame)
    }
}
