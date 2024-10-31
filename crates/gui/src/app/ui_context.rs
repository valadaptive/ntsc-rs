use eframe::egui;

pub trait UIContext: Clone + Send + Sync {
    fn request_repaint(&self);
    fn current_time(&self) -> f64;
}

impl UIContext for egui::Context {
    fn request_repaint(&self) {
        self.request_repaint();
    }

    fn current_time(&self) -> f64 {
        self.input(|input| input.time)
    }
}
