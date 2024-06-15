use eframe::egui::{self, pos2, Rect};

#[derive(Debug)]
pub struct VideoZoom {
    pub scale: f64,
    pub fit: bool,
}

#[derive(Debug)]
pub struct VideoScale {
    pub scale: usize,
    pub enabled: bool,
}

#[derive(Debug)]
pub struct AudioVolume {
    pub gain: f64,
    // If the user drags the volume slider all the way to 0, we want to keep track of what it was before they did that
    // so we can reset the volume to it when they click the unmute button. This prevents e.g. the user setting the
    // volume to 25%, dragging it down to 0%, then clicking unmute and having it reset to some really loud default
    // value.
    pub gain_pre_mute: f64,
    pub mute: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EffectPreviewMode {
    #[default]
    Enabled,
    Disabled,
    SplitScreen,
}

#[derive(Debug)]
pub struct EffectPreviewSettings {
    pub mode: EffectPreviewMode,
    pub preview_rect: Rect,
}

impl Default for EffectPreviewSettings {
    fn default() -> Self {
        Self {
            mode: Default::default(),
            preview_rect: Rect::from_min_max(pos2(0.0, 0.0), pos2(0.5, 1.0)),
        }
    }
}

impl Default for AudioVolume {
    fn default() -> Self {
        Self {
            gain: 1.0,
            gain_pre_mute: 1.0,
            mute: false,
        }
    }
}

#[derive(Default, PartialEq, Eq)]
pub enum LeftPanelState {
    #[default]
    EffectSettings,
    RenderSettings,
}

#[derive(Default, PartialEq, Eq)]
pub enum ColorTheme {
    Dark,
    Light,
    #[default]
    System,
}

impl ColorTheme {
    pub fn visuals(&self, info: &eframe::IntegrationInfo) -> egui::Visuals {
        match &self {
            ColorTheme::Dark => egui::Visuals::dark(),
            ColorTheme::Light => egui::Visuals::light(),
            ColorTheme::System => match info.system_theme {
                Some(eframe::Theme::Dark) => egui::Visuals::dark(),
                Some(eframe::Theme::Light) => egui::Visuals::light(),
                None => egui::Visuals::default(),
            },
        }
    }
}

impl From<&ColorTheme> for &str {
    fn from(value: &ColorTheme) -> Self {
        match value {
            ColorTheme::Dark => "Dark",
            ColorTheme::Light => "Light",
            ColorTheme::System => "System",
        }
    }
}

impl TryFrom<&str> for ColorTheme {
    type Error = ();
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "Dark" => Ok(ColorTheme::Dark),
            "Light" => Ok(ColorTheme::Light),
            "System" => Ok(ColorTheme::System),
            _ => Err(()),
        }
    }
}
