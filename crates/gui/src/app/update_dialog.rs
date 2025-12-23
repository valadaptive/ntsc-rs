use std::{
    collections::HashMap,
    thread::{self, JoinHandle},
};

use eframe::egui;
use snafu::{OptionExt, ResultExt, Snafu};
use tinyjson::JsonValue;
use ureq::{config::Config, tls::TlsConfig};

use crate::app::layout_helper::LayoutHelper;

pub struct UpdateResponse {
    latest_release_label: String,
    download_url: String,
    up_to_date: bool,
}

#[derive(Debug, Snafu)]
pub enum CheckForUpdatesError {
    #[snafu(display("HTTP request error: {source}"))]
    Request { source: ureq::Error },
    #[snafu(display("Error parsing JSON: {source}"))]
    JSON { source: tinyjson::JsonParseError },
    #[snafu(display("Incorrect type for field {field}"))]
    IncorrectFieldType { field: &'static str },
    #[snafu(display("Missing field {field}"))]
    MissingField { field: &'static str },
}

pub enum UpdateDialogState {
    Closed,
    Loading(JoinHandle<Result<UpdateResponse, CheckForUpdatesError>>),
    Loaded(UpdateResponse),
    Error(CheckForUpdatesError),
}

impl UpdateDialogState {
    pub(crate) fn show(&mut self, ctx: &egui::Context) {
        if let UpdateDialogState::Loading(handle) = &self {
            if handle.is_finished() {
                let handle = match std::mem::replace(self, UpdateDialogState::Closed) {
                    UpdateDialogState::Loading(handle) => handle,
                    _ => unreachable!(),
                };
                *self = match handle.join().unwrap() {
                    Ok(resp) => UpdateDialogState::Loaded(resp),
                    Err(e) => UpdateDialogState::Error(e),
                }
            }
        }
        let mut open = !matches!(self, UpdateDialogState::Closed);
        egui::Window::new("Check for Updates")
            .open(&mut open)
            .show(ctx, |ui| match &self {
                UpdateDialogState::Closed => {}
                UpdateDialogState::Loading(_) => {
                    ui.ltr(|ui| {
                        ui.label("Checking for updates... ");
                        ui.add(egui::Spinner::new());
                    });
                }
                UpdateDialogState::Loaded(update_response) => {
                    ui.heading(&update_response.latest_release_label);
                    if update_response.up_to_date {
                        ui.label("✓ Up to date");
                    } else {
                        if ui.button("Download from GitHub ⤴").clicked() {
                            ctx.open_url(egui::OpenUrl::new_tab(&update_response.download_url));
                        }
                    }
                }
                UpdateDialogState::Error(error) => {
                    ui.label("Error checking for updates:");
                    ui.monospace(error.to_string());
                }
            });

        if !open {
            *self = UpdateDialogState::Closed;
        }
    }

    pub(crate) fn open(&mut self) {
        if let UpdateDialogState::Closed = &self {
            let request_thread = thread::spawn(|| {
                let config = Config::builder()
                    .tls_config(
                        TlsConfig::builder()
                            .provider(ureq::tls::TlsProvider::NativeTls)
                            .build(),
                    )
                    .build();
                let agent = config.new_agent();
                let mut resp = agent
                    .get("https://api.github.com/repositories/682371900/releases")
                    .header("Accept", "application/vnd.github+json")
                    .header("X-GitHub-Api-Version", "2022-11-28")
                    .query("per_page", "1")
                    .call()
                    .context(RequestSnafu)?;

                let resp = resp.body_mut().read_to_string().context(RequestSnafu)?;
                let resp = resp.parse::<JsonValue>().context(JSONSnafu)?;

                let releases: &Vec<_> = resp
                    .get()
                    .context(IncorrectFieldTypeSnafu { field: "<root>" })?;
                let release: &HashMap<_, _> = releases
                    .get(0)
                    .context(MissingFieldSnafu { field: "0" })?
                    .get()
                    .context(IncorrectFieldTypeSnafu { field: "root[0]" })?;
                let tag_name: &String = release
                    .get("tag_name")
                    .context(MissingFieldSnafu { field: "tag_name" })?
                    .get()
                    .context(IncorrectFieldTypeSnafu { field: "tag_name" })?;
                let html_url: &String = release
                    .get("html_url")
                    .context(MissingFieldSnafu { field: "html_url" })?
                    .get()
                    .context(IncorrectFieldTypeSnafu { field: "html_url" })?;

                let up_to_date = tag_name.strip_prefix("v") == Some(env!("CARGO_PKG_VERSION"));

                Ok(UpdateResponse {
                    latest_release_label: format!("Latest version: {tag_name}"),
                    download_url: html_url.clone(),
                    up_to_date,
                })
            });
            *self = UpdateDialogState::Loading(request_thread);
        }
    }
}
