use std::{
    borrow::Cow,
    collections::HashSet,
    ffi::{OsStr, OsString},
    fs,
    hash::RandomState,
    io::Write,
    path::PathBuf,
    str::FromStr,
};

use blocking::unblock;
use eframe::egui::{self, TextBuffer};
use futures_lite::Future;
use ntscrs::ntsc::NtscEffectFullSettings;
use snafu::prelude::*;

use super::{
    AppFn, NtscApp,
    error::{
        ApplicationError, CreatePresetFileSnafu, CreatePresetJSONSnafu,
        CreatePresetsDirectorySnafu, DeletePresetSnafu, FsSnafu, InstallPresetSnafu,
        JSONParseSnafu, JSONReadSnafu, RenamePresetSnafu,
    },
    layout_helper::LayoutHelper,
};

#[derive(Debug)]
pub struct Preset {
    pub path: PathBuf,
}

#[derive(Debug, Default)]
enum PresetsDirState {
    #[default]
    NotLoaded,
    Loading(Option<Vec<Preset>>),
    Error,
    Loaded(Vec<Preset>),
}

#[derive(Debug)]
struct SelectedPreset {
    path: PathBuf,
    settings: NtscEffectFullSettings,
}

#[derive(Debug)]
enum Action {
    /// Load a preset from the given path and set it as the selected preset.
    LoadPreset { path: PathBuf },
    /// Save the current settings to a new path.
    CreatePreset { path: PathBuf },
    /// Overwrite the preset at the given path with the current settings.
    OverwritePreset { path: PathBuf },
    /// Rename or move a preset.
    RenamePreset {
        old_path: PathBuf,
        new_path: PathBuf,
    },
    /// Delete a preset file entirely.
    DeletePreset { path: PathBuf },
    /// Open the presets directory in the file manager.
    OpenPresetsDir,
    /// Explicitly reload the presets directory.
    ReloadPresetsDir,
}

#[derive(Debug, Default)]
pub struct PresetsState {
    presets_dir: PresetsDirState,
    selected_preset: Option<SelectedPreset>,
    renamed_preset: Option<(usize, String)>,
    new_preset_name: Option<String>,
    /// Count the number of active FS operations occurring on the presets directory. Once this hits 0, we will refresh
    /// the presets list.
    active_fs_operations: u32,
}

impl PresetsState {
    fn show(
        &mut self,
        ui: &mut egui::Ui,
        current_effect_settings: &NtscEffectFullSettings,
    ) -> Option<Action> {
        let mut just_pressed_save = false;
        let mut action = None::<Action>;

        let selected_preset_modified = self
            .selected_preset
            .as_ref()
            .is_some_and(|selected_preset| &selected_preset.settings != current_effect_settings);

        ui.horizontal(|ui| {
            if ui.button("Open folder").clicked() {
                action = Some(Action::OpenPresetsDir);
            }
            if ui.button("Reload").clicked() {
                match self.presets_dir {
                    PresetsDirState::Loading(_) => {}
                    _ => {
                        action = Some(Action::ReloadPresetsDir);
                    }
                }
            }

            if ui
                .add_enabled(selected_preset_modified, egui::Button::new("Overwrite"))
                .clicked()
            {
                if let Some(selected_preset) = self.selected_preset.as_ref() {
                    action = Some(Action::OverwritePreset {
                        path: selected_preset.path.clone(),
                    })
                }
            }

            if ui.button("Save as").clicked() {
                let mut preset_name = String::from("preset.json");
                just_pressed_save = true;

                let preset_names: Option<HashSet<&OsStr, RandomState>> =
                    if let PresetsDirState::Loaded(presets)
                    | PresetsDirState::Loading(Some(presets)) = &self.presets_dir
                    {
                        Some(HashSet::from_iter(
                            presets.iter().filter_map(|preset| preset.path.file_name()),
                        ))
                    } else {
                        None
                    };

                if let Some(preset_names) = preset_names {
                    let mut i = 1;
                    while preset_names.contains(
                        OsString::from_str(preset_name.as_str())
                            .unwrap()
                            .as_os_str(),
                    ) {
                        preset_name = format!("preset_{i}.json");
                        i += 1;
                    }
                }

                self.new_preset_name = Some(preset_name);
            }
        });
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .min_scrolled_height(0.0)
            .show(ui, |ui| {
                if let PresetsDirState::NotLoaded = self.presets_dir {
                    action = Some(Action::ReloadPresetsDir);
                }
                match &self.presets_dir {
                    PresetsDirState::NotLoaded | PresetsDirState::Loading(None) => {
                        ui.add(egui::Spinner::new());
                    }
                    PresetsDirState::Error => {
                        ui.label("Error loading presets directory");
                    }
                    PresetsDirState::Loaded(presets) | PresetsDirState::Loading(Some(presets)) => {
                        ui.with_layout(egui::Layout::top_down_justified(egui::Align::LEFT), |ui| {
                            if matches!(&self.presets_dir, PresetsDirState::Loading(_)) {
                                ui.disable();
                            }

                            let renamed_preset = &mut self.renamed_preset;
                            let mut close_rename = false;
                            for (index, preset) in presets.iter().enumerate() {
                                let preset_path = preset.path.as_path();

                                let selected =
                                    self.selected_preset
                                        .as_ref()
                                        .is_some_and(|selected_preset| {
                                            selected_preset.path == preset_path
                                        });

                                let mut file_name = preset_path
                                    .file_name()
                                    .unwrap_or(preset_path.as_os_str())
                                    .to_string_lossy();

                                if selected && selected_preset_modified {
                                    file_name = Cow::Owned(format!("* {file_name}"));
                                }

                                let rename_name =
                                    renamed_preset.as_mut().and_then(|(renamed_index, name)| {
                                        if *renamed_index == index {
                                            Some(name)
                                        } else {
                                            None
                                        }
                                    });

                                if let Some(name) = rename_name {
                                    let name_edit = ui.text_edit_singleline(name);

                                    if name_edit.lost_focus() {
                                        close_rename = true;
                                        if !ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                                            action = Some(Action::RenamePreset {
                                                old_path: preset_path.to_owned(),
                                                new_path: PathBuf::from_iter([
                                                    preset_path.parent().unwrap().as_os_str(),
                                                    &OsString::from_str(name).unwrap(),
                                                ]),
                                            });
                                        }
                                    }
                                    name_edit.request_focus();
                                } else {
                                    let preset_label = ui.add(egui::SelectableLabel::new(
                                        selected,
                                        file_name.as_str(),
                                    ));

                                    preset_label.context_menu(|ui| {
                                        if ui.button("Delete").clicked() {
                                            action = Some(Action::DeletePreset {
                                                path: preset_path.to_owned(),
                                            });
                                            ui.close_menu();
                                        }

                                        if ui.button("Rename").clicked() {
                                            *renamed_preset = Some((index, file_name.to_string()));
                                            ui.close_menu();
                                        }
                                    });

                                    if preset_label.clicked() {
                                        action = Some(Action::LoadPreset {
                                            path: preset_path.to_path_buf(),
                                        });
                                    }
                                }

                                if close_rename {
                                    *renamed_preset = None;
                                }
                            }

                            let clear_preset_name = if let Some(new_preset_name) =
                                &mut self.new_preset_name
                            {
                                ui.rtl(|ui| {
                                    let close_button = ui.button("🗙");
                                    let save_button = ui.button("Save");
                                    let text_edit = ui.add_sized(
                                        ui.available_size(),
                                        egui::TextEdit::singleline(new_preset_name),
                                    );
                                    if just_pressed_save {
                                        text_edit.scroll_to_me(None);
                                        text_edit.request_focus();
                                    }

                                    let (esc_pressed, enter_pressed) = ui.input(|i| {
                                        (
                                            i.key_pressed(egui::Key::Escape),
                                            i.key_pressed(egui::Key::Enter),
                                        )
                                    });
                                    if (text_edit.lost_focus() && esc_pressed)
                                        || close_button.clicked()
                                    {
                                        true
                                    } else if (text_edit.lost_focus() && enter_pressed)
                                        || save_button.clicked()
                                    {
                                        if !new_preset_name.is_empty() {
                                            let mut preset_path = NtscApp::presets_dir().unwrap();
                                            preset_path.push(new_preset_name);
                                            action =
                                                Some(Action::CreatePreset { path: preset_path });
                                        }
                                        true
                                    } else {
                                        false
                                    }
                                })
                                .inner
                            } else {
                                false
                            };

                            if clear_preset_name {
                                self.new_preset_name = None;
                            }
                        });
                    }
                }
            });

        action
    }
}

impl NtscApp {
    pub fn load_preset(&mut self, path: PathBuf) {
        self.spawn(async {
            let json_path = path.clone();
            let json = unblock(|| fs::read_to_string(json_path))
                .await
                .context(JSONReadSnafu);

            Some(Box::new(|app: &mut NtscApp| {
                let json = json?;
                let settings = app.settings_list.from_json(&json).context(JSONParseSnafu)?;
                app.set_effect_settings(settings.clone());
                app.presets_state.selected_preset = Some(SelectedPreset { path, settings });
                Ok(())
            }) as _)
        });
    }

    fn presets_dir() -> Option<PathBuf> {
        let mut presets_dir = eframe::storage_dir(Self::APP_ID)?;
        presets_dir.push("presets");
        Some(presets_dir)
    }

    fn reload_presets_dir(&mut self) {
        let Some(presets_dir) = Self::presets_dir() else {
            return;
        };

        self.presets_state.presets_dir =
            PresetsDirState::Loading(match std::mem::take(&mut self.presets_state.presets_dir) {
                PresetsDirState::Loaded(presets) => Some(presets),
                _ => None,
            });

        self.spawn(async move {
            let presets_dir2 = presets_dir.clone();
            let res = unblock(move || fs::create_dir_all(presets_dir2))
                .await
                .context(CreatePresetsDirectorySnafu);
            if let Err(e) = res {
                return Some(Box::new(move |app: &mut NtscApp| {
                    app.presets_state.presets_dir = PresetsDirState::Error;
                    Err(e)
                }) as _);
            }
            let presets_dir2 = presets_dir.clone();
            let presets = unblock(move || fs::read_dir(presets_dir2).context(FsSnafu)).await;
            let presets = match presets {
                Ok(presets) => presets,
                Err(err) => {
                    return Some(Box::new(move |app: &mut NtscApp| {
                        app.presets_state.presets_dir = PresetsDirState::Error;
                        Err(err)
                    }) as _);
                }
            };

            let mut presets = presets
                .filter_map(|entry| entry.ok().map(|entry| Preset { path: entry.path() }))
                .collect::<Vec<_>>();

            presets.sort_by(|a, b| a.path.cmp(&b.path));

            Some(Box::new(move |app: &mut NtscApp| {
                app.presets_state.presets_dir = PresetsDirState::Loaded(presets);
                Ok(())
            }) as _)
        })
    }

    pub fn show_presets_pane(&mut self, ui: &mut egui::Ui) {
        let Some(presets_dir) = NtscApp::presets_dir() else {
            return;
        };

        let action = self.presets_state.show(ui, &self.effect_settings);

        match action {
            Some(Action::LoadPreset { path }) => {
                self.load_preset(path);
            }
            Some(Action::RenamePreset { old_path, new_path }) => {
                if new_path != old_path {
                    self.do_fs_operation_then_refresh(unblock(|| {
                        if new_path.try_exists().context(RenamePresetSnafu)? {
                            return Err(ApplicationError::RenamePreset {
                                source: std::io::Error::new(
                                    std::io::ErrorKind::AlreadyExists,
                                    "A preset with that name already exists",
                                ),
                            });
                        }
                        fs::rename(old_path, new_path).context(RenamePresetSnafu)?;

                        Ok(None)
                    }));
                }
            }
            Some(Action::CreatePreset { path }) => {
                let new_selected_preset = SelectedPreset {
                    path,
                    settings: self.effect_settings.clone(),
                };
                match self
                    .settings_list
                    .to_json_string(&new_selected_preset.settings)
                    .context(CreatePresetJSONSnafu)
                {
                    Ok(json) => self.do_fs_operation_then_refresh(async move {
                        if new_selected_preset
                            .path
                            .try_exists()
                            .context(CreatePresetFileSnafu)?
                        {
                            return Err(ApplicationError::CreatePresetFile {
                                source: std::io::Error::new(
                                    std::io::ErrorKind::AlreadyExists,
                                    "A preset with that name already exists",
                                ),
                            });
                        }
                        let mut destination = fs::File::create(&new_selected_preset.path)
                            .context(CreatePresetFileSnafu)?;
                        destination
                            .write_all(json.as_bytes())
                            .context(CreatePresetFileSnafu)?;
                        Ok(Some(Box::new(|app: &mut NtscApp| {
                            app.presets_state.selected_preset = Some(new_selected_preset);
                            Ok(())
                        }) as _))
                    }),
                    Err(e) => {
                        self.handle_error(&e);
                    }
                }
            }
            Some(Action::OverwritePreset { path }) => {
                let new_selected_preset = SelectedPreset {
                    path,
                    settings: self.effect_settings.clone(),
                };
                match self
                    .settings_list
                    .to_json_string(&new_selected_preset.settings)
                    .context(CreatePresetJSONSnafu)
                {
                    Ok(json) => self.do_fs_operation_then_refresh(async move {
                        let mut destination = fs::File::create(&new_selected_preset.path)
                            .context(CreatePresetFileSnafu)?;
                        destination
                            .write_all(json.as_bytes())
                            .context(CreatePresetFileSnafu)?;
                        Ok(Some(Box::new(|app: &mut NtscApp| {
                            app.presets_state.selected_preset = Some(new_selected_preset);
                            Ok(())
                        }) as _))
                    }),
                    Err(e) => {
                        self.handle_error(&e);
                    }
                }
            }
            Some(Action::DeletePreset { path }) => {
                self.do_fs_operation_then_refresh(unblock(|| {
                    trash::delete(path).context(DeletePresetSnafu)?;
                    Ok(None)
                }));
            }

            Some(Action::OpenPresetsDir) => {
                self.handle_result(open::that_detached(presets_dir));
            }
            Some(Action::ReloadPresetsDir) => {
                self.reload_presets_dir();
            }

            None => {}
        }
    }

    fn do_fs_operation_then_refresh(
        &mut self,
        op: impl Future<Output = Result<Option<AppFn>, ApplicationError>> + Send + 'static,
    ) {
        self.presets_state.active_fs_operations += 1;

        self.spawn(async {
            let res = op.await;

            Some(Box::new(|app: &mut NtscApp| {
                match res {
                    Ok(Some(func)) => func(app)?,
                    Ok(None) => {}
                    Err(e) => app.handle_error(&e),
                }

                app.presets_state.active_fs_operations -= 1;

                if app.presets_state.active_fs_operations == 0 {
                    app.reload_presets_dir();
                }

                Ok(())
            }) as _)
        })
    }

    pub fn install_presets<I: IntoIterator<Item = PathBuf> + Send + 'static>(&mut self, presets: I)
    where
        <I as IntoIterator>::IntoIter: Send,
    {
        let presets = presets.into_iter();
        self.do_fs_operation_then_refresh(async move {
            for preset in presets {
                let Some(file_name) = preset.file_name() else {
                    continue;
                };
                let mut dst_path = Self::presets_dir().unwrap();
                dst_path.push(file_name);
                unblock(|| fs::copy(preset, dst_path))
                    .await
                    .context(InstallPresetSnafu)?;
            }

            Ok(None)
        });
    }
}
