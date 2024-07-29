use std::{
    borrow::Cow,
    cell::{Cell, RefCell},
    collections::HashSet,
    ffi::{OsStr, OsString},
    fs,
    hash::RandomState,
    path::PathBuf,
    str::FromStr,
};

use blocking::unblock;
use eframe::egui::{self, TextBuffer};
use futures_lite::Future;
use ntscrs::ntsc::NtscEffectFullSettings;
use snafu::prelude::*;

use super::{
    error::{
        ApplicationError, CreatePresetSnafu, CreatePresetsDirectorySnafu, DeletePresetSnafu,
        FsSnafu, JSONParseSnafu, JSONReadSnafu, RenamePresetSnafu,
    },
    layout_helper::LayoutHelper,
    AppFn, NtscApp,
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

#[derive(Debug, Default)]
pub struct PresetsState {
    presets_dir: PresetsDirState,
    selected_preset: Option<SelectedPreset>,
    renamed_preset: RefCell<Option<(usize, String)>>,
    new_preset_name: RefCell<Option<String>>,
    /// Count the number of active FS operations occurring on the presets directory. Once this hits 0, we will refresh
    /// the presets list.
    active_fs_operations: Cell<u32>,
}

impl NtscApp {
    fn load_preset(&mut self, path: PathBuf) {
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
        let mut just_pressed_save = false;
        let mut overwrite_selected_preset = false;

        let selected_preset_modified = self
            .presets_state
            .selected_preset
            .as_ref()
            .is_some_and(|selected_preset| selected_preset.settings != self.effect_settings);

        ui.horizontal(|ui| {
            if ui.button("Open folder").clicked() {
                self.handle_result(open::that_detached(presets_dir));
            }
            if ui.button("Reload").clicked() {
                match self.presets_state.presets_dir {
                    PresetsDirState::Loading(_) => {}
                    _ => self.reload_presets_dir(),
                }
            }

            if ui
                .add_enabled(selected_preset_modified, egui::Button::new("Overwrite"))
                .clicked()
            {
                overwrite_selected_preset = true;
            }

            if ui.button("Save as").clicked() {
                let mut preset_name = String::from("preset.json");
                just_pressed_save = true;

                let preset_names: Option<HashSet<&OsStr, RandomState>> =
                    if let PresetsDirState::Loaded(presets)
                    | PresetsDirState::Loading(Some(presets)) = &self.presets_state.presets_dir
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

                *self.presets_state.new_preset_name.get_mut() = Some(preset_name);
            }
        });
        egui::ScrollArea::vertical().auto_shrink([false, false]).min_scrolled_height(0.0).show(ui, |ui| {
            if let PresetsDirState::NotLoaded = self.presets_state.presets_dir {
                self.reload_presets_dir();
            }
            let mut load_preset_path = None;
            match &self.presets_state.presets_dir {
                PresetsDirState::NotLoaded | PresetsDirState::Loading(None) => {
                    ui.add(egui::Spinner::new());
                }
                PresetsDirState::Error => {
                    ui.label("Error loading presets directory");
                }
                PresetsDirState::Loaded(presets) | PresetsDirState::Loading(Some(presets)) => {
                    ui.with_layout(egui::Layout::top_down_justified(egui::Align::LEFT), |ui| {
                        if matches!(&self.presets_state.presets_dir, PresetsDirState::Loading(_)) {
                            ui.disable();
                        }

                        let mut renamed_preset = self.presets_state.renamed_preset.borrow_mut();
                        let mut close_rename = false;
                        for (index, preset) in presets.iter().enumerate() {
                            let preset_path = preset.path.as_path();

                            let selected = self.presets_state.selected_preset.as_ref().is_some_and(
                                |selected_preset| {
                                    selected_preset.path == preset_path
                                },
                            );

                            let mut file_name = preset_path
                                .file_name()
                                .unwrap_or(preset_path.as_os_str())
                                .to_string_lossy();

                            if selected && selected_preset_modified {
                                file_name = Cow::Owned(format!("* {file_name}"));
                            }

                            let rename =
                                renamed_preset.as_mut().and_then(|(renamed_index, name)| {
                                    if *renamed_index == index {
                                        Some(name)
                                    } else {
                                        None
                                    }
                                });

                            if let Some(name) = rename {
                                let name_edit = ui.text_edit_singleline(name);

                                if name_edit.lost_focus() {
                                    close_rename = true;
                                    if !ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                                        let new_path = PathBuf::from_iter([
                                            preset_path.parent().unwrap().as_os_str(),
                                            &OsString::from_str(name).unwrap(),
                                        ]);
                                        if new_path != preset_path {
                                            let old_path = preset_path.to_owned();
                                            self.do_fs_operation_then_refresh(unblock(
                                                || {
                                                    if new_path.try_exists().context(RenamePresetSnafu)? {
                                                        return Err(ApplicationError::RenamePreset {
                                                            source: std::io::Error::new(
                                                                std::io::ErrorKind::AlreadyExists,
                                                                "A preset with that name already exists"
                                                            )
                                                        });
                                                    }
                                                    fs::rename(old_path, new_path)
                                                        .context(RenamePresetSnafu)?;

                                                    Ok(None)
                                                },
                                            ));
                                        }
                                    }
                                }
                                name_edit.request_focus();
                            } else {
                                let preset_label = ui
                                    .add(egui::SelectableLabel::new(selected, file_name.as_str()));

                                preset_label.context_menu(|ui| {
                                    if ui.button("Delete").clicked() {
                                        let delete_path = preset_path.to_owned();
                                        self.do_fs_operation_then_refresh(unblock(
                                            || {
                                                trash::delete(delete_path)
                                                    .context(DeletePresetSnafu)?;
                                                Ok(None)
                                            },
                                        ));
                                        ui.close_menu();
                                    }

                                    if ui.button("Rename").clicked() {
                                        *renamed_preset = Some((index, file_name.to_string()));
                                        ui.close_menu();
                                    }
                                });

                                if preset_label.clicked() {
                                    load_preset_path = Some(preset_path.to_path_buf());
                                }
                            }

                            if close_rename {
                                *renamed_preset = None;
                            }
                        }

                        let mut save_new_preset = false;
                        let mut clear_preset_name = false;
                        if let Some(new_preset_name) = &mut *self.presets_state.new_preset_name.borrow_mut() {
                            ui.rtl(|ui| {
                                let close_button = ui.button("🗙");
                                let save_button = ui.button("Save");
                                let text_edit = ui.add_sized(ui.available_size(), egui::TextEdit::singleline(new_preset_name));
                                if just_pressed_save {
                                    text_edit.scroll_to_me(None);
                                    text_edit.request_focus();
                                }
                                if (text_edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Escape))) || close_button.clicked() {
                                    clear_preset_name = true;
                                }
                                if (text_edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) || save_button.clicked() {
                                    save_new_preset = true;
                                }
                            });
                        }

                        if save_new_preset {
                            let preset_name = std::mem::take(&mut *self.presets_state.new_preset_name.borrow_mut()).unwrap();
                            let mut preset_path = Self::presets_dir().unwrap();
                            preset_path.push(preset_name);
                            let preset_json = self.settings_list.to_json(&self.effect_settings);
                            let new_selected_preset = SelectedPreset { path: preset_path.clone(), settings: self.effect_settings.clone() };
                            self.do_fs_operation_then_refresh(async move {
                                if preset_path.try_exists().context(CreatePresetSnafu)? {
                                    return Err(ApplicationError::CreatePreset {
                                        source: std::io::Error::new(
                                            std::io::ErrorKind::AlreadyExists,
                                            "A preset with that name already exists"
                                        )
                                    });
                                }
                                let mut destination = fs::File::create(preset_path).context(CreatePresetSnafu)?;
                                preset_json.write_to(&mut destination).context(CreatePresetSnafu)?;
                                Ok(Some(Box::new(|app: &mut NtscApp| {
                                    app.presets_state.selected_preset = Some(new_selected_preset);
                                    Ok(())
                                }) as _))
                            })
                        } else if clear_preset_name {
                            *self.presets_state.new_preset_name.borrow_mut() = None;
                        }

                        if let (true, Some(selected_preset)) = (overwrite_selected_preset, self.presets_state.selected_preset.as_ref()) {
                            let preset_json = self.settings_list.to_json(&self.effect_settings);
                            let preset_path = selected_preset.path.clone();
                            let new_selected_preset = SelectedPreset {path: preset_path.clone(), settings: self.effect_settings.clone() };
                            self.do_fs_operation_then_refresh(async move {
                                let mut destination = fs::File::create(preset_path).context(CreatePresetSnafu)?;
                                preset_json.write_to(&mut destination).context(CreatePresetSnafu)?;
                                Ok(Some(Box::new(|app: &mut NtscApp| {
                                    app.presets_state.selected_preset = Some(new_selected_preset);
                                    Ok(())
                                }) as _))
                            })
                        }
                    });
                }
            }

            if let Some(path) = load_preset_path {
                self.load_preset(path);
            }
        });
    }

    fn do_fs_operation_then_refresh(
        &self,
        op: impl Future<Output = Result<Option<AppFn>, ApplicationError>> + Send + 'static,
    ) {
        self.presets_state
            .active_fs_operations
            .set(self.presets_state.active_fs_operations.get() + 1);

        self.spawn(async {
            let res = op.await;

            Some(Box::new(|app: &mut NtscApp| {
                match res {
                    Ok(Some(func)) => func(app)?,
                    Ok(None) => {}
                    Err(e) => app.handle_error(&e),
                }

                let active_fs_operations = app.presets_state.active_fs_operations.get_mut();
                *active_fs_operations -= 1;

                if *active_fs_operations == 0 {
                    app.reload_presets_dir();
                }

                Ok(())
            }) as _)
        })
    }
}
