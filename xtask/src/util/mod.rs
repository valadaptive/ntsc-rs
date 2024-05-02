use std::{path::{Path, PathBuf}, sync::OnceLock};

static WORKSPACE_DIR: OnceLock<PathBuf> = OnceLock::new();

pub fn workspace_dir() -> &'static Path {
    return WORKSPACE_DIR.get_or_init(|| {
        let output = std::process::Command::new(env!("CARGO"))
            .arg("locate-project")
            .arg("--workspace")
            .arg("--message-format=plain")
            .output()
            .unwrap()
            .stdout;
        let cargo_path = Path::new(std::str::from_utf8(&output).unwrap().trim());
        cargo_path.parent().unwrap().to_path_buf()
    });
}
