pub mod targets;

use std::{
    path::{Path, PathBuf},
    process::ExitStatus,
    sync::OnceLock,
};

static WORKSPACE_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Return the path to the root Cargo workspace, even if we're in a subcrate.
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

pub trait PathBufExt {
    /// Chainably append another segment to the given path, returning the result as a new path.
    fn plus<T: AsRef<Path>>(&self, additional: T) -> PathBuf;
    /// Chainably append many segments to the given path, returning the result as a new path.
    fn plus_iter<T: AsRef<Path>, I: IntoIterator<Item = T>>(&self, additional: I) -> PathBuf;
}

impl<P: AsRef<Path>> PathBufExt for P {
    fn plus<T: AsRef<Path>>(&self, additional: T) -> PathBuf {
        let mut new_path = self.as_ref().to_path_buf();
        new_path.push(additional);
        new_path
    }

    fn plus_iter<T: AsRef<Path>, I: IntoIterator<Item = T>>(&self, additional: I) -> PathBuf {
        let mut new_path = self.as_ref().to_path_buf();
        new_path.extend(additional);
        new_path
    }
}

pub trait StatusExt {
    /// Converts a non-zero exit status when running a command into an error.
    /// In lieu of https://github.com/rust-lang/rfcs/pull/3362.
    fn expect_success(self) -> std::io::Result<()>;
}

impl StatusExt for std::io::Result<ExitStatus> {
    fn expect_success(self) -> std::io::Result<()> {
        match self {
            Err(e) => Err(e),
            Ok(status) => {
                if status.success() {
                    Ok(())
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        status.to_string(),
                    ))
                }
            }
        }
    }
}
