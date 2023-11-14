#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::{env, process};

pub fn main() -> Result<(), std::io::Error> {
    let mut launcher_path = env::current_exe()?;
    launcher_path.pop();
    launcher_path.extend(["bin", "ntsc-rs-standalone.exe"]);

    if let Err(e) = process::Command::new(launcher_path).spawn() {
        rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Error)
            .set_title("Could not launch ntsc-rs")
            .set_description(format!("ntsc-rs could not be launched. This may happen if you move the ntsc-rs launcher \
                out of its folder without copying the \"bin\" and \"lib\" folders along with it.\n\n\
                Error message: {}", e))
            .show();
    }

    Ok(())
}
