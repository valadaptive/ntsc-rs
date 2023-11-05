#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::{env, process};

pub fn main() -> Result<(), std::io::Error> {
    let mut launcher_path = env::current_exe()?;
    launcher_path.pop();
    launcher_path.extend(["bin", "ntsc-rs-standalone.exe"]);

    process::Command::new(launcher_path).spawn()?;

    Ok(())
}
