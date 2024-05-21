#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::error::Error;

use gui::app::main::run;

fn main() -> Result<(), Box<dyn Error>> {
    run()
}
