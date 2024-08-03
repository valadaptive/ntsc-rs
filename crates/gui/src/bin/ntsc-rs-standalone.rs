#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::error::Error;

use gui::app::main::run;

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(windows)]
    std::panic::set_hook(Box::new(|info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        rfd::MessageDialog::new()
            .set_buttons(rfd::MessageButtons::Ok)
            .set_level(rfd::MessageLevel::Error)
            .set_description(format!("{info}\n\nBacktrace:\n{backtrace}"))
            .set_title("Error")
            .show();
    }));

    run()
}
