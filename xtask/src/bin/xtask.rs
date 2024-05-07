use std::process;

use xtask::{build_ofx_plugin, macos_bundle};

fn main() {
    let cmd = clap::Command::new("xtask")
        .subcommand_required(true)
        .subcommand(build_ofx_plugin::command())
        .subcommand(macos_bundle::command());

    let matches = cmd.get_matches();

    let (task, args) = matches.subcommand().unwrap();

    match task {
        "macos-bundle" => {
            macos_bundle::main(&args).unwrap();
        }
        "build-ofx-plugin" => {
            build_ofx_plugin::main(&args).unwrap();
        }
        _ => {
            println!("Invalid xtask: {task}");
            process::exit(1);
        }
    }
}
