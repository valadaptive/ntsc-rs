use std::process;

use xtask::build_plugin;

fn main() {
    let cmd = clap::Command::new("xtask")
        .subcommand_required(true)
    .subcommand(build_plugin::command());

    let matches = cmd.get_matches();

    let (task, args) = matches.subcommand().unwrap();

    match task {
        "macos-bundle" => {
            todo!("Building a macOS bundle is not yet implemented");
        },
        "build-plugin"=> {
            xtask::build_plugin::main(&args).unwrap();
        },
        _ => {
            println!("Invalid xtask: {task}");
            process::exit(1);
        }
    }
}
