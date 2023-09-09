use std::ffi::OsStr;
use std::process::Command;
use std::env::args;
use std::path::{Path, PathBuf};
use std::fs;

const CURRENT_TARGET: &str = env!("TARGET");
const OUT_DIR: &str = env!("OUT_DIR");
const CARGO_MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

struct Target {
    /// Cargo target triple for this target
    target_triple: &'static str,
    /// OpenFX architecture string for this target
    /// https://openfx.readthedocs.io/en/master/Reference/ofxPackaging.html#installation-directory-hierarchy
    ofx_architecture: &'static str,
    /// File extension for a dynamic library on this platform, excluding the leading dot
    library_extension: &'static str,
}

// "Supported" target triples
const TARGETS: &'static [Target] = &[
    Target {
        target_triple: "x86_64-unknown-linux-gnu",
        ofx_architecture: "Linux-x86-64",
        library_extension: "so",
    },
    Target {
        target_triple: "i686-unknown-linux-gnu",
        ofx_architecture: "Linux-x86",
        library_extension: "so",
    },
    Target {
        target_triple: "x86_64-pc-windows-msvc",
        ofx_architecture: "Win64",
        library_extension: "dll",
    },
    Target {
        target_triple: "i686-pc-windows-msvc",
        ofx_architecture: "Win32",
        library_extension: "dll",
    },
    // These two are completely untested. If your macOS build fails, don't be afraid to change these.
    Target {
        target_triple: "x86_64-apple-darwin",
        ofx_architecture: "MacOS-x86-64",
        library_extension: "dylib",
    },
    Target {
        target_triple: "aarch64-apple-darwin",
        ofx_architecture: "MacOS",
        library_extension: "dylib",
    },
];

pub fn main() -> std::io::Result<()> {
    let target = TARGETS
        .iter()
        .find(|candidate_target| candidate_target.target_triple == CURRENT_TARGET)
        .expect(&format!(
            "Your target \"{}\" is not supported",
            CURRENT_TARGET
        ));

    let mut cargo_args: Vec<_> = vec![String::from("build")];
    cargo_args.extend(args().skip(1));
    Command::new("cargo").args(&cargo_args).status()?;

    let mut target_dir_path = PathBuf::from(OUT_DIR);
    while target_dir_path.parent().is_some() && !target_dir_path.ends_with("target") {
        target_dir_path.pop();
    }
    if !target_dir_path.ends_with("target") {
        panic!("Could not find target dir");
    }

    if cargo_args.contains(&String::from("--release")) {
        target_dir_path.push("release");
    } else {
        target_dir_path.push("debug");
    }

    let output_dir = Path::new(CARGO_MANIFEST_DIR).join("build");
    let mut plugin_bin_path = output_dir.clone();
    plugin_bin_path.push("NtscRs.ofx.bundle/Contents");
    plugin_bin_path.push(target.ofx_architecture);
    plugin_bin_path.push("NtscRs.ofx");

    let mut built_library_path = target_dir_path.clone();
    built_library_path.push("libopenfx_plugin");
    built_library_path.set_extension(target.library_extension);

    fs::create_dir_all(plugin_bin_path.parent().unwrap())?;
    fs::copy(built_library_path, plugin_bin_path)?;


    Ok(())
}
