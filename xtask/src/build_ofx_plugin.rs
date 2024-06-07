use crate::util::{workspace_dir, PathBufExt, StatusExt};

use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug)]
struct Target {
    /// Cargo target triple for this target
    target_triple: &'static str,
    /// OpenFX architecture string for this target
    /// https://openfx.readthedocs.io/en/master/Reference/ofxPackaging.html#installation-directory-hierarchy
    ofx_architecture: &'static str,
    /// File extension for a dynamic library on this platform, excluding the leading dot
    library_extension: &'static str,
    /// Prefix for the library output filename. Platform-dependant; thanks Cargo!
    library_prefix: &'static str,
}

// "Supported" target triples
const TARGETS: &[Target] = &[
    Target {
        target_triple: "x86_64-unknown-linux-gnu",
        ofx_architecture: "Linux-x86-64",
        library_extension: "so",
        library_prefix: "lib",
    },
    Target {
        target_triple: "i686-unknown-linux-gnu",
        ofx_architecture: "Linux-x86",
        library_extension: "so",
        library_prefix: "lib",
    },
    Target {
        target_triple: "x86_64-pc-windows-msvc",
        ofx_architecture: "Win64",
        library_extension: "dll",
        library_prefix: "",
    },
    Target {
        target_triple: "i686-pc-windows-msvc",
        ofx_architecture: "Win32",
        library_extension: "dll",
        library_prefix: "",
    },
    // These two are completely untested. If your macOS build fails, don't be afraid to change these.
    Target {
        target_triple: "x86_64-apple-darwin",
        ofx_architecture: "MacOS",
        library_extension: "dylib",
        library_prefix: "lib",
    },
    Target {
        target_triple: "aarch64-apple-darwin",
        ofx_architecture: "MacOS",
        library_extension: "dylib",
        library_prefix: "lib",
    },
];

pub fn command() -> clap::Command {
    clap::Command::new("build-ofx-plugin")
        .arg(
            clap::Arg::new("release")
                .long("release")
                .help("Build the plugin in release mode")
                .conflicts_with("debug")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            clap::Arg::new("debug")
                .long("debug")
                .help("Build the plugin in debug mode")
                .conflicts_with("release")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            clap::Arg::new("target")
                .long("target")
                .help("Set the target triple to compile for")
                .default_value(current_platform::CURRENT_PLATFORM),
        )
        .arg(
            clap::Arg::new("macos-universal")
                .long("macos-universal")
                .help("Build a macOS universal library (x86_64 and aarch64)")
                .action(clap::ArgAction::SetTrue)
                .conflicts_with("target"),
        )
}

fn get_info_plist() -> String {
    let cargo_toml_path = workspace_dir().plus_iter(["crates", "openfx-plugin", "Cargo.toml"]);
    let manifest = cargo_toml::Manifest::from_path(cargo_toml_path).unwrap();
    let version = manifest.package().version();
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>CFBundleInfoDictionaryVersion</key>
            <string>6.0</string>
            <key>CFBundleDevelopmentRegion</key>
            <string>en</string>
            <key>CFBundlePackageType</key>
            <string>BNDL</string>
            <key>CFBundleIdentifier</key>
            <string>rs.ntsc.openfx</string>
            <key>CFBundleExecutable</key>
            <string>ntsc-rs-standalone</string>
            <key>CFBundleVersion</key>
            <string>{version}</string>
            <key>CFBundleShortVersionString</key>
            <string>{version}</string>
            <key>NSHumanReadableCopyright</key>
            <string>© 2023-2024 valadaptive</string>
            <key>CFBundleSignature</key>
            <string>????</string>
        </dict>
        </plist>
        "#
    )
}

fn build_plugin_for_target(target: &Target, release_mode: bool) -> std::io::Result<PathBuf> {
    println!("Building OpenFX plugin for target {}", target.target_triple);

    let mut cargo_args: Vec<_> = vec![
        String::from("build"),
        String::from("--package=openfx-plugin"),
        String::from("--lib"),
        String::from("--target"),
        target.target_triple.to_string(),
    ];
    if release_mode {
        cargo_args.push(String::from("--release"));
    }
    Command::new("cargo")
        .args(&cargo_args)
        .status()
        .expect_success()?;

    let mut target_dir_path = workspace_dir().to_path_buf();
    target_dir_path.extend(&[
        "target",
        target.target_triple,
        if cargo_args.contains(&String::from("--release")) {
            "release"
        } else {
            "debug"
        },
    ]);

    let mut built_library_path = target_dir_path.clone();
    built_library_path.push(target.library_prefix.to_owned() + "openfx_plugin");
    built_library_path.set_extension(target.library_extension);

    Ok(built_library_path)
}

pub fn main(args: &clap::ArgMatches) -> std::io::Result<()> {
    let release_mode = args.get_flag("release");
    let (built_library_path, ofx_architecture) = if args.get_flag("macos-universal") {
        let x86_64_target = TARGETS
            .iter()
            .find(|target| target.target_triple == "x86_64-apple-darwin")
            .unwrap();
        let aarch64_target = TARGETS
            .iter()
            .find(|target| target.target_triple == "aarch64-apple-darwin")
            .unwrap();
        let x86_64_path = build_plugin_for_target(x86_64_target, release_mode)?;
        let aarch64_path = build_plugin_for_target(aarch64_target, release_mode)?;

        let mut dst_path = std::env::temp_dir();
        dst_path.push(format!(
            "ntsc-rs-ofx-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));

        Command::new("lipo")
            .args(&[
                OsString::from("-create"),
                OsString::from("-output"),
                dst_path.clone().into(),
                x86_64_path.into(),
                aarch64_path.into(),
            ])
            .status()
            .expect_success()?;

        // both targets have ofx_architecture: "MacOS"
        assert_eq!(
            x86_64_target.ofx_architecture,
            aarch64_target.ofx_architecture
        );
        (dst_path, x86_64_target.ofx_architecture)
    } else {
        let target_triple = args.get_one::<String>("target").unwrap();
        let target = TARGETS
            .iter()
            .find(|candidate_target| candidate_target.target_triple == target_triple)
            .unwrap_or_else(|| panic!("Your target \"{}\" is not supported", target_triple));

        (
            build_plugin_for_target(target, release_mode)?,
            target.ofx_architecture,
        )
    };

    let mut output_dir = workspace_dir().to_path_buf();
    output_dir.extend(&["crates", "openfx-plugin", "build"]);

    let plugin_bundle_path = output_dir.plus_iter(["NtscRs.ofx.bundle", "Contents"]);
    let plugin_bin_path = plugin_bundle_path.plus_iter([ofx_architecture, "NtscRs.ofx"]);

    fs::create_dir_all(plugin_bin_path.parent().unwrap())?;
    fs::copy(built_library_path, plugin_bin_path)?;
    if ofx_architecture == "MacOS" {
        fs::write(plugin_bundle_path.plus("Info.plist"), get_info_plist())?;
    }

    Ok(())
}
