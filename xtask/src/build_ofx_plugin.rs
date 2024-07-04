//! Builds the OpenFX plugin and bundles it according to the OpenFX specification, optionally taking care of producing
//! a universal binary for macOS.
//! For more information, see https://openfx.readthedocs.io/en/main/Reference/ofxPackaging.html.

use crate::util::targets::{Target, MACOS_AARCH64, MACOS_X86_64, TARGETS};
use crate::util::{workspace_dir, PathBufExt, StatusExt};

use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

pub fn command() -> clap::Command {
    clap::Command::new("build-ofx-plugin")
        .about(
            "Builds and bundles the OpenFX plugin, which is then output to `crates/openfx/build`.",
        )
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

/// Creates the contents of the Info.plist file for the bundle when building for macOS.
/// I may want to consider using PlistBuddy, but string formatting is likely a lot faster.
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

/// Build the plugin for a given target, in either debug or release mode. This is called once in most cases, but when
/// creating a macOS universal binary, it's called twice--once per architecture.
/// This returns the path to the built library.
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

    let target_dir_path = workspace_dir().to_path_buf().plus_iter(&[
        "target",
        target.target_triple,
        if cargo_args.contains(&String::from("--release")) {
            "release"
        } else {
            "debug"
        },
    ]);

    let mut built_library_path =
        target_dir_path.plus(target.library_prefix.to_owned() + "openfx_plugin");
    built_library_path.set_extension(target.library_extension);

    Ok(built_library_path)
}

pub fn main(args: &clap::ArgMatches) -> std::io::Result<()> {
    let release_mode = args.get_flag("release");

    // TODO: remove previous built bundle?

    let (built_library_path, ofx_architecture) = if args.get_flag("macos-universal") {
        let x86_64_target = MACOS_X86_64;
        let aarch64_target = MACOS_AARCH64;
        let x86_64_path = build_plugin_for_target(x86_64_target, release_mode)?;
        let aarch64_path = build_plugin_for_target(aarch64_target, release_mode)?;

        let dst_path = std::env::temp_dir().plus(format!(
            "ntsc-rs-ofx-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));

        // Combine the x86_64 and aarch64 builds into one using `lipo`, and output to the temp file we created
        // above.
        // TODO: Create the directories beforehand, output into that with lipo, and just rename it afterwards?
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

        // Both targets should have ofx_architecture: "MacOS" since it's a universal binary. Some platforms have
        // different bundle directories depending on the architecture, but as of Apple Silicon, that's not done for
        // macOS:
        // https://openfx.readthedocs.io/en/main/Reference/ofxPackaging.html#macos-architectures-and-universal-binaries
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
