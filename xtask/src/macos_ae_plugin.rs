//! Builds and bundles the After Effects plugin for macOS.
//! Adapted from https://github.com/AdrianEddy/after-effects/blob/cbcaf4b/AdobePlugin.just.

use crate::util::targets::{Target, MACOS_AARCH64, MACOS_X86_64, TARGETS};
use crate::util::{workspace_dir, PathBufExt, StatusExt};

use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

pub fn command() -> clap::Command {
    clap::Command::new("macos-ae-plugin")
        .about("Builds and bundles the After Effects plugin for macOS, handling Apple-specific things like creating a universal binary and a bundle.")
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

/// Build the After Effects plugin for a specific target, returning the paths to 1. the plugin library itself and
/// 2. the Carbon resource file (.rsrc) to include with it in the bundle.
fn build_plugin_for_target(
    target: &Target,
    release_mode: bool,
) -> std::io::Result<(PathBuf, PathBuf)> {
    println!(
        "Building After Effects plugin for target {}",
        target.target_triple
    );

    let mut cargo_args: Vec<_> = vec![
        String::from("build"),
        String::from("--package=ae-plugin"),
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
    built_library_path.push(target.library_prefix.to_owned() + "ae_plugin");
    built_library_path.set_extension(target.library_extension);

    let built_rsrc_path = target_dir_path.plus("ae-plugin.rsrc");

    Ok((built_library_path, built_rsrc_path))
}

pub fn main(args: &clap::ArgMatches) -> std::io::Result<()> {
    let release_mode = args.get_flag("release");

    let build_dir_path = workspace_dir().plus("build");
    let plugin_dir_path = build_dir_path.plus("ntsc-rs.plugin");

    // Clean up the previous build. If there is no previous build, this will fail; that's OK.
    let _ = fs::remove_dir_all(&plugin_dir_path);

    let contents_dir_path = plugin_dir_path.plus("Contents");
    fs::create_dir_all(&contents_dir_path)?;

    let macos_dir_path = contents_dir_path.plus("MacOS");
    fs::create_dir_all(&macos_dir_path)?;

    let resources_dir_path = contents_dir_path.plus("Resources");
    fs::create_dir_all(&resources_dir_path)?;

    fs::write(contents_dir_path.plus("PkgInfo"), "eFKTFXTC")?;

    let info_plist_path = contents_dir_path.plus("Info.plist");

    let plist_buddy = |cmd: &str| {
        Command::new("/usr/libexec/PlistBuddy")
            .args([
                OsString::from("-c"),
                OsString::from(cmd),
                OsString::from(&info_plist_path),
            ])
            .status()
            .expect_success()
    };

    plist_buddy("add CFBundlePackageType string eFKT")?;
    plist_buddy("add CFBundleSignature string FXTC")?;
    plist_buddy("add CFBundleIdentifier string rs.ntsc.openfx")?;

    let (built_library_path, built_rsrc_path) = if args.get_flag("macos-universal") {
        let x86_64_target = MACOS_X86_64;
        let aarch64_target = MACOS_AARCH64;

        let (x86_64_lib_path, x86_64_rsrc_path) =
            build_plugin_for_target(x86_64_target, release_mode)?;
        let (aarch64_lib_path, _) = build_plugin_for_target(aarch64_target, release_mode)?;

        let dst_path = std::env::temp_dir().plus(format!(
            "ntsc-rs-ae-{}",
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
                x86_64_lib_path.into(),
                aarch64_lib_path.into(),
            ])
            .status()
            .expect_success()?;

        // I hope the .rsrc files are the same between builds--I haven't checked and don't want to compare the contents
        // in case they do differ but it's OK--but the Justfile mentioned in the docs at the top use the x86_64 .rsrc.
        (dst_path, x86_64_rsrc_path)
    } else {
        let target_triple = args.get_one::<String>("target").unwrap();
        let target = TARGETS
            .iter()
            .find(|candidate_target| candidate_target.target_triple == target_triple)
            .unwrap_or_else(|| panic!("Your target \"{}\" is not supported", target_triple));

        build_plugin_for_target(target, release_mode)?
    };

    fs::copy(built_library_path, macos_dir_path.plus("ntsc-rs"))?;
    fs::copy(built_rsrc_path, resources_dir_path.plus("ntsc-rs.rsrc"))?;

    Ok(())
}
