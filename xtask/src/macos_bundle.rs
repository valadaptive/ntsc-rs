//! Builds and bundles the standalone GUI as a macOS .app bundle.

use crate::util::{workspace_dir, PathBufExt, StatusExt};

use std::collections::HashSet;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::builder::PathBufValueParser;
use walkdir::WalkDir;

pub fn command() -> clap::Command {
    clap::Command::new("macos-bundle")
        .about("Builds the standalone app and creates a macOS application bundle.")
        .arg(
            clap::Arg::new("release")
                .long("release")
                .help("Build the software in release mode")
                .conflicts_with("debug")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            clap::Arg::new("debug")
                .long("debug")
                .help("Build the software in debug mode")
                .conflicts_with("release")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            clap::Arg::new("destdir")
                .long("destdir")
                .help("The directory that the app bundle will be output to")
                .value_parser(PathBufValueParser::new())
                .default_value(workspace_dir().plus("build").as_os_str().to_owned()),
        )
}

/// Build the plugin for a given target, in either debug or release mode. This is called once in most cases, but when
/// creating a macOS universal binary, it's called twice--once per architecture.
/// This returns the path to the built binary.
fn build_for_target(target: &str, release_mode: bool) -> std::io::Result<PathBuf> {
    println!("Building application for target {}", target);

    let mut cargo_args: Vec<_> = vec![
        String::from("build"),
        String::from("--package=gui"),
        String::from("--target"),
        target.to_string(),
    ];
    if release_mode {
        cargo_args.push(String::from("--release"));
    }
    Command::new("cargo")
        .args(&cargo_args)
        // When cross-compiling, pkg-config will complain and fail by default. Cross-compilation works just fine, so we
        // disable the check.
        .env("PKG_CONFIG_ALLOW_CROSS", "1")
        .status()
        .expect_success()?;

    let mut target_dir_path = workspace_dir().to_path_buf();
    target_dir_path.extend(&[
        "target",
        target,
        if cargo_args.contains(&String::from("--release")) {
            "release"
        } else {
            "debug"
        },
    ]);

    let mut built_app_path = target_dir_path.clone();
    built_app_path.push("ntsc-rs-standalone");

    Ok(built_app_path)
}

/// Use the `sips` utility built into macOS to resize an image (used for the application icon).
/// See https://ss64.com/mac/sips.html.
fn resize_image(
    src_path: impl AsRef<Path>,
    dst_path: impl AsRef<Path>,
    size: u32,
) -> std::io::Result<()> {
    let size_str = OsString::from(size.to_string());
    let args = [
        OsString::from("-z"),
        size_str.clone(),
        size_str.clone(),
        OsString::from(src_path.as_ref()),
        OsString::from("--out"),
        OsString::from(dst_path.as_ref()),
    ];
    Command::new("sips").args(args).status().expect_success()?;
    Ok(())
}

pub fn main(args: &clap::ArgMatches) -> Result<(), Box<dyn Error>> {
    let release_mode = args.get_flag("release");

    // Build x86_64 and aarch64 binaries.
    // TODO: unlike the other macOS xtasks, this doesn't yet support choosing the targets.
    println!("Building binaries...");
    let x86_64_path = build_for_target("x86_64-apple-darwin", release_mode)?;
    let aarch64_path = build_for_target("aarch64-apple-darwin", release_mode)?;

    // Extract gui version from Cargo.toml.
    println!("Getting version for Info.plist and creating bundle directories...");
    let mut cargo_toml_path = workspace_dir().to_path_buf();
    cargo_toml_path.extend(["crates", "gui", "Cargo.toml"]);
    let gui_manifest = cargo_toml::Manifest::from_path(cargo_toml_path)?;
    let gui_version = gui_manifest.package().version();

    // Construct Info.plist and bundle structure.
    let info_plist_contents = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIdentifier</key>
    <string>rs.ntsc</string>
    <key>CFBundleExecutable</key>
    <string>ntsc-rs-standalone</string>
    <key>CFBundleIconFile</key>
    <string>icon.icns</string>
    <key>CFBundleDisplayName</key>
    <string>ntsc-rs</string>
    <key>CFBundleName</key>
    <string>ntsc-rs</string>
    <key>CFBundleVersion</key>
    <string>{gui_version}</string>
    <key>CFBundleShortVersionString</key>
    <string>{gui_version}</string>
    <key>NSHumanReadableCopyright</key>
    <string>© 2023-2024 valadaptive</string>
    <key>CFBundleSignature</key>
    <string>????</string>
</dict>
</plist>
"#
    );

    let build_dir_path = args.get_one::<PathBuf>("destdir").unwrap();
    let app_dir_path = build_dir_path.plus("ntsc-rs.app");
    let iconset_dir_path = build_dir_path.plus("ntsc-rs.iconset");

    // Clean up the previous build. If there is no previous build, this will fail; that's OK.
    let _ = fs::remove_dir_all(&app_dir_path);
    let _ = fs::remove_dir_all(&iconset_dir_path);

    let contents_dir_path = app_dir_path.plus("Contents");
    fs::create_dir_all(&contents_dir_path)?;

    let macos_dir_path = contents_dir_path.plus("MacOS");
    fs::create_dir_all(&macos_dir_path)?;

    let resources_dir_path = contents_dir_path.plus("Resources");
    fs::create_dir_all(&resources_dir_path)?;

    fs::write(
        contents_dir_path.plus("Info.plist"),
        info_plist_contents.as_bytes(),
    )?;

    let app_executable_path = macos_dir_path.plus("ntsc-rs-standalone");

    println!("Creating universal binary...");
    // Combine x86_64 and aarch64 binaries and place the result in the bundle.
    Command::new("lipo")
        .args(&[
            OsString::from("-create"),
            OsString::from("-output"),
            app_executable_path.clone().into(),
            x86_64_path.into(),
            aarch64_path.into(),
        ])
        .status()
        .expect_success()?;

    // Copy gstreamer libraries into the bundle.
    println!("Copying gstreamer libraries...");
    let src_lib_path = PathBuf::from("/Library/Frameworks/GStreamer.framework/Versions/1.0/lib");
    let dst_lib_path = contents_dir_path.plus_iter([
        "Frameworks",
        "GStreamer.framework",
        "Versions",
        "1.0",
        "lib",
    ]);
    let mut created_dirs = HashSet::<PathBuf>::new();
    for file in WalkDir::new(&src_lib_path).into_iter() {
        let file = match file {
            Ok(entry) => entry,
            Err(e) => return Err(Box::new(e)),
        };

        let ty = file.file_type();
        if !ty.is_file() {
            continue;
        }
        let src_path = file.path();
        let rel_lib_path = src_path.strip_prefix(&src_lib_path)?;
        let Some(ext) = rel_lib_path.extension() else {
            continue;
        };
        // We only want dylibs, not the static libs also present.
        if ext != "dylib" {
            continue;
        }
        let dst_path = dst_lib_path.plus(rel_lib_path);
        let dst_dir = dst_path.parent().unwrap().to_path_buf();
        // Avoid making one create_dir_all call per file (could be expensive?)
        let dst_dir_does_not_exist = created_dirs.insert(dst_dir.clone());
        if dst_dir_does_not_exist {
            std::fs::create_dir_all(&dst_dir)?;
        }
        std::fs::copy(src_path, &dst_path)?;
    }

    // Add gstreamer rpath to the executable, so it can load the gstreamer libraries.
    // According to https://gstreamer.freedesktop.org/documentation/deploying/mac-osx.html?gi-language=c#location-of-dependent-dynamic-libraries,
    // macOS doesn't locate libraries relative to the executable. That page's prescribed solution is a convoluted
    // `osxrelocator.py` script that I've seen several versions of floating around, but I just use `install_name_tool`
    // and it *seems* to work fine. GStreamer includes many binaries which would also need to be `install_name_tool`'d
    // and apparently the paths *to* those binaries need to be properly set via environment variables(?), but we don't
    // copy any binaries anyway (see the above recursive copy, which only copies .dylibs), so hopefully it's okay.
    println!("Adding gstreamer rpath...");
    Command::new("install_name_tool")
        .args([
            OsString::from("-add_rpath"),
            OsString::from("@executable_path/../Frameworks/GStreamer.framework/Versions/1.0/lib"),
            OsString::from(&app_executable_path),
        ])
        .status()
        .expect_success()?;

    // Create the iconset. Adapted from https://stackoverflow.com/a/20703594.

    // First, we resize the icons to all the sizes that Apple specifies:
    // https://developer.apple.com/design/human-interface-guidelines/app-icons#macOS-app-icon-sizes
    // Note that we actually have 2 icons: one for larger sizes, and one for smaller sizes where the thin lines on the
    // VHS label are removed.
    println!("Resizing icons...");
    let src_icon_folder_path = workspace_dir().plus("assets");
    let icon_lg_path = src_icon_folder_path.plus("macos_icon.png");
    let icon_sm_path = src_icon_folder_path.plus("macos_icon_less_detail.png");

    fs::create_dir_all(&iconset_dir_path)?;

    resize_image(&icon_sm_path, iconset_dir_path.plus("icon_16x16.png"), 16)?;
    let icon_32_path = iconset_dir_path.plus("icon_32x32.png");
    resize_image(&icon_sm_path, &icon_32_path, 32)?;
    fs::copy(&icon_32_path, iconset_dir_path.plus("icon_16x16@2x.png"))?;

    resize_image(
        &icon_sm_path,
        iconset_dir_path.plus("icon_128x128.png"),
        128,
    )?;
    let icon_256_path = iconset_dir_path.plus("icon_256x256.png");
    resize_image(&icon_lg_path, &icon_256_path, 256)?;
    fs::copy(&icon_256_path, iconset_dir_path.plus("icon_128x128@2x.png"))?;

    let icon_512_path = iconset_dir_path.plus("icon_512x512.png");
    resize_image(&icon_lg_path, &icon_512_path, 512)?;
    fs::copy(&icon_512_path, iconset_dir_path.plus("icon_256x256@2x.png"))?;

    resize_image(
        &icon_lg_path,
        iconset_dir_path.plus("icon_512x512@2x.png"),
        1024,
    )?;

    // Combine the iconset files into a single .icns.
    println!("Creating iconset...");
    Command::new("iconutil")
        .args([
            OsString::from("-c"),
            OsString::from("icns"),
            OsString::from("-o"),
            OsString::from(resources_dir_path.plus("icon.icns")),
            OsString::from(iconset_dir_path),
        ])
        .status()
        .expect_success()?;

    // TODO: code signing and notarization

    println!("Done!");

    Ok(())
}
