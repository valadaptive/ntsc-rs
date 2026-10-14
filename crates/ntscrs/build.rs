use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/ntsc/metal/renderer.mm");
    println!("cargo:rerun-if-changed=src/ntsc/metal/kernels.metal");
    if env::var_os("CARGO_FEATURE_METAL").is_none()
        || env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64")
    {
        return;
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let arch = match env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
        "aarch64" => "arm64",
        "x86_64" => "x86_64",
        other => panic!("unsupported Metal architecture: {other}"),
    };
    // No external compiler dependency; use the selected Xcode toolchain.
    assert!(
        Command::new("xcrun")
            .args([
                "clang++",
                "-c",
                "-O3",
                "-std=c++17",
                "-fobjc-arc",
                "-fblocks",
                "-arch",
                arch,
                "-mmacosx-version-min=11.0",
                "src/ntsc/metal/renderer.mm",
                "-o"
            ])
            .arg(out.join("metal.o"))
            .status()
            .unwrap()
            .success()
    );
    assert!(
        Command::new("xcrun")
            .args(["ar", "crs"])
            .arg(out.join("libntscrs_metal.a"))
            .arg(out.join("metal.o"))
            .status()
            .unwrap()
            .success()
    );
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=ntscrs_metal");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=c++");
}
