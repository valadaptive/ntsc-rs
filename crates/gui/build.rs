use std::env;
extern crate embed_resource;

fn main() {
    if env::var_os("CARGO_CFG_WINDOWS").is_some() {
        embed_resource::compile("icon.rc", embed_resource::NONE);
    }

    if env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos") {
        println!("cargo:rustc-link-arg=-headerpad_max_install_names");
    }
}
