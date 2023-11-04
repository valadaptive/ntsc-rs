use std::env;
extern crate embed_resource;

fn main() {
    if env::var_os("CARGO_CFG_WINDOWS").is_some() {
        embed_resource::compile("icon.rc", embed_resource::NONE);
    }
}
