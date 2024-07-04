# Drafting a new release

1. Bump the version of the `gui` crate. I don't update the version numbers of some other crates, even if maybe I should. This should be done even if there are no changes to the GUI itself--it's considered the "canonical" version number that shows up on the releases page.
2. Update the third-party credits and licenses for the GUI. Instructions for this are in that crate's README.
3. If you make changes to the OpenFX plugin, After Effects plugin, or effect settings, bump those versions as well. The plugin versions are taken from the code (src/lib.rs for OpenFX and build.rs for AE) and not the Cargo.toml.
4. Commit the changes and tag them. The tag will trigger a CI build that produces artifacts.
5. Once the CI build finishes, there will be a draft release on the Releases page. You'll need to provide a summary of the changes since last version, and then you can publish the release.

# Various notes

- You'll need to clone with `--recurse-submodules` (or use `git submodule update --init --recursive` in the repo if you've already cloned it) in order to build the OpenFX plugin, because it imports the `openfx` repo as a submodule.
- Benchmarking is done with `RAYON_NUM_THREADS=1 cargo bench --bench filter_profile`. Benchmarks with multithreading enabled are probably also a good idea.
- Development of the GUI will require gstreamer to be installed. You can do so following [the gstreamer-rs instructions](https://lib.rs/crates/gstreamer), though note that the MacOS instructions are a bit outdated and you can install it with `brew` just fine now.
