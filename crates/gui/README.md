# ntsc-rs standalone GUI

ntsc-rs can be used as a standalone GUI application. It uses gstreamer for media encoding, decoding, and playback.

## Building

See [the documentation on the ntsc-rs website](https://ntsc.rs/docs/building-from-source/) for up-to-date information.

## Updating third-party license credits

After installing or updating Cargo dependencies, you'll need to regenerate the list of third-party licenses using [cargo-about](https://github.com/EmbarkStudios/cargo-about):

```bash
$ cargo about generate --format=json -o about.json
```

when inside the `gui` crate folder.

If you get a "failed to satisfy license requirements" error, you'll need to add the failing third-party crate's license identifier to [`about.toml`](../../about.toml).