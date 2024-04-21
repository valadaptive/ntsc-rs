# ntsc-rs standalone GUI

ntsc-rs can be used as a standalone GUI application. It uses gstreamer for media encoding, decoding, and playback.

## Building

Building the GUI requires Rust and Cargo. Since it also uses GStreamer, you'll need to download the GStreamer development packages as well:

<details>
<summary>Windows</summary>

Download the MSVC versions of GStreamer's runtime and development packages from [its website](https://gstreamer.freedesktop.org/download/). Before building, you'll also need to add GStreamer's development utils to your PATH, which can be done in PowerShell with:

```pwsh
$Env:PATH += ";C:\gstreamer\1.0\msvc_x86_64\bin"
```

</details>

<details>
<summary>Ubuntu / Debian</summary>

Install the GStreamer development packages using apt:

```bash
$ sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libatk1.0-dev libgtk-3-dev
```

</details>

<details>
<summary>Fedora</summary>

Install the GStreamer development packages using dnf:

```bash
$ sudo dnf install gstreamer1-devel gstreamer1-plugins-base-devel atk-devel gtk3-devel
```

</details>

## Updating third-party license credits

After installing or updating Cargo dependencies, you'll need to regenerate the list of third-party licenses using [cargo-about](https://github.com/EmbarkStudios/cargo-about):

```bash
$ cargo about generate --format=json > about.json
```

when inside the `gui` crate folder.

If you get a "failed to satisfy license requirements" error, you'll need to add the failing third-party crate's license identifier to [`about.toml`](../../about.toml).