<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="./docs/img/logo-darkmode.svg">
        <img alt="ntsc-rs logo" src="./docs/img/logo-lightmode.svg">
    </picture>
</p>

---

**ntsc-rs** is a tool for replicating NTSC and VHS video artifacts, available as a standalone tool or as a plugin for After Effects or OpenFX hosts.

![Screenshot of the ntsc-rs standalone application](./docs/img/appdemo.png)

## Download

The latest version of ntsc-rs can be downloaded from [the releases page](https://github.com/valadaptive/ntsc-rs/releases).

If you're using Linux, the GUI in particular requires GStreamer and some of its plugins to be installed:

<details>
<summary>Ubuntu / Debian</summary>

```bash
$ sudo apt-get install libgstreamer1.0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-alsa
```
</details>

<details>
<summary>Fedora</summary>

In order to decode and encode H.264 video, you'll need packages from the [RPM Fusion "free" repository](https://rpmfusion.org/Configuration).

After enabling the RPM Fusion "free" repository:

```bash
$ sudo dnf install gstreamer1 gstreamer1-plugins-base gstreamer1-plugins-good gstreamer1-plugins-bad-free gstreamer1-plugins-bad-freeworld gstreamer1-plugins-ugly gstreamer1-plugin-libav libavcodec-freeworld
```
</details>

## More information

ntsc-rs is a rough Rust port of [ntscqt](https://github.com/JargeZ/ntscqt), a PyQt-based GUI for [ntsc](https://github.com/zhuker/ntsc), itself a Python port of [composite-video-simulator](https://github.com/joncampbell123/composite-video-simulator). Reimplementing the image processing in multithreaded Rust allows it to run at (mostly) real-time speeds.

It's not an exact port--some processing passes have visibly different results, and some new ones have been added.
