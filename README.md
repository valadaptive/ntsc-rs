<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="./docs/img/logo-darkmode.svg">
        <img alt="ntsc-rs logo" src="./docs/img/logo-lightmode.svg">
    </picture>
</p>

---

**ntsc-rs** is a video effect which emulates NTSC and VHS video artifacts. It can be used as an After Effects or OpenFX plugin, or as a standalone application.

![Screenshot of the ntsc-rs standalone application](./docs/img/appdemo.png)

## Download

The latest version of ntsc-rs can be downloaded from [the releases page](https://github.com/valadaptive/ntsc-rs/releases).

If you're using Linux, the GUI in particular requires GStreamer and some of its plugins to be installed:

<details>
<summary>Ubuntu / Debian</summary>

```bash
$ sudo apt-get install libgstreamer1.0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-alsa
```
It should work! But if it doesn't (access denied, can't open) . . .
--
Locate the executable

![Screenshot from 2024-01-15 10-27-52](https://github.com/valadaptive/ntsc-rs/assets/149022474/59685de5-b80d-4ae7-b25d-4cb8965e767d)

Right click. Go to `Properties > Permissions`

![Allow-Excecute](https://github.com/valadaptive/ntsc-rs/assets/149022474/81387a98-6d7f-4305-8162-b08ed2556973)

Click the button that says `Allow executing file as a program`

Try again in terminal!

```bash
./ntsc-rs-standalone
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

[README](README.md) edited by [Degamisu](https://github.com/Degamisu)
