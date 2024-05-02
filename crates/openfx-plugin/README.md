# ntsc-rs OpenFX plugin

ntsc-rs can be used as an OpenFX plugin! The plugin can be found under the "Filter" category.

A lot of different video software supports OpenFX plugins, including:
- Natron (tested; working)
- DaVinci Resolve / Fusion (tested; working)
- HitFilm (untested)
- Vegas (untested)
- Nuke (untested)

If your video editing software supports OpenFX but has trouble with ntsc-rs, feel free to open an issue; just remember
to include as much information as possible.

## Building

Building the OpenFX plugin requires Cargo and [rust-bindgen](
https://rust-lang.github.io/rust-bindgen/requirements.html). A custom build script is used to package the plugin into
an OpenFX bundle. To build, run:

```sh
cargo xtask build-plugin --release
```

That'll build a release build--for a debug build, just leave off the `--release` part.

Building for macOS likely requires signing and notarization. I don't have a Mac or an Apple Developer account, so I
can't provide any assistance there.

## Installing

OpenFX plugins are typically installed to a common folder. Your editing software might look for them somewhere
else--consult its documentation to be sure.

To install the plugin, copy the `NtscRs.ofx.bundle` folder itself to the [common directory for your
platform](https://openfx.readthedocs.io/en/main/Reference/ofxPackaging.html#installation-location).

## Usage Notes

### Effect Order and Transforms

NTSC video itself is quite low-resolution--only 480 lines of vertical resolution. As such, you should apply it to 480p
footage for best results (both for performance reasons and correctness reasons).

When doing so, you should be aware of your timeline resolution, and whether effects like ntsc-rs are applied before or
after its recipient video clip gets resized to fit the timeline.

If, for example, you place a 480p clip in a 1080p timeline, and add the ntsc-rs effect to it, things could go one of two
ways, depending on what editing software you use:

- Your editing software applies the ntsc-rs effect to the 480p clip, and then scales it up to 1080p to fit the timeline.
  All is well.
- Your editing software *first* scales the 480p clip up to 1008p, *then* applies ntsc-rs. This will produce sub-par,
  possibly unintended results, and ntsc-rs will run much slower because it has to process a 1080p clip.

In particular, effects applied to a clip in DaVinci Resolve behave the second way. Don't apply the ntsc-rs effect to a
low-resolution clip in a high-resolution timeline! Instead, either create a new timeline that matches your clip's
resolution and apply the effect there, or apply the effect in the Fusion panel, where it will be applied prior to
scaling the clip.

### sRGB and Gamma

OpenFX doesn't specify the color space that effects should operate in. Some editing software (e.g. Natron) performs all
effect processing in linear space, whereas other software (e.g. Resolve) seems to do it in sRGB space.

ntsc-rs expects its input to be in sRGB space. If it isn't, the output will appear incorrect--dark areas of the image
will appear to "glow" and become oversaturated.

Long story short:
- If you use ntsc-rs and notice that dark, desaturated areas of the image become brighter and more saturated, while the
  rest of the image appears more washed-out, check the "Apply sRGB Gamma" box at the bottom of the effect controls.
- If you use ntsc-rs and notice that dark areas of the image become even darker and blown-out, *un*check the "Apply sRGB
  Gamma" box at the bottom of the effect controls.