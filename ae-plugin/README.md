# ntsc-rs After Effects plugin

ntsc-rs can be used from After Effects! When built, the plugin can be found under the "Stylize" category.

## Prerequisites to building

I've set up a workflow to build the plugin on Windows. If you use macOS, you're on your own--I don't have a Mac, so I
can't set up a build environment for it myself, but contributions are welcome. Here's what you'll need:

### Visual Studio w/ Windows SDK
You'll need Visual Studio and probably the Windows SDK installed in order to build the plugin. In particular, errors
regarding a missing "ws2_32", "userenv", "ntdll", or "bcrypt" library are probably an indication that you're missing the
Windows SDK.

### Meson
This project is built using [Meson](https://mesonbuild.com/Getting-meson.html). It's the worst C/C++ build system,
except for all the others.

### Python
You'll need Python installed, solely to run a shim script that moves Cargo's build artifacts into a place where Meson
won't complain about them. This is an extremely stupid issue caused by a bikeshedding war between a [Meson issue
left unresolved since 2017](https://github.com/mesonbuild/meson/issues/2320) and a [Cargo issue left unresolved since
2018](https://github.com/rust-lang/cargo/issues/6790).

### The After Effects SDK
You'll need the [After Effects SDK](https://developer.adobe.com/after-effects/). Version 15.0 is what I developed
against (I'd love to use a newer version of After Effects, but anything past CC 2018 keeps freezing), but you should be
able to use a newer version. Unzip the SDK into a subfolder named "sdk" inside this folder. You'll want to unzip the
*contents* of the .zip's root folder, which at this time is just a PDF of the SDK guide and an "Examples" folder which
bafflingly contains the actual headers. The final path should be something like `ntsc-rs/ae-plugin/sdk/Examples`.

### Cargo and Rust
Of course, ntsc-rs itself is written in Rust, so you'll need [the Rust toolchain](https://rustup.rs/). The stable
version will work just fine.

## Building the plugin
In this directory, run:

```powershell
meson setup build
cd build
```

to set up the Meson build directory.

`build` is the name of the Meson build directory, not any special keyword. You could name it whatever you want, but
`build` is in the .gitignore.

If you want to be able to build directly into your After Effects plugin folder using `meson install`, you may instead
want to do:
```powershell
meson setup --prefix="C:\Program Files\[path to After Effects]\Support Files\Plug-ins\" build
cd build
```

Keep in mind you'll need to run `meson install` in an elevated terminal ("Run as Administrator") to have permission to
write into the After Effects plugin folder.

Once inside the build directory, you might want to switch to building in release mode (with optimization):
```powershell
meson configure --buildtype=release
```
Debug builds of ntsc-rs are *very* slow.

To actually compile everything, run:
```powershell
meson compile
```

Or to compile and then copy the artifacts directly into your After Effects plugin folder (run as an admin, see above):
```powershell
meson install
```
