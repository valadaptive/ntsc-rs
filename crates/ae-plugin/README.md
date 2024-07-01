# ntsc-rs After Effects plugin

ntsc-rs can be used from After Effects and Premiere! When built, the plugin can be found under the "Stylize" category.

## Building the plugin (Windows)

After running

```powershell
cargo build -p ae-plugin --release
```

the built plugin will be under the `target` folder in the root directory of this repository, as `ae_plugin.dll`.

Install it to the After Effects/Premiere/Media Encoder shared plugin folder:

```powershell
Copy-Item -Force ..\..\target\release\ae_plugin.dll 'C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\ntsc-rs-ae.aex'
```
