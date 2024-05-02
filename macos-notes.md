- `PKG_CONFIG_ALLOW_CROSS=1 cargo build` seems to work fine when cross compiling
- Resolve seems to load my unsigned OFX plugin fine, but the Mac App Store version notes it may not support all third-party OFX plugins. Is code signing required for that?
- After Effects needs code signing, I'm pretty sure

Todo:
- Auto code signing and notarization for everything
- App bundle for standalone application
- Build universal binary for OFX plugin (done)
- Build AE plugin
- Do it on CI