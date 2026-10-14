#!/bin/bash
# Build an opt-in Apple Silicon Metal bundle, retaining the NTSC-rs plugin identity.
# This script only creates artifacts under target; it does not install them.
set -euo pipefail
cd "$(dirname "$0")/.."
ntsc_build_name=arm64
if [[ "${1:-}" == "--native" ]]; then
    export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"
    ntsc_build_name=native-arm64
elif [[ -n "${1:-}" ]]; then
    echo "Usage: $0 [--native]" >&2
    exit 2
fi
export CARGO_PROFILE_RELEASE_LTO=thin
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
cargo build --locked --release --target aarch64-apple-darwin -p ntsc-rs-openfx-plugin --features metal
ntsc_bundle="$PWD/target/metal-upstream-release/NtscRs.ofx.bundle"
mkdir -p "$ntsc_bundle/Contents/MacOS"
# Replace the inode so rebuilding cannot modify a binary mapped by a running host.
cp target/aarch64-apple-darwin/release/libntsc_rs_openfx_plugin.dylib "$ntsc_bundle/Contents/MacOS/NtscRs.ofx.new"
mv "$ntsc_bundle/Contents/MacOS/NtscRs.ofx.new" "$ntsc_bundle/Contents/MacOS/NtscRs.ofx"
python3 - "$ntsc_bundle" <<'PY'
import pathlib,plistlib,subprocess,sys
bundle=pathlib.Path(sys.argv[1])
with (bundle/'Contents/Info.plist').open('wb') as f:
    plistlib.dump({'CFBundleInfoDictionaryVersion':'6.0','CFBundlePackageType':'BNDL',
        'CFBundleIdentifier':'wtf.vala.ntsc-rs','CFBundleName':'NTSC-rs Metal',
        'CFBundleExecutable':'NtscRs.ofx','CFBundleVersion':'0.1.8',
        'CFBundleShortVersionString':'0.1.8','NSHumanReadableCopyright':'NTSC-rs contributors',
        'NtscRsBuild':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()},f)
PY
codesign --force --sign - "$ntsc_bundle"
codesign --verify --deep --strict "$ntsc_bundle"
ditto -c -k --sequesterRsrc --keepParent "$ntsc_bundle" "target/metal-upstream-release/ntsc-rs-metal-$ntsc_build_name.zip"
shasum -a 256 "target/metal-upstream-release/ntsc-rs-metal-$ntsc_build_name.zip"
pkgbuild --analyze --root target/metal-upstream-release target/metal-components.plist
python3 - <<'PY'
import pathlib, plistlib
path = pathlib.Path('target/metal-components.plist')
components = plistlib.loads(path.read_bytes())
for component in components:
    component['BundleIsRelocatable'] = False
path.write_bytes(plistlib.dumps(components))
PY
pkgbuild --root target/metal-upstream-release --filter '.*\.zip' \
    --component-plist target/metal-components.plist \
    --identifier wtf.vala.ntsc-rs --version 0.1.8 \
    --install-location /Library/OFX/Plugins "target/ntsc-rs-metal-$ntsc_build_name.pkg"
shasum -a 256 "target/ntsc-rs-metal-$ntsc_build_name.pkg"
