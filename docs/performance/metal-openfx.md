# Experimental Apple Silicon Metal rendering

The optional `metal` feature runs the OpenFX effect on a host's Metal command queue on Apple Silicon. The default build uses the existing CPU implementation. The plugin identifier remains `wtf.vala:NtscRs`, so the optimized bundle replaces the existing plugin rather than adding a second effect.

This branch includes the independent Rayon and NEON changes proposed in upstream PRs #840 and #841. It is based on upstream main at `450c6ee`, including the new 2D and adaptive 2D chroma filters. The backend uses Rust for deterministic random decisions and filter coefficients, with Objective-C++ managing Metal resources and Metal shaders processing pixels. No language rewrite is required for the rest of the application.

## Build and test

```sh
cargo test --locked -p ntsc-rs --features metal
cargo test --locked -p ntsc-rs --features metal -- --ignored
cargo run --locked --release -p ntsc-rs --features metal --example metal_validate -- 3840 2160 1
NTSC_METAL_DIRECT=1 cargo run --locked --release -p ntsc-rs --features metal --example metal_validate -- 3840 2160 60
bash scripts/build-metal-ofx.sh
```

The script creates an ARM64 bundle, ZIP, and installer under `target`. It does not install them. Its optional `--native` argument is for a build used on the same machine; omit it for distribution. Install only one bundle with this plugin identifier, and restart the host after replacing it.

The Metal feature requires Apple Silicon, macOS 11 or newer, and the Xcode command-line compiler. Intel macOS, Windows, and Linux compile the CPU implementation even when the feature is requested. Builds without the feature do not link Metal or invoke the Objective-C++ compiler. The existing default deployment target is unchanged.

## Validation and limits

Physical-machine validation uses an M4 Max with 64 GB memory, macOS 26.5.2, and Resolve Studio 21.1. The CSV files beside this document record current-main parity and timing. Across 20 configurations at 3840x2160, the largest absolute RGB error was 0.000045002. In a separate 60-frame run, the median was 12.968 ms for the optimized CPU and 11.382 ms for the Metal host-buffer path, about 1.14x faster. These timings exclude decode and encode. Tests cover field modes, padded and flipped rows, RGB/RGBX, byte/short/float pixels, gamma conversion, scaled filters, concurrent renders, and the new chroma modes. The parity harness processes 20 effect configurations at 4K without writing raw video frames.

An earlier v0.9.6-based build rendered a 59-frame 4K Resolve clip in a median 1.773 seconds versus 10.223 seconds for the stock plugin over three pairs. That 5.77x comparison includes enabling CPU threading and is not the gain from Metal alone. A fresh launch of the installed earlier build rendered the same clip in 1.672 seconds. These are historical measurements, not benchmarks of this rebased build.

Metal changes floating-point operation grouping. Output is close to the CPU reference, not bit-identical. Recursive filters with near-unit poles use a serial GPU path to limit numerical error. Heavy snow settings can spend more time generating metadata on the CPU. The scratch cache retains at most two 4K buffers, about 380 MiB; concurrent frames can temporarily allocate more.

The OpenFX host must supply supported positive-stride, equally sized full-frame Metal buffers. CPU rendering remains available when the host does not enable Metal. An unsupported GPU layout returns an error; the plugin cannot treat a Metal buffer as a CPU pixel pointer. Sustained long-form rendering, other OpenFX hosts, Intel GPU rendering, and the complete preset space have not been validated. This remains a draft integration pending broader testing and review of host negotiation, resource lifetime, and packaging.
