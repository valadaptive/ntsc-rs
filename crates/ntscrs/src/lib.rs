#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod ctx;
mod filter;
mod noise;
mod ntsc;
mod random;
pub mod settings;
mod shift;
mod thread_pool;
pub mod yiq_fielding;

pub use ctx::Context;
pub use settings::standard::NtscEffect;
pub use yiq_fielding::{YiqOwned, YiqView};

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "metal"))]
pub use ntsc::metal;
