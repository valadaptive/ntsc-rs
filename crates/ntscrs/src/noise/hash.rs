//! Adapted from Clatter (https://github.com/Ralith/clatter/)
//! which is available under MIT/Apache-2.0/zlib

use fearless_simd::prelude::*;

#[inline(always)]
pub fn hash_1d<S: Simd>(mut v: S::u32s) -> S::u32s {
    // ESGTSA hash from "Hash Functions for GPU Rendering", detailed in the supplementary paper.
    //
    // The paper recommends the 1D PCG hash, which it says is slightly faster and fails very slightly fewer BigCrush
    // tests. However, it contains a variable right shift, which is slow on AVX2 and scalarized entirely in SSE4.2 and
    // below. In practice, ESGTSA is faster on CPU.
    v = (v ^ 2747636419) * 2654435769;
    v = (v ^ (v >> 16)) * 2654435769;
    v = (v ^ (v >> 16)) * 2654435769;
    v
}

#[inline(always)]
pub fn pcg_3d<S: Simd>([mut vx, mut vy, mut vz]: [S::u32s; 3]) -> [S::u32s; 3] {
    // PCG3D hash function from "Hash Functions for GPU Rendering"
    vx = vx * 1664525 + 1013904223;
    vy = vy * 1664525 + 1013904223;
    vz = vz * 1664525 + 1013904223;

    vx += vy * vz;
    vy += vz * vx;
    vz += vx * vy;

    vx = vx ^ (vx >> 16);
    vy = vy ^ (vy >> 16);
    vz = vz ^ (vz >> 16);

    vx += vy * vz;
    vy += vz * vx;
    vz += vx * vy;

    [vx, vy, vz]
}
