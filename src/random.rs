use std::hash::Hasher;

use siphasher::sip::SipHasher;

pub fn key_seed(seed: u64, key: u64, line_num: usize, frame_num: usize) -> u64 {
    let mut hasher = SipHasher::new_with_keys(seed, key);
    hasher.write_u64(line_num as u64);
    hasher.write_u64(frame_num as u64);
    hasher.finish()
}
