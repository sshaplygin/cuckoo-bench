//! Shared hashing utilities for both filter implementations.

use std::hash::{Hash, Hasher};

const SEED: u64 = 0x517c_c1b7_2722_0a95;

/// FxHash-style hasher: fast multiplicative hashing, identical cost for both
/// filters so benchmark differences come from probing, not hashing.
#[derive(Default)]
struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(buf));
        }
    }

    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.add(n as u64);
    }

    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.add(n as u64);
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }

    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }
}

/// Fx mixing alone leaves the high bits weak, and both the fingerprint and the
/// bucket index are carved out of the same 64-bit value.
#[inline]
fn splitmix_finalize(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[inline]
pub fn hash_of<T: Hash + ?Sized>(item: &T) -> u64 {
    let mut hasher = FxHasher::default();
    item.hash(&mut hasher);
    splitmix_finalize(hasher.finish())
}

/// 8-bit fingerprint; 0 is reserved as the empty-slot sentinel.
#[inline]
pub fn fingerprint(hash: u64) -> u8 {
    let fp = (hash >> 32) as u8;
    if fp == 0 { 1 } else { fp }
}

/// 16-bit fingerprint; 0 is reserved as the empty-slot sentinel.
#[inline]
pub fn fingerprint_u16(hash: u64) -> u16 {
    let fp = (hash >> 32) as u16;
    if fp == 0 { 1 } else { fp }
}

/// Partial-key cuckoo hashing: the alternate bucket is derived from the
/// current bucket and the fingerprint only, so eviction never needs the key.
#[inline]
pub fn alt_index(index: usize, fp: usize, bucket_mask: usize) -> usize {
    index ^ (fp.wrapping_mul(0x5bd1_e995) & bucket_mask)
}

/// Deterministic stream of pseudo-random u64s (used for eviction choices and
/// benchmark key generation).
#[inline]
pub fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
