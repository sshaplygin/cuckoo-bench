//! Classic Bloom filter baseline.
//!
//! Kirsch–Mitzenmacher double hashing: k probe positions are derived from one
//! 64-bit hash as `h1 + i*h2`, which costs one real hash per operation like
//! the cuckoo filters. Power-of-two bit count keeps indexing to a mask.
//!
//! Contrast with the cuckoo variants: no deletion support, and a lookup
//! touches up to k scattered cache lines (a hit always touches all k, a miss
//! exits on the first zero bit ~half the time per probe), while a cuckoo
//! lookup touches at most 2.

use std::hash::Hash;

use crate::hash::hash_of;

pub struct BloomFilter {
    words: Vec<u64>,
    bit_mask: usize,
    k: u32,
    len: usize,
}

impl BloomFilter {
    /// Creates a filter with `bits` capacity (rounded up to a power of two)
    /// and `k` hash probes per item.
    pub fn new(bits: usize, k: u32) -> Self {
        let bits = bits.next_power_of_two().max(64);
        Self {
            words: vec![0u64; bits / 64],
            bit_mask: bits - 1,
            k,
            len: 0,
        }
    }

    /// k minimizing FPR for the given table size and expected item count:
    /// k = ln2 * bits/item.
    pub fn optimal_k(bits: usize, expected_items: usize) -> u32 {
        (((bits as f64 / expected_items as f64) * std::f64::consts::LN_2).round() as u32).max(1)
    }

    #[inline]
    fn hashes<T: Hash + ?Sized>(item: &T) -> (u64, u64) {
        let h = hash_of(item);
        // h2 is odd, so successive probes cycle through the pow2 bit space.
        (h, (h >> 32) | 1)
    }

    pub fn insert<T: Hash + ?Sized>(&mut self, item: &T) {
        let (mut bit, step) = Self::hashes(item);
        for _ in 0..self.k {
            let idx = (bit as usize) & self.bit_mask;
            self.words[idx >> 6] |= 1u64 << (idx & 63);
            bit = bit.wrapping_add(step);
        }
        self.len += 1;
    }

    #[inline]
    pub fn contains<T: Hash + ?Sized>(&self, item: &T) -> bool {
        let (mut bit, step) = Self::hashes(item);
        for _ in 0..self.k {
            let idx = (bit as usize) & self.bit_mask;
            if self.words[idx >> 6] & (1u64 << (idx & 63)) == 0 {
                return false;
            }
            bit = bit.wrapping_add(step);
        }
        true
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn k(&self) -> u32 {
        self.k
    }

    #[inline]
    pub fn bits(&self) -> usize {
        self.bit_mask + 1
    }

    /// Table memory in bytes (bit array only).
    pub fn memory_bytes(&self) -> usize {
        self.words.len() * 8
    }
}
