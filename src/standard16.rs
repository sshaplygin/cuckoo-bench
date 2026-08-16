//! Classic cuckoo filter layout with wide fingerprints: 4 slots of 16-bit
//! fingerprints per bucket, scalar probing.
//!
//! This is the equal-bits-per-key control for [`crate::wide`]: both spend
//! 2 bytes per slot, but the 4-slot bucket matches a query against 8
//! candidates instead of 16, so its false-positive rate is ~2x lower
//! (~2*4/65536 = 0.012% at full load). The bucket is only 8 bytes — half a
//! SIMD register — which is exactly why SIMD-friendly designs widen it.

use std::hash::Hash;

use crate::hash::{alt_index, fingerprint_u16, hash_of, xorshift};

pub const SLOTS: usize = 4;
const MAX_KICKS: usize = 500;

pub struct CuckooFilter16 {
    buckets: Vec<[u16; SLOTS]>,
    bucket_mask: usize,
    len: usize,
    rng: u64,
}

impl CuckooFilter16 {
    /// Creates a filter with room for at least `capacity` fingerprint slots
    /// (rounded up to a power-of-two bucket count). Fills reliably to ~95%.
    pub fn with_capacity(capacity: usize) -> Self {
        let buckets = (capacity.div_ceil(SLOTS)).next_power_of_two().max(1);
        Self {
            buckets: vec![[0u16; SLOTS]; buckets],
            bucket_mask: buckets - 1,
            len: 0,
            rng: 0x9e37_79b9_7f4a_7c15,
        }
    }

    #[inline]
    fn index_and_fp<T: Hash + ?Sized>(&self, item: &T) -> (usize, u16) {
        let hash = hash_of(item);
        ((hash as usize) & self.bucket_mask, fingerprint_u16(hash))
    }

    #[inline]
    fn try_insert_at(&mut self, index: usize, fp: u16) -> bool {
        let bucket = &mut self.buckets[index];
        for slot in bucket.iter_mut() {
            if *slot == 0 {
                *slot = fp;
                return true;
            }
        }
        false
    }

    /// Returns `false` if the filter is too full to accept the item.
    pub fn insert<T: Hash + ?Sized>(&mut self, item: &T) -> bool {
        let (i1, fp) = self.index_and_fp(item);
        let i2 = alt_index(i1, fp as usize, self.bucket_mask);

        if self.try_insert_at(i1, fp) || self.try_insert_at(i2, fp) {
            self.len += 1;
            return true;
        }

        let mut index = if xorshift(&mut self.rng) & 1 == 0 { i1 } else { i2 };
        let mut fp = fp;
        for _ in 0..MAX_KICKS {
            let slot = (xorshift(&mut self.rng) as usize) % SLOTS;
            std::mem::swap(&mut fp, &mut self.buckets[index][slot]);
            index = alt_index(index, fp as usize, self.bucket_mask);
            if self.try_insert_at(index, fp) {
                self.len += 1;
                return true;
            }
        }
        false
    }

    #[inline]
    pub fn contains<T: Hash + ?Sized>(&self, item: &T) -> bool {
        let (i1, fp) = self.index_and_fp(item);
        if self.buckets[i1].contains(&fp) {
            return true;
        }
        let i2 = alt_index(i1, fp as usize, self.bucket_mask);
        self.buckets[i2].contains(&fp)
    }

    /// Benchmark helper: `Some(1)` if the fingerprint is in the primary
    /// bucket, `Some(2)` if only in the alternate one, `None` if absent.
    pub fn probe_depth<T: Hash + ?Sized>(&self, item: &T) -> Option<u8> {
        let (i1, fp) = self.index_and_fp(item);
        if self.buckets[i1].contains(&fp) {
            return Some(1);
        }
        let i2 = alt_index(i1, fp as usize, self.bucket_mask);
        if self.buckets[i2].contains(&fp) { Some(2) } else { None }
    }

    /// Removes one copy of the item's fingerprint. Only call for items that
    /// were actually inserted, otherwise unrelated entries may be evicted.
    pub fn remove<T: Hash + ?Sized>(&mut self, item: &T) -> bool {
        let (i1, fp) = self.index_and_fp(item);
        let i2 = alt_index(i1, fp as usize, self.bucket_mask);
        for index in [i1, i2] {
            if let Some(slot) = self.buckets[index].iter().position(|&s| s == fp) {
                self.buckets[index][slot] = 0;
                self.len -= 1;
                return true;
            }
        }
        false
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
    pub fn capacity(&self) -> usize {
        self.buckets.len() * SLOTS
    }

    pub fn load_factor(&self) -> f64 {
        self.len as f64 / self.capacity() as f64
    }

    /// Table memory in bytes (fingerprint storage only).
    pub fn memory_bytes(&self) -> usize {
        self.buckets.len() * SLOTS * 2
    }
}
