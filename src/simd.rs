//! Swiss-table-style cuckoo filter: 16 slots per bucket, SIMD probing.
//!
//! Each bucket is a 16-byte group of 8-bit fingerprints — exactly the shape of
//! a Swiss table / hashbrown control group. A probe is one 128-bit load, one
//! byte-wise compare against the splatted fingerprint, and one movemask:
//!
//! - aarch64: `vceqq_u8` + the `vshrn` narrowing trick hashbrown uses to
//!   build a nibble-per-lane bitmask (NEON has no movemask instruction).
//! - x86_64: `_mm_cmpeq_epi8` + `_mm_movemask_epi8` (SSE2, always available).
//! - other targets: scalar fallback with identical semantics.
//!
//! Fingerprint 0 is the empty sentinel, so "find a free slot" is the same
//! SIMD compare against 0.
//!
//! Trade-off vs the 4-slot baseline at equal memory: 4x fewer, wider buckets
//! mean higher achievable load (~98%) but a higher false-positive rate
//! (~2*16/256 = 12.5% vs ~3.1%), since a query now matches against 32
//! candidate fingerprints instead of 8. Same probing cost per lookup: two
//! bucket checks either way — but here each check is one vector op.

use std::hash::Hash;

use crate::hash::{alt_index, fingerprint, hash_of, xorshift};

pub const SLOTS: usize = 16;
const MAX_KICKS: usize = 500;

#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct Bucket(pub [u8; SLOTS]);

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod group {
    use super::Bucket;
    use core::arch::aarch64::*;

    /// Bitmask with 4 bits set per matching lane (the vshrn narrowing trick:
    /// 16 lanes of 0x00/0xFF collapse into a u64 of 0x0/0xF nibbles).
    #[inline]
    fn eq_mask(bucket: &Bucket, byte: u8) -> u64 {
        unsafe {
            let group = vld1q_u8(bucket.0.as_ptr());
            let eq = vceqq_u8(group, vdupq_n_u8(byte));
            let nibbles = vshrn_n_u16::<4>(vreinterpretq_u16_u8(eq));
            vget_lane_u64::<0>(vreinterpret_u64_u8(nibbles))
        }
    }

    #[inline]
    pub fn any_eq(bucket: &Bucket, byte: u8) -> bool {
        eq_mask(bucket, byte) != 0
    }

    #[inline]
    pub fn first_eq(bucket: &Bucket, byte: u8) -> Option<usize> {
        let mask = eq_mask(bucket, byte);
        if mask == 0 {
            None
        } else {
            Some((mask.trailing_zeros() >> 2) as usize)
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod group {
    use super::Bucket;
    use core::arch::x86_64::*;

    /// Bitmask with 1 bit set per matching lane.
    #[inline]
    fn eq_mask(bucket: &Bucket, byte: u8) -> u32 {
        unsafe {
            let group = _mm_load_si128(bucket.0.as_ptr() as *const __m128i);
            let eq = _mm_cmpeq_epi8(group, _mm_set1_epi8(byte as i8));
            _mm_movemask_epi8(eq) as u32
        }
    }

    #[inline]
    pub fn any_eq(bucket: &Bucket, byte: u8) -> bool {
        eq_mask(bucket, byte) != 0
    }

    #[inline]
    pub fn first_eq(bucket: &Bucket, byte: u8) -> Option<usize> {
        let mask = eq_mask(bucket, byte);
        if mask == 0 {
            None
        } else {
            Some(mask.trailing_zeros() as usize)
        }
    }
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    target_arch = "x86_64"
)))]
mod group {
    use super::Bucket;

    #[inline]
    pub fn any_eq(bucket: &Bucket, byte: u8) -> bool {
        bucket.0.contains(&byte)
    }

    #[inline]
    pub fn first_eq(bucket: &Bucket, byte: u8) -> Option<usize> {
        bucket.0.iter().position(|&slot| slot == byte)
    }
}

/// Scalar probe over the same 16-slot layout — kept around so benchmarks can
/// separate "wider buckets" from "SIMD probing".
mod scalar_group {
    use super::Bucket;

    #[inline]
    pub fn any_eq(bucket: &Bucket, byte: u8) -> bool {
        bucket.0.contains(&byte)
    }
}

pub struct SimdCuckooFilter {
    buckets: Vec<Bucket>,
    bucket_mask: usize,
    len: usize,
    rng: u64,
}

impl SimdCuckooFilter {
    /// Creates a filter with room for at least `capacity` fingerprint slots
    /// (rounded up to a power-of-two bucket count). Fills reliably to ~98%.
    pub fn with_capacity(capacity: usize) -> Self {
        let buckets = (capacity.div_ceil(SLOTS)).next_power_of_two().max(1);
        Self {
            buckets: vec![Bucket([0u8; SLOTS]); buckets],
            bucket_mask: buckets - 1,
            len: 0,
            rng: 0x9e37_79b9_7f4a_7c15,
        }
    }

    #[inline]
    fn index_and_fp<T: Hash + ?Sized>(&self, item: &T) -> (usize, u8) {
        let hash = hash_of(item);
        ((hash as usize) & self.bucket_mask, fingerprint(hash))
    }

    #[inline]
    fn try_insert_at(&mut self, index: usize, fp: u8) -> bool {
        match group::first_eq(&self.buckets[index], 0) {
            Some(slot) => {
                self.buckets[index].0[slot] = fp;
                true
            }
            None => false,
        }
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
            std::mem::swap(&mut fp, &mut self.buckets[index].0[slot]);
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
        if group::any_eq(&self.buckets[i1], fp) {
            return true;
        }
        let i2 = alt_index(i1, fp as usize, self.bucket_mask);
        group::any_eq(&self.buckets[i2], fp)
    }

    /// Same lookup with a scalar byte-by-byte probe (benchmark control).
    #[inline]
    pub fn contains_scalar<T: Hash + ?Sized>(&self, item: &T) -> bool {
        let (i1, fp) = self.index_and_fp(item);
        if scalar_group::any_eq(&self.buckets[i1], fp) {
            return true;
        }
        let i2 = alt_index(i1, fp as usize, self.bucket_mask);
        scalar_group::any_eq(&self.buckets[i2], fp)
    }

    /// Benchmark helper: `Some(1)` if the fingerprint is in the primary
    /// bucket, `Some(2)` if only in the alternate one, `None` if absent.
    pub fn probe_depth<T: Hash + ?Sized>(&self, item: &T) -> Option<u8> {
        let (i1, fp) = self.index_and_fp(item);
        if group::any_eq(&self.buckets[i1], fp) {
            return Some(1);
        }
        let i2 = alt_index(i1, fp as usize, self.bucket_mask);
        if group::any_eq(&self.buckets[i2], fp) { Some(2) } else { None }
    }

    /// Removes one copy of the item's fingerprint. Only call for items that
    /// were actually inserted, otherwise unrelated entries may be evicted.
    pub fn remove<T: Hash + ?Sized>(&mut self, item: &T) -> bool {
        let (i1, fp) = self.index_and_fp(item);
        let i2 = alt_index(i1, fp as usize, self.bucket_mask);
        for index in [i1, i2] {
            if let Some(slot) = group::first_eq(&self.buckets[index], fp) {
                self.buckets[index].0[slot] = 0;
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
        self.buckets.len() * SLOTS
    }
}
