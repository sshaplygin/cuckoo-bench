//! SIMD cuckoo filter with wide fingerprints: 8 slots of 16-bit fingerprints
//! per bucket.
//!
//! The bucket is still one 16-byte group probed with a single 128-bit compare
//! (`vceqq_u16` on aarch64, `_mm_cmpeq_epi16` on x86_64), but each slot costs
//! 2 bytes instead of 1. In exchange the false-positive rate drops from
//! ~2*16/256 = 12.5% (swiss16) to ~2*8/65536 = 0.024%: a query matches
//! against 16 candidate fingerprints of 16 bits each.
//!
//! Fingerprint 0 is the empty sentinel, as in the other variants.

use std::hash::Hash;

use crate::hash::{alt_index, fingerprint_u16, hash_of, xorshift};

pub const SLOTS: usize = 8;
const MAX_KICKS: usize = 500;

#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct Bucket(pub [u16; SLOTS]);

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod group {
    use super::Bucket;
    use core::arch::aarch64::*;

    /// Bitmask with 8 bits set per matching lane (vshrn narrows each
    /// 0xFFFF/0x0000 u16 lane to a 0xFF/0x00 byte of a u64).
    #[inline]
    fn eq_mask(bucket: &Bucket, fp: u16) -> u64 {
        unsafe {
            let group = vld1q_u16(bucket.0.as_ptr());
            let eq = vceqq_u16(group, vdupq_n_u16(fp));
            let bytes = vshrn_n_u16::<4>(eq);
            vget_lane_u64::<0>(vreinterpret_u64_u8(bytes))
        }
    }

    #[inline]
    pub fn any_eq(bucket: &Bucket, fp: u16) -> bool {
        eq_mask(bucket, fp) != 0
    }

    #[inline]
    pub fn first_eq(bucket: &Bucket, fp: u16) -> Option<usize> {
        let mask = eq_mask(bucket, fp);
        if mask == 0 {
            None
        } else {
            Some((mask.trailing_zeros() >> 3) as usize)
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod group {
    use super::Bucket;
    use core::arch::x86_64::*;

    /// Bitmask with 2 bits set per matching lane.
    #[inline]
    fn eq_mask(bucket: &Bucket, fp: u16) -> u32 {
        unsafe {
            let group = _mm_load_si128(bucket.0.as_ptr() as *const __m128i);
            let eq = _mm_cmpeq_epi16(group, _mm_set1_epi16(fp as i16));
            _mm_movemask_epi8(eq) as u32
        }
    }

    #[inline]
    pub fn any_eq(bucket: &Bucket, fp: u16) -> bool {
        eq_mask(bucket, fp) != 0
    }

    #[inline]
    pub fn first_eq(bucket: &Bucket, fp: u16) -> Option<usize> {
        let mask = eq_mask(bucket, fp);
        if mask == 0 {
            None
        } else {
            Some((mask.trailing_zeros() >> 1) as usize)
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
    pub fn any_eq(bucket: &Bucket, fp: u16) -> bool {
        bucket.0.contains(&fp)
    }

    #[inline]
    pub fn first_eq(bucket: &Bucket, fp: u16) -> Option<usize> {
        bucket.0.iter().position(|&slot| slot == fp)
    }
}

pub struct WideCuckooFilter {
    buckets: Vec<Bucket>,
    bucket_mask: usize,
    len: usize,
    rng: u64,
}

impl WideCuckooFilter {
    /// Creates a filter with room for at least `capacity` fingerprint slots
    /// (rounded up to a power-of-two bucket count). Fills reliably to ~97%.
    pub fn with_capacity(capacity: usize) -> Self {
        let buckets = (capacity.div_ceil(SLOTS)).next_power_of_two().max(1);
        Self {
            buckets: vec![Bucket([0u16; SLOTS]); buckets],
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
        self.buckets.len() * std::mem::size_of::<Bucket>()
    }
}
