//! Two cuckoo filter implementations for benchmarking:
//!
//! - [`standard::CuckooFilter`] — the classic layout: 4-slot buckets of 8-bit
//!   fingerprints, scalar probing.
//! - [`simd::SimdCuckooFilter`] — Swiss-table-style: 16-slot (16-byte) bucket
//!   groups probed with a single vector compare (NEON on aarch64, SSE2 on
//!   x86_64), the way hashbrown probes its control bytes.
//! - [`wide::WideCuckooFilter`] — same SIMD group probe, but 8 × 16-bit
//!   fingerprints per bucket: 2 bytes/slot buys a ~500x lower false-positive
//!   rate.
//! - [`bloom::BloomFilter`] — classic Bloom filter baseline (double hashing,
//!   k probes, no deletion).
//!
//! Both use partial-key cuckoo hashing (alternate bucket derived from the
//! fingerprint), the same hash function, and 1 byte per slot, so at equal
//! capacity they use equal memory.

pub mod bloom;
pub mod hash;
pub mod simd;
pub mod standard;
pub mod wide;
