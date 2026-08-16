//! Quick demo: fill all filters to the same load factor, report memory, false
//! positive rate, and rough lookup timings. Use `cargo bench` for the real
//! measurements.

use std::time::Instant;

use cuckoo_bench::bloom::BloomFilter;
use cuckoo_bench::hash::xorshift;
use cuckoo_bench::simd::SimdCuckooFilter;
use cuckoo_bench::standard::CuckooFilter;
use cuckoo_bench::wide::WideCuckooFilter;

const TOTAL_SLOTS: usize = 1 << 20;
const LOAD_FACTOR: f64 = 0.85;

fn keys(n: usize, mut seed: u64) -> Vec<u64> {
    (0..n).map(|_| xorshift(&mut seed)).collect()
}

fn main() {
    let n = (TOTAL_SLOTS as f64 * LOAD_FACTOR) as usize;
    let present = keys(n, 0xdead_beef);
    let absent = keys(1 << 20, 0x1234_5678_9abc_def0);

    let mut standard = CuckooFilter::with_capacity(TOTAL_SLOTS);
    let mut simd = SimdCuckooFilter::with_capacity(TOTAL_SLOTS);
    let mut wide = WideCuckooFilter::with_capacity(TOTAL_SLOTS);
    // Same memory as the byte-fingerprint tables: 8 bits per slot.
    let bloom_bits = TOTAL_SLOTS * 8;
    let mut bloom = BloomFilter::new(bloom_bits, BloomFilter::optimal_k(bloom_bits, n));

    for key in &present {
        assert!(standard.insert(key), "standard filter rejected an insert");
        assert!(simd.insert(key), "simd filter rejected an insert");
        assert!(wide.insert(key), "wide filter rejected an insert");
        bloom.insert(key);
    }

    println!(
        "{n} keys inserted into each filter ({TOTAL_SLOTS} slots, {:.0}% load)\n",
        LOAD_FACTOR * 100.0
    );

    println!("memory (fingerprint table only):");
    for (name, bytes) in [
        ("standard 4x8bit scalar", standard.memory_bytes()),
        ("swiss 16x8bit simd", simd.memory_bytes()),
        ("wide 8x16bit simd", wide.memory_bytes()),
        ("bloom k=7 scalar", bloom.memory_bytes()),
    ] {
        println!(
            "  {name:24} {:6} KiB   {:.2} bits/key",
            bytes / 1024,
            bytes as f64 * 8.0 / n as f64
        );
    }
    println!();

    type Probe<'a> = (&'a str, Box<dyn Fn(&u64) -> bool + 'a>);
    let probes: [Probe; 5] = [
        ("standard 4x8bit scalar", Box::new(|k| standard.contains(k))),
        ("swiss 16x8bit simd", Box::new(|k| simd.contains(k))),
        ("swiss 16x8bit scalar", Box::new(|k| simd.contains_scalar(k))),
        ("wide 8x16bit simd", Box::new(|k| wide.contains(k))),
        ("bloom k=7 scalar", Box::new(|k| bloom.contains(k))),
    ];
    for (name, contains) in probes {
        let mut hits = 0usize;
        let start = Instant::now();
        for key in &present {
            hits += contains(key) as usize;
        }
        let hit_time = start.elapsed();
        assert_eq!(hits, present.len(), "{name}: lost an inserted key");

        let mut false_positives = 0usize;
        let start = Instant::now();
        for key in &absent {
            false_positives += contains(key) as usize;
        }
        let miss_time = start.elapsed();

        println!(
            "{name:24} lookup(hit) {:6.2} ns/op   lookup(miss) {:6.2} ns/op   FPR {:.4}%",
            hit_time.as_nanos() as f64 / present.len() as f64,
            miss_time.as_nanos() as f64 / absent.len() as f64,
            100.0 * false_positives as f64 / absent.len() as f64,
        );
    }
}
