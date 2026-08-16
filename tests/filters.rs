use cuckoo_bench::bloom::BloomFilter;
use cuckoo_bench::hash::xorshift;
use cuckoo_bench::simd::SimdCuckooFilter;
use cuckoo_bench::standard::CuckooFilter;
use cuckoo_bench::standard16::CuckooFilter16;
use cuckoo_bench::wide::WideCuckooFilter;

fn keys(n: usize, mut seed: u64) -> Vec<u64> {
    (0..n).map(|_| xorshift(&mut seed)).collect()
}

const SLOTS: usize = 1 << 16;

#[test]
fn standard_insert_then_contains_at_high_load() {
    let items = keys((SLOTS as f64 * 0.9) as usize, 1);
    let mut filter = CuckooFilter::with_capacity(SLOTS);
    for key in &items {
        assert!(filter.insert(key), "insert failed at load {:.3}", filter.load_factor());
    }
    assert_eq!(filter.len(), items.len());
    // Cuckoo filters have no false negatives.
    for key in &items {
        assert!(filter.contains(key));
    }
}

#[test]
fn simd_insert_then_contains_at_high_load() {
    let items = keys((SLOTS as f64 * 0.95) as usize, 1);
    let mut filter = SimdCuckooFilter::with_capacity(SLOTS);
    for key in &items {
        assert!(filter.insert(key), "insert failed at load {:.3}", filter.load_factor());
    }
    assert_eq!(filter.len(), items.len());
    for key in &items {
        assert!(filter.contains(key));
        assert!(filter.contains_scalar(key));
    }
}

#[test]
fn standard16_insert_then_contains_and_tiny_fpr() {
    let items = keys((SLOTS as f64 * 0.85) as usize, 1);
    let probes = keys(1 << 21, 0xabcdef);
    let mut filter = CuckooFilter16::with_capacity(SLOTS);
    for key in &items {
        assert!(filter.insert(key), "insert failed at load {:.3}", filter.load_factor());
    }
    for key in &items {
        assert!(filter.contains(key));
    }
    let hits = probes.iter().filter(|k| filter.contains(*k)).count();
    let fpr = hits as f64 / probes.len() as f64;
    // Theory: ~2 * 4/65535 * 0.85 ~= 0.010% — half of wide8x16.
    assert!(fpr < 0.0005, "standard16 FPR too high: {fpr}");
}

#[test]
fn wide_insert_then_contains_at_high_load() {
    let items = keys((SLOTS as f64 * 0.95) as usize, 1);
    let mut filter = WideCuckooFilter::with_capacity(SLOTS);
    for key in &items {
        assert!(filter.insert(key), "insert failed at load {:.3}", filter.load_factor());
    }
    assert_eq!(filter.len(), items.len());
    for key in &items {
        assert!(filter.contains(key));
    }
}

#[test]
fn wide_fpr_is_tiny() {
    let items = keys((SLOTS as f64 * 0.85) as usize, 3);
    let probes = keys(1 << 21, 0xabcdef);
    let mut filter = WideCuckooFilter::with_capacity(SLOTS);
    for key in &items {
        assert!(filter.insert(key));
    }
    let hits = probes.iter().filter(|k| filter.contains(*k)).count();
    let fpr = hits as f64 / probes.len() as f64;
    // Theoretical ~2 * 8/65535 * 0.85 ~= 0.02%; allow generous slack.
    assert!(fpr < 0.001, "wide FPR too high: {fpr}");
}

#[test]
fn simd_and_scalar_probes_agree() {
    let items = keys((SLOTS as f64 * 0.9) as usize, 7);
    let mut filter = SimdCuckooFilter::with_capacity(SLOTS);
    for key in &items {
        assert!(filter.insert(key));
    }
    for key in keys(100_000, 999) {
        assert_eq!(filter.contains(&key), filter.contains_scalar(&key));
    }
}

#[test]
fn false_positive_rates_within_expected_bounds() {
    let items = keys((SLOTS as f64 * 0.85) as usize, 3);
    let probes = keys(1 << 20, 0xabcdef);

    let mut standard = CuckooFilter::with_capacity(SLOTS);
    let mut simd = SimdCuckooFilter::with_capacity(SLOTS);
    for key in &items {
        assert!(standard.insert(key));
        assert!(simd.insert(key));
    }

    let fpr = |hits: usize| hits as f64 / probes.len() as f64;
    let standard_fpr = fpr(probes.iter().filter(|k| standard.contains(*k)).count());
    let simd_fpr = fpr(probes.iter().filter(|k| simd.contains(*k)).count());

    // Theoretical worst case ~= 2 * slots_per_bucket / 255 at full load;
    // at 85% load: standard ~2.7%, swiss16 ~10.7%. Allow slack.
    assert!(standard_fpr < 0.04, "standard FPR too high: {standard_fpr}");
    assert!(simd_fpr < 0.13, "simd FPR too high: {simd_fpr}");
}

#[test]
fn remove_works() {
    let items = keys(1000, 5);
    let mut standard = CuckooFilter::with_capacity(SLOTS);
    let mut simd = SimdCuckooFilter::with_capacity(SLOTS);
    let mut wide = WideCuckooFilter::with_capacity(SLOTS);
    for key in &items {
        standard.insert(key);
        simd.insert(key);
        wide.insert(key);
    }
    for key in &items {
        assert!(standard.remove(key));
        assert!(simd.remove(key));
        assert!(wide.remove(key));
    }
    assert!(standard.is_empty());
    assert!(simd.is_empty());
    assert!(wide.is_empty());
}

#[test]
fn bloom_no_false_negatives_and_sane_fpr() {
    let items = keys((SLOTS as f64 * 0.85) as usize, 3);
    let probes = keys(1 << 20, 0xabcdef);
    let bits = SLOTS * 8;
    let mut bloom = BloomFilter::new(bits, BloomFilter::optimal_k(bits, items.len()));
    for key in &items {
        bloom.insert(key);
    }
    for key in &items {
        assert!(bloom.contains(key));
    }
    let hits = probes.iter().filter(|k| bloom.contains(*k)).count();
    let fpr = hits as f64 / probes.len() as f64;
    // Theory at 9.41 bits/key, k=7: ~1.1%.
    assert!(fpr < 0.02, "bloom FPR too high: {fpr}");
}

#[test]
fn memory_accounting() {
    // Equal slot capacity: byte-fingerprint tables are equal, wide is 2x.
    let standard = CuckooFilter::with_capacity(1 << 16);
    let simd = SimdCuckooFilter::with_capacity(1 << 16);
    let standard16 = CuckooFilter16::with_capacity(1 << 16);
    let wide = WideCuckooFilter::with_capacity(1 << 16);
    let bloom = BloomFilter::new((1 << 16) * 8, 7);
    assert_eq!(standard.memory_bytes(), 1 << 16);
    assert_eq!(standard16.memory_bytes(), 2 << 16);
    assert_eq!(simd.memory_bytes(), 1 << 16);
    assert_eq!(wide.memory_bytes(), 2 << 16);
    assert_eq!(bloom.memory_bytes(), 1 << 16);
}

#[test]
fn rejects_when_full() {
    let mut filter = CuckooFilter::with_capacity(64);
    let mut inserted = 0;
    for key in keys(10_000, 11) {
        if filter.insert(&key) {
            inserted += 1;
        }
    }
    assert!(inserted >= 56, "filled only {inserted}/64 slots");
    assert!(inserted <= 64);
}
