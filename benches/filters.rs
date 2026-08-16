//! Criterion benchmarks: standard scalar cuckoo filters vs Swiss-table-style
//! SIMD cuckoo filters vs a classic Bloom filter, at equal load factor.
//!
//! Series (bits/key at 85% load in parens):
//! - `standard4/scalar`  — 4 × 8-bit,  scalar (9.4)
//! - `standard4x16/scalar` — 4 × 16-bit, scalar (18.8) — equal-bits control
//!   for wide8x16
//! - `swiss16/simd`      — 16 × 8-bit, SIMD (9.4)
//! - `swiss16/scalar`    — same filter, scalar probe (isolates vectorization)
//! - `wide8x16/simd`     — 8 × 16-bit, SIMD (18.8)
//! - `bloom/scalar`      — k=7 Bloom (9.4)
//!
//! Lookup scenarios:
//! - `hit_first` — the fingerprint is found in the primary bucket (best-case
//!   hit, one bucket probed);
//! - `hit_last`  — the fingerprint is only in the alternate bucket
//!   (worst-case hit, both buckets probed);
//! - `miss`      — the key was never inserted.
//!
//! Each scenario runs at two table sizes: 2^20 slots (1-2 MiB tables,
//! L2-resident) and 2^26 slots (64-128 MiB tables, DRAM-resident) — group
//! names carry a `_2p20` / `_2p26` suffix. The DRAM runs show how much of
//! the SIMD advantage survives once every probe is a cache miss.
//!
//! Query sets are built per filter via `probe_depth`, because the same key
//! can land in the primary bucket of one filter and the alternate bucket of
//! another. The Bloom filter has no bucket structure — every hit costs the
//! same k probes — so it gets one random hit sample in both hit scenarios.

use criterion::measurement::WallTime;
use criterion::{BatchSize, BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use cuckoo_bench::bloom::BloomFilter;
use cuckoo_bench::hash::xorshift;
use cuckoo_bench::simd::SimdCuckooFilter;
use cuckoo_bench::standard::CuckooFilter;
use cuckoo_bench::standard16::CuckooFilter16;
use cuckoo_bench::wide::WideCuckooFilter;

const LOAD_FACTOR: f64 = 0.85;

fn keys(n: usize, mut seed: u64) -> Vec<u64> {
    (0..n).map(|_| xorshift(&mut seed)).collect()
}

struct Filters {
    standard: CuckooFilter,
    standard16: CuckooFilter16,
    simd: SimdCuckooFilter,
    wide: WideCuckooFilter,
    bloom: BloomFilter,
}

fn bloom_for(slots: usize, items: usize) -> BloomFilter {
    // Same memory as the byte-fingerprint tables: 8 bits per slot.
    let bits = slots * 8;
    BloomFilter::new(bits, BloomFilter::optimal_k(bits, items))
}

fn filled_filters(present: &[u64], total_slots: usize) -> Filters {
    let mut standard = CuckooFilter::with_capacity(total_slots);
    let mut standard16 = CuckooFilter16::with_capacity(total_slots);
    let mut simd = SimdCuckooFilter::with_capacity(total_slots);
    let mut wide = WideCuckooFilter::with_capacity(total_slots);
    let mut bloom = bloom_for(total_slots, present.len());
    for key in present {
        assert!(standard.insert(key));
        assert!(standard16.insert(key));
        assert!(simd.insert(key));
        assert!(wide.insert(key));
        bloom.insert(key);
    }
    Filters { standard, standard16, simd, wide, bloom }
}

/// `queries` inserted keys that this filter finds at the given probe depth
/// (1 = primary bucket, 2 = alternate bucket).
fn take_by_depth(
    present: &[u64],
    depth: u8,
    queries: usize,
    probe: impl Fn(&u64) -> Option<u8>,
) -> Vec<u64> {
    let set: Vec<u64> = present
        .iter()
        .copied()
        .filter(|key| probe(key) == Some(depth))
        .take(queries)
        .collect();
    assert_eq!(set.len(), queries, "not enough depth-{depth} keys");
    set
}

fn bench_contains(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    queries: &[u64],
    contains: impl Fn(&u64) -> bool,
) {
    group.bench_function(name, |b| {
        b.iter(|| {
            let mut found = 0usize;
            for key in queries {
                found += contains(black_box(key)) as usize;
            }
            found
        })
    });
}

fn bench_lookups_at(c: &mut Criterion, log2_slots: u32) {
    let total_slots = 1usize << log2_slots;
    let n = (total_slots as f64 * LOAD_FACTOR) as usize;
    let present = keys(n, 0xdead_beef);
    let filters = filled_filters(&present, total_slots);

    // The query batch is reused across criterion iterations, so its touched
    // cache lines get warm. For the DRAM-resident tables the batch must be
    // large enough that those lines (queries x buckets x 128B) blow past L2,
    // otherwise the benchmark quietly measures L2 again.
    let queries = if log2_slots >= 24 { 1 << 18 } else { 8192 };

    let standard_sets =
        [1, 2].map(|d| take_by_depth(&present, d, queries, |k| filters.standard.probe_depth(k)));
    let standard16_sets =
        [1, 2].map(|d| take_by_depth(&present, d, queries, |k| filters.standard16.probe_depth(k)));
    let simd_sets =
        [1, 2].map(|d| take_by_depth(&present, d, queries, |k| filters.simd.probe_depth(k)));
    let wide_sets =
        [1, 2].map(|d| take_by_depth(&present, d, queries, |k| filters.wide.probe_depth(k)));

    let mut seed = 42u64;
    let bloom_hits: Vec<u64> = (0..queries)
        .map(|_| present[(xorshift(&mut seed) as usize) % present.len()])
        .collect();
    let misses = keys(queries, 0x1234_5678_9abc_def0);

    for (scenario, idx) in [("hit_first", 0), ("hit_last", 1)] {
        let mut group = c.benchmark_group(format!("lookup_{scenario}_2p{log2_slots}"));
        group.throughput(Throughput::Elements(queries as u64));
        bench_contains(&mut group, "standard4/scalar", &standard_sets[idx], |k| {
            filters.standard.contains(k)
        });
        bench_contains(&mut group, "standard4x16/scalar", &standard16_sets[idx], |k| {
            filters.standard16.contains(k)
        });
        bench_contains(&mut group, "swiss16/simd", &simd_sets[idx], |k| filters.simd.contains(k));
        bench_contains(&mut group, "swiss16/scalar", &simd_sets[idx], |k| {
            filters.simd.contains_scalar(k)
        });
        bench_contains(&mut group, "wide8x16/simd", &wide_sets[idx], |k| filters.wide.contains(k));
        bench_contains(&mut group, "bloom/scalar", &bloom_hits, |k| filters.bloom.contains(k));
        group.finish();
    }

    let mut group = c.benchmark_group(format!("lookup_miss_2p{log2_slots}"));
    group.throughput(Throughput::Elements(queries as u64));
    bench_contains(&mut group, "standard4/scalar", &misses, |k| filters.standard.contains(k));
    bench_contains(&mut group, "standard4x16/scalar", &misses, |k| {
        filters.standard16.contains(k)
    });
    bench_contains(&mut group, "swiss16/simd", &misses, |k| filters.simd.contains(k));
    bench_contains(&mut group, "swiss16/scalar", &misses, |k| filters.simd.contains_scalar(k));
    bench_contains(&mut group, "wide8x16/simd", &misses, |k| filters.wide.contains(k));
    bench_contains(&mut group, "bloom/scalar", &misses, |k| filters.bloom.contains(k));
    group.finish();
}

fn bench_lookups_l2(c: &mut Criterion) {
    bench_lookups_at(c, 20);
}

fn bench_lookups_dram(c: &mut Criterion) {
    bench_lookups_at(c, 26);
}

fn bench_insert(c: &mut Criterion) {
    const INSERT_SLOTS: usize = 1 << 16;
    let n = (INSERT_SLOTS as f64 * LOAD_FACTOR) as usize;
    let items = keys(n, 0xdead_beef);

    let mut group = c.benchmark_group("insert");
    group.throughput(Throughput::Elements(items.len() as u64));

    group.bench_function("standard4/scalar", |b| {
        b.iter_batched(
            || CuckooFilter::with_capacity(INSERT_SLOTS),
            |mut filter| {
                for key in &items {
                    black_box(filter.insert(key));
                }
                filter
            },
            BatchSize::LargeInput,
        )
    });
    group.bench_function("standard4x16/scalar", |b| {
        b.iter_batched(
            || CuckooFilter16::with_capacity(INSERT_SLOTS),
            |mut filter| {
                for key in &items {
                    black_box(filter.insert(key));
                }
                filter
            },
            BatchSize::LargeInput,
        )
    });
    group.bench_function("swiss16/simd", |b| {
        b.iter_batched(
            || SimdCuckooFilter::with_capacity(INSERT_SLOTS),
            |mut filter| {
                for key in &items {
                    black_box(filter.insert(key));
                }
                filter
            },
            BatchSize::LargeInput,
        )
    });
    group.bench_function("wide8x16/simd", |b| {
        b.iter_batched(
            || WideCuckooFilter::with_capacity(INSERT_SLOTS),
            |mut filter| {
                for key in &items {
                    black_box(filter.insert(key));
                }
                filter
            },
            BatchSize::LargeInput,
        )
    });
    group.bench_function("bloom/scalar", |b| {
        b.iter_batched(
            || bloom_for(INSERT_SLOTS, items.len()),
            |mut filter| {
                for key in &items {
                    black_box(filter.insert(key));
                }
                filter
            },
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench_lookups_l2, bench_lookups_dram, bench_insert);
criterion_main!(benches);
