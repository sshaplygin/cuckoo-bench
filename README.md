# cuckoo-bench

Benchmark of a **Swiss-table-style SIMD cuckoo filter** against a classic
cuckoo filter and a Bloom filter.

## Implementations

| | `standard::CuckooFilter` | `standard16::CuckooFilter16` | `simd::SimdCuckooFilter` | `wide::WideCuckooFilter` |
| --- | --- | --- | --- | --- |
| Bucket layout | 4 × 8-bit fingerprints | 4 × 16-bit fingerprints | 16 × 8-bit fingerprints (one 16-byte group, like a hashbrown control group) | 8 × 16-bit fingerprints (same 16-byte group) |
| Probe | scalar byte-by-byte compare | scalar (8-byte bucket — half a SIMD register) | one 128-bit load + vector compare + movemask (NEON `vceqq_u8`/`vshrn` on aarch64, SSE2 `_mm_cmpeq_epi8` on x86_64, scalar fallback elsewhere) | same, over 16-bit lanes (`vceqq_u16` / `_mm_cmpeq_epi16`) |
| Memory | 1 byte/slot | 2 bytes/slot | 1 byte/slot | 2 bytes/slot |
| Max load | ~95% | ~95% | ~98% | ~97% |
| FPR @ 85% load | ~2.7% | ~0.010% | ~10.2% | ~0.023% |

`standard16` is the equal-bits-per-key control for `wide`: both spend
18.8 bits/key, but the narrow 4-slot bucket matches a query against 8
candidates instead of 16 — so on the FPR-per-bit axis the classic layout
stays ~2x ahead of the SIMD-friendly one.

For comparison, `bloom::BloomFilter` is a classic Bloom filter
(Kirsch–Mitzenmacher double hashing, k probes per operation, no deletion
support). In the benchmarks it gets the same memory as the byte-fingerprint
tables (8 bits per slot) and the optimal k=7.

All cuckoo variants use partial-key cuckoo hashing (alternate bucket =
`i ^ h(fp)`), the same FxHash+splitmix hash, eviction capped at 500 kicks, and
support `insert` / `contains` / `remove`. Fingerprint `0` is reserved as the
empty-slot sentinel, so finding a free slot is the same SIMD compare against
zero.

`SimdCuckooFilter` also exposes `contains_scalar` — the same 16-slot layout
probed byte-by-byte — so the benchmark separates the effect of wider buckets
from the effect of vectorization.

## Memory (2^20 slots, 891,289 keys, 85% load)

Measured via `memory_bytes()` — the fingerprint table itself (struct overhead
is a few bytes, there is no per-slot overhead).

```text
standard 4x8bit  scalar    1024 KiB    9.41 bits/key   FPR  2.67%
standard 4x16bit scalar    2048 KiB   18.82 bits/key   FPR  0.0099%
swiss   16x8bit  simd      1024 KiB    9.41 bits/key   FPR 10.15%
wide    8x16bit  simd      2048 KiB   18.82 bits/key   FPR  0.023%
bloom   k=7      scalar    1024 KiB    9.41 bits/key   FPR  1.09%
```

At equal memory the Bloom filter beats the classic cuckoo filter on FPR
(theory: reaching 0.023% would take it ~17.5 bits/key — almost exactly the
wide variant's budget), but it cannot delete items and touches up to k
scattered cache lines per query versus 1–2 for cuckoo.

## Benchmark environment

All numbers in this README were measured on:

- **CPU**: Apple M1 Max (aarch64) — 8 performance + 2 efficiency cores;
  128-byte cache lines; 12 MiB shared L2 (P-cluster) + 4 MiB (E-cluster).
  SIMD paths use NEON (128-bit).
- **Memory**: 32 GiB unified.
- **OS**: macOS 26.2 (build 25C56).
- **Toolchain**: rustc 1.94.0, cargo 1.94.0, edition 2024; bench profile with
  `lto = true`, `codegen-units = 1`, default `target-cpu`.
- **Method**: criterion 0.7 defaults (100 samples, warm-up); machine on AC
  power. Single-threaded, one filter instance per benchmark; lookup batches
  of 8192 queries per iteration for L2-resident tables and 2^18 for
  DRAM-resident ones.

Absolute numbers will differ on other machines (especially x86_64, which
takes the SSE2 path), but the relative picture should hold.

## Speed

Lookup scenarios:

- `hit_first` — the fingerprint is found in the primary bucket (best-case
  hit: one probe);
- `hit_last` — the fingerprint is only in the alternate bucket (worst-case
  hit: both probes);
- `miss` — the key was never inserted.

Query sets are built per filter via `probe_depth`: the same key can sit in
the primary bucket of one filter and the alternate bucket of another. The
Bloom filter has no bucket structure (every hit costs the same k probes), so
it gets a random sample of inserted keys in both hit scenarios.

Every scenario runs at two table sizes: **2^20 slots** (1–2 MiB, tables and
query working set fit in the M1 Max's 12 MiB L2 — measures probe cost) and
**2^26 slots** (64–128 MiB, DRAM-resident — measures what survives when every
probe is a cache/TLB miss). The DRAM run uses 2^18-key query batches so the
touched cache lines cannot become L2-warm across criterion iterations.

### L2-resident (2^20 slots, 85% load)

```text
                        hit_first   hit_last    miss      (Melem/s)
standard4/scalar           402         190       255
standard4x16/scalar        416         208       276
swiss16/simd               594         379       356
swiss16/scalar             264          46       175
wide8x16/simd              594         406       406
bloom/scalar               147         146        73
```

### DRAM-resident (2^26 slots, 85% load)

```text
                        hit_first   hit_last    miss      (Melem/s)
standard4/scalar           163          39        69
standard4x16/scalar        148          35        55
swiss16/simd               168          97        92
swiss16/scalar             111          22        54
wide8x16/simd              152          84        83
bloom/scalar                35          35        29
```

### Inserts (2^16 slots, 85% load)

```text
insert/standard4/scalar               81 Melem/s
insert/standard4x16/scalar            76 Melem/s
insert/swiss16/simd                  409 Melem/s   (~5.1x)
insert/wide8x16/simd                 294 Melem/s   (~3.6x)
insert/bloom/scalar                  169 Melem/s
```

Takeaways:

- **In cache, SIMD probing wins ~1.5–2x on lookups** and ~3.6–5x on inserts
  (empty-slot search is a single vector compare, and wider buckets also cause
  far fewer cuckoo evictions at the same load).
- **Out of cache, the best-case advantage evaporates**: `hit_first` is
  168 vs 163 Melem/s (~3%) — one memory access dominates and it costs the
  same for every layout. What survives is a 1.3–2.4x edge in `hit_last` and
  `miss`, and notably not because of the wide compare itself: the branchless
  SIMD probe lets the CPU issue the second bucket load speculatively, while
  the scalar early-exit loop's unpredictable branches serialize the two cache
  misses.
- The 16-slot scalar control shows the in-cache win really is vectorization:
  the same layout probed byte-by-byte collapses to 46 Melem/s in its worst
  case, slower than the 4-slot baseline.
- **At equal bits per key the classic layout wins the FPR axis**:
  `standard4x16` reaches 0.0099% vs `wide8x16`'s 0.023% with identical
  memory. Wider, SIMD-friendly buckets always pay ~2x FPR for their speed —
  which is why 4-slot buckets remain the default in production filters.
- `swiss16` pays for speed with false positives (~10% vs ~2.7%): a query
  matches against 32 candidates instead of 8 with the same 8-bit
  fingerprints.
- The Bloom filter is insensitive to the hit scenario, but k=7 scattered
  probes hurt everywhere: 2.7–4x behind SIMD in cache, and a hard fall to
  29–35 Melem/s from DRAM.

## Usage

```sh
cargo test              # correctness: no false negatives, FPR bounds, remove, memory
cargo run --release     # quick demo: memory + FPR + ns/op
cargo bench             # criterion benchmarks (report in target/criterion)
```

## Related work

Papers:

- Fan, Andersen, Kaminsky, Mitzenmacher — [Cuckoo Filter: Practically Better
  Than Bloom](https://www.cs.cmu.edu/~dga/papers/cuckoo-conext2014.pdf)
  (CoNEXT 2014) — the original paper; our `standard` variant is its 4-slot
  layout.
- Kirsch, Mitzenmacher — [Less Hashing, Same Performance: Building a Better
  Bloom Filter](https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf)
  — the double-hashing scheme used by our `bloom` module.
- Lang, Neumann, Kemper, Boncz — [Performance-Optimal Filtering: Bloom
  Overtakes Cuckoo at High Throughput](https://www.vldb.org/pvldb/vol12/p502-lang.pdf)
  (VLDB 2019) — SIMD Bloom vs cuckoo study;
  [reproduction repo](https://github.com/peterboncz/bloomfilter-repro).
- Breslow, Jayasena — [Morton Filters: Faster, Space-Efficient Cuckoo Filters
  via Biasing, Compression, and Decoupled Logical Sparsity](https://www.vldb.org/pvldb/vol11/p1041-breslow.pdf)
  (VLDB 2018) — a compressed, batched cuckoo filter designed for wide loads.
- Pandey, Conway, Durie, Bender, Farach-Colton, Johnson — [Vector Quotient
  Filters](https://dl.acm.org/doi/10.1145/3448016.3452841) (SIGMOD 2021) — an
  explicitly SIMD-first filter design.

Implementations:

- [efficient/cuckoofilter](https://github.com/efficient/cuckoofilter) — the
  authors' C++ reference implementation (includes SSE-assisted semi-sorted
  buckets).
- [cuckoofilter](https://crates.io/crates/cuckoofilter)
  ([axiomhq/rust-cuckoofilter](https://github.com/axiomhq/rust-cuckoofilter)) —
  the canonical Rust crate: 4 × 8-bit buckets, scalar probing.
- [scalable_cuckoo_filter](https://crates.io/crates/scalable_cuckoo_filter)
  and its fork
  [autoscale_cuckoo_filter](https://crates.io/crates/autoscale_cuckoo_filter) —
  auto-scaling cuckoo filters (a chain of filters, doubling capacity);
  branchless but not SIMD.
- [hashbrown](https://github.com/rust-lang/hashbrown) /
  [Swiss tables](https://abseil.io/about/design/swisstables) — the SIMD group
  probe this project borrows, including the NEON `vshrn` movemask trick.
- [Cuckoo hashing improves SIMD hash tables](https://reiner.org/cuckoo-hashing)
  — a write-up of the cuckoo + SIMD-group idea for hash tables.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
