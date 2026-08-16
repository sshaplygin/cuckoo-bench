# cuckoo-bench

Benchmark of a **Swiss-table-style SIMD cuckoo filter** against a classic
cuckoo filter and a Bloom filter.

## Implementations

| | `standard::CuckooFilter` | `simd::SimdCuckooFilter` | `wide::WideCuckooFilter` |
| --- | --- | --- | --- |
| Bucket layout | 4 × 8-bit fingerprints | 16 × 8-bit fingerprints (one 16-byte group, like a hashbrown control group) | 8 × 16-bit fingerprints (same 16-byte group) |
| Probe | scalar byte-by-byte compare | one 128-bit load + vector compare + movemask (NEON `vceqq_u8`/`vshrn` on aarch64, SSE2 `_mm_cmpeq_epi8` on x86_64, scalar fallback elsewhere) | same, over 16-bit lanes (`vceqq_u16` / `_mm_cmpeq_epi16`) |
| Memory | 1 byte/slot | 1 byte/slot | 2 bytes/slot |
| Max load | ~95% | ~98% | ~97% |
| FPR @ 85% load | ~2.7% | ~10.2% | ~0.023% |

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
standard 4x8bit scalar     1024 KiB    9.41 bits/key   FPR  2.67%
swiss   16x8bit simd       1024 KiB    9.41 bits/key   FPR 10.15%
wide    8x16bit simd       2048 KiB   18.82 bits/key   FPR  0.023%
bloom   k=7     scalar     1024 KiB    9.41 bits/key   FPR  1.09%
```

At equal memory the Bloom filter beats the classic cuckoo filter on FPR
(theory: reaching 0.023% would take it ~17.5 bits/key — almost exactly the
wide variant's budget), but it cannot delete items and touches up to k
scattered cache lines per query versus 1–2 for cuckoo.

## Speed (Apple Silicon, 2^20 slots, 85% load)

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

```text
lookup_hit_first/standard4/scalar    398 Melem/s
lookup_hit_first/swiss16/simd        577 Melem/s   (~1.5x)
lookup_hit_first/swiss16/scalar      257 Melem/s
lookup_hit_first/wide8x16/simd       586 Melem/s   (~1.5x)
lookup_hit_first/bloom/scalar        145 Melem/s

lookup_hit_last/standard4/scalar     270 Melem/s
lookup_hit_last/swiss16/simd         375 Melem/s   (~1.4x)
lookup_hit_last/swiss16/scalar        45 Melem/s
lookup_hit_last/wide8x16/simd        397 Melem/s   (~1.5x)
lookup_hit_last/bloom/scalar         145 Melem/s

lookup_miss/standard4/scalar         250 Melem/s
lookup_miss/swiss16/simd             362 Melem/s   (~1.4x)
lookup_miss/swiss16/scalar           175 Melem/s
lookup_miss/wide8x16/simd            399 Melem/s   (~1.6x)
lookup_miss/bloom/scalar              71 Melem/s

insert/standard4/scalar               78 Melem/s
insert/swiss16/simd                  398 Melem/s   (~5.1x)
insert/wide8x16/simd                 262 Melem/s   (~3.4x)
insert/bloom/scalar                  168 Melem/s
```

Takeaways:

- SIMD probing wins ~1.4–1.6x on lookups and ~3.4–5x on inserts (empty-slot
  search is a single vector compare, and wider buckets also cause far fewer
  cuckoo evictions at the same load).
- The 16-slot scalar control shows the win really is vectorization: the same
  layout probed byte-by-byte collapses to 45 Melem/s in its worst case
  (`hit_last`), slower than the 4-slot baseline, while SIMD loses only ~35%
  of its best case.
- `swiss16` pays for speed with false positives (~10% vs ~2.7%): a query
  matches against 32 candidates instead of 8 with the same 8-bit
  fingerprints.
- `wide8x16` removes that trade-off: same lookup speed, ~100x lower FPR than
  the baseline — at 2x the memory. Inserts are a bit slower than swiss16
  because 8-slot buckets fill up sooner and evict more.
- The Bloom filter is insensitive to the hit scenario, but its flat
  145 Melem/s is 2.7–4x behind the SIMD variants, misses drop to 71 Melem/s
  (k scattered probes with poorly predicted early exits), and it cannot
  delete.

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
