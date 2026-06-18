//! EXP-020 (claim B2) -- hashrope vs Ropey incremental-edit benchmark.
//!
//! Downstream-only: `hashrope` is linked as a READ-ONLY path dependency; this bench
//! lives outside the production crate and changes nothing in it.
//!
//! Two regimes (verbatim-criterion-aligned):
//!
//!   Regime A -- raw edit latency. Per size N, time K_A churn ops in one closure;
//!     per-edit = closure / K_A. Both arms do the IDENTICAL seeded churn (a delete of a
//!     random span + insert of a same-length random ASCII span at a random position, so
//!     |buffer| is stable at N). -> criterion (ii): hashrope log-log slope <= 0.3 for N>=10k,
//!     Ropey slope reported alongside; criterion (iv): the edit-only constant factor C at 1M.
//!
//!   Regime B -- edit + whole-buffer content identity. Per size N and query:edit ratio r,
//!     time K_B cycles of [ (1/r) churn ops -> one whole-buffer fingerprint query ].
//!       hashrope query = arena.hash(root)            -- O(1) read of the maintained fingerprint
//!       Ropey   query  = scratch.hash_bytes(&bytes)  -- hashrope's OWN PolynomialHash, recomputed
//!                                                       from scratch over Ropey's materialized
//!                                                       bytes (O(N)); maintained-vs-recomputed,
//!                                                       same hash algorithm (apples-to-apples).
//!     r=1 is the headline. -> criterion (iii): at r=1, N=1M, hashrope per-cycle < Ropey,
//!     mean speedup >= 2x, advantage monotone-growing above a crossover N*_id.
//!
//! Fairness controls (locked):
//!   * hashrope uses Arena::new() (NON-LAZY), so fingerprint maintenance cost is INCLUDED in
//!     the edit timing (new_lazy() would defer the hash and void the claim).
//!   * ASCII content (0x20..=0x7e) so char_idx == byte_idx (no byte/char confound).
//!   * iter_batched + BatchSize::PerIteration; the timed closure RETURNS the arena/rope so its
//!     destructor runs OUTSIDE the timed window (a ~1.3 MB arena drop is ~700 us and would
//!     otherwise dominate a ~us operation).
//!   * the PolynomialHash power-cache is warmed to N in (untimed) setup, so timings are
//!     steady-state, not first-touch.
//!
//! Honest-negative probe (criterion v), NON-timed: run with HASHROPE_B2_MEMPROBE=1 to print the
//! persistent-arena node_count growth under linear churn (the cost of persistence -- the same
//! property that buys O(1) branch/snapshot and content-identity; the flip side of EXP-004).
//!
//! Cross-run error model (criterion vi): seed via HASHROPE_B2_SEED (default 1); the driver runs
//! this binary >=3 seeds x >=3 invocations and aggregates mean +/- std (+ paired sign for B).
//!
//! Env knobs: HASHROPE_B2_SEED, HASHROPE_B2_SIZES="1000,3000,..." (subset; also used for fast
//! sandbox validation), HASHROPE_B2_MEMPROBE=1.

use criterion::{black_box, BatchSize, BenchmarkId, Criterion, Throughput};
use hashrope::Arena;
use ropey::Rope;

// ---- tunables ----
const K_A: usize = 200; // regime A: churn ops per timed closure (per-edit = closure / K_A)
const K_B: usize = 50; // regime B: cycles per timed closure (per-cycle = closure / K_B)
const SPAN_CAP: usize = 32; // max delete/insert span length
const MEM_PROBE_OPS: usize = 2000;

// Query:edit ratios r and their edits-per-query = round(1/r). r=1 is the headline.
const R_LABELS: [&str; 3] = ["r1", "r0_1", "r0_01"];
const R_EDITS_PER_QUERY: [usize; 3] = [1, 10, 100];

// Default decimal, non-power-of-two size sweep (expansive: up to 10M).
const DEFAULT_SIZES: [usize; 9] = [
    1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000,
];

fn sizes() -> Vec<usize> {
    match std::env::var("HASHROPE_B2_SIZES") {
        Ok(s) if !s.trim().is_empty() => s
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .collect(),
        _ => DEFAULT_SIZES.to_vec(),
    }
}

fn seed() -> u64 {
    std::env::var("HASHROPE_B2_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

// Expansive-but-tractable per-size sample counts; cross-run n>=9 carries the error bar.
fn samples_for(n: usize) -> usize {
    if n <= 300_000 {
        100
    } else if n <= 1_000_000 {
        60
    } else if n <= 3_000_000 {
        40
    } else {
        25
    }
}

// ---- deterministic churn (identical to the correctness harness) ----
struct Rng {
    state: u64,
}
impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }
    fn ascii(&mut self) -> u8 {
        0x20u8 + self.below(95) as u8
    }
}
fn rand_ascii_vec(rng: &mut Rng, n: usize) -> Vec<u8> {
    (0..n).map(|_| rng.ascii()).collect()
}
fn ascii_string(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| b as char).collect()
}
fn ropey_bytes(r: &Rope) -> Vec<u8> {
    r.to_string().into_bytes()
}

// hashrope: one churn op via split/concat primitives; returns the new root.
fn churn_hashrope(arena: &mut Arena, root: hashrope::Node, rng: &mut Rng) -> hashrope::Node {
    let len = arena.len(root) as usize;
    let l = 1 + rng.below(SPAN_CAP.min(len - 1));
    let del_start = rng.below(len - l + 1);
    let ins_pos = rng.below((len - l) + 1);
    let inserted = rand_ascii_vec(rng, l);
    let (left, rest) = arena.split(root, del_start as u64);
    let (_mid, right) = arena.split(rest, l as u64);
    let root = arena.concat(left, right);
    let ins_node = arena.from_bytes(&inserted);
    let (l2, r2) = arena.split(root, ins_pos as u64);
    let lr = arena.concat(l2, ins_node);
    arena.concat(lr, r2)
}

// ropey: one churn op via remove/insert (same RNG-driven parameters as hashrope).
fn churn_ropey(rope: &mut Rope, rng: &mut Rng) {
    let len = rope.len_chars();
    let l = 1 + rng.below(SPAN_CAP.min(len - 1));
    let del_start = rng.below(len - l + 1);
    let ins_pos = rng.below((len - l) + 1);
    let inserted = rand_ascii_vec(rng, l);
    rope.remove(del_start..del_start + l);
    rope.insert(ins_pos, &ascii_string(&inserted));
}

// Build a warmed hashrope arena+root of length n; power-cache grown to n in (untimed) setup.
fn build_hashrope(n: usize, seed: u64) -> (Arena, hashrope::Node, Rng) {
    let mut rng = Rng::new(seed);
    let init = rand_ascii_vec(&mut rng, n);
    let mut arena = Arena::new();
    let root = arena.from_bytes(&init);
    let _ = arena.substr_hash(root, 0, n as u64); // force the length-n power into the cache
    let _ = arena.hash(root);
    (arena, root, rng)
}

fn build_ropey(n: usize, seed: u64) -> (Rope, Rng) {
    let mut rng = Rng::new(seed);
    let init = rand_ascii_vec(&mut rng, n);
    let rope = Rope::from_str(&ascii_string(&init));
    (rope, rng)
}

// ---------------- Regime A: raw edit latency ----------------
fn regime_a(c: &mut Criterion, seed: u64) {
    let mut g = c.benchmark_group("b2_regimeA_edit");
    g.throughput(Throughput::Elements(K_A as u64)); // per-edit readout
    for &n in &sizes() {
        g.sample_size(samples_for(n));

        g.bench_with_input(BenchmarkId::new("hashrope", n), &n, |b, &n| {
            b.iter_batched(
                move || build_hashrope(n, seed),
                |(mut arena, mut root, mut rng)| {
                    for _ in 0..K_A {
                        root = churn_hashrope(&mut arena, root, &mut rng);
                    }
                    (arena, root) // returned -> Drop runs outside the timed window
                },
                BatchSize::PerIteration,
            );
        });

        g.bench_with_input(BenchmarkId::new("ropey", n), &n, |b, &n| {
            b.iter_batched(
                move || build_ropey(n, seed),
                |(mut rope, mut rng)| {
                    for _ in 0..K_A {
                        churn_ropey(&mut rope, &mut rng);
                    }
                    rope
                },
                BatchSize::PerIteration,
            );
        });
    }
    g.finish();
}

// ---------------- Regime B: edit + whole-buffer content identity ----------------
fn regime_b(c: &mut Criterion, seed: u64) {
    for (ri, &epq) in R_EDITS_PER_QUERY.iter().enumerate() {
        let mut g = c.benchmark_group(format!("b2_regimeB_{}", R_LABELS[ri]));
        g.throughput(Throughput::Elements(K_B as u64)); // per-cycle readout
        for &n in &sizes() {
            g.sample_size(samples_for(n));

            // hashrope: O(1) maintained-fingerprint read per cycle.
            g.bench_with_input(BenchmarkId::new("hashrope", n), &n, |b, &n| {
                b.iter_batched(
                    move || build_hashrope(n, seed),
                    |(mut arena, mut root, mut rng)| {
                        for _ in 0..K_B {
                            for _ in 0..epq {
                                root = churn_hashrope(&mut arena, root, &mut rng);
                            }
                            black_box(arena.hash(root)); // O(1)
                        }
                        (arena, root)
                    },
                    BatchSize::PerIteration,
                );
            });

            // ropey: recompute hashrope's OWN PolynomialHash over materialized bytes (O(N)).
            g.bench_with_input(BenchmarkId::new("ropey", n), &n, |b, &n| {
                b.iter_batched(
                    move || {
                        let (rope, rng) = build_ropey(n, seed);
                        let scratch = Arena::new(); // only to call hash_bytes (same algorithm)
                        (rope, rng, scratch)
                    },
                    |(mut rope, mut rng, scratch)| {
                        for _ in 0..K_B {
                            for _ in 0..epq {
                                churn_ropey(&mut rope, &mut rng);
                            }
                            let bytes = ropey_bytes(&rope); // O(N) materialize
                            black_box(scratch.hash_bytes(&bytes)); // O(N) recompute
                        }
                        rope
                    },
                    BatchSize::PerIteration,
                );
            });
        }
        g.finish();
    }
}

// ---------------- Honest-negative probe (criterion v): persistent-arena growth ----------------
fn memory_probe(seed: u64) {
    println!("# MEMPROBE seed={} ops_per_size={}", seed, MEM_PROBE_OPS);
    println!("n,nodes_initial,nodes_after,delta,nodes_per_op");
    for &n in &sizes() {
        let mut rng = Rng::new(seed);
        let init = rand_ascii_vec(&mut rng, n);
        let mut arena = Arena::new();
        let mut root = arena.from_bytes(&init);
        let n0 = arena.node_count();
        for _ in 0..MEM_PROBE_OPS {
            root = churn_hashrope(&mut arena, root, &mut rng);
        }
        let n1 = arena.node_count();
        println!(
            "{},{},{},{},{:.3}",
            n,
            n0,
            n1,
            n1 - n0,
            (n1 - n0) as f64 / MEM_PROBE_OPS as f64
        );
        black_box(root);
    }
}

fn main() {
    let seed = seed();
    if std::env::var("HASHROPE_B2_MEMPROBE")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        memory_probe(seed);
        return;
    }
    let mut c = Criterion::default().configure_from_args();
    regime_a(&mut c, seed);
    regime_b(&mut c, seed);
    c.final_summary();
}
