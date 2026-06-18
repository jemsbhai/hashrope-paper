//! EXP-020 (claim B2) -- correctness harness, red-first.
//!
//! Drives an identical seeded churn sequence (delete a random span + insert a
//! same-length random ASCII span at a random position, keeping |buffer| stable)
//! through three arms in lockstep:
//!   * hashrope     -- Arena::new() (NON-LAZY: the whole-buffer polynomial
//!                     fingerprint is maintained eagerly along the O(log N) spine
//!                     on every edit), edited via the split/concat primitives
//!                     (hashrope exposes no insert/delete API);
//!   * Ropey 1.6.1  -- Rope::insert / Rope::remove (maintains no hash);
//!   * a Vec<u8> oracle -- the ground-truth byte buffer.
//!
//! ASCII content (bytes 0x20..=0x7e) so char_idx == byte_idx, removing the
//! byte/char confound between the two libraries.
//!
//! Per the EXP-020 verbatim promotion criterion clause (i) [HARD], after every
//! edit step this asserts:
//!   (a) byte-identity    : hashrope bytes == oracle == ropey bytes;
//!   (b) hash maintenance : arena.hash(root)       (maintained incrementally)
//!                          == arena.hash_bytes(&bytes) (an independent
//!                          from-scratch polynomial hash of the materialized
//!                          bytes).
//! It also calls validate (the BB[2/7] invariant) periodically and at the end.
//! A mismatch fails B2 outright -- it would be a hash-maintenance-under-churn
//! bug, investigated, never retrofitted.
//!
//! Red-first: this file `use ropey::Rope;`. Before `ropey` is added to
//! [dev-dependencies] it fails to COMPILE (confirmed-red); adding ropey 1.6.1
//! turns it GREEN (the arms agree).

use hashrope::Arena;
use ropey::Rope;

/// Deterministic, dependency-free SplitMix64 PRNG so the churn sequence is
/// identical across arms and reproducible across runs, with no `rand` dep.
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

    /// Value in [0, n); requires n > 0.
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }

    /// A printable ASCII byte in 0x20..=0x7e (always single-byte UTF-8).
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

/// One full churn run at (seed, n) for `k` ops, asserting criterion (i) every step.
fn run_one(seed: u64, n: usize, k: usize) {
    let mut rng = Rng::new(seed);

    // Identical initial buffer for all three arms.
    let init = rand_ascii_vec(&mut rng, n);

    let mut arena = Arena::new(); // NON-LAZY: fingerprint maintained eagerly.
    let mut root = arena.from_bytes(&init);

    let mut rope = Rope::from_str(&ascii_string(&init));

    let mut oracle: Vec<u8> = init.clone();

    // Sanity at start.
    assert_eq!(arena.to_bytes(root), oracle, "init hashrope != oracle (seed {seed}, n {n})");
    assert_eq!(ropey_bytes(&rope), oracle, "init ropey != oracle (seed {seed}, n {n})");
    let h_m0 = arena.hash(root);
    let h_r0 = arena.hash_bytes(&oracle);
    assert_eq!(h_m0, h_r0, "init maintained-hash != recompute (seed {seed}, n {n})");
    arena.validate(root); // panics internally on a BB[2/7] invariant violation

    let span_cap = 32usize;

    for step in 0..k {
        let len = oracle.len();
        assert_eq!(len, n, "length drifted to {len} at step {step} (seed {seed}, n {n})");

        // Span length L in [1, min(span_cap, len-1)].
        let l = 1 + rng.below(span_cap.min(len - 1));
        // Delete window [del_start, del_start + l).
        let del_start = rng.below(len - l + 1);
        // Insert position in the post-delete buffer of length (len - l).
        let post_len = len - l;
        let ins_pos = rng.below(post_len + 1);
        // L random ASCII bytes to insert.
        let inserted = rand_ascii_vec(&mut rng, l);

        // --- oracle ---
        oracle.drain(del_start..del_start + l);
        oracle.splice(ins_pos..ins_pos, inserted.iter().copied());

        // --- hashrope (split/concat primitives) ---
        // delete: (left, rest) = split(root, del_start); (_, right) = split(rest, l);
        //         root = concat(left, right)
        let (left, rest) = arena.split(root, del_start as u64);
        let (_mid, right) = arena.split(rest, l as u64);
        root = arena.concat(left, right);
        // insert: ins = from_bytes(inserted); (l2, r2) = split(root, ins_pos);
        //         root = concat(concat(l2, ins), r2)
        let ins_node = arena.from_bytes(&inserted);
        let (l2, r2) = arena.split(root, ins_pos as u64);
        let lr = arena.concat(l2, ins_node);
        root = arena.concat(lr, r2);

        // --- ropey ---
        rope.remove(del_start..del_start + l);
        rope.insert(ins_pos, &ascii_string(&inserted));

        // --- criterion (i) assertions, every step ---
        let hr_bytes = arena.to_bytes(root);
        assert_eq!(hr_bytes, oracle, "byte mismatch hashrope vs oracle, step {step} (seed {seed}, n {n})");
        assert_eq!(ropey_bytes(&rope), oracle, "byte mismatch ropey vs oracle, step {step} (seed {seed}, n {n})");
        let h_m = arena.hash(root);
        let h_r = arena.hash_bytes(&oracle);
        assert_eq!(h_m, h_r, "maintained-hash != recompute, step {step} (seed {seed}, n {n})");

        if step % 25 == 0 {
            // validate(...) panics internally on a BB[2/7] invariant violation.
            arena.validate(root);
        }
    }

    arena.validate(root); // panics internally on a BB[2/7] invariant violation
}

#[test]
fn b2_churn_byte_identity_and_hash_maintenance() {
    for &seed in &[1u64, 2, 3] {
        for &n in &[1_000usize, 5_000] {
            run_one(seed, n, 200);
        }
    }
}
