// Sandbox-only: does `repeat` work correctly when the base is a CHUNKED (>512B) tree
// under the patched from_bytes? The existing e3 test only uses a 100-byte (single-leaf) base.
use hashrope::{Arena, PolynomialHash};

fn main() {
    let ph = PolynomialHash::default_hash();
    let mut all_ok = true;
    for &base_len in &[100usize, 513, 600, 2000, 10_000, 50_000] {
        let base: Vec<u8> = (0..base_len).map(|i| (i % 251) as u8).collect();
        let mut a = Arena::new();
        let r = a.from_bytes(&base);
        let h_init = a.height(r);
        for &q in &[1u64, 2, 7, 50] {
            let rep = a.repeat(r, q);
            let materialized: Vec<u8> =
                base.iter().copied().cycle().take(base_len * q as usize).collect();
            let hash_ok = a.hash(rep) == ph.hash(&materialized);
            let bytes_ok = a.to_bytes(rep) == materialized;
            a.validate(rep); // panics on BB[2/7] violation
            if !(hash_ok && bytes_ok) {
                all_ok = false;
            }
            println!(
                "base_len={:>6} h_init={:>2} chunked={:<5} q={:>3} hash_ok={} bytes_ok={}",
                base_len,
                h_init,
                base_len > 512,
                q,
                hash_ok,
                bytes_ok
            );
        }
    }
    println!("\nALL_OK={}", all_ok);
}
