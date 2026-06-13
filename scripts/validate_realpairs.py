"""Quick validation of realpairs.py on real datasets."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.realpairs import load_conversations, load_real_pairs

sg = load_conversations("data/canonical/sharegpt_sample.jsonl", min_turns=4)
lm = load_conversations("data/canonical/lmsys_sample.jsonl", min_turns=4)
print(f"ShareGPT >= 4 turns: {len(sg)}")
print(f"LMSYS >= 4 turns: {len(lm)}")

pairs = load_real_pairs("data/canonical/sharegpt_sample.jsonl", n_pairs=5, seed=42)
print(f"ShareGPT 5 pairs: LCPs={[p['oracle_token_lcp'] for p in pairs]}, cached_lens={[p['cached_len'] for p in pairs]}")

pairs = load_real_pairs("data/canonical/lmsys_sample.jsonl", n_pairs=5, seed=42)
print(f"LMSYS 5 pairs: LCPs={[p['oracle_token_lcp'] for p in pairs]}, cached_lens={[p['cached_len'] for p in pairs]}")

print("OK")
