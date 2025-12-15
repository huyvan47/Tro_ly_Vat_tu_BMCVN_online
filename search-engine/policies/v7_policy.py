from dataclasses import dataclass

@dataclass(frozen=True)
class PolicyV7:
    name: str = "v7"
    min_score_main = 0.35
    direct_conf_min: float = 0.55
    strict_conf_min: float = 0.25
    direct_gap_min: float = 0.08
    frag_gap_max: float = 0.03
    # --- suggestions ---
    min_suggest_score: float = 0.40
    max_suggest: int = 3
policy = PolicyV7()
