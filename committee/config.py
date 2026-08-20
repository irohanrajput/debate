from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # secrets
    gemini_api_key: str = ""

    # models
    flash_model: str = "gemini-2.5-flash"
    pro_model: str = "gemini-2.5-pro"
    embedding_model: str = "models/gemini-embedding-001"
    llm_temperature: float = 0.0
    llm_timeout_s: int = 120
    flash_thinking_budget: int = 0
    pro_thinking_budget: int = 512
    repair_retries: int = 1
    chars_per_token: int = 3
    schema_overhead_tokens: int = 1200
    plan_schema_overhead_tokens: int = 400
    min_output_tokens: int = 512
    plan_min_output_tokens: int = 256
    judge_output_tokens: int = 256
    tiebreak_output_tokens: int = 1024
    max_tiebreaks: int = 3

    # budget pools (fractions of total)
    research_pool_frac: float = 0.20
    debate_pool_frac: float = 0.65
    synthesis_pool_frac: float = 0.15

    # per-round allocation fractions (of the relevant remaining pool)
    discover_research_frac: float = 0.60
    discover_argue_frac: float = 0.35
    explore_research_frac: float = 0.80
    explore_argue_frac: float = 0.55
    exploit_argue_frac: float = 0.70
    explore_final_frac: float = 0.85
    plan_tokens: int = 768
    argue_floor_tokens: int = 1024
    argue_realistic_tokens: int = 3600
    judge_tokens: int = 512
    tiebreak_tokens: int = 2048

    # debate shape
    consensus_importance_min: int = 3
    min_rounds: int = 3
    max_rounds: int = 4
    theta_converged: float = 0.40
    convergence_delta_cap: float = 0.10
    similarity_candidate_threshold: float = 0.55
    contradiction_threshold: float = 0.60
    max_judge_pairs: int = 8
    stance_weight: float = 0.5
    contested_weight: float = 0.5

    # claim quality
    comparison_cues: list[str] = ["vs", "than", "compared", "versus", "above", "below", "percentile",
                                  "range", "peer", "historical", "average", "prior", "decelerat",
                                  "accelerat", "expand", "contract", "yoy", "year-over-year", "trend",
                                  "scenario", "implied", "relative"]
    comparison_min_ratio: float = 0.5

    # research / tools
    max_research_queries: int = 4
    tool_result_char_cap: int = 1500
    evidence_snippet_char_cap: int = 600
    others_summary_char_cap: int = 2000
    synthesis_input_char_cap: int = 15000
    evidence_render_char_cap: int = 3200
    retrieval_k: int = 5

    # corpus / chunking
    chunk_chars: int = 2000
    chunk_overlap_chars: int = 200
    reliability_map: dict[str, float] = {"high": 0.9, "medium": 0.6, "low": 0.3}
    chroma_dir: str = ".chroma"
    corpus_collection: str = "corpus"

    # market data
    volume_window_days: int = 30
    volume_baseline_days: int = 90
    quarters_shown: int = 8
    years_shown: int = 4
    timeline_entry_char_cap: int = 160
    timeline_max_entries: int = 25
    history_period: str = "max"
    macro_tickers: list[str] = ["^GSPC", "^SOX", "^TNX", "^VIX"]
    vol_window_days: int = 30
    trading_days_per_year: int = 252
    ma_short: int = 50
    ma_long: int = 200
    relative_strength_days: int = 126
    drawdown_window_days: int = 252
    risk_on_vix_max: float = 20.0

    # io
    data_dir: str = "data"
    runs_dir: str = "runs"
    default_budget: int = 60000
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
