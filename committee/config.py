from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # secrets
    gemini_api_key: str = ""

    # models
    flash_model: str = "gemini-2.5-flash"
    pro_model: str = "gemini-2.5-pro"
    embedding_model: str = "models/text-embedding-004"
    llm_temperature: float = 0.0
    llm_timeout_s: int = 120
    repair_retries: int = 1

    # budget pools (fractions of total)
    research_pool_frac: float = 0.30
    debate_pool_frac: float = 0.55
    synthesis_pool_frac: float = 0.15

    # per-round allocation fractions (of the relevant remaining pool)
    discover_research_frac: float = 0.60
    discover_argue_frac: float = 0.40
    explore_research_frac: float = 0.50
    explore_argue_frac: float = 0.50
    exploit_argue_frac: float = 0.70
    plan_tokens: int = 512
    argue_floor_tokens: int = 600
    judge_tokens: int = 384
    tiebreak_tokens: int = 2048

    # debate shape
    min_rounds: int = 3
    max_rounds: int = 4
    theta_converged: float = 0.40
    convergence_delta_cap: float = 0.10
    similarity_candidate_threshold: float = 0.55
    contradiction_threshold: float = 0.60
    stance_weight: float = 0.5
    contested_weight: float = 0.5

    # research / tools
    max_research_queries: int = 4
    tool_result_char_cap: int = 1500
    evidence_snippet_char_cap: int = 600
    others_summary_char_cap: int = 4000
    retrieval_k: int = 8

    # corpus / chunking
    chunk_chars: int = 2000
    chunk_overlap_chars: int = 200
    reliability_map: dict[str, float] = {"high": 0.9, "medium": 0.6, "low": 0.3}
    chroma_dir: str = ".chroma"
    corpus_collection: str = "corpus"

    # market data
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
    default_budget: int = 40000
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
