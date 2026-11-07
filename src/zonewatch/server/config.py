"""Review-server settings (env prefix ZW_SERVER_)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ZW_SERVER_", env_file=".env", extra="ignore")

    data_dir: str = "data/server"
    db_path: str = ""  # default <data_dir>/zonewatch.db
    evidence_dir: str = ""  # default <data_dir>/evidence
    videos_dir: str = "data/videos"  # original sample videos for "원본 장면" playback (optional)
    grpc_listen: str = "0.0.0.0:50052"
    http_host: str = "0.0.0.0"
    http_port: int = 8000
    console_dir: str = "console"
    device_offline_after_s: float = 8.0

    # VLM
    vlm_provider: str = "mock"  # mock | anthropic | openai_compat | transformers | none
    vlm_model: str = ""
    vlm_trigger: str = "closed"  # closed | dwell_or_closed | never
    vlm_timeout_s: float = 90.0
    vlm_max_concurrency: int = 1
    vlm_max_attempts: int = 2
    vlm_frames: str = "context_before,entry,dwell,exit,context_after"
    vlm_frame_max_side: int = 768
    vlm_use_annotated: bool = True
    vlm_openai_base_url: str = "http://127.0.0.1:8001/v1"
    vlm_openai_api_key: str = "none"
    vlm_transformers_model: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    vlm_max_tokens: int = 700
    vlm_budget_usd: float = 5.0  # hard stop for paid providers (estimated spend)
    vlm_mock_delay_s: float = 0.2  # fault injection for tests/scenarios (mock provider only)
    vlm_mock_fail_every: int = 0
    vlm_mock_invalid_every: int = 0

    # demo mode
    replay_bundle: str = ""  # path to a recorded run dir (events.jsonl + evidence/) imported at startup
    mode: str = "live"  # live | replay  (replay = public demo without a device)
    public_url: str = ""

    def resolved_db(self) -> str:
        return self.db_path or f"{self.data_dir}/zonewatch.db"

    def resolved_evidence(self) -> str:
        return self.evidence_dir or f"{self.data_dir}/evidence"
