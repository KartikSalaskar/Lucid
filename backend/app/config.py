from pathlib import Path
from pydantic_settings import BaseSettings

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    max_upload_mb: int = 20

    model_path: str = str(ROOT / "models/classifier.pt")
    temperature_path: str = str(ROOT / "trust_artifacts/temperature.json")

    metrics_path: str = str(ROOT / "models/metrics.json")
    dataset_stats_path: str = str(ROOT / "data/processed/stats.json")

    db_url: str = f"sqlite:///{ROOT / 'data/app.db'}"

    img_size: int = 224
    clahe: bool = True

    class Config:
        env_file = ".env"


settings = Settings()