from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SnapTalk AI Service"
    env: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_image_bytes: int = 10_000_000
    max_remote_fetch_bytes: int = 8_000_000
    max_audio_fetch_bytes: int = 4_000_000
    masked_image_max_dim: int = 640
    masked_image_png_max_bytes: int = 350_000
    allow_private_network_urls: bool = False
    allowed_external_hosts: str = ""

    yolo_world_vocab_path: str = "./data/yolo_world_vocab.txt"

    # DeepL Translation API
    deepl_api_key: str = ""  # Set in .env file
    
    # Detection & Segmentation settings
    detector: str = "ldet"  # Options: ldet
    segmentation: str = "mobilesam"  # Options: mobilesam, bbox
    ollama_base_url: str = "http://localhost:11434/v1"  # Ollama API
    
    translation_default_model: str = "llama3.2:3b"
    translation_db_path: str = "./data/translation_memory.db"
    translation_google_mode: str = "official_with_fallback"  # official_with_fallback, deep_translator_only
    google_cloud_project_id: str = ""
    google_cloud_location: str = "global"
    google_cloud_credentials_path: str = ""
    google_cloud_timeout_seconds: float = 12.0
    tts_primary_langs: str = "en,fr,ja,ko,zh"
    tts_primary_engine: str = "edge"  # Edge-TTS cloud synthesis
    tts_fallback_engine: str = "silence"  # Offline silence fallback
    kokoro_service_url: str = "http://localhost:8201/tts"
    tts_output_dir: str = "./data/audio"
    public_base_url: str = "http://127.0.0.1:8000"

    pronunciation_service_url: str = "http://localhost:8301/score"
    pronunciation_mode: str = "hybrid"  # remote -> local -> simulation
    pronunciation_local_enabled: bool = False
    pronunciation_whisper_model: str = "openai/whisper-tiny"
    pronunciation_wav2vec_model: str = "facebook/wav2vec2-base-960h"
    pronunciation_model_device: str = "cpu"  # cpu or cuda
    pronunciation_model_timeout_seconds: float = 20.0
    pronunciation_score_green_threshold: float = 0.80
    pronunciation_score_orange_threshold: float = 0.55

    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/snaptalk"
    llm_base_url: str = "http://localhost:11434/v1"  # Ollama default port
    llm_api_key: str = "ollama"  # Ollama doesn't require API key

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
