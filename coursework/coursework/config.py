from typing import Union

from injector import singleton
from pydantic_settings import BaseSettings, SettingsConfigDict


@singleton
class Config(BaseSettings):
    capture_source: Union[str, int] = 0
    capture_fps: int = 30
    capture_width: int = 640
    capture_height: int = 480

    model_config = SettingsConfigDict(
        extra="ignore",
        env_nested_delimiter="__",
        cli_parse_args=True,
    )
