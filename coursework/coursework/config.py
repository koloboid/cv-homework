from typing import Literal, Union

from injector import singleton
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict

from coursework.common_types import Box


@singleton
class Config(BaseSettings):
    debug: CliImplicitFlag[bool] = False
    capture_source: Union[str, int] = 0
    capture_fps: int = 30
    capture_width: int = 640
    capture_height: int = 480

    lane_crop: Box = Box(0.0, 0.6, 1.0, 0.25)

    draw_metrics: bool = True
    draw_found_lines: bool = True
    draw_filtered_lines: bool = True
    draw_roi: Literal["blurred", "edges", "original", "masked"] = "original"

    model_config = SettingsConfigDict(
        extra="ignore",
        env_nested_delimiter="__",
        cli_parse_args=True,
    )
