from typing import Union
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    input_source: Union[str, int] = 0
    input_fps: int = 30
