from typing import NamedTuple


class Box(NamedTuple):
    @classmethod
    def from_tlrb(cls, tlrb: tuple[float, float, float, float]) -> "Box":
        x1, y1, x2, y2 = tlrb
        return cls(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    x: float
    y: float
    width: float
    height: float

    def multiply(self, mul_x: float, mul_y: float) -> "Box":
        return Box(
            x=self.x * mul_x,
            y=self.y * mul_y,
            width=self.width * mul_x,
            height=self.height * mul_y,
        )

    def to_int(self) -> "Box":
        return Box(
            x=int(self.x),
            y=int(self.y),
            width=int(self.width),
            height=int(self.height),
        )
