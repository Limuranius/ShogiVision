from __future__ import annotations

from extra.figures import Figure, Direction

JP_DIGITS = {
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
}

USI_LETTERS = "abcdefghi"


class Move:
    figure: Figure
    direction: Direction

    # (y, x), 0 <= x, y <= 8
    # Array coordinate system (origin in upper left corner)
    array_destination: tuple[int, int]
    array_origin: tuple[int, int] | None

    # (x, y), 1 <= x, y <= 9
    # Board coordinate system (origin in upper right corner)
    destination: tuple[int, int]
    origin: tuple[int, int] | None

    is_drop: bool
    is_promotion: bool

    def __init__(
            self,
            array_destination: tuple[int, int],  # (y, x)
            figure: Figure,
            direction: Direction,
            array_origin: tuple[int, int] = None,  # (y, x)
            is_drop: bool = False,
            is_promotion: bool = False,
    ):
        self.array_destination = array_destination
        self.destination = (9 - array_destination[1], array_destination[0] + 1)
        self.array_origin = array_origin
        if array_origin is not None:
            self.origin = (9 - array_origin[1], array_origin[0] + 1)
        else:
            self.origin = None
        self.figure = figure
        self.direction = direction
        self.is_drop = is_drop
        self.is_promotion = is_promotion


    def to_usi(self) -> str:
        if self.is_drop:
            fmt = "{fig_chr}*{x_dest_num}{y_dest_chr}"
            return fmt.format(
                x_dest_num=self.destination[0],
                y_dest_chr=USI_LETTERS[self.destination[1] - 1],
                fig_chr=self.figure.value.upper()
            )
        else:
            fmt = "{x_orig_num}{y_orig_chr}{x_dest_num}{y_dest_chr}"
            if self.is_promotion:
                fmt += "+"
            return fmt.format(
                x_orig_num=self.origin[0],
                y_orig_chr=USI_LETTERS[self.origin[1] - 1],
                x_dest_num=self.destination[0],
                y_dest_chr=USI_LETTERS[self.destination[1] - 1],
            )

    @classmethod
    def from_coords(
            cls,
            board,
            start: tuple[int, int],  # (y, x)
            end: tuple[int, int],  # (y, x)
    ) -> Move:
        figure = board.figures[start[0]][start[1]]
        direction = board.directions[start[0]][start[1]]
        return Move(end, figure, direction, start)


    def to_kif(self, flip=False) -> str:
        """
        Returns kif signature of move
        """

        dest_coords_str = "{x}{y_jp}".format(
            x=self.destination[0] if not flip else 10 - self.destination[0],
            y_jp=JP_DIGITS[self.destination[1] if not flip else 10 - self.destination[1]]
        )
        if self.is_drop:
            s = "{dest}{fig_jp}打".format(
                dest=dest_coords_str,
                fig_jp=self.figure.to_jp()
            )
            return s
        else:
            origin_coords_str = "{x}{y}".format(
                x=self.origin[0] if not flip else 10 - self.origin[0],
                y=self.origin[1] if not flip else 10 - self.origin[1]
            )
            prom_str = "成" if self.is_promotion else ""
            s = "{dest}{fig_jp}{prom}({origin})".format(
                dest=dest_coords_str,
                fig_jp=self.figure.to_jp(),
                prom=prom_str,
                origin=origin_coords_str
            )
            return s

    def __repr__(self):
        return f"{self.array_origin}-{self.array_destination}|{self.figure} {self.direction}|{self.is_drop=} {self.is_promotion=}"