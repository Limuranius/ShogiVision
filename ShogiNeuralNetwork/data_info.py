from extra.figures import Figure, Direction


CATEGORIES_FIGURE_TYPE = [
    Figure.PAWN,
    Figure.BISHOP,
    Figure.ROOK,
    Figure.LANCE,
    Figure.KNIGHT,
    Figure.SILVER,
    Figure.GOLD,
    Figure.KING,
    Figure.EMPTY,

    Figure.PAWN_PROM,
    Figure.LANCE_PROM,
    Figure.KNIGHT_PROM,
    Figure.SILVER_PROM,
    Figure.BISHOP_PROM,
    Figure.ROOK_PROM
]

CATEGORIES_DIRECTION = [Direction.UP, Direction.DOWN, Direction.NONE]

DIRECTION_TO_INDEX = {direction: CATEGORIES_DIRECTION.index(direction) for direction in CATEGORIES_DIRECTION}
FIGURE_TO_INDEX = {figure: CATEGORIES_FIGURE_TYPE.index(figure) for figure in CATEGORIES_FIGURE_TYPE}