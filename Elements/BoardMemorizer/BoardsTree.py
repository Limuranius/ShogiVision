from __future__ import annotations

import itertools

import cv2

from Elements import BoardChangeStatus
from Elements.Board import Board
from Elements.BoardMemorizer.Move import Move


class BoardNode:
    board: Board
    children: list[BoardNode]
    parent: BoardNode | None

    def __init__(self, board: Board, parent: BoardNode = None):
        self.board = board
        self.parent = parent
        self.children = []

    def add_child(self, node: BoardNode) -> None:
        self.children.append(node)


class BoardsTree:
    """
    Stores boards in tree structure
    Boards are connected if one can be obtained from other in 1 move
    """
    nodes_by_turn: list[list[BoardNode]]  # Nodes for certain index of turn

    def __init__(self):
        self.nodes_by_turn = []

    def insert(self, board: Board) -> BoardChangeStatus:
        """
        Tries to insert board in tree
        Returns:
            BoardChangeStatus.NOTHING_CHANGED - board already in tree
            BoardChangeStatus.VALID_MOVE - board added to tree
            BoardChangeStatus.INVALID_MOVE - all boards in tree don't match
        One board can be matched with several boards in tree. In this case multiple nodes will be inserted
        """
        DEPTH = 100

        if len(self.nodes_by_turn) == 0:
            # First board to be added
            self.nodes_by_turn = [[BoardNode(board)]]
            return BoardChangeStatus.VALID_MOVE
        has_match = False
        for i, node_group in reversed(list(enumerate(self.nodes_by_turn))[-DEPTH:]):
            # Starting from last moves and moving to first

            for node in node_group:
                status = node.board.get_change_status(board)
                if status == BoardChangeStatus.NOTHING_CHANGED:
                    return BoardChangeStatus.NOTHING_CHANGED  # board is already in tree, skipping
                if status == BoardChangeStatus.VALID_MOVE:
                    has_match = True
                    new_node = BoardNode(board, parent=node)
                    node.add_child(new_node)
                    if i == len(self.nodes_by_turn) - 1:  # Need to create room for new turns
                        self.nodes_by_turn.append([])
                    self.nodes_by_turn[i + 1].append(new_node)
        if has_match:
            return BoardChangeStatus.VALID_MOVE
        return BoardChangeStatus.INVALID_MOVE

    def get_last_boards(self) -> list[Board]:
        """Returns possible boards on last turn"""
        return [node.board for node in self.nodes_by_turn[-1]]

    @staticmethod
    def __get_move_history(node: BoardNode) -> list[Move]:
        moves = []
        while node.parent is not None:
            move = node.parent.board.get_move(node.board)
            moves.append(move)
            node = node.parent
        return moves[::-1]

    def move_histories(self) -> list[list[Move]]:
        return [self.__get_move_history(node) for node in self.nodes_by_turn[-1]]

    def clear(self) -> None:
        self.nodes_by_turn.clear()

    def nodes_count(self) -> int:
        return sum([len(turn) for turn in self.nodes_by_turn])

    def show(self):
        from pyvis.network import Network
        import base64

        nodes = list(itertools.chain(*self.nodes_by_turn))
        edges = []
        for node in nodes:
            if node.parent is None:
                continue
            edges.append((
                nodes.index(node.parent),
                nodes.index(node),
            ))

        # Create a network
        net = Network(directed=True)

        for i, node in enumerate(nodes):
            img = node.board.to_image()

            _, buffer = cv2.imencode('.png', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')  # Convert to base64

            net.add_node(
                i,
                shape="image",
                image=f"data:image/png;base64,{img_base64}",
                size=25
            )  # adjust size as needed

        for e in edges:
            start, end = e
            net.add_edge(
                start,
                end,
                title=nodes[start].board.get_move(nodes[end].board).to_usi(),
            )

        # Show the network
        net.show("graph_with_images.html", notebook=False)
