import os
import pickle
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import tqdm

from Elements.Board import BoardChangeStatus, Board
from Elements.BoardMemorizer.BoardMemorizerTree import BoardMemorizerTree
from Elements.ImageGetters import Video
from extra import factories
from extra.figures import Direction


def smart_convert():
    reader = factories.camera_reader()
    video = Video(r"C:\Users\Gleb\Desktop\Shogi\videos\партия1.mp4")
    reader.get_board_splitter().set_image_getter(video)
    fps = int(video.video.get(cv2.CAP_PROP_FPS))
    idle_step = fps // 2
    count = video.frames_count()

    results = []
    boards_count = defaultdict(int)

    def skip_frames(n: int):
        for _ in range(n):
            video.video.grab()

    pbar = tqdm.tqdm(total=count)
    i = 0
    while i < count:
        pbar.set_description(f"{len(boards_count)} moves")
        reader.update()
        pbar.update(1)
        i += 1
        s = reader.get_board().__str__()
        if boards_count[s] > 5:
            skip_frames(idle_step)
            pbar.update(idle_step)
            i += idle_step
        boards_count[s] += 1
        board = reader.get_board()
        score = reader.confidence_score
        results.append((board, score))

    with open("boards1.pickle", "wb") as file:
        pickle.dump(results, file)


def boards_to_kif_2():
    with open("boards2.pickle", "rb") as file:
        results = pickle.load(file)
    print(len(results))
    results = [r for r in results if r[1] > 0.99]
    memorizer = BoardMemorizerTree()

    # for board, score in results:
    #     memorizer.update(board.figures, board.directions, score)
    #     # if len(memorizer.get_moves()) >= 30:
    #     #     print(memorizer.update_status)
    #     #     if memorizer.update_status == BoardChangeStatus.VALID_MOVE:
    #     #         print(memorizer.get_moves()[-1])
    #     #     plt.imshow(board.to_image()); plt.show()

    patience = 100
    i = 0
    last_good_i = 0
    while i < len(results):
        board, score = results[i]

        board.update()

        memorizer.update(board.figures, board.directions, score)
        if memorizer.update_status in [BoardChangeStatus.NOTHING_CHANGED, BoardChangeStatus.VALID_MOVE]:
            last_good_i = i
        else:
            if (i - last_good_i) > patience:
                # patience expired, returning to last good board and trying to fill missing boards
                i = last_good_i + 1
                # board_old = results[last_good_i][0]
                board_old = memorizer.get_board()
                next_possible_boards = board_old.get_all_variants_smart()
                pbar = tqdm.tqdm(desc="Generating missing board")
                missing_board_found = False
                while not missing_board_found:
                    board_curr = results[i][0]

                    # if (i - last_good_i) >= 38:
                    #     print()
                    #
                    #     _, ax = plt.subplots(ncols=2)
                    #     memorizer.update(board_curr.figures, board_curr.directions, 1.0)
                    #     print(memorizer.get_board().__hash__())
                    #     print(board_curr.__hash__())
                    #     print(memorizer.update_status)
                    #     print(board_utils.get_changed_cells(board_old, board_curr))
                    #     ax[0].imshow(board_old.to_image())
                    #     ax[1].imshow(board_curr.to_image())
                    #     plt.show()
                    # else:
                    #     pbar.update(1)
                    #     i += 1
                    #     continue

                    _, ax = plt.subplots(ncols=2)
                    print(board_old.get_changed_cells(board_curr))
                    memorizer.update(board_curr.figures, board_curr.directions, 1.0)
                    print(memorizer.get_board().__hash__())
                    print(board_curr.__hash__())
                    print(memorizer.update_status)
                    ax[0].imshow(board_old.to_image())
                    ax[1].imshow(board_curr.to_image())
                    plt.show()

                    for skipped_board in next_possible_boards:
                        if board_old.is_sequence_valid([
                            board_old,
                            skipped_board,
                            board_curr
                        ]):
                            missing_board_found = True
                            last_good_i = i
                            memorizer.update(skipped_board.figures, skipped_board.directions, 1.0)
                            memorizer.update(board_curr.figures, board_curr.directions, 1.0)
                            break
                    pbar.update(1)
                    i += 1
                pbar.close()

        i += 1

    print(memorizer.get_kif())


def boards_to_kif_3():
    with open("boards1.pickle", "rb") as file:
        results = pickle.load(file)
    results = [r for r in results if r[1] > 0.99]

    # Remove duplicated boards
    tmp = []
    prev = results[-1]
    for res in results:
        if res[0] == prev[0]:
            continue
        tmp.append(res)
        prev = res
    results = tmp

    for board, _ in results:
        for i in range(9):
            for j in range(9):
                if board.directions[i][j] == Direction.NONE:
                    board.directions[i][j] = Direction.UP

    print(len(results))

    save_boards([res[0] for res in results], "boards_logs")

    memorizer = BoardMemorizerTree()

    pbar = tqdm.tqdm(results)

    for board, score in pbar:
        memorizer.update(board.figures, board.directions, score)
        n_turns = len(memorizer._tree.nodes_by_turn)
        # if n_turns >= 41:
        #     print(memorizer.update_status)
        #     change = memorizer.get_board().get_changed_cells(board)
        #     print(change)
        #     memorizer.update(board.figures, board.directions, score)
        #     _, ax = plt.subplots(ncols=2)
        #     ax[0].imshow(memorizer.get_board().to_image())
        #     ax[1].imshow(board.to_image())
        #     # plt.imshow(memorizer.get_board().to_image())
        #     # plt.show()
        pbar.set_description(f"Turns: {n_turns}, Nodes: {memorizer._tree.nodes_count()}")
    memorizer._tree.show()
    print(memorizer.get_kif())


def save_boards(boards: list[Board], folder_name: str):
    os.makedirs(folder_name, exist_ok=True)
    for i, board in tqdm.tqdm(list(enumerate(boards))):
        img = board.to_image()
        cv2.imwrite(os.path.join(folder_name, f"{i}.png"), img)


# video_to_boards()
# smart_convert()
# boards_to_kif()
# boards_to_kif_2()
boards_to_kif_3()
