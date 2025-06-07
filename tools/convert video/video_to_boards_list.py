import os
import pathlib
import pickle
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib import animation

from Elements import BoardChangeStatus, Board
from Elements.BoardMemorizer.BoardMemorizerTree import BoardMemorizerTree
from Elements.Board.Move import Move
from Elements.ImageGetters import Video
from extra import factories
from extra.figures import Direction
from extra.types import FilePath


def animate(imgs):
    # fig = plt.figure()
    # im = plt.imshow(imgs[0], animated=True)
    # def updatefig(i, *args):
    #     im.set_array(imgs[i])
    #     return im,
    # ani = animation.FuncAnimation(fig, updatefig, frames=len(imgs), interval=50, blit=True)
    # return ani

    # Create dummy frames (replace with your actual frames)
    num_frames = len(imgs)
    frames = imgs

    fig, ax = plt.subplots()
    img = ax.imshow(frames[0])

    def update(frame):
        img.set_array(frames[frame])
        return img,

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)

    plt.show()


def get_frames_from_video(video: Video, idx: list[int]) -> list[np.ndarray]:
    video = video.__copy__()
    idx = sorted(idx)
    video.restart()
    frame_i = 0
    pointer = 0
    frames = []
    while frame_i != idx[-1] + 1:
        if frame_i < idx[pointer]:
            video.skip_frame()
        else:
            frames.append(video.get_image())
            pointer += 1
        frame_i += 1
    return frames



def smart_convert(
        video_path: FilePath,
        output_path: FilePath = None,
) -> pathlib.Path:
    if output_path is None:
        output_path = pathlib.Path(video_path).stem + "_processed.pkl"

    reader = factories.camera_reader()
    video = Video(video_path)
    reader.get_board_splitter().set_image_getter(video)
    fps = video.fps()
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
        results.append({
            "board": board,
            "score": score,
            "frame_i": i,
        })

    with open(output_path, "wb") as file:
        pickle.dump(results, file)
    return output_path


def boards_to_kif(
        file_path: FilePath
):
    with open(file_path, "rb") as file:
        results = pickle.load(file)

    # Filter high confidence frames
    results = [r for r in results if r["score"] > 0.99]

    # Remove duplicated boards
    tmp = []
    prev = results[-1]
    for res in results:
        if res["board"] == prev["board"]:
            continue
        tmp.append(res)
        prev = res
    results = tmp

    # replace Direction.NONE
    for r in results:
        for i in range(9):
            for j in range(9):
                if r["board"].directions[i][j] == Direction.NONE:
                    r["board"].directions[i][j] = Direction.UP

    frames_i = [i["frame_i"] for i in results]
    boards = [i["board"] for i in results]

    print(len(results))

    # save_boards([res["board"] for res in results], paths.LOGS_DIR / "boards_logs")
    # subset = [0, 12, 21, 39, 63, 74, 91, 112, 125, 136]
    # results = [results[i] for i in subset]

    memorizer = BoardMemorizerTree()

    pbar = tqdm.tqdm(results)

    for info in pbar:
        board = info["board"]
        score = info["score"]
        memorizer.update(board.figures, board.directions, score)
        n_turns = len(memorizer._tree.nodes_by_turn)

        if memorizer.update_status == BoardChangeStatus.NEED_MANUAL:
            memorizer._tree.show()
            frames = get_frames_from_video(
                Video(video_path),
                frames_i[boards.index(board) - 100: boards.index(board) - 70]
            )
            animate(frames)
            plt.show()

            inp = input("Need manual:")
            moves = []
            for s in inp.split():
                i1, j1, i2, j2 = [int(i) for i in s]
                move = Move.from_coords(memorizer.get_board(), (j1, i1), (j2, i2))
                moves.append(move)
            memorizer.provide_manual_info(moves)

            # 8271 0617
            # 5747 0516
            # 6656 0213
            # 6655
            # 2030
            # 2838

        pbar.set_description(f"Turns: {n_turns}, Nodes: {memorizer._tree.nodes_count()}")
    print(memorizer.get_kif())
    memorizer._tree.show()


def save_boards(boards: list[Board], folder_name: str):
    os.makedirs(folder_name, exist_ok=True)
    for i, board in tqdm.tqdm(list(enumerate(boards))):
        img = board.to_image()
        cv2.imwrite(os.path.join(folder_name, f"{i}.png"), img)


if __name__ == '__main__':
    video_path = r"C:\Users\Gleb\Desktop\Shogi\videos\партия1.mp4"
    meta_path = "boards1.pickle"
    # smart_convert(video_path, meta_path)
    boards_to_kif(meta_path)
