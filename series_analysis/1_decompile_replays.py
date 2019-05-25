from pathlib import Path
from typing import List, Sequence

from carball.json_parser.game import Game as JsonParserGame
from carball.decompile_replays import decompile_replay

from series_analysis.folders import *


def get_all_replay_filepaths(folder: Path) -> List[Path]:
    replay_filepaths = [file for file in folder.glob("**/*.replay")]
    print(f"Found {len(replay_filepaths)} replay files.")
    return replay_filepaths


def decompile_replays(replay_filepaths: Sequence[Path], output_folder: Path):
    for replay_filepath in replay_filepaths:
        replay_filename = replay_filepath.name[:-7]

        output_path = output_folder / (replay_filename + ".json")
        _json = decompile_replay(str(replay_filepath), output_path=str(output_path))
        game = JsonParserGame()
        game.initialize(loaded_json=_json, parse_replay=False, clean_player_names=True)

        # Move decompiled file if game.id is not replay_filename
        if game.id != replay_filename:
            new_output_path = output_folder / (game.id + ".json")
            try:
                output_path.rename(new_output_path)
            except FileExistsError:
                pass


if __name__ == '__main__':
    _replay_filepaths = get_all_replay_filepaths(folder_path_to_analyse)
    decompile_replays(_replay_filepaths, output_folder=decompiled_output_folder)
