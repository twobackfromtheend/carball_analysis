from pathlib import Path
from typing import Optional


BASE_PATH = r"D:\Replays\RLCS\Season 7"
FOLDER_TO_ANALYSE = "RLCS EU League Play"
# FOLDER_TO_ANALYSE = "RLCS NA League Play"

BASE_PATH = r"D:\Replays\RLCS\Season 6\RLCS"
FOLDER_TO_ANALYSE = "RLCS EU League Play"

XG_MODEL: Optional[Path] = Path(r"additional_stats/x_goals/x_goals.876-0.76482.hdf5")

XG_MODEL_PATH = Path(__file__).parent.parent / XG_MODEL

assert XG_MODEL_PATH is None or XG_MODEL_PATH.is_file(), "XG_MODEL must be Path to existing file or None"

BASE_PATH = Path(BASE_PATH)
FOLDER_TO_ANALYSE = Path(FOLDER_TO_ANALYSE)

folder_path_to_analyse: Path = BASE_PATH / FOLDER_TO_ANALYSE
assert folder_path_to_analyse.is_dir(), "FOLDER_TO_ANALYSE is not directory in BASE_PATH"

decompiled_folder: Path = BASE_PATH / "decompiled"
decompiled_folder.mkdir(exist_ok=True)

decompiled_output_folder: Path = decompiled_folder / FOLDER_TO_ANALYSE
decompiled_output_folder.mkdir(exist_ok=True)

analysis_folder: Path = BASE_PATH / "analysis"
analysis_folder.mkdir(exist_ok=True)

analysis_output_folder: Path = analysis_folder / FOLDER_TO_ANALYSE
analysis_output_folder.mkdir(exist_ok=True)

__all__ = [
    "BASE_PATH", "FOLDER_TO_ANALYSE", "XG_MODEL_PATH",
    "folder_path_to_analyse",
    "decompiled_folder", "decompiled_output_folder",
    "analysis_folder", "analysis_output_folder",
]
