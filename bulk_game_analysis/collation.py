import datetime
import os
import shutil
from pathlib import Path

from series_analysis.folders import *

ANALYSES_FOLDER = r"C:\Users\harry\Documents\rocket_league\carball_analysis\series_analysis\analyses"

for file in analysis_folder.glob("**/*.xlsx"):
    target_path: Path = Path(ANALYSES_FOLDER) / file.name
    timestamp = os.path.getmtime(file)
    dt = datetime.datetime.fromtimestamp(timestamp)
    # print(dt, dt > datetime.datetime(2019, 4, 15))
    if dt > datetime.datetime(2019, 4, 15):
        shutil.copy(file, target_path)
