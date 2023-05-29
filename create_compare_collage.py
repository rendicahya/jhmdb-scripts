from pathlib import Path

import numpy as np
from moviepy.editor import VideoFileClip, clips_array, concatenate_videoclips

dataset_path = Path("/nas.dbms/randy/datasets/jhmdb")
output_path = Path("/nas.dbms/randy/datasets/jhmdb-masked-collage")
masked_path = Path("/nas.dbms/randy/datasets/jhmdb-masked")

output_path.mkdir(parents=True, exist_ok=True)

for action in masked_path.iterdir():
    count = 0
    clips = []
    pairs = []

    for masked in action.iterdir():
        orig_video = dataset_path / action.name / masked.with_suffix(".avi").name
        count += 1

        pairs.append([VideoFileClip(str(orig_video)), VideoFileClip(str(masked))])

        if count % 3 == 0:
            clip = np.array(pairs).T
            clip = clips_array(clip)
            pairs = []

            clips.append(clip)

    final_video = concatenate_videoclips(clips)
    final_video.without_audio().write_videofile(
        str(output_path / action.with_suffix(".mp4").name)
    )
