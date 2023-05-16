from pathlib import Path

import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
from scipy.io import loadmat

dataset_path = Path("/nas.dbms/randy/datasets/jhmdb")
output_path = Path("/nas.dbms/randy/datasets/jhmdb-mask-scene")
mask_path = Path("/nas.dbms/randy/projects/jhmdb-scripts/mask-annotations")
n_mat = sum(
    1 for f in mask_path.glob("**/*") if f.is_file() and f.name.endswith(".mat")
)

for action in mask_path.iterdir():
    for video_name in action.iterdir():
        mat = loadmat(video_name / "puppet_mask.mat")
        mask = np.array(mat["part_mask"])
        output_frames = []

        input_video_path = dataset_path / action.name / (video_name.name + ".avi")
        output_video_path = output_path / action.name / (video_name.name + ".mp4")

        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx = 0
        n_frames_in_mask = mask.shape[-1]

        while cap.isOpened():
            ret, frame = cap.read()

            if frame_idx < n_frames_in_mask:
                frame_mask = 1 - mask[:, :, frame_idx]
                masked_frame = cv2.bitwise_and(frame, frame, mask=frame_mask)
                masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)

                output_frames.append(masked_frame)

            frame_idx += 1

            if not ret:
                break

        if len(output_frames) > 0:
            clip = ImageSequenceClip(output_frames, fps=fps)
            clip.write_videofile(str(output_video_path), audio=False)
