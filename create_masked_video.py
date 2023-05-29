from pathlib import Path

import cv2
import numpy as np
import utils
from moviepy.editor import ImageSequenceClip, VideoFileClip
from scipy.io import loadmat

input_path = Path("/nas.dbms/randy/datasets/jhmdb")
output_path = Path("/nas.dbms/randy/datasets/jhmdb-masked")
mask_path = Path("/nas.dbms/randy/projects/jhmdb-scripts/mask-annotations")
n_mat = utils.count_files(mask_path, extension="mat")
actor = True


def operation(action, video):
    mat = loadmat(video / "puppet_mask.mat")
    masks = np.array(mat["part_mask"])
    output_frames = []

    input_video_path = input_path / action.name / (video.name + ".avi")
    output_video_path = output_path / action.name / (video.name + ".mp4")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    clip = VideoFileClip(str(input_video_path))
    frames = clip.iter_frames()
    n_mask_frames = masks.shape[-1]
    dilation_kernel = np.ones((5, 5), np.uint8)

    for i, frame in enumerate(frames):
        if i < n_mask_frames:
            mask = masks[..., i] if actor else 1 - masks[..., i]
            mask = cv2.dilate(mask, dilation_kernel, iterations=4)
            masked = cv2.bitwise_and(frame, frame, mask=mask)

            output_frames.append(masked)

    if len(output_frames) > 0:
        clip = ImageSequenceClip(output_frames, fps=clip.fps)
        clip.write_videofile(str(output_video_path), audio=False)


if __name__ == "__main__":
    utils.iterate(mask_path, operation, progress_bar=False)
