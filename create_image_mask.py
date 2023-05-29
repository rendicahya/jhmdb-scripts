from pathlib import Path

import cv2
import numpy as np
import utils
from scipy.io import loadmat

output_path = Path("/nas.dbms/randy/datasets/jhmdb-mask")
mask_path = Path("/nas.dbms/randy/projects/jhmdb-scripts/mask-annotations")
n_mat = utils.count_files(mask_path, recursive=True, extension=".mat")


def operation(action, video):
    mat = loadmat(video / "puppet_mask.mat")
    mask = np.array(mat["part_mask"])
    n_frames = mask.shape[-1]

    for frame_idx in range(n_frames):
        output = output_path / action.name / video.name / f"{frame_idx:04}.jpg"
        frame = mask[..., frame_idx]
        frame *= 255

        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), frame)


utils.iterate(mask_path, operation, extension=".mat")
