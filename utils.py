from pathlib import Path

from tqdm import tqdm


def iterate(path: Path, op, extension=None):
    n_files = count_files(path, recursive=True, extension=extension)

    with tqdm(total=n_files) as bar:
        for action in path.iterdir():
            for video in action.iterdir():
                bar.set_description(video.name[:30])
                op(action, video)
                bar.update(1)


def count_files(path: Path, recursive=False, extension=None):
    pattern = "**/*" if recursive else "*"
    filter = (
        (lambda f: f.is_file())
        if extension is None
        else (lambda f: f.is_file() and f.suffix == f".{extension}")
    )

    return sum(1 for f in path.glob(pattern) if filter(f))
