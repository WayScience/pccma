"""Utility function to find the root directory of the repository, so the notebooks build every path
from the root instead of from the directory they are run in."""

import pathlib


def get_root_dir(start_dir: pathlib.Path | None = None) -> pathlib.Path:
    """
    Find the root directory of the Git repository (the first parent directory with a .git folder).

    We search up from `start_dir` (the current working directory by default), then from this file,
    so the root is found when running a notebook from its module, a script from the root, or a
    script from outside the repository (utils is installed in editable mode, so this file is in
    the repository).

    Parameters
    ----------
    start_dir : pathlib.Path | None
        Directory to start the search from. If None, use the current working directory.

    Returns
    -------
    pathlib.Path
        The root directory of the repository.
    """
    start_dirs = [pathlib.Path.cwd() if start_dir is None else start_dir, pathlib.Path(__file__)]
    for start in start_dirs:
        start = start.resolve()
        for directory in (start, *start.parents):
            # .git is a folder in a clone and a file in a git worktree
            if (directory / ".git").exists():
                return directory
    raise FileNotFoundError(
        f"Could not find the root of the repository (a .git folder) above {start_dirs}"
    )
