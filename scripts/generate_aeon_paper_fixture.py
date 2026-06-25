"""Generate the aeon-vs-Rust equivalence fixture from an example dataset.

By default this script clones the paper-flextail-vs-camera repository into
target/example-data/ and uses a small subset of its RSF recordings as a real-data
example. The cloned repository is an example data source only; it is not a crate
submodule and is not required by the RDST model format.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
from aeon.classification.shapelet_based import RDSTClassifier

from aeon_rdst_export import export_rdst_classifier


DEFAULT_ACTIVITIES = ("cutting", "eating", "loading_dishwasher")
DEFAULT_PAPER_REPO_URL = "https://github.com/onecalfman/paper-flextail-vs-camera.git"
DEFAULT_PAPER_REPO_DIR = Path("target/example-data/paper-flextail-vs-camera")


def ensure_paper_repo(path: Path, url: str) -> Path:
    if (path / "data").is_dir():
        return path
    if path.exists():
        raise ValueError(f"{path} exists but does not look like the paper repo")

    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(path)], check=True)
    return path


def read_rsf_matrix(path: Path, n_channels: int) -> np.ndarray:
    raw = path.read_bytes()
    if raw[:4] != b"RSF1":
        raise ValueError(f"{path} is not an RSF1 file")

    payload = gzip.decompress(raw[4:])
    for encoding in ("utf-8", "ISO-8859-1", "cp1252"):
        try:
            doc = json.loads(payload.decode(encoding))
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"could not decode {path}")

    rows = [line for line in doc["content"].splitlines() if line.strip()]
    matrix = np.asarray(
        [[float(value) for value in row.split(",")] for row in rows], dtype=np.float64
    )
    if matrix.shape[1] < n_channels:
        raise ValueError(f"{path} has {matrix.shape[1]} columns, need {n_channels}")
    return matrix[:, :n_channels]


def make_windows(matrix: np.ndarray, window_size: int, count: int, offset: int) -> np.ndarray:
    needed = (offset + count) * window_size
    if len(matrix) < needed:
        raise ValueError(f"not enough rows for {count} windows at offset {offset}")
    windows = matrix[:needed].reshape(-1, window_size, matrix.shape[1])
    windows = windows[offset : offset + count]
    return np.transpose(windows, (0, 2, 1))


def build_dataset(
    data_dir: Path,
    participant: str,
    activities: tuple[str, ...],
    n_channels: int,
    window_size: int,
    train_windows_per_class: int,
    test_windows_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train_parts = []
    y_train: list[str] = []
    x_test_parts = []
    y_test: list[str] = []

    for activity in activities:
        matrix = read_rsf_matrix(data_dir / participant / f"{activity}.rsf", n_channels)
        x_train = make_windows(matrix, window_size, train_windows_per_class, 0)
        x_test = make_windows(
            matrix, window_size, test_windows_per_class, train_windows_per_class
        )
        x_train_parts.append(x_train)
        x_test_parts.append(x_test)
        y_train.extend([activity] * len(x_train))
        y_test.extend([activity] * len(x_test))

    return (
        np.concatenate(x_train_parts).astype(np.float64),
        np.asarray(y_train),
        np.concatenate(x_test_parts).astype(np.float64),
        np.asarray(y_test),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-repo",
        type=Path,
        default=DEFAULT_PAPER_REPO_DIR,
        help="Clone destination or existing checkout for the example data repo",
    )
    parser.add_argument(
        "--paper-repo-url",
        default=DEFAULT_PAPER_REPO_URL,
        help="Git URL used when --paper-repo does not exist",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/aeon_paper_equivalence.json"),
        help="Fixture JSON path",
    )
    parser.add_argument("--participant", default="M1")
    parser.add_argument("--activities", nargs="+", default=list(DEFAULT_ACTIVITIES))
    parser.add_argument("--n-channels", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--train-windows-per-class", type=int, default=6)
    parser.add_argument("--test-windows-per-class", type=int, default=3)
    parser.add_argument("--max-shapelets", type=int, default=16)
    parser.add_argument("--shapelet-length", type=int, default=7)
    parser.add_argument("--random-state", type=int, default=7)
    args = parser.parse_args()

    paper_repo = ensure_paper_repo(args.paper_repo, args.paper_repo_url)
    data_dir = paper_repo / "data"
    activities = tuple(args.activities)
    x_train, y_train, x_test, y_test = build_dataset(
        data_dir,
        args.participant,
        activities,
        args.n_channels,
        args.window_size,
        args.train_windows_per_class,
        args.test_windows_per_class,
    )

    clf = RDSTClassifier(
        max_shapelets=args.max_shapelets,
        shapelet_lengths=[args.shapelet_length],
        random_state=args.random_state,
        n_jobs=1,
    )
    clf.fit(x_train, y_train)
    expected = [str(label) for label in clf.predict(x_test)]

    fixture = {
        "source": {
            "repo": paper_repo.as_posix(),
            "repo_url": args.paper_repo_url,
            "participant": args.participant,
            "activities": list(activities),
            "n_channels": args.n_channels,
            "window_size": args.window_size,
            "train_windows_per_class": args.train_windows_per_class,
            "test_windows_per_class": args.test_windows_per_class,
            "max_shapelets": args.max_shapelets,
            "shapelet_length": args.shapelet_length,
            "random_state": args.random_state,
        },
        "classes": [str(label) for label in clf.classes_],
        "y_test": [str(label) for label in y_test],
        "expected_predictions": expected,
        "test_X": x_test.tolist(),
        "model": export_rdst_classifier(clf),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(
        "wrote",
        args.output,
        "train_shape=",
        x_train.shape,
        "test_shape=",
        x_test.shape,
    )


if __name__ == "__main__":
    main()
