"""Canonical, guarded acquisition of the D1 three-model target-node CSV.

The reduction and stimulus/inference settings are ported from the archived
D1_run_pilot.py. This generator creates only the node-level acquisition table;
it does not recalculate D1 inferential statistics or Table S5B.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/precomputed/supplementary/biological_constraints/D1/pilot/pilot_target_node_distribution.csv"
EXPECTED_SHA256 = "fbb384f5e7cbfc11fdad7feae0df57c95e00f4d1dd8b9342a40946e9a5d970ca"
EXPECTED_ROWS = 1_661_184
COLUMNS = (
    "target_type", "source_type", "target_node_id",
    "mean_current_frames_10_49", "model_id", "phase_index", "phase_deg", "condition",
)

# Frozen acquisition contract copied from the archived D1 pilot.
FLYVIS_VERSION = "1.2.0"
FLYVIS_COMMIT = "92b3845cc426dd309a1a0e1b3890156c42e14021"
MODEL_IDS = ("flow/0000/000", "flow/0000/005", "flow/0000/010")
PHASES_DEG = tuple(i * 1.25 for i in range(12))
CONDITIONS = ("original", "mirror")
ON_SOURCES = ("Mi1", "Tm3")
OFF_SOURCES = ("Tm1", "Tm2", "Tm4", "Tm9")
ON_TARGETS = tuple(f"T4{x}" for x in "abcd")
OFF_TARGETS = tuple(f"T5{x}" for x in "abcd")
DT = 0.02
FRAMES = 50
FADE_SECONDS = 1.0
ANALYSIS_START = 10
ANALYSIS_STOP = 50  # Python-exclusive; frames 10–49 inclusive.
BOXEYE_EXTENT = 15
BOXEYE_KERNEL_SIZE = 13
IMAGE_SIZE = 382
INNER_DIAMETER = 55
OUTER_DIAMETER = 382


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_csv(path: Path) -> tuple[int, tuple[str, ...]]:
    """Count CSV records and read its header without loading the 87 MB table."""
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {path}") from exc
        rows = sum(1 for _ in reader)
    return rows, header


def verify_existing(path: Path = OUTPUT, *, require_canonical_hash: bool = True) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected saved D1 CSV is missing: {path}")
    rows, columns = inspect_csv(path)
    digest = sha256_file(path)
    if columns != COLUMNS:
        raise ValueError(f"Unexpected D1 CSV columns: {columns}")
    if rows != EXPECTED_ROWS:
        raise ValueError(f"Unexpected D1 CSV row count: {rows:,} (expected {EXPECTED_ROWS:,})")
    if require_canonical_hash and digest != EXPECTED_SHA256:
        raise ValueError(f"Canonical D1 CSV hash mismatch: {digest}")
    return {"path": str(path), "rows": rows, "columns": columns, "sha256": digest,
            "canonical_hash_match": digest == EXPECTED_SHA256}


def compare_csv_content(reference: Path, candidate: Path) -> bool:
    """Compare all table values in bounded memory after a byte-hash mismatch."""
    import numpy as np
    import pandas as pd

    left_iter = pd.read_csv(reference, chunksize=50_000, dtype={"model_id": str})
    right_iter = pd.read_csv(candidate, chunksize=50_000, dtype={"model_id": str})
    for left, right in zip(left_iter, right_iter):
        if tuple(left.columns) != COLUMNS or tuple(right.columns) != COLUMNS:
            return False
        if len(left) != len(right):
            return False
        for column in COLUMNS:
            if column == "mean_current_frames_10_49":
                if not np.array_equal(left[column].to_numpy(), right[column].to_numpy(), equal_nan=True):
                    return False
            elif not left[column].astype(str).equals(right[column].astype(str)):
                return False
    # Detect unequal chunk counts.
    try:
        next(left_iter)
        return False
    except StopIteration:
        pass
    try:
        next(right_iter)
        return False
    except StopIteration:
        pass
    return True


def _external_data_root() -> Path:
    candidates = []
    if os.environ.get("FLYVIS_ROOT_DIR"):
        candidates.append(Path(os.environ["FLYVIS_ROOT_DIR"]).expanduser())
    candidates.append(Path.home() / "flyvis" / "data")
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "results" / "flow" / "0000" / "000").is_dir():
            os.environ["FLYVIS_ROOT_DIR"] = str(resolved)
            return resolved
    raise FileNotFoundError(
        "FlyVis pretrained checkpoints are unavailable. Install the pinned FlyVis package, "
        "run `flyvis download-pretrained`, and set FLYVIS_ROOT_DIR to the FlyVis data directory."
    )


def _check_pinned_runtime(data_root: Path):
    import flyvis
    import torch
    from flyvis.connectome.connectome import ReceptiveFields
    from flyvis.network import NetworkView
    from flyvis.utils.activity_utils import SourceCurrentView

    dist = importlib_metadata.distribution("flyvis")
    try:
        direct = json.loads((dist._path / "direct_url.json").read_text())
    except (AttributeError, OSError, json.JSONDecodeError):
        direct = {}
    commit = direct.get("vcs_info", {}).get("commit_id")
    if flyvis.__version__ != FLYVIS_VERSION or commit != FLYVIS_COMMIT:
        raise RuntimeError(
            f"Pinned FlyVis required: version {FLYVIS_VERSION}, commit {FLYVIS_COMMIT}; "
            f"found version {getattr(flyvis, '__version__', 'unknown')}, commit {commit!r}."
        )
    record_hashes_ok = True
    for package_file in dist.files or ():
        expected_hash = package_file.hash
        if expected_hash is None:
            continue
        installed_file = dist.locate_file(package_file)
        if not installed_file.is_file():
            record_hashes_ok = False
            break
        actual_hash = base64.urlsafe_b64encode(
            hashlib.new(expected_hash.mode, installed_file.read_bytes()).digest()
        ).decode().rstrip("=")
        if actual_hash != expected_hash.value:
            record_hashes_ok = False
            break
    if not record_hashes_ok:
        raise RuntimeError("Installed FlyVis files do not match the distribution RECORD hashes.")
    if not hasattr(flyvis.Network, "current_response"):
        raise RuntimeError("Pinned FlyVis Network.current_response API is unavailable.")
    if SourceCurrentView is None or ReceptiveFields is None:
        raise RuntimeError("Pinned FlyVis source-current/connectome APIs are unavailable.")

    for model_id in MODEL_IDS:
        model_path = data_root / "results" / model_id
        if not model_path.is_dir() or not (model_path / "best_chkpt").is_file():
            raise FileNotFoundError(f"Missing official best checkpoint: {model_path / 'best_chkpt'}")
        view = NetworkView(model_id, root_dir=data_root / "results")
        net = view.init_network("best")
        edges = net.connectome.edges.to_df().reset_index(drop=True)
        valid = True
        for sources, targets in ((ON_SOURCES, ON_TARGETS), (OFF_SOURCES, OFF_TARGETS)):
            for target in targets:
                receptive = ReceptiveFields(target, edges)
                valid &= all(source in receptive.source_types and len(receptive[source]) > 0 for source in sources)
        if net.training or not hasattr(net, "current_response") or not valid:
            raise RuntimeError(f"Checkpoint/connectome preflight failed for {model_id}.")
        del net, view
    return flyvis, torch, NetworkView, ReceptiveFields, SourceCurrentView


def _acquire_to(output_path: Path) -> None:
    """Run exactly 72 fixed current_response integrations and save target rows."""
    import numpy as np
    import pandas as pd
    import torch

    from flyvis_midd.stimuli import build_3midd_sequences, render_sector_image
    from flyvis.datasets.rendering import BoxEye

    data_root = _external_data_root()
    flyvis, torch, NetworkView, ReceptiveFields, SourceCurrentView = _check_pinned_runtime(data_root)
    phase_manifest_path = ROOT / "data/precomputed/fig04/phase_stimulus_manifest.csv"
    # The B1 table is provenance context only; it is not read by the D1 reducer.
    provenance_input = ROOT / "data/precomputed/supplementary/biological_constraints/B1/t4t5_activity_model_phase.csv"
    for required in (phase_manifest_path, provenance_input):
        if not required.is_file():
            raise FileNotFoundError(f"Required committed provenance input is missing: {required}")
    phase_manifest = pd.read_csv(phase_manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite acquisition output: {output_path}")

    class OneMovieDataset:
        def __init__(self, movie):
            self.movie = movie
            self.dt = None
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return {"lum": self.movie}

    def make_movie(phase_deg: float, condition: str):
        sequence = build_3midd_sequences(48)[condition]
        image, _ = render_sector_image(sequence, image_size=IMAGE_SIZE,
                                       inner_diameter=INNER_DIAMETER,
                                       outer_diameter=OUTER_DIAMETER,
                                       phase_deg=float(phase_deg))
        phase_index = int(round(float(phase_deg) / 1.25))
        key = f"3MIDD_048_p{phase_index:02d}__{condition}"
        row = phase_manifest.loc[phase_manifest.condition.eq(key)]
        if len(row) != 1 or hashlib.sha256(image.tobytes()).hexdigest() != row.iloc[0].image_sha256:
            raise AssertionError(f"Stimulus differs from the saved canonical image: {key}")
        movie = np.repeat(image[None, None, :, :], FRAMES, axis=1).astype(np.float32) / 255.0
        rendered = BoxEye(extent=BOXEYE_EXTENT, kernel_size=BOXEYE_KERNEL_SIZE)(torch.as_tensor(movie))
        if tuple(rendered.shape) != (1, FRAMES, 1, 721):
            raise ValueError(f"Unexpected BoxEye output shape: {tuple(rendered.shape)}")
        return OneMovieDataset(rendered[0])

    def reduce_target_rows(net, current, model_id: str, phase_index: int,
                           phase_deg: float, condition: str):
        edges = net.connectome.edges.to_df().reset_index(drop=True)
        nodes = net.connectome.nodes.to_df().reset_index(drop=True)
        if current.shape != (FRAMES, len(edges)) or not np.isfinite(current).all():
            raise ValueError(f"Unexpected/nonfinite current tensor: {current.shape}")
        if sorted(nodes["index"].astype(int)) != list(range(len(nodes))):
            raise ValueError("Unexpected node-index mapping.")
        source_map = {**{target: ON_SOURCES for target in ON_TARGETS},
                      **{target: OFF_SOURCES for target in OFF_TARGETS}}
        rows = []
        window = current[ANALYSIS_START:ANALYSIS_STOP].astype(np.float64, copy=False)
        for target, sources in source_map.items():
            target_nodes = nodes.loc[nodes.type.eq(target)].sort_values("index")
            node_ids = target_nodes["index"].to_numpy(dtype=int)
            local_index = {int(node_id): i for i, node_id in enumerate(node_ids)}
            receptive = ReceptiveFields(target, edges)
            view = SourceCurrentView(receptive, current)
            source_means = {}
            for source in sources:
                if source not in view.source_types:
                    raise ValueError(f"Missing source {source}->{target}")
                edge_indices = np.asarray(receptive[source].index, dtype=int)
                observed = np.asarray(view[source])
                if observed.shape != (FRAMES, len(edge_indices)) or not np.array_equal(observed, current[:, edge_indices]):
                    raise AssertionError("SourceCurrentView edge-index mapping changed.")
                selected_edges = edges.iloc[edge_indices]
                mask = (selected_edges.source_type.eq(source) & selected_edges.target_type.eq(target)).to_numpy()
                edge_indices = edge_indices[mask]
                selected_edges = edges.iloc[edge_indices]
                if not len(edge_indices):
                    raise ValueError(f"No source edges {source}->{target}")
                target_positions = selected_edges.target_index.map(local_index)
                if target_positions.isna().any():
                    raise ValueError("Target node mapping failed.")
                summed = np.zeros((ANALYSIS_STOP - ANALYSIS_START, len(node_ids)), dtype=np.float64)
                target_positions = target_positions.to_numpy(dtype=int)
                for frame_index, frame in enumerate(window):
                    np.add.at(summed[frame_index], target_positions, frame[edge_indices])
                source_means[source] = summed.mean(axis=0)
            selected_sum = np.sum(np.stack([source_means[source] for source in sources]), axis=0)
            for source in (*sources, "__selected_sources_sum__"):
                values = selected_sum if source == "__selected_sources_sum__" else source_means[source]
                rows.extend({
                    "target_type": target,
                    "source_type": source,
                    "target_node_id": int(node_id),
                    "mean_current_frames_10_49": float(value),
                    "model_id": model_id,
                    "phase_index": phase_index,
                    "phase_deg": phase_deg,
                    "condition": condition,
                } for node_id, value in zip(node_ids, values))
        return rows

    if not output_path.parent.is_dir():
        raise FileNotFoundError(output_path.parent)
    for model_id in MODEL_IDS:
        view = NetworkView(model_id, root_dir=data_root / "results")
        network = view.init_network("best")
        network.eval()
        for phase_index, phase_deg in enumerate(PHASES_DEG):
            for condition in CONDITIONS:
                dataset = make_movie(phase_deg, condition)
                result = list(network.current_response(dataset, DT, t_pre=0.0, t_fade_in=FADE_SECONDS))
                if len(result) != 1:
                    raise RuntimeError("Expected one stimulus response per integration.")
                _, _, current = map(np.asarray, result[0])
                rows = reduce_target_rows(network, current, model_id, phase_index, phase_deg, condition)
                if not rows:
                    raise RuntimeError("The current reducer returned no target-node values.")
                # Preserve archived row order and pandas CSV serialization.
                pd.DataFrame(rows, columns=COLUMNS).to_csv(
                    output_path, mode="a", header=not output_path.exists(), index=False
                )
                del dataset, result, current, rows
        del network, view
    if output_path.exists():
        rows, columns = inspect_csv(output_path)
        if rows != EXPECTED_ROWS or columns != COLUMNS:
            raise RuntimeError(f"Generated output violates contract: {rows} rows, {columns} columns")


def regenerate_to_temp() -> dict[str, object]:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="d1_target_node_distribution_", suffix=".csv")
    os.close(fd)
    temp_path = Path(name)
    temp_path.unlink()
    try:
        _acquire_to(temp_path)
        rows, columns = inspect_csv(temp_path)
        digest = sha256_file(temp_path)
        if columns != COLUMNS or rows != EXPECTED_ROWS:
            raise RuntimeError("Temporary regeneration failed the output contract.")
        if OUTPUT.exists():
            reference = verify_existing(OUTPUT)
            if digest == reference["sha256"]:
                content_match = True
                comparison = "byte-identical"
            else:
                content_match = compare_csv_content(OUTPUT, temp_path)
                comparison = "numeric/content identical" if content_match else "content differs"
            return {"temporary_path": str(temp_path), "rows": rows, "columns": columns,
                    "sha256": digest, "existing_sha256": reference["sha256"],
                    "existing_content_match": content_match, "comparison": comparison,
                    "canonical_replaced": False}
        return {"temporary_path": str(temp_path), "rows": rows, "columns": columns,
                "sha256": digest, "existing_content_match": None,
                "comparison": "no canonical file existed for comparison", "canonical_replaced": False}
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def generate_canonical_if_missing() -> dict[str, object]:
    """Create the canonical file only when absent and only on exact hash match."""
    if OUTPUT.exists():
        raise FileExistsError(
            "Canonical output already exists. It was not overwritten; use --verify-existing "
            "or --regenerate-to-temp for a non-destructive comparison."
        )
    fd, name = tempfile.mkstemp(prefix="d1_target_node_distribution_", suffix=".csv")
    os.close(fd)
    temp_path = Path(name)
    temp_path.unlink()
    try:
        _acquire_to(temp_path)
        info = verify_existing(temp_path, require_canonical_hash=True)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_path, OUTPUT)
        return {**info, "path": str(OUTPUT), "created_canonical": True}
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--verify-existing", action="store_true",
                         help="verify the committed canonical CSV only (default; no inference)")
    actions.add_argument("--regenerate-to-temp", action="store_true",
                         help="run the fixed 72 integrations to a temporary CSV and compare without replacing canonical data")
    actions.add_argument("--generate-canonical-if-missing", action="store_true",
                         help="create canonical output only if absent and generated bytes match its recorded SHA-256")
    args = parser.parse_args(argv)
    if args.regenerate_to_temp:
        result = regenerate_to_temp()
    elif args.generate_canonical_if_missing:
        result = generate_canonical_if_missing()
    else:
        result = verify_existing()
    print(json.dumps(result, indent=2, default=list))
    if args.regenerate_to_temp and result.get("existing_content_match") is False:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
