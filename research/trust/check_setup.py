"""Read-only checks of files, hashes, and split membership."""

import hashlib
import json
import re
from itertools import combinations
from pathlib import Path


def filename_group(path):
    stem = Path(path.replace("\\", "/")).stem
    match = re.match(r"^(person\d+)(?:_|$)", stem, re.I)
    if match:
        return match.group(1).lower()

    match = re.match(r"^((?:NORMAL2-)?IM-\d+)(?:-|$)", stem, re.I)
    return match.group(1).upper() if match else None


def main():
    split = Path("data/processed/splits.json")
    model = Path("models/classifier.pt")

    for path in (split, model):
        if not path.is_file():
            raise FileNotFoundError(str(path))
        print(path, hashlib.sha256(path.read_bytes()).hexdigest())

    data = json.loads(split.read_text())
    if data["classes"] != ["NORMAL", "PNEUMONIA"]:
        raise ValueError("Unexpected class order")

    filenames = {}
    patients = {}
    missing_ids = 0
    unmatched = 0

    for name in ("train", "val", "test"):
        rows = data[name]
        if not rows:
            raise ValueError(f"Empty split: {name}")

        filenames[name] = set()
        patients[name] = set()
        split_missing_ids = 0
        split_unmatched = 0

        for row in rows:
            path = Path(row["path"])
            if not path.is_file():
                raise FileNotFoundError(str(path))
            if row["label"] not in (0, 1):
                raise ValueError(f"Invalid label: {row}")

            group = filename_group(row["path"])
            if group is None:
                split_unmatched += 1
            else:
                filenames[name].add(group)

            pid = row.get("patient")
            if pid is None or str(pid).strip() == "":
                pid = row.get("patient_id")

            if pid is None or str(pid).strip() == "":
                split_missing_ids += 1
            else:
                patients[name].add(str(pid).strip())

        missing_ids += split_missing_ids
        unmatched += split_unmatched
        print(
            f"{name}: {len(rows)} images; "
            f"{split_missing_ids} missing patient IDs; "
            f"{split_unmatched} unmatched filenames"
        )

    for a, b in combinations(("train", "val", "test"), 2):
        overlap = filenames[a] & filenames[b]
        print(f"{a}/{b}: {len(overlap)} overlapping filename groups")
        if overlap:
            raise ValueError(
                f"Filename-group overlap: {a}/{b}: {sorted(overlap)[:5]}"
            )

        overlap = patients[a] & patients[b]
        if overlap:
            raise ValueError(
                f"Recorded patient-ID overlap: {a}/{b}: "
                f"{sorted(overlap)[:5]}"
            )

    if unmatched:
        print("Filename audit INCOMPLETE: some filenames were unmatched.")
    else:
        print("Filename-group audit passed.")

    if missing_ids:
        print("Patient-ID audit INCOMPLETE: some or all IDs are missing.")
    else:
        print("Recorded patient-ID audit passed.")

    print("Filename groups alone do not establish verified patient identity.")
    print("Compare hashes with the team's agreed baseline before shared evaluation.")


if __name__ == "__main__":
    main()