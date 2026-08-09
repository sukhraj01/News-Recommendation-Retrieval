"""Raw data presence check and MIND download.

`make data`'s default path only calls `ensure_raw_data()` — a presence
check that fails loudly with instructions rather than downloading, since
this project's raw bundles are already on disk. `download_mind_bundle()` is
a real, callable function (not a stub) for the case where they're not, using
the URLs the actively-maintained recommenders-team/recommenders MIND loader
uses (verified live and resolving as of this writing:
https://github.com/recommenders-team/recommenders/blob/main/recommenders/datasets/mind.py).

EB-NeRD is deliberately NOT auto-downloadable here: its distribution is
consent/terms-gated (registration required), so no URL can be scripted
around that gate — it must be downloaded manually per the assignment's
instructions.
"""
from pathlib import Path

from src.utils.config import EBNERD_RAW_DIR, MIND_RAW_DIR

_HF_MIND_BASE = "https://huggingface.co/datasets/Recommenders/MIND/resolve/main"

MIND_BUNDLE_URLS = {
    "MINDsmall_train.zip": f"{_HF_MIND_BASE}/MINDsmall_train.zip",
    "MINDsmall_dev.zip": f"{_HF_MIND_BASE}/MINDsmall_dev.zip",
    "MINDlarge_train.zip": f"{_HF_MIND_BASE}/MINDlarge_train.zip",
    "MINDlarge_dev.zip": f"{_HF_MIND_BASE}/MINDlarge_dev.zip",
    "MINDlarge_test.zip": f"{_HF_MIND_BASE}/MINDlarge_test.zip",
}


class RawDataMissingError(Exception):
    pass


def ensure_raw_data(
    mind_bundles: tuple[str, ...] = ("MINDsmall_train.zip", "MINDsmall_dev.zip"),
    ebnerd_bundles: tuple[str, ...] = ("ebnerd_demo.zip",),
) -> None:
    """Fail loudly, with instructions, if required raw bundles are absent.

    Does not download automatically — this is a deliberate scope decision
    (see module docstring), not an oversight.
    """
    missing: list[str] = []

    for name in mind_bundles:
        if not (MIND_RAW_DIR / name).exists():
            missing.append(str(MIND_RAW_DIR / name))
    for name in ebnerd_bundles:
        if not (EBNERD_RAW_DIR / name).exists():
            missing.append(str(EBNERD_RAW_DIR / name))

    if missing:
        raise RawDataMissingError(
            "Missing raw data files:\n  - " + "\n  - ".join(missing) +
            "\n\nMIND bundles can be fetched with "
            "src.pipeline.download.download_mind_bundle(). "
            "EB-NeRD must be downloaded manually (registration-gated) from "
            "https://recsys.eb.dk/ and placed under data/raw/ebnerd/."
        )


def download_mind_bundle(filename: str, dest_dir: Path = MIND_RAW_DIR) -> Path:
    """Download one MIND zip (e.g. 'MINDsmall_train.zip') if not already present."""
    import urllib.request

    if filename not in MIND_BUNDLE_URLS:
        raise ValueError(
            f"Unknown MIND bundle {filename!r}; known bundles: "
            f"{sorted(MIND_BUNDLE_URLS)}"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    if dest_path.exists():
        return dest_path

    url = MIND_BUNDLE_URLS[filename]
    urllib.request.urlretrieve(url, dest_path)
    return dest_path
