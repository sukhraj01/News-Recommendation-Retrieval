"""Generates the tiny hand-built MIND/EB-NeRD fixture zips under tests/fixtures/.

Run once to (re)produce the committed fixtures if their content needs to
change. Fixture zips mirror real MIND/EB-NeRD internal structure exactly
(folder name == zip stem for MIND; top-level articles.parquet + per-split
train/validation dirs for EB-NeRD, including __MACOSX/ junk) so the same
loader code path that reads real data reads these fixtures unmodified.

Covers: empty MIND abstract, empty MIND history string, a MIND user with two
impression rows sharing identical history (static-history dedup check), a
MINDlarge_test-style unlabeled-candidates bundle, and an EB-NeRD null vs
non-null context article_id (front-page derivation).
"""
import io
import zipfile
from pathlib import Path

import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def _write_mind_zip(zip_name: str, news_rows: list[str], behaviors_rows: list[str]) -> None:
    folder = zip_name  # convention: internal folder name == zip stem
    path = FIXTURES_DIR / f"{zip_name}.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{folder}/news.tsv", "\n".join(news_rows) + "\n")
        z.writestr(f"{folder}/behaviors.tsv", "\n".join(behaviors_rows) + "\n")


def build_mind_labeled_fixture() -> None:
    news_rows = [
        "\t".join(["N1", "news", "newsworld", "Title One", "Some abstract text.",
                   "http://x/1", "[]", "[]"]),
        "\t".join(["N2", "news", "newsworld", "Title Two", "",
                   "http://x/2", "[]", "[]"]),  # empty abstract
        "\t".join(["N3", "sports", "football", "Title Three", "Sports abstract.",
                   "http://x/3",
                   '[{"Label": "Team A", "Type": "O", "WikidataId": "Q1", '
                   '"Confidence": 1.0, "OccurrenceOffsets": [0], "SurfaceForms": ["Team A"]}]',
                   "[]"]),
        "\t".join(["N4", "lifestyle", "royals", "Title Four", "Royal abstract.",
                   "http://x/4", "[]", "[]"]),
    ]
    behaviors_rows = [
        "\t".join(["1", "U1", "11/09/2019 8:00:00 AM", "N1 N2", "N3-1 N4-0"]),
        "\t".join(["2", "U2", "11/09/2019 9:00:00 AM", "", "N1-0 N2-1"]),  # empty history
        # U1 again, identical history -> static-history dedup check
        "\t".join(["3", "U1", "11/10/2019 10:00:00 AM", "N1 N2", "N4-1 N3-0"]),
    ]
    _write_mind_zip("MINDsmall_train_sample", news_rows, behaviors_rows)


def build_mind_unlabeled_fixture() -> None:
    news_rows = [
        "\t".join(["N1", "news", "newsworld", "Title One", "Some abstract text.",
                   "http://x/1", "[]", "[]"]),
        "\t".join(["N2", "news", "newsworld", "Title Two", "Abstract two.",
                   "http://x/2", "[]", "[]"]),
    ]
    behaviors_rows = [
        # MINDlarge_test style: candidate tokens have no "-label" suffix
        "\t".join(["1", "U3", "11/16/2019 8:00:00 AM", "N1 N2", "N1 N2"]),
        "\t".join(["2", "U4", "11/16/2019 9:00:00 AM", "", "N1"]),
    ]
    _write_mind_zip("MINDlarge_test_sample", news_rows, behaviors_rows)


def build_ebnerd_fixture() -> None:
    articles = pd.DataFrame({
        "article_id": [101, 102, 103],
        "title": ["Article A", "Article B", "Article C"],
        "subtitle": ["Sub A", "Sub B", "Sub C"],
        "body": ["Body A", "Body B", "Body C"],
        "published_time": pd.to_datetime(["2023-04-01", "2023-04-02", "2023-04-03"]),
        "category_str": ["nyheder", "sport", "nyheder"],
        "subcategory": [[1], [2, 3], [1]],
        "ner_clusters": [["Foo"], [], ["Bar"]],
        "entity_groups": [["PER"], [], ["ORG"]],
        "topics": [["politik"], ["fodbold"], []],
    })

    def _behaviors(split_impression_ids):
        return pd.DataFrame({
            "impression_id": split_impression_ids,
            "article_id": [None, 103],  # context article: null -> front page, non-null -> not
            "impression_time": pd.to_datetime(["2023-05-19 10:00:00", "2023-05-20 11:00:00"]),
            "read_time": [12.0, 30.0],
            "scroll_percentage": [50.0, 100.0],
            "article_ids_inview": [[101, 102], [102, 103]],
            "article_ids_clicked": [[101], [103]],
            "user_id": [1, 2],
            "session_id": [11, 12],
        })

    def _history():
        return pd.DataFrame({
            "user_id": [1, 2],
            "article_id_fixed": [[101, 102], [103]],
            "impression_time_fixed": [
                list(pd.to_datetime(["2023-04-27 09:00:00", "2023-04-27 10:00:00"])),
                list(pd.to_datetime(["2023-04-28 09:00:00"])),
            ],
            "read_time_fixed": [[5.0, 6.0], [7.0]],
        })

    path = FIXTURES_DIR / "ebnerd_demo_sample.zip"
    with zipfile.ZipFile(path, "w") as z:
        buf = io.BytesIO()
        articles.to_parquet(buf, index=False)
        z.writestr("articles.parquet", buf.getvalue())
        z.writestr("__MACOSX/._articles.parquet", b"junk-metadata-not-real-parquet")

        for split, ids in (("train", [1, 2]), ("validation", [3, 4])):
            buf = io.BytesIO()
            _behaviors(ids).to_parquet(buf, index=False)
            z.writestr(f"{split}/behaviors.parquet", buf.getvalue())

            buf = io.BytesIO()
            _history().to_parquet(buf, index=False)
            z.writestr(f"{split}/history.parquet", buf.getvalue())

            z.writestr(f"__MACOSX/{split}/._behaviors.parquet", b"junk")


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    build_mind_labeled_fixture()
    build_mind_unlabeled_fixture()
    build_ebnerd_fixture()
    print(f"Fixtures written to {FIXTURES_DIR}")
