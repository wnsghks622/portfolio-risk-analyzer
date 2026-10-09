import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def readme():
    return (ROOT / "README.md").read_text(encoding="utf-8").replace("\r\n", "\n")


def frontmatter():
    text = readme()
    assert text.startswith("---\n")
    block = text.split("---\n")[1]
    return dict(line.split(": ", 1) for line in block.strip().splitlines())


def test_frontmatter_matches_docker_space():
    meta = frontmatter()
    assert meta["sdk"] == "docker"
    assert meta["app_port"] == "7860"
    assert len(meta["short_description"]) <= 60


def test_referenced_screenshots_exist():
    paths = re.findall(r"\((docs/screenshots/[^)]+\.png)\)", readme())
    assert len(paths) == 4
    for path in paths:
        assert (ROOT / path).exists(), path
