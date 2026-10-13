import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def readme():
    return (ROOT / "README.md").read_text(encoding="utf-8").replace("\r\n", "\n")


def test_readme_has_no_hugging_face_setup():
    text = readme()
    assert text.startswith("# Portfolio Risk Analyzer")
    assert "huggingface" not in text.lower()


def test_readme_explains_running_locally_and_with_docker():
    text = readme()
    assert "streamlit run app.py" in text
    assert "docker run" in text


def test_referenced_screenshots_exist():
    paths = re.findall(r"\((docs/screenshots/[^)]+\.png)\)", readme())
    assert len(paths) == 4
    for path in paths:
        assert (ROOT / path).exists(), path
