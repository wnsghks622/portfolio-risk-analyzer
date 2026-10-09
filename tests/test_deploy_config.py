from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(name):
    return (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")


def test_dockerfile_serves_streamlit_on_7860_with_python_312():
    dockerfile = read("Dockerfile")
    assert dockerfile.startswith("FROM python:3.12-slim")
    assert "--server.port=7860" in dockerfile
    assert "--server.address=0.0.0.0" in dockerfile
    assert "EXPOSE 7860" in dockerfile


def test_dockerignore_keeps_sample_data_in_image():
    ignored = read(".dockerignore").split()
    assert "data" not in ignored
    assert ".venv" in ignored


def test_ci_checks_out_lfs_and_deploys_only_after_tests():
    ci = read(".github/workflows/ci.yml")
    assert "lfs: true" in ci
    assert "needs: test" in ci
    assert 'python-version: "3.12"' in ci
