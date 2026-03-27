import subprocess
import pytest
from pathlib import Path


def _notebooks():
    root = Path(__file__).parent
    # Skip the data-download notebook — requires network access
    skip = {"00.1 - Mona Loa download.py"}
    return sorted(nb for nb in root.glob("ch*/*.py") if nb.name not in skip)


@pytest.mark.parametrize("notebook", _notebooks(), ids=lambda p: str(p.relative_to(Path(__file__).parent)))
def test_notebook_runs(notebook, tmp_path):
    result = subprocess.run(
        ["uv", "run", "marimo", "export", "html", str(notebook), "-o", str(tmp_path / "out.html")],
        capture_output=True,
        text=True,
        cwd=notebook.parent,  # ensures relative data paths (co2.npy, faithful.csv, etc.) resolve
    )
    assert result.returncode == 0, f"{notebook}:\n{result.stderr}"
