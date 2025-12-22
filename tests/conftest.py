import pytest


@pytest.fixture
def sample_image(tmp_path):
    p = tmp_path / "img.png"
    p.write_text("")
    return str(p)
