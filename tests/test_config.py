import pytest
from pydantic import ValidationError

from sceneproof.edge.config import EdgeConfig, load_config


def test_example_config_loads_and_hashes():
    cfg = load_config("configs/edge.yaml")
    assert cfg.camera("cam-worker") is not None
    z = cfg.camera("cam-worker").zones[0].to_spec()
    assert len(z.polygon) == 5 and z.dwell_seconds == 5
    assert len(cfg.config_hash()) == 12
    assert cfg.config_hash() == load_config("configs/edge.yaml").config_hash()


def test_polygon_must_be_normalized():
    with pytest.raises(ValidationError):
        EdgeConfig.model_validate({"cameras": [{"camera_id": "c", "source": "x", "zones": [
            {"zone_id": "z", "name": "z", "polygon": [[0, 0], [2, 0], [1, 1]]}]}]})
