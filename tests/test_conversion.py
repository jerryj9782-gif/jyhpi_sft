import json
from pathlib import Path
import numpy as np
import pytest
import torch
from navpi.config import ModelConfig
from navpi.convert_checkpoint import convert_parameters
from navpi.model import NavPi


@pytest.mark.parametrize("wrapped", [False, True])
def test_official_jax_layout_maps_to_complete_tiny_model(wrapped):
    # Names and full-size shapes from the official cached pi05_base Orbax metadata.
    # Shrink dimensions to test every mapping on CPU, without loading real weights.
    shapes = json.loads((Path(__file__).parent / "fixtures/pi05_jax_shapes.json").read_text())
    replacements = {1152: 32, 2048: 32, 1024: 16, 27: 2, 18: 2, 8: 2, 16: 4,
                    72: 8, 256: 16, 4304: 64, 16384: 64, 4096: 32, 3072: 48, 257152: 128, 32: 3}
    params = {}
    rng = np.random.default_rng(12)
    for key, shape in shapes.items():
        small = [replacements.get(dim, dim) for dim in shape]
        if key.endswith("img/pos_embedding"):
            small = [1, 4, 32]
        value = rng.standard_normal(small).astype(np.float32)
        node = params
        parts = key.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = {"value": value} if wrapped else value
    converted = convert_parameters(params, ModelConfig.tiny())
    expected = NavPi(ModelConfig.tiny()).state_dict()
    assert converted.keys() == expected.keys()
    for key, tensor in converted.items():
        assert tensor.shape == expected[key].shape, (key, tensor.shape, expected[key].shape)
    source = params["time_mlp_in"]["kernel"]
    if wrapped: source = source["value"]
    torch.testing.assert_close(converted["time_mlp_in.weight"], torch.from_numpy(source.T))
    # The directory name is empty: adaptive norm selection must use pi05=True.
    assert "paligemma_with_expert.gemma_expert.model.norm.dense.weight" in converted
