import numpy as np
from wavelet import w2d


def test_the_output_is_a_single_channel_image_of_the_same_size() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    result = w2d(image)

    assert result.shape == (32, 32)
    assert result.dtype == np.uint8


def test_a_flat_image_has_no_edges() -> None:
    assert w2d(np.full((32, 32, 3), 128, dtype=np.uint8)).max() == 0


def test_an_edge_in_the_image_produces_a_response() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 15:] = 255

    assert w2d(image, level=5).max() > 0
