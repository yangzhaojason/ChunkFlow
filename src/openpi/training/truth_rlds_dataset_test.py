import numpy as np
import pytest

from openpi.training.truth_rlds_dataset import paired_episode_action_windows


def test_paired_episode_windows_require_two_full_chunks():
    actions = np.arange(8, dtype=np.float32)[:, None]

    starts, previous, current, history, history_mask = paired_episode_action_windows(
        actions,
        horizon=4,
        stride=2,
        history_length=2,
    )

    np.testing.assert_array_equal(starts, [0, 2])
    np.testing.assert_array_equal(previous[:, :, 0], [[0, 1, 2, 3], [2, 3, 4, 5]])
    np.testing.assert_array_equal(current[:, :, 0], [[2, 3, 4, 5], [4, 5, 6, 7]])
    np.testing.assert_array_equal(history[:, :, 0], [[0, 1], [2, 3]])
    np.testing.assert_array_equal(history_mask, np.ones((2, 2), dtype=bool))


def test_paired_episode_windows_do_not_repeat_terminal_actions():
    starts, previous, current, history, history_mask = paired_episode_action_windows(
        np.arange(5, dtype=np.float32)[:, None],
        horizon=4,
        stride=2,
        history_length=1,
    )

    assert starts.shape == (0,)
    assert previous.shape == (0, 4, 1)
    assert current.shape == (0, 4, 1)
    assert history.shape == (0, 1, 1)
    assert history_mask.shape == (0, 1)


@pytest.mark.parametrize(
    ("horizon", "stride", "history_length"),
    [(0, 1, 0), (4, 0, 0), (4, 2, 3), (4, 5, 1), (4, 2, -1)],
)
def test_paired_episode_window_configuration_is_validated(horizon, stride, history_length):
    with pytest.raises(ValueError, match=r"horizon|stride|history_length"):
        paired_episode_action_windows(
            np.zeros((8, 2), dtype=np.float32),
            horizon=horizon,
            stride=stride,
            history_length=history_length,
        )
