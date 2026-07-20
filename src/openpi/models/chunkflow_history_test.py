import jax
import jax.numpy as jnp
import pytest

from openpi.models.chunkflow_history import corrupt_history
from openpi.models.chunkflow_history import scheduled_sampling_alpha
from openpi.models.chunkflow_history import validate_history


def test_alpha_ramps_after_warmup():
    assert scheduled_sampling_alpha(4, warmup_steps=5, ramp_steps=10, max_alpha=0.8) == 0.0
    assert scheduled_sampling_alpha(10, warmup_steps=5, ramp_steps=10, max_alpha=0.8) == pytest.approx(0.4)
    assert scheduled_sampling_alpha(20, warmup_steps=5, ramp_steps=10, max_alpha=0.8) == pytest.approx(0.8)


def test_alpha_does_not_underflow_for_unsigned_step_before_warmup():
    value = scheduled_sampling_alpha(
        jnp.array(0, dtype=jnp.uint32),
        warmup_steps=5,
        ramp_steps=10,
        max_alpha=0.8,
    )

    assert value == 0.0


def test_alpha_schedule_is_jittable_with_static_configuration():
    compiled = jax.jit(
        scheduled_sampling_alpha,
        static_argnames=("warmup_steps", "ramp_steps", "max_alpha"),
    )

    value = compiled(
        jnp.array(10),
        warmup_steps=5,
        ramp_steps=10,
        max_alpha=0.8,
    )

    assert value == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("step", "warmup_steps", "expected"),
    [
        (jnp.array(16_777_218, dtype=jnp.uint32), 16_777_217, 0.1),
        (jnp.array(4_294_967_295, dtype=jnp.uint32), 4_294_967_290, 0.5),
    ],
)
def test_alpha_preserves_large_integer_step_differences(step, warmup_steps, expected):
    eager = scheduled_sampling_alpha(
        step,
        warmup_steps=warmup_steps,
        ramp_steps=10,
        max_alpha=1.0,
    )
    compiled = jax.jit(
        scheduled_sampling_alpha,
        static_argnames=("warmup_steps", "ramp_steps", "max_alpha"),
    )
    traced = compiled(
        step,
        warmup_steps=warmup_steps,
        ramp_steps=10,
        max_alpha=1.0,
    )

    assert eager == pytest.approx(expected)
    assert traced == pytest.approx(expected)


def test_alpha_normalizes_large_python_integer_or_rejects_out_of_range():
    value = scheduled_sampling_alpha(
        4_294_967_295,
        warmup_steps=4_294_967_290,
        ramp_steps=10,
        max_alpha=1.0,
    )

    assert value == pytest.approx(0.5)
    with pytest.raises(ValueError, match="step.*uint32"):
        scheduled_sampling_alpha(
            4_294_967_296,
            warmup_steps=0,
            ramp_steps=10,
            max_alpha=1.0,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warmup_steps": -1, "ramp_steps": 10, "max_alpha": 0.5},
        {"warmup_steps": 0, "ramp_steps": 0, "max_alpha": 0.5},
        {"warmup_steps": 0, "ramp_steps": 10, "max_alpha": -0.1},
        {"warmup_steps": 0, "ramp_steps": 10, "max_alpha": 1.1},
        {"warmup_steps": 0, "ramp_steps": 10, "max_alpha": float("nan")},
    ],
)
def test_alpha_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError, match="scheduled-sampling"):
        scheduled_sampling_alpha(0, **kwargs)


def test_corruption_zeros_dropped_vectors_but_preserves_padding_mask():
    clean = jnp.ones((1, 3, 2))
    valid = jnp.array([[False, True, True]])
    predicted = 3 * clean

    values, output_mask = corrupt_history(
        jax.random.key(0),
        clean,
        valid,
        predicted,
        noise_std=0.0,
        dropout_probability=1.0,
        alpha=0.0,
    )

    assert jnp.array_equal(output_mask, valid)
    assert jnp.all(values == 0)


def test_scheduled_sampling_is_convex_interpolation():
    clean = jnp.ones((1, 2, 1))
    predicted = 5 * clean
    valid = jnp.ones((1, 2), dtype=bool)

    values, _ = corrupt_history(
        jax.random.key(1),
        clean,
        valid,
        predicted,
        noise_std=0.0,
        dropout_probability=0.0,
        alpha=0.25,
    )

    assert jnp.allclose(values, 2.0)


def test_uncovered_history_positions_keep_corrupted_demo_values():
    clean = jnp.ones((1, 4, 1))
    predicted = 5 * clean
    coverage = jnp.array([[False, False, True, True]])
    values, _ = corrupt_history(
        jax.random.key(0),
        clean,
        jnp.ones((1, 4), dtype=bool),
        predicted,
        prediction_mask=coverage,
        noise_std=0.0,
        dropout_probability=1.0,
        alpha=1.0,
    )
    assert values.tolist() == [[[0.0], [0.0], [5.0], [5.0]]]


def test_gaussian_noise_matches_split_rng_exactly():
    key = jax.random.key(7)
    clean = jnp.ones((1, 2, 2))
    valid = jnp.ones((1, 2), dtype=bool)
    noise_key, _ = jax.random.split(key)
    expected = clean + 0.25 * jax.random.normal(noise_key, clean.shape, dtype=clean.dtype)

    values, _ = corrupt_history(
        key,
        clean,
        valid,
        clean,
        noise_std=0.25,
        dropout_probability=0.0,
        alpha=0.0,
    )

    assert jnp.array_equal(values, expected)


def test_partial_dropout_is_per_position_and_precedes_interpolation():
    key = jax.random.key(8)
    clean = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3) + 1
    predicted = clean + 100
    valid = jnp.ones((2, 4), dtype=bool)
    _, dropout_key = jax.random.split(key)
    dropped = jax.random.bernoulli(dropout_key, p=0.5, shape=valid.shape)
    assert jnp.any(dropped)
    assert jnp.any(~dropped)
    corrupted = jnp.where(dropped[..., None], 0.0, clean)
    expected = 0.75 * corrupted + 0.25 * predicted

    values, output_mask = corrupt_history(
        key,
        clean,
        valid,
        predicted,
        noise_std=0.0,
        dropout_probability=0.5,
        alpha=0.25,
    )

    assert jnp.array_equal(values, expected)
    assert jnp.array_equal(output_mask, valid)


def test_padding_positions_stay_zero_with_noise_and_prediction():
    clean = jnp.ones((1, 3, 2))
    predicted = 9 * clean
    valid = jnp.array([[False, True, False]])

    values, output_mask = corrupt_history(
        jax.random.key(2),
        clean,
        valid,
        predicted,
        noise_std=3.0,
        dropout_probability=0.0,
        alpha=0.5,
    )

    assert jnp.array_equal(output_mask, valid)
    assert jnp.all(values[:, (0, 2), :] == 0)


def test_predicted_history_is_stop_gradient():
    clean = jnp.ones((1, 2, 1))
    valid = jnp.ones((1, 2), dtype=bool)

    gradient = jax.grad(
        lambda predicted: jnp.sum(
            corrupt_history(
                jax.random.key(3),
                clean,
                valid,
                predicted,
                noise_std=0.0,
                dropout_probability=0.0,
                alpha=0.5,
            )[0]
        )
    )(4 * clean)

    assert jnp.array_equal(gradient, jnp.zeros_like(clean))


def test_corruption_is_deterministic_for_same_key():
    clean = jnp.arange(12, dtype=jnp.float32).reshape(2, 3, 2)
    valid = jnp.ones((2, 3), dtype=bool)
    kwargs = {
        "noise_std": 0.2,
        "dropout_probability": 0.4,
        "alpha": 0.3,
    }

    first = corrupt_history(jax.random.key(4), clean, valid, clean, **kwargs)
    second = corrupt_history(jax.random.key(4), clean, valid, clean, **kwargs)

    assert jnp.array_equal(first[0], second[0])
    assert jnp.array_equal(first[1], second[1])


def test_invalid_history_shape_is_rejected():
    with pytest.raises(ValueError, match="history"):
        validate_history(
            jnp.ones((2, 3)),
            jnp.ones((2, 3), dtype=bool),
            action_dim=3,
        )


def test_invalid_history_mask_is_rejected():
    history = jnp.ones((2, 3, 4))
    with pytest.raises(ValueError, match="mask"):
        validate_history(history, jnp.ones((2, 3)), action_dim=4)
    with pytest.raises(ValueError, match="mask"):
        validate_history(history, jnp.ones((2, 2), dtype=bool), action_dim=4)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("noise_std", -0.1),
        ("noise_std", float("nan")),
        ("dropout_probability", -0.1),
        ("dropout_probability", 1.1),
        ("dropout_probability", float("nan")),
    ],
)
def test_corruption_rejects_invalid_configuration(attribute, value):
    kwargs = {
        "noise_std": 0.0,
        "dropout_probability": 0.0,
        "alpha": 0.0,
    }
    kwargs[attribute] = value
    clean = jnp.ones((1, 2, 1))
    valid = jnp.ones((1, 2), dtype=bool)

    with pytest.raises(ValueError, match="history corruption"):
        corrupt_history(jax.random.key(5), clean, valid, clean, **kwargs)


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [
        (jnp.array(-0.1), 1.0),
        (jnp.array(1.1), 5.0),
        (jnp.array(jnp.nan), 1.0),
        (jnp.array(-jnp.inf), 1.0),
        (jnp.array(jnp.inf), 5.0),
    ],
)
def test_corruption_projects_alpha_consistently_in_eager_and_jit(alpha, expected):
    clean = jnp.ones((1, 2, 1))
    predicted = 5 * clean
    valid = jnp.ones((1, 2), dtype=bool)
    kwargs = {"noise_std": 0.0, "dropout_probability": 0.0}

    eager, _ = corrupt_history(jax.random.key(9), clean, valid, predicted, alpha=alpha, **kwargs)
    compiled = jax.jit(
        corrupt_history,
        static_argnames=("noise_std", "dropout_probability"),
    )
    traced, _ = compiled(jax.random.key(9), clean, valid, predicted, alpha=alpha, **kwargs)

    assert jnp.all(eager == expected)
    assert jnp.array_equal(traced, eager)


def test_inactive_interpolation_branch_cannot_contaminate_endpoint():
    clean = jnp.ones((1, 2, 1))
    predicted = 5 * clean
    valid = jnp.ones((1, 2), dtype=bool)
    kwargs = {"noise_std": 0.0, "dropout_probability": 0.0}

    demo_only, _ = corrupt_history(
        jax.random.key(13),
        clean,
        valid,
        jnp.full_like(predicted, jnp.nan),
        alpha=jnp.array(0.0),
        **kwargs,
    )
    prediction_only, _ = corrupt_history(
        jax.random.key(13),
        jnp.full_like(clean, jnp.nan),
        valid,
        predicted,
        alpha=jnp.array(1.0),
        **kwargs,
    )

    assert jnp.array_equal(demo_only, clean)
    assert jnp.array_equal(prediction_only, predicted)


def test_corruption_rejects_scalar_and_complex_history_cleanly():
    with pytest.raises(ValueError, match="history"):
        corrupt_history(
            jax.random.key(10),
            jnp.array(1.0),
            jnp.array(True),
            jnp.array(1.0),
            noise_std=0.0,
            dropout_probability=0.0,
            alpha=0.0,
        )
    complex_history = jnp.ones((1, 2, 1), dtype=jnp.complex64)
    with pytest.raises(ValueError, match="floating dtype"):
        validate_history(
            complex_history,
            jnp.ones((1, 2), dtype=bool),
            action_dim=1,
        )


def test_zero_length_history_is_supported_eager_and_jit():
    clean = jnp.empty((2, 0, 3), dtype=jnp.float32)
    valid = jnp.empty((2, 0), dtype=bool)
    kwargs = {"noise_std": 0.1, "dropout_probability": 0.2, "alpha": jnp.array(0.3)}

    eager = corrupt_history(jax.random.key(11), clean, valid, clean, **kwargs)
    compiled = jax.jit(
        corrupt_history,
        static_argnames=("noise_std", "dropout_probability"),
    )
    traced = compiled(jax.random.key(11), clean, valid, clean, **kwargs)

    assert eager[0].shape == clean.shape
    assert eager[1].shape == valid.shape
    assert jnp.array_equal(traced[0], eager[0])
    assert jnp.array_equal(traced[1], eager[1])


def test_clean_history_gradient_respects_alpha_and_padding():
    clean = jnp.ones((1, 3, 1))
    predicted = 5 * clean
    valid = jnp.array([[False, True, True]])

    gradient = jax.grad(
        lambda value: jnp.sum(
            corrupt_history(
                jax.random.key(12),
                value,
                valid,
                predicted,
                noise_std=0.0,
                dropout_probability=0.0,
                alpha=jnp.array(0.25),
            )[0]
        )
    )(clean)

    expected = jnp.array([[[0.0], [0.75], [0.75]]])
    assert jnp.array_equal(gradient, expected)


def test_corruption_is_jittable_with_static_configuration():
    clean = jnp.ones((1, 2, 1))
    valid = jnp.ones((1, 2), dtype=bool)
    compiled = jax.jit(
        corrupt_history,
        static_argnames=("noise_std", "dropout_probability"),
    )

    values, mask = compiled(
        jax.random.key(6),
        clean,
        valid,
        clean,
        noise_std=0.1,
        dropout_probability=0.2,
        alpha=jnp.array(0.3),
    )

    assert values.shape == clean.shape
    assert jnp.array_equal(mask, valid)
