"""Shared test configuration."""

import os

from hypothesis import HealthCheck, settings

# JAX traces and compiles on the first call with a given shape/dtype, so a
# per-example deadline measures compilation, not the property. `function_scoped
# _fixture` is suppressed because the numeric fixtures here are deterministic
# constants -- rebuilding them per example would only cost time.
settings.register_profile(
    "galax",
    deadline=None,
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.register_profile("ci", parent=settings.get_profile("galax"), max_examples=200)
settings.load_profile("ci" if os.environ.get("CI") else "galax")
