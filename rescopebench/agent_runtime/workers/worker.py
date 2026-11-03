"""Final Robot Hermes worker with a 100-turn semantic limit."""

from driver_semantic_limit import run_wrapped_driver


if __name__ == "__main__":
    run_wrapped_driver(
        "driver_native_base", max_iterations=100, learner_wall_clock_budget=False
    )
