from inference.agent.python_tool_sandbox import run_sandboxed_python


def _frame(grid, *, step, level=1):
    return {
        "ascii": "",
        "step": step,
        "level": level,
        "shape": [len(grid), len(grid[0])],
        "grid": grid,
    }


def _run(code, history, current):
    return run_sandboxed_python(
        code=code,
        timeout_seconds=5,
        initial_state={
            "current_frame": current,
            "history": history,
            "valid_actions": ["SPACE", "LEFT", "MOUSE"],
            "last_action_result": {},
        },
        action_handler=lambda actions: {},
    )


def _history():
    # Seeded frame (no action), then SPACE, LEFT, MOUSE, SPACE.
    frames = [_frame([[0, 0]], step=step) for step in range(5)]
    actions = ["", "SPACE", "LEFT", "MOUSE(row=4, col=7)", "SPACE"]
    return (
        [
            {"action": action, "frame": frame}
            for action, frame in zip(actions, frames)
        ],
        frames[-1],
    )


def test_steps_by_action_groups_base_names_chronologically():
    history, current = _history()

    outcome = _run("result = transitions.steps_by_action()", history, current)

    assert not outcome["error"]
    assert outcome["result"] == {"SPACE": [1, 4], "LEFT": [2], "MOUSE": [3]}


def test_for_action_matches_base_name_case_insensitively():
    history, current = _history()

    outcome = _run(
        "result = [t.step for t in transitions.for_action('space')]",
        history,
        current,
    )

    assert not outcome["error"]
    assert outcome["result"] == [1, 4]


def test_for_action_full_display_matches_exactly():
    history, current = _history()

    outcome = _run(
        "matches = transitions.for_action('MOUSE(row=4, col=7)')\n"
        "misses = transitions.for_action('MOUSE(row=9, col=9)')\n"
        "result = [[t.step for t in matches], [t.step for t in misses]]",
        history,
        current,
    )

    assert not outcome["error"]
    assert outcome["result"] == [[3], []]


def test_at_step_returns_matching_transition_or_none():
    history, current = _history()

    outcome = _run(
        "hit = transitions.at_step(2)\n"
        "result = [hit.action, hit.step, transitions.at_step(99)]",
        history,
        current,
    )

    assert not outcome["error"]
    assert outcome["result"] == ["LEFT", 2, None]


def test_for_action_result_supports_diff_inspection():
    before = _frame([[0, 0]], step=0)
    after = _frame([[5, 0]], step=1)
    history = [
        {"action": "", "frame": before},
        {"action": "SPACE", "frame": after},
    ]

    outcome = _run(
        "result = [t.diff['cells_changed'] for t in transitions.for_action('SPACE')]",
        history,
        after,
    )

    assert not outcome["error"]
    assert outcome["result"] == [1]
