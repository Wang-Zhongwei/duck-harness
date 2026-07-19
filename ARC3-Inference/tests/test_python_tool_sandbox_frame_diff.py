from inference.agent.python_tool_sandbox import run_sandboxed_python


def _frame(grid, *, step):
    return {
        "ascii": "",
        "step": step,
        "level": 1,
        "shape": [len(grid), len(grid[0])],
        "grid": grid,
    }


def _run(code, before, after):
    return run_sandboxed_python(
        code=code,
        timeout_seconds=5,
        initial_state={
            "current_frame": after,
            "history": [
                {"action": "", "frame": before},
                {"action": "SPACE", "frame": after},
            ],
            "valid_actions": ["SPACE"],
            "last_action_result": {},
        },
        action_handler=lambda actions: {},
    )


def test_frame_diff_preserves_isolated_cells_and_groups_color_transitions():
    before = _frame([[0, 0, 0], [0, 5, 0], [0, 0, 0]], step=58)
    after = _frame([[1, 0, 0], [0, 0, 0], [0, 0, 5]], step=59)

    outcome = _run("result = last_transition.diff", before, after)

    assert not outcome["error"]
    assert outcome["result"] == {
        "cells_changed": 3,
        "groups": [
            {
                "from": "B",
                "to": "W",
                "count": 1,
                "bbox": [[1, 1], [1, 1]],
                "cells": [[1, 1]],
            },
            {
                "from": "W",
                "to": "B",
                "count": 1,
                "bbox": [[2, 2], [2, 2]],
                "cells": [[2, 2]],
            },
            {
                "from": "W",
                "to": "w",
                "count": 1,
                "bbox": [[0, 0], [0, 0]],
                "cells": [[0, 0]],
            },
        ],
    }


def test_frame_diff_folds_only_the_printed_representation():
    before = _frame([[0] * 13], step=58)
    after = _frame([[5] * 13], step=59)

    outcome = _run(
        'print(last_transition.diff); result = last_transition.diff["groups"][0]["cells"]',
        before,
        after,
    )

    assert not outcome["error"]
    assert "<13 cells; inspect group['cells'] for coordinates>" in outcome["stdout"]
    assert outcome["result"] == [[0, c] for c in range(13)]


def test_frame_diff_alias_and_no_change_result():
    frame = _frame([[0, 5], [5, 0]], step=1)

    outcome = _run("result = diff(previous_frame, current_frame)", frame, frame)

    assert not outcome["error"]
    assert outcome["result"] == {"cells_changed": 0, "groups": []}


def test_frame_diff_rejects_different_shapes():
    before = _frame([[0]], step=1)
    after = _frame([[0, 0]], step=2)

    outcome = _run("result = frame_diff(previous_frame, current_frame)", before, after)

    assert "ValueError: frame_diff requires equal frame shapes" in outcome["error"]
