from inference.agent.python_tool_sandbox import run_sandboxed_python


def _frame(grid, *, step):
    return {
        "ascii": "",
        "step": step,
        "level": 1,
        "shape": [len(grid), len(grid[0])],
        "grid": grid,
    }


def _run(code, memory_handler=None):
    frame = _frame([[0, 5], [5, 0]], step=1)
    return run_sandboxed_python(
        code=code,
        timeout_seconds=5,
        initial_state={
            "current_frame": frame,
            "history": [{"action": "", "frame": frame}],
            "valid_actions": ["SPACE"],
            "last_action_result": {},
        },
        action_handler=lambda actions: {},
        memory_handler=memory_handler,
    )


def test_update_memory_forwards_fields_to_the_host():
    received = []

    def handler(fields):
        received.append(fields)
        return {"updated": sorted(fields)}

    outcome = _run(
        "result = update_memory(world_model='A token moves.', plan='Probe right.')",
        memory_handler=handler,
    )

    assert not outcome["error"]
    assert received == [{"world_model": "A token moves.", "plan": "Probe right."}]
    assert outcome["result"] == {"updated": ["plan", "world_model"]}


def test_update_memory_host_error_surfaces_as_a_runtime_error():
    outcome = _run(
        "update_memory(world_model='x')",
        memory_handler=lambda fields: {"error": "Model fields must be strings: plan"},
    )

    assert "RuntimeError: Model fields must be strings: plan" in outcome["error"]


def test_update_memory_without_a_handler_reports_unavailability():
    outcome = _run("update_memory(world_model='x')", memory_handler=None)

    assert "update_memory is not available" in outcome["error"]


def test_update_memory_rejects_empty_and_non_string_calls_in_the_sandbox():
    calls = []

    def handler(fields):
        calls.append(fields)
        return {"updated": []}

    outcome = _run("update_memory()", memory_handler=handler)
    assert "needs at least one field" in outcome["error"]

    outcome = _run("update_memory(world_model=42)", memory_handler=handler)
    assert "must be strings" in outcome["error"]

    outcome = _run("update_memory(notes='x')", memory_handler=handler)
    assert "unexpected keyword argument" in outcome["error"]

    assert calls == []  # nothing invalid ever reached the host
