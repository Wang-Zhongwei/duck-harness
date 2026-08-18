from inference.agent.tool_agent import _extract_labeled_blocks, _extract_scientist_note


def test_extract_labeled_blocks_accepts_qualified_labels() -> None:
    content = """\\
World model updated: The stamp is currently above the canvas.
Goal model (revised): Paint the lower half purple.
Recent findings update: DOWN changes position only after rotation.
Plan for next attempt: Rotate, move down, and stamp.
"""

    assert _extract_labeled_blocks(
        content,
        ["World model", "Goal model", "Recent findings", "Plan"],
    ) == {
        "World model": "The stamp is currently above the canvas.",
        "Goal model": "Paint the lower half purple.",
        "Recent findings": "DOWN changes position only after rotation.",
        "Plan": "Rotate, move down, and stamp.",
    }


def test_extract_labeled_blocks_accepts_leading_qualified_labels() -> None:
    content = """\\
Revised world model: The stamp is now below the canvas.
Updated goal model (level 2): Paint the upper half green.
Revision of the action model: DOWN moves two cells.
"""

    assert _extract_labeled_blocks(
        content,
        ["World model", "Goal model", "Action model"],
    ) == {
        "World model": "The stamp is now below the canvas.",
        "Goal model": "Paint the upper half green.",
        "Action model": "DOWN moves two cells.",
    }


def test_extract_labeled_blocks_accepts_level_prefix_as_new_block() -> None:
    content = """\\
Revised world model: Level one description.
Level 2 world model: Level two description.
"""

    assert _extract_labeled_blocks(content, ["World model"]) == {
        "World model": "Level one description. Level two description."
    }


def test_extract_labeled_blocks_accepts_observed_plan_prefixes() -> None:
    content = """\\
Probe plan: Test LEFT once.
Next-run plan (32-action budget): Replay the verified sequence.
"""

    assert _extract_labeled_blocks(content, ["Plan"]) == {
        "Plan": "Test LEFT once. Replay the verified sequence."
    }


def test_extract_labeled_blocks_accepts_long_heading_qualifier() -> None:
    content = (
        "World model — " + "important evidence " * 6 + ": Rebuild the map."
    )

    assert _extract_labeled_blocks(content, ["World model"]) == {
        "World model": "Rebuild the map."
    }


def test_extract_labeled_blocks_accepts_markdown_wrapped_qualified_label() -> None:
    content = "* **World model updated:** The canvas is all white."

    assert _extract_labeled_blocks(content, ["World model"]) == {
        "World model": "** The canvas is all white."
    }


def test_extract_labeled_blocks_does_not_match_label_prefix_inside_word() -> None:
    assert _extract_labeled_blocks(
        "World modelling note: this is ordinary prose.",
        ["World model"],
    ) == {}


def test_extract_scientist_note_maps_observed_qwen_variant() -> None:
    assert _extract_scientist_note(
        "World model updated: The vertical stamp is to the right of the canvas."
    )["world_model"] == "The vertical stamp is to the right of the canvas."


def test_extract_scientist_note_maps_observed_leading_variants() -> None:
    for heading in (
        "Revised world model",
        "Updated world model",
        "Final world model",
        "Level 2 world model",
        "Updating the world model",
        "Revision of the world model",
    ):
        assert _extract_scientist_note(f"{heading}: New evidence.")["world_model"] == (
            "New evidence."
        )
