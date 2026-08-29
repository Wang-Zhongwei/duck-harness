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
