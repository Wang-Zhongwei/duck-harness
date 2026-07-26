from __future__ import annotations

import argparse
from typing import Any


def validate_tokenizer_compatibility(teacher: Any, student: Any) -> set[int]:
    """Return student-only special IDs that must be rejected at scoring time."""
    teacher_vocab = teacher.get_vocab()
    student_vocab = student.get_vocab()
    shared = teacher_vocab.keys() & student_vocab.keys()
    changed = [
        token for token in shared if teacher_vocab[token] != student_vocab[token]
    ]
    if changed:
        raise ValueError(f"{len(changed)} shared tokens have different IDs")

    student_only = student_vocab.keys() - teacher_vocab.keys()
    forbidden_ids = {student_vocab[token] for token in student_only}
    non_special = forbidden_ids - set(student.all_special_ids)
    if non_special:
        raise ValueError(
            f"student has {len(non_special)} ordinary token IDs missing from the teacher"
        )
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(teacher, field, None) != getattr(student, field, None):
            raise ValueError(f"teacher and student {field} values differ")
    return forbidden_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check teacher/student token-id compatibility"
    )
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    teacher = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    student = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    try:
        forbidden_ids = validate_tokenizer_compatibility(teacher, student)
    except ValueError as exc:
        raise SystemExit(f"INCOMPATIBLE tokenizers: {exc}") from exc
    shared_count = len(teacher.get_vocab().keys() & student.get_vocab().keys())
    print(f"Compatible shared token IDs: {shared_count}")
    if forbidden_ids:
        print(
            "Student-only special IDs will be rejected in rollouts: "
            + ", ".join(map(str, sorted(forbidden_ids)))
        )


if __name__ == "__main__":
    main()
