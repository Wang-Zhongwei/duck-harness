from types import SimpleNamespace

import pytest

from inference.distillation.check import validate_tokenizer_compatibility


class TokenizerStub(SimpleNamespace):
    def get_vocab(self):
        return self.vocab


def tokenizer(vocab, special_ids=()):
    return TokenizerStub(
        vocab=vocab,
        all_special_ids=list(special_ids),
        bos_token_id=None,
        eos_token_id=2,
        pad_token_id=0,
    )


def test_allows_student_only_special_ids_and_returns_guard_set():
    teacher = tokenizer({"pad": 0, "word": 1, "eos": 2}, {0, 2})
    student = tokenizer({"pad": 0, "word": 1, "eos": 2, "audio": 3}, {0, 2, 3})
    assert validate_tokenizer_compatibility(teacher, student) == {3}


def test_rejects_changed_shared_token_id():
    teacher = tokenizer({"pad": 0, "word": 1, "eos": 2}, {0, 2})
    student = tokenizer({"pad": 0, "word": 4, "eos": 2}, {0, 2})
    with pytest.raises(ValueError, match="different IDs"):
        validate_tokenizer_compatibility(teacher, student)


def test_rejects_student_only_ordinary_token():
    teacher = tokenizer({"pad": 0, "eos": 2}, {0, 2})
    student = tokenizer({"pad": 0, "word": 1, "eos": 2}, {0, 2})
    with pytest.raises(ValueError, match="ordinary token"):
        validate_tokenizer_compatibility(teacher, student)
