from __future__ import annotations

import pytest

from pysalsa import CycleError, Database, tracked


def test_backdates_when_intermediate_output_is_equal() -> None:
    db = Database()
    source = db.input(22)
    log: list[str] = []

    @tracked
    def intermediate(db: Database, value):
        log.append("intermediate")
        return value.get() // 2

    @tracked
    def final(db: Database, value):
        log.append("final")
        return intermediate(db, value) * 2

    assert final(db, source) == 22
    assert log == ["final", "intermediate"]

    log.clear()
    assert source.set(23)
    assert final(db, source) == 22
    assert log == ["intermediate"]

    log.clear()
    assert source.set(24)
    assert final(db, source) == 24
    assert log == ["intermediate", "final"]


def test_unrelated_input_change_does_not_reexecute() -> None:
    db = Database()
    source = db.input(22)
    log: list[str] = []

    @tracked
    def final(db: Database, value):
        log.append("final")
        return value.get() * 2

    assert final(db, source) == 44
    assert log == ["final"]

    red_herring = db.input(10)
    assert red_herring.set(11)

    log.clear()
    assert final(db, source) == 44
    assert log == []


def test_setting_equivalent_input_does_not_advance_revision() -> None:
    db = Database()
    source = db.input({"a": 1})
    revision = db.current_revision

    assert not source.set({"a": 1})
    assert db.current_revision == revision


def test_cycles_are_reported() -> None:
    db = Database()

    @tracked
    def recursive(db: Database, n: int):
        return recursive(db, n)

    with pytest.raises(CycleError):
        recursive(db, 0)
