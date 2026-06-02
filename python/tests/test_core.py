from __future__ import annotations

import pytest

from pysalsa import CycleError, Database, Durability, tracked


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


def test_input_values_are_copied_at_boundaries() -> None:
    db = Database()
    original: list[int] = []
    source = db.input(original)

    @tracked
    def length(db: Database, value):
        return len(value.get())

    assert length(db, source) == 0
    original.append(1)
    assert length(db, source) == 0

    read_value = source.get()
    read_value.append(2)
    assert length(db, source) == 0

    assert source.set([1, 2])
    assert length(db, source) == 2


def test_foreign_input_and_config_reads_are_rejected() -> None:
    db1 = Database()
    db2 = Database()
    source = db2.input(1)
    config = db2.config({"a": 1})

    @tracked
    def read_input(db: Database, value):
        return value.get()

    @tracked
    def read_config(db: Database, value):
        return value.read("a")

    with pytest.raises(ValueError, match="different Database"):
        read_input(db1, source)
    with pytest.raises(ValueError, match="different Database"):
        read_config(db1, config)


def test_heterogeneous_container_arguments_key_structurally() -> None:
    db = Database()

    @tracked
    def count(db: Database, value):
        return len(value)

    assert count(db, {1: "one", "two": 2}) == 2
    assert count(db, {1, "two"}) == 2


def test_config_slice_accepts_list_paths() -> None:
    db = Database()
    config = db.config({"a": {"b": 1}})

    assert config.slice([["a", "b"]]) == {("a", "b"): 1}


def test_durability_changed_at_is_conservative() -> None:
    db = Database()
    high = db.input(1, durability=Durability.HIGH)
    assert high.set(2)

    assert db.last_changed_at(Durability.HIGH) == db.current_revision
    assert db.last_changed_at(Durability.MEDIUM) == db.current_revision
    assert db.last_changed_at(Durability.LOW) == db.current_revision

    low_revision = db.current_revision
    low = db.input(1, durability=Durability.LOW)
    assert low.set(2)

    assert db.last_changed_at(Durability.LOW) == db.current_revision
    assert db.last_changed_at(Durability.MEDIUM) == low_revision
    assert db.last_changed_at(Durability.HIGH) == low_revision


def test_cycles_are_reported() -> None:
    db = Database()

    @tracked
    def recursive(db: Database, n: int):
        return recursive(db, n)

    with pytest.raises(CycleError):
        recursive(db, 0)
