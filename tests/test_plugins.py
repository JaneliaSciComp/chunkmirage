"""Plugins that fail to load are skipped, and say so."""

import logging

from chunkmirage.ops import base


class _EntryPoint:
    name, value = "broken_op", "missing_module:BrokenOp"

    def load(self):
        raise ImportError("No module named 'missing_module'")


def test_a_broken_op_plugin_is_skipped_with_a_warning(monkeypatch, caplog):
    monkeypatch.setattr(base, "entry_points", lambda group: [_EntryPoint()])
    monkeypatch.setattr(base, "_ENTRYPOINTS_LOADED", False)
    with caplog.at_level(logging.WARNING, logger="chunkmirage"):
        ops = base.list_ops()
    assert "threshold" in ops and "broken_op" not in ops
    assert "broken_op" in caplog.text and "missing_module" in caplog.text
