"""Unit tests for ingestion batching and GraphStore write logic (mocked driver)."""

from unittest.mock import MagicMock

import pytest

from sgrag.graph import schema
from sgrag.graph.config import GraphConfig
from sgrag.graph.ingest import GraphStore, batched


def test_batched_splits():
    rows = [{"i": i} for i in range(2500)]
    parts = list(batched(rows, 1000))
    assert [len(p) for p in parts] == [1000, 1000, 500]
    assert list(batched([], 100)) == []


def _store_with_mock():
    cfg = GraphConfig()
    store = GraphStore(cfg)
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    store._driver = driver
    return store, session


def test_run_write_batches_unwind():
    store, session = _store_with_mock()
    rows = [{"chunk_id": f"c{i}"} for i in range(5)]
    total = store.run_write(schema.UPSERT_CHUNKS, rows, batch_size=2)
    assert total == 5
    assert session.execute_write.call_count == 3  # 2 + 2 + 1


def test_upsert_helpers_delegate_cypher():
    store, session = _store_with_mock()
    store.upsert_articles([{"pmid": "1"}])
    tx_fn = session.execute_write.call_args[0][0]
    tx = MagicMock()
    tx_fn(tx, [{"pmid": "1"}])
    tx.run.assert_called_once()
    assert tx.run.call_args[0][0] == schema.UPSERT_ARTICLES


def test_ensure_schema_runs_all_ddl():
    store, _ = _store_with_mock()
    store._run = MagicMock()
    store.ensure_schema()
    stmts = [c[0][0] for c in store._run.call_args_list]
    for expected in (schema.CONSTRAINT_STATEMENTS
                     + schema.INDEX_STATEMENTS
                     + (schema.vector_index_statement("chunk_embedding_idx", 384, "cosine"),)):
        assert expected in stmts


def test_connect_requires_password(monkeypatch):
    monkeypatch.delenv("SGRAG_NEO4J_PASSWORD", raising=False)
    cfg = GraphConfig()  # password == ""
    store = GraphStore(cfg)
    with pytest.raises(RuntimeError, match="SGRAG_NEO4J_PASSWORD"):
        store.connect()
