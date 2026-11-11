import json
import sqlite3
import threading
from datetime import UTC, datetime


def now():
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "catalog.sqlite3"
        self.lock = threading.RLock()
        with self.connect() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("CREATE TABLE IF NOT EXISTS recording (id TEXT PRIMARY KEY, document TEXT NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=15)

    def get(self, identifier):
        with self.connect() as c:
            row = c.execute("SELECT document FROM recording WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError(identifier)
        return json.loads(row[0])

    def all(self):
        with self.connect() as c:
            rows = c.execute("SELECT document FROM recording ORDER BY rowid DESC").fetchall()
        return [json.loads(r[0]) for r in rows]

    def create(self, identifier, **fields):
        doc = dict(
            id=identifier,
            title=identifier,
            state="new",
            stage="Imported",
            progress=0,
            created_at=now(),
            updated_at=now(),
            revision=0,
            ends=[],
            clips=[],
            warnings=[],
            expected_ends=None,
            error=None,
        )
        doc.update(fields)
        with self.connect() as c:
            c.execute("INSERT INTO recording VALUES (?,?)", (identifier, json.dumps(doc)))
        return doc

    def update(self, identifier, **fields):
        with self.lock, self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT document FROM recording WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError(identifier)
            doc = {**json.loads(row[0]), **fields, "updated_at": now()}
            c.execute("UPDATE recording SET document=? WHERE id=?", (json.dumps(doc), identifier))
        return doc
