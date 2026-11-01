"""Where every converted asset lives, and how usd-world knows it is up to date.

Layout of a cache folder (ids are paths relative to the data set root, so they never change):

    index.sqlite                       what was scanned and converted, with keys (each leaf also
                                       carries its key in asset.extras.usd_world)
    units/<unit>/geo.gltf              what an asset draws itself, without what it places (glTF 2.0);
                                       <unit> is <layer>/<prim>, plus {set=variant,...} when its
                                       placement selects variants
    <leaf>.<look>.gltf                 a leaf with a look applied (--look), sharing the leaf's .bin
    units/<unit>/assembly.gltf         the asset placing all of the above (glTF 2.1); options other
                                       than the defaults add tags: assembly.coarsest.<look>.gltf
    logs/                              converter warnings

A key is the hash of what the output depends on: the converter version, the options that change
the output, what is converted (unit, variant selections), and the size and modification time of
every layer the conversion read (and, for a leaf, every texture, MDL and MaterialX file). A record is up to date when the key recomputed from the current
state of its recorded layers is the stored one, which needs no USD.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

# Bump when a change to the conversion changes its output: every leaf is converted again.
# A fix that concerns a few assets is applied to them alone with --force instead.
CONVERTER_VERSION = 7
# Bump when the scan's rules change what it finds. Every leaf's key includes its unit's scan key, so
# this too converts every leaf again (an hour on a large scene): for a new field that changes nothing
# else, a forced scan (--force, or "Scan again" in the UI) records it and keeps the keys.
SCAN_VERSION = 11


def stat(path: str | Path) -> list[int] | None:
    try:
        s = os.stat(path)
    except OSError:
        return None
    return [s.st_size, s.st_mtime_ns]


def make_key(identity: dict, inputs: dict) -> str:
    return hashlib.sha1(json.dumps({**identity, "inputs": inputs}, sort_keys=True).encode()).hexdigest()


def layer_inputs(root: Path, layers) -> dict:
    """{path: [size, mtime]} of the layer files that exist (anonymous layers have none). Paths are
    relative to ``root``, or absolute for files outside it (a shared material library)."""
    out = {}
    for path in layers:
        s = stat(path)
        if s is not None:
            try:
                rel = os.path.relpath(os.path.abspath(path), root)
            except ValueError:  # another drive
                rel = ".."
            out[Path(rel).as_posix() if not rel.startswith("..") else Path(path).as_posix()] = s
    return out


class Paths:
    """Ids are paths relative to the data set root. Worker processes need these rules, not the index.
    ``variants`` ({set: value}) are selections made everywhere while composing: they change what
    every asset converts to, so they are part of every id (see ``scan.unit_id``)."""

    def __init__(self, root: Path, variants: dict | None = None):
        self.root = Path(root)
        self.variants = variants or {}

    OUTSIDE = "~outside/"  # ids of files outside the root: their absolute path, the drive a folder

    def rel(self, path: str | Path) -> str:
        path = os.path.abspath(path)
        try:
            rel = os.path.relpath(path, self.root)
        except ValueError:  # another drive
            rel = ".."
        if rel.startswith(".."):
            return self.OUTSIDE + Path(path).as_posix().replace(":", "", 1).lstrip("/")
        return Path(rel).as_posix()

    def abs(self, rel: str) -> str:
        if rel.startswith(self.OUTSIDE):
            rest = rel[len(self.OUTSIDE):]
            return str(Path(f"{rest[0]}:/{rest[2:]}" if os.name == "nt" else f"/{rest}"))
        return str(self.root / rel)

    def inputs(self, layers) -> dict:
        return layer_inputs(self.root, layers)

    def fresh(self, record: dict | None, identity: dict) -> bool:
        if not record or "key" not in record:
            return False
        current = {rel: stat(self.abs(rel)) for rel in record.get("inputs", {})}
        return make_key(identity, current) == record["key"]


class Cache(Paths):
    TABLES = ("units", "leaves")

    def __init__(self, folder: Path, root: Path | None):
        self.dir = folder.resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        # The UI reads from its own threads while a job writes from another connection.
        self.db = sqlite3.connect(self.dir / "index.sqlite", check_same_thread=False, timeout=60)
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT)")
        for table in self.TABLES:
            self.db.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, record TEXT)")
        stored = self.db.execute("SELECT value FROM meta WHERE name='root'").fetchone()
        if stored and root and Path(stored[0]) != root.resolve():
            raise SystemExit(f"error: {self.dir} caches {stored[0]}, not {root}")
        if not stored:
            if not root:
                raise SystemExit(f"error: {self.dir} is a new cache: give the data set root with --root")
            self.db.execute("INSERT INTO meta VALUES ('root', ?)", (str(root.resolve()),))
            self.db.commit()
        self.root = Path(stored[0]) if stored else root.resolve()
        self.variants: dict = {}

    # ---- files ----
    # Windows (and vk_gltf_renderer's file loading) can't open a path of 260 characters or more. Room
    # is kept for the longest file name of a folder (an assembly tagged with its options, a look).
    MAX_FOLDER = 200

    def path(self, leaf_or_file: str, suffix: str = ".gltf") -> Path:
        """``units/<unit>/geo`` (or ``.../assembly...``) -> its file in the cache. A unit whose
        folder would be too long for Windows goes to ``units/~/<hash>/`` instead: its id stays in
        the index, only the files move (and a moved cache converts those again, by itself)."""
        folder, _, name = leaf_or_file.rpartition("/")
        if len(str(self.dir / folder)) > self.MAX_FOLDER:
            folder = "units/~/" + hashlib.sha1(folder.encode()).hexdigest()[:20]
        return self.dir / folder / (name + suffix)

    # ---- records ----
    def get(self, table: str, id: str) -> dict | None:
        row = self.db.execute(f"SELECT record FROM {table} WHERE id=?", (id,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, table: str, id: str, record: dict) -> None:
        self.db.execute(f"INSERT OR REPLACE INTO {table} VALUES (?, ?)", (id, json.dumps(record)))

    def commit(self) -> None:
        self.db.commit()

    def count(self, table: str) -> int:
        return self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def records(self, table: str):
        for id, rec in self.db.execute(f"SELECT id, record FROM {table}"):
            yield id, json.loads(rec)
