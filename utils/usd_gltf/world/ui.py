"""The usd-world UI: a local web page (``usd-world.py ui``).

Start from a USD stage ("Open USD stage..." shows the system's file dialog) or from a recent
project. Opening a stage finds its data set root (the folder holding the layers it uses), makes a
cache folder for it in the caches folder, and scans it at once, showing what it finds: the assets,
the switches (levels of detail, proxies), the variant sets that offer choices, and how many leaves
there are to convert. One button then converts and assembles, with the choices next to it (KTX2
textures, look, variants, level of detail); the scene graph and every setting stay available below.

One job runs at a time, in a thread of this server with its own worker processes and its own
connection to the index; the page polls its progress. Python's standard library only: the page is
``ui.html`` next to this file, the API is JSON. Recent projects and the caches folder are kept in
``~/.usd-world.json``.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .assemble import geo_leaf, short_name
from .scan import asset_variant_sets, unit_id as make_unit_id
from .project import DEFAULTS, STEPS, Project, available_looks, guess_root

CONFIG = Path.home() / ".usd-world.json"
USD_FILES = [("USD stages", "*.usd *.usda *.usdc *.usdz"), ("All files", "*.*")]


def read_config() -> dict:
    try:
        return {"caches": str(Path.home() / "usd-world"), "recent": [], **json.loads(CONFIG.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return {"caches": str(Path.home() / "usd-world"), "recent": []}


def write_config(config: dict) -> None:
    CONFIG.write_text(json.dumps(config, indent=1), encoding="utf-8")


def pick(kind: str, title: str, start: str | None) -> str | None:
    """The system's file (``kind`` "file") or folder dialog, from a process of its own (Tk wants
    the main thread, the server's requests don't run there). None when cancelled."""
    if kind == "file":
        call = f"fd.askopenfilename(title={title!r}, filetypes={USD_FILES!r}, initialdir={start or ''!r})"
    else:
        call = f"fd.askdirectory(title={title!r}, initialdir={start or ''!r})"
    code = ("import tkinter as tk, tkinter.filedialog as fd\n"
            "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
            f"print({call} or '')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
    return os.path.normpath(out) if out else None


class Stopped(Exception):
    """Raised in the job's thread at its next progress report once Stop was pressed."""


class Job:
    """The running (or last) job: its steps, progress and log, read by the page."""

    def __init__(self, title: str = ""):
        self.lock = threading.Lock()
        self.running, self.title, self.step, self.done, self.total = False, title, "", 0, 0
        self.lines: list[str] = []
        self.started = self.ended = 0.0
        self.error = ""
        self.result: dict | None = None  # an estimate's
        self.stop = False  # Stop pressed: the steps end at their next progress report

    def log(self, text: str) -> None:
        with self.lock:
            self.lines.append(text)

    def progress(self, step: str, done: int, total: int) -> None:
        with self.lock:
            self.step, self.done, self.total = step, done, total
        if self.stop:
            raise Stopped()

    def state(self, since: int) -> dict:
        with self.lock:
            return {"running": self.running, "title": self.title, "step": self.step, "done": self.done,
                    "total": self.total, "lines": self.lines[since:], "count": len(self.lines), "stopping": self.stop,
                    "error": self.error, "result": self.result, "seconds": round((time.time() if self.running else self.ended) - self.started, 1)
                    if self.started else 0}


class App:
    def __init__(self, folder: Path | None, root: Path | None, renderer: str | None):
        self.renderer = renderer or default_renderer()
        self.lock = threading.Lock()  # the page's requests share self.project
        self.job = Job()
        self.project: Project | None = None
        self.folder: Path | None = None
        self._assembler = None
        self._variant_sets: dict[str, dict] = {}  # unit id -> its root's variant sets (read with USD)
        if folder:
            self._open(folder, root)

    # ---- projects ----
    def home(self) -> dict:
        config = read_config()
        recent = []
        for folder in config["recent"]:
            info = _project_info(Path(folder))
            if info:
                recent.append(info)
        return {"caches": config["caches"], "recent": recent, "renderer": self.renderer, "sep": os.sep}

    def open_project(self, folder: Path, root: Path | None = None) -> dict:
        if self.job.running:
            return {"error": "a job is running"}
        return self._open(folder, root)

    def _open(self, folder: Path, root: Path | None = None) -> dict:
        with self.lock:
            self.folder = Path(folder).resolve()
            self.project = Project.open(self.folder, root)
            self.project.load()
            self._assembler = None
            self._variant_sets = {}
        _remember(self.folder)
        return self.state()

    def new_project(self, stage: str, caches: str | None) -> dict:
        """Open a stage: its project (found again if it was opened before), scanned at once."""
        if self.job.running:
            return {"error": "a job is running"}
        stage = os.path.normpath(os.path.abspath(stage))
        if not os.path.isfile(stage):
            return {"error": f"no such file: {stage}"}
        config = read_config()
        if caches:
            config["caches"] = caches
            write_config(config)
        for folder in config["recent"]:  # the same stage: its project
            info = _project_info(Path(folder))
            if info and info["input"] and os.path.normcase(info["input"]) == os.path.normcase(stage):
                self._open(Path(folder))
                return self.run(["scan"], [], False, title="Scanning " + Path(stage).name)
        folder = Path(config["caches"]) / f"{Path(stage).stem}-gltf"
        n = 2
        while folder.exists() and any(folder.iterdir()):
            folder, n = Path(config["caches"]) / f"{Path(stage).stem}-gltf-{n}", n + 1
        return self.run(["scan"], [], False, title="Opening " + Path(stage).name, new=(stage, folder))

    def close(self) -> dict:
        if self.job.running:
            return {"error": "a job is running"}
        with self.lock:
            self.project = self.folder = self._assembler = None
        return {"closed": True}

    # ---- the page's view of the project ----
    def move_project(self, dest: str) -> dict:
        """Move the project folder (and the scene file in it) to ``dest``: a new or empty folder."""
        if self.job.running:
            return {"error": "a job is running"}
        if self.project is None:
            return {"error": "no project open"}
        src, dest = self.folder, Path(dest).resolve()
        if dest == src:
            return {"error": "that is the project folder already"}
        if dest.is_relative_to(src):
            return {"error": "the new folder can't be inside the project folder"}
        if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
            return {"error": f"{dest} isn't empty: choose a new or empty folder"}
        with self.lock:
            self.project.cache.db.close()  # Windows can't move an open database
            self.project = self._assembler = None
            if dest.exists():
                dest.rmdir()
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
        config = read_config()
        config["recent"] = [f for f in config["recent"] if os.path.normcase(f) != os.path.normcase(str(src))]
        write_config(config)
        return self._open(dest)

    def reload(self) -> None:
        if self.folder:
            with self.lock:
                self.project = Project.open(self.folder)
                self.project.load()
                self._assembler = None

    def assembler(self):
        if self._assembler is None and self.project and self.project.top:
            self._assembler = self.project.assembler()
        return self._assembler

    def state(self) -> dict:
        p = self.project
        if p is None:
            return {"project": None, "renderer": self.renderer}
        scene = p.scene_path() if p.settings["input"] else None
        with self.lock:
            summary = p.summary() if p.scanner and p.scanner.units else None
            recommended = p.recommend(summary) if summary else None
            if summary and summary["stage"]["convert"] and not self.job.running and p.get_meta("recommended") is None:
                # The look starts as recommended, once: later changes are the user's. The level of
                # detail stays as authored until chosen: another policy scans and converts every
                # level (Caldera: 63,000 assets instead of 245), which is the user's call.
                changes = {"look": recommended["look"] or ""}
                p.settings.update(changes)
                p.save()
                p.set_meta("recommended", changes)
                summary = p.summary()
        return {"project": True, "cache": str(p.cache.dir), "root": str(p.cache.root), "settings": p.settings,
                "defaults": DEFAULTS, "looks": available_looks(), "top": p.top, "summary": summary,
                "recommended": recommended, "applied": p.get_meta("recommended"),
                "scene": str(scene) if scene else None, "scene_exists": bool(scene and scene.exists()),
                "renderer": self.renderer, "renderer_found": bool(self.renderer and Path(self.renderer).exists())}

    def unit(self, unit_id: str) -> dict:
        """A unit as the page shows it: its own leaf, and what it places, grouped by unit."""
        with self.lock:
            p, a = self.project, self.assembler()
            rec = p.scanner.units.get(unit_id) if p and p.scanner else None
            if rec is None:
                return {"id": unit_id, "missing": True}
            leaf = p.cache.get("leaves", geo_leaf(unit_id)) if rec["geo"] else None
            placed: dict[str, dict] = {}
            switches = []
            self._walk(rec["children"], placed, switches, a)
            for child in placed.values():
                child.update(self._summary(child["id"], a))
            for sw in switches:
                for option in sw["options"]:
                    option.update(self._summary(option["id"], a))
            base = p.scanner.chosen_from.get(unit_id, (unit_id, None))[0]
            return {"id": unit_id, "name": short_name(unit_id), "layer": rec["layer"], "prim": rec["prim"],
                    "selections": rec.get("selections", {}), "variant_sets": self._root_sets(unit_id),
                    "chosen": p.settings["asset_variants"].get(base, {}),
                    "draws": rec["geo"], "leaf": _leaf_view(p, unit_id, leaf),
                    "drawn": a.triangles(unit_id) if a else None,
                    "placed": sorted(placed.values(), key=lambda c: (-c["count"], c["name"])), "switches": switches}

    def _root_sets(self, unit_id: str) -> dict:
        """The variant sets of a unit's root prim (scan.asset_variant_sets): opens USD, once per unit."""
        if unit_id not in self._variant_sets:
            rec = self.project.scanner.units[unit_id]
            self._variant_sets[unit_id] = asset_variant_sets(self.project.cache.abs(rec["layer"]), rec["prim"],
                                                             rec.get("selections", {}))
        return self._variant_sets[unit_id]

    def assets_with(self, variant_set: str) -> dict:
        """The assets whose own prim has ``variant_set``: what the Convert stage lists to choose per asset."""
        with self.lock:
            p, out = self.project, []
            if p is None or p.scanner is None:
                return {"error": "no project"}
            for unit_id, rec in sorted(p.scanner.units.items(), key=lambda kv: short_name(kv[0])):
                if variant_set not in rec.get("variant_sets", {}):
                    continue
                own = self._root_sets(unit_id).get(variant_set)
                if own:
                    base = p.scanner.chosen_from.get(unit_id, (unit_id, None))[0]
                    out.append({"id": unit_id, "name": short_name(unit_id), **own,
                                "chosen": p.settings["asset_variants"].get(base, {}).get(variant_set)})
            return {"set": variant_set, "assets": out}

    def _walk(self, entries, placed, switches, a) -> None:
        for e in entries:
            if "unit" in e:
                c = placed.setdefault(e["unit"], {"id": e["unit"], "name": short_name(e["unit"]), "count": 0})
                c["count"] += 1
            elif "switch" in e:
                choice = a._choice(e) if a else e["selected"]
                switches.append({"name": e["name"], "set": e["switch"], "selected": e["selected"], "choice": choice,
                                 "options": [{"variant": v, "id": u, "name": short_name(u)} for v, u in e["options"].items()]})
            else:
                self._walk(e.get("children", []), placed, switches, a)

    def _summary(self, unit_id: str, a) -> dict:
        rec = self.project.scanner.units.get(unit_id)
        if rec is None:
            return {"status": "not scanned"}
        leaf = self.project.cache.get("leaves", geo_leaf(unit_id)) if rec["geo"] else None
        return {"status": _leaf_view(self.project, unit_id, leaf)["status"], "places": _count(rec["children"]),
                "drawn": a.triangles(unit_id) if a else None}

    # ---- changes ----
    def set_settings(self, changes: dict) -> dict:
        if self.job.running:
            return {"error": "a job is running"}
        if self.project is None:
            return {"error": "no project"}
        unknown = set(changes) - set(DEFAULTS)
        if unknown:
            return {"error": f"unknown settings {sorted(unknown)}"}
        with self.lock:
            try:
                Project.open(self.folder, None, changes)
            except (SystemExit, ValueError) as e:
                return {"error": str(e)}
        self.reload()
        return self.state()

    def choose(self, switch_set: str, name: str | None, value: str | None) -> dict:
        """Place ``value`` at the switches named ``name`` of ``switch_set`` (every one of the set when
        ``name`` is None; for a set that is not a switch, the selection made while composing). None
        value: as authored."""
        specs = []
        for s in self.project.settings["variants"]:
            if not _names(s, switch_set, name):
                specs.append(s)
            elif name:  # a rule for several switches keeps the others
                head, _, names = s.partition(":")
                others = [n for n in names.split(",") if n != name]
                if others:
                    specs.append(f"{head}:{','.join(others)}")
        if value:
            specs.append(f"{switch_set}={value}" + (f":{name}" if name else ""))
        return self.set_settings({"variants": specs})

    def choose_asset(self, unit_id: str, variant_set: str, value: str | None) -> dict:
        """Convert the asset ``unit_id`` (as placed, wherever it is) with ``value`` of its
        ``variant_set`` (None: as placed). Returns the unit it becomes, for the scan to find."""
        p = self.project
        rec = p.scanner.units.get(unit_id) if p and p.scanner else None
        if rec is None:
            return {"error": f"no unit {unit_id}"}
        base, placed_with = p.scanner.chosen_from.get(unit_id, (unit_id, rec.get("selections", {})))
        choices = {k: dict(v) for k, v in p.settings["asset_variants"].items()}
        mine = choices.setdefault(base, {})
        if value:  # even the placement's own: it wins over a scene-wide choice
            mine[variant_set] = value
        else:
            mine.pop(variant_set, None)
        if not mine:
            del choices[base]
        layer, prim = p.cache.abs(rec["layer"]), rec["prim"]
        state = self.set_settings({"asset_variants": choices})
        if "error" not in state:
            chosen = {**placed_with, **choices.get(base, {})}
            state["unit"] = make_unit_id(p.cache, layer, prim, chosen) if base in choices else base
        return state

    def run(self, steps: list[str], units: list[str], force: bool, title: str | None = None, new=None) -> dict:
        if self.job.running:
            return {"error": "a job is running"}
        if self.project is None and not new:
            return {"error": "no project: open a USD stage first"}
        steps = [s for s in (*STEPS, "estimate") if s in steps]
        job = self.job = Job(title or (" + ".join(steps) + (f" on {len(units)} unit(s)" if units else "")))
        job.running, job.started = True, time.time()
        if force and steps == ["scan"]:
            forced = [f"{u}*" for u in units] or ["*"]  # unit ids: the scan's
        else:
            forced = [f"units/{u}/geo" for u in units] if force and units else (["*"] if force else [])
        threading.Thread(target=self._run, args=(job, steps, units, forced, new), daemon=True).start()
        return {"started": job.title}

    def stop(self) -> dict:
        if not self.job.running:
            return {"error": "nothing is running"}
        self.job.stop = True
        return {"stopping": self.job.title}

    def _run(self, job: Job, steps, units, forced, new) -> None:
        project = None
        try:
            if new:
                stage, folder = new
                job.progress("finding the data set", 0, 0)
                job.log(f"Finding the folder holding the layers of {stage}...")
                root = guess_root(stage)
                job.log(f"  data set root: {root}")
                job.log(f"  cache: {folder}")
                Project.open(folder, root, {"input": stage})
                self._open(folder)
            project = Project.open(self.folder, log=job.log, progress=job.progress)
            with multiprocessing.Pool(project.settings["jobs"]) as pool:
                job.result = project.run(steps, pool, units, forced)
        except Stopped:
            # Leaving the pool's block ends its workers. Each finished asset is already in the
            # project folder: record the last ones, and running again continues from there.
            if project:
                project.cache.commit()
            job.title += " (stopped)"
            job.log("Stopped. What was finished is kept: run it again to continue where it stopped.")
        except BaseException as e:  # noqa: BLE001 - reported in the page
            job.error = f"{type(e).__name__}: {e}"
            job.log(traceback.format_exc())
        finally:
            job.running, job.ended = False, time.time()
            self.reload()

    def open_renderer(self) -> dict:
        scene = self.project.scene_path() if self.project else None
        if not scene or not scene.exists():
            return {"error": "no scene yet: convert and assemble first"}
        if not self.renderer or not Path(self.renderer).exists():
            return {"error": f"vk_gltf_renderer not found: {self.renderer} (start the UI with --renderer)"}
        # The renderer's own tonemapper and auto-exposure, as the user left them: not forced here.
        args = [self.renderer, "--scenefile", str(scene)]
        subprocess.Popen(args, cwd=Path(self.renderer).parent)
        return {"opened": str(scene)}

    def reveal(self, what: str) -> dict:
        """Show the scene file, the project folder or the projects folder in the file browser."""
        if what == "caches":
            path = Path(read_config()["caches"])
        elif self.project is None:
            return {"error": "no project open"}
        else:
            path = self.project.scene_path() if what == "scene" else self.project.cache.dir
        if not path.exists():
            return {"error": f"{path} doesn't exist yet"}
        if sys.platform == "win32":
            subprocess.Popen(["explorer", f"/select,{path}"] if path.is_file() else ["explorer", str(path)])
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path if path.is_dir() else path.parent)])
        return {"shown": str(path)}


def default_renderer() -> str | None:
    """This repository's Release build, else the one of a ``vk_gltf_renderer`` checkout next to it
    (a worktree of this repository has no build of its own)."""
    repo = Path(__file__).resolve().parents[3]
    name = "vk_gltf_renderer.exe" if os.name == "nt" else "vk_gltf_renderer"
    for exe in (repo / "_bin" / "Release" / name, repo.parent / "vk_gltf_renderer" / "_bin" / "Release" / name):
        if exe.exists():
            return str(exe)
    return None


def _remember(folder: Path) -> None:
    config = read_config()
    recent = [f for f in config["recent"] if os.path.normcase(f) != os.path.normcase(str(folder))]
    config["recent"] = [str(folder)] + recent[:19]
    write_config(config)


def _project_info(folder: Path) -> dict | None:
    """A recent project for the start page, read without opening USD."""
    index = folder / "index.sqlite"
    if not index.exists():
        return None
    import sqlite3

    db = sqlite3.connect(index)
    try:
        row = db.execute("SELECT value FROM meta WHERE name='project'").fetchone()
        settings = json.loads(row[0]) if row else {}
        leaves = db.execute("SELECT COUNT(*) FROM leaves").fetchone()[0]
    finally:
        db.close()
    stage = settings.get("input")
    return {"cache": str(folder), "input": stage, "name": Path(stage).name if stage else folder.name,
            "leaves": leaves, "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(index.stat().st_mtime))}


def _leaf_view(p: Project, unit_id: str, leaf: dict | None) -> dict:
    if not p.scanner.units[unit_id]["geo"]:
        return {"status": "places only"}
    if leaf is None:
        return {"status": "to convert"}
    if "error" in leaf:
        return {"status": "failed", "error": leaf["error"]}
    view = {k: leaf.get(k) for k in ("triangles", "meshes", "nodes", "point_instances", "lights", "cameras",
                                     "materials", "images", "seconds", "bounds")}
    view["status"] = "empty" if leaf.get("empty") else "converted"
    view["ktx"], view["looks"] = bool(leaf.get("ktx")), sorted(leaf.get("looks", {}))
    view["file"] = str(p.cache.path(geo_leaf(unit_id)))
    return view


def _count(entries) -> int:
    return sum(1 if ("unit" in e or "switch" in e) else _count(e.get("children", [])) for e in entries)


def _names(spec: str, switch_set: str, name: str | None) -> bool:
    """Whether the --variant ``spec`` is the choice for ``switch_set`` (at ``name``, or everywhere)."""
    set_, _, rest = spec.partition("=")
    names = rest.partition(":")[2]
    return set_ == switch_set and (name in names.split(",") if name else not names)


def handler_for(app: App):
    page = (Path(__file__).with_name("ui.html")).read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet: the page polls
            pass

        def _send(self, body: bytes, kind: str = "application/json", code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value, code: int = 200):
            self._send(json.dumps(value).encode(), code=code)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    self._send(page, "text/html; charset=utf-8")
                elif url.path == "/api/home":
                    self._json(app.home())
                elif url.path == "/api/state":
                    self._json(app.state())
                elif url.path == "/api/assets_with":
                    self._json(app.assets_with(q.get("set", "")))
                elif url.path == "/api/unit":
                    self._json(app.unit(q.get("id") or (app.project.top if app.project else "") or ""))
                elif url.path == "/api/job":
                    self._json(app.job.state(int(q.get("since", 0))))
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self):
            # Only the page itself: a JSON post from another site needs a preflight, which this server
            # never answers, and a browser names the site it comes from.
            origin = self.headers.get("Origin")
            if (self.headers.get("Content-Type", "").split(";")[0] != "application/json"
                    or (origin and urlparse(origin).netloc != self.headers.get("Host"))):
                self._json({"error": "forbidden"}, 403)
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                if self.path == "/api/pick":
                    self._json({"path": pick(body.get("kind", "file"), body.get("title", "Open"), body.get("start"))})
                elif self.path == "/api/new":
                    self._json(app.new_project(body["stage"], body.get("caches")))
                elif self.path == "/api/caches":
                    config = read_config()
                    config["caches"] = body["path"]
                    write_config(config)
                    self._json(app.home())
                elif self.path == "/api/open_project":
                    self._json(app.open_project(Path(body["cache"])))
                elif self.path == "/api/close":
                    self._json(app.close())
                elif self.path == "/api/settings":
                    self._json(app.set_settings(body))
                elif self.path == "/api/choose":
                    self._json(app.choose(body["set"], body.get("name"), body.get("value")))
                elif self.path == "/api/run":
                    self._json(app.run(body.get("steps", []), body.get("units", []), bool(body.get("force"))))
                elif self.path == "/api/open":
                    self._json(app.open_renderer())
                elif self.path == "/api/choose_asset":
                    self._json(app.choose_asset(body["unit"], body["set"], body.get("value")))
                elif self.path == "/api/stop":
                    self._json(app.stop())
                elif self.path == "/api/move_project":
                    self._json(app.move_project(body["path"]))
                elif self.path == "/api/reveal":
                    self._json(app.reveal(body.get("what", "project")))
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def serve(args) -> int:
    app = App(Path(args.cache) if args.cache else None, Path(args.root) if args.root else None, args.renderer)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(app))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"usd-world: {url} (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
