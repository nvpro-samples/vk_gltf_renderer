"""Convert FBX / OBJ / USD to glTF using Blender as the conversion backend.

Run with no arguments to launch a Tk GUI with a live Blender log; pass
both input and output positional arguments for headless CLI mode.

Sections:
    Discovery       - Locate Blender and probe its version.
    Job             - Validation, ``Job`` dataclass, error types.
    Embedded script - Python program executed inside Blender (string).
    Runner          - Single streaming runner; sync wrapper for CLI.
    OS / Theme      - File-manager opener, optional sv-ttk integration.
    GUI             - Tk window with live log and progress.
    CLI             - argparse entry point.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator, Optional


# --------------------------------------------------------------------------------------
# Module constants
# --------------------------------------------------------------------------------------

SUPPORTED_INPUTS = (".fbx", ".obj", ".usd", ".usda", ".usdc", ".usdz")
GLTF_OUTPUTS = (".gltf", ".glb")
STAGE_MARKERS = (
    ("[stage] importing", 1, "Importing"),
    ("[stage] ensuring materials", 2, "Adding materials"),
    ("[stage] exporting", 3, "Exporting"),
)


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------

def normalize_blender_path(candidate: str | os.PathLike[str] | None) -> Optional[Path]:
    """Resolve ``candidate`` to a Blender executable path or return ``None``."""
    if not candidate:
        return None
    path = Path(candidate).expanduser()

    if path.is_dir():
        for name in ("blender.exe", "blender"):
            exe = path / name
            if exe.exists():
                return exe
        return None

    if path.is_file():
        return path

    if sys.platform == "win32" and path.suffix == "":
        exe = path.with_suffix(".exe")
        if exe.exists():
            return exe

    return None


def resolve_blender_bin(user_path: str | os.PathLike[str] | None = None) -> Optional[Path]:
    """Find a usable Blender executable. Searches user override, env, PATH, and Windows defaults."""
    candidates: list[str | os.PathLike[str]] = []
    if user_path:
        candidates.append(user_path)
    env_path = os.environ.get("BLENDER_BIN")
    if env_path:
        candidates.append(env_path)
    which_path = shutil.which("blender") or shutil.which("blender.exe")
    if which_path:
        candidates.append(which_path)

    if sys.platform == "win32":
        candidates.extend([
            r"C:\Program Files\Blender Foundation",
            r"C:\Program Files (x86)\Blender Foundation",
        ])

    for candidate in candidates:
        resolved = normalize_blender_path(candidate)
        if resolved:
            return resolved

    if sys.platform == "win32":
        for root in (
            Path(r"C:\Program Files\Blender Foundation"),
            Path(r"C:\Program Files (x86)\Blender Foundation"),
        ):
            if not root.exists():
                continue
            for exe in root.glob(r"Blender*\blender.exe"):
                return exe

    return None


@lru_cache(maxsize=None)
def get_blender_version(blender_bin: str) -> str:
    """First non-empty line of ``blender --version`` (cached); empty on failure.

    Cache keys are strings rather than ``Path`` because :func:`functools.lru_cache`
    requires hashable args; callers stringify before passing.
    """
    try:
        result = subprocess.run(
            [blender_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return ""
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


# --------------------------------------------------------------------------------------
# Job (validation + frozen parameter bundle)
# --------------------------------------------------------------------------------------

def infer_export_format(output_path: Path, format_arg: Optional[str], embed: bool) -> str:
    if embed and format_arg == "glb":
        raise ValueError("--embed cannot be used with --format glb")

    if format_arg == "glb":
        return "GLB"
    if format_arg == "gltf":
        return "GLTF_EMBEDDED" if embed else "GLTF_SEPARATE"

    suffix = output_path.suffix.lower()
    if suffix == ".glb":
        return "GLB"
    if suffix == ".gltf":
        return "GLTF_EMBEDDED" if embed else "GLTF_SEPARATE"

    raise ValueError("Output file must end with .gltf or .glb, or pass --format.")


class JobError(ValueError):
    """Raised by :func:`prepare_job` with a single human-readable message."""


@dataclass(frozen=True)
class Job:
    """Validated parameters for a single Blender conversion."""
    blender_bin: Path
    input_path: Path
    output_path: Path
    export_format: str
    apply_transform: bool
    y_up: bool
    textures_dir: Optional[str]


def prepare_job(
    *,
    input_str: Optional[str],
    output_str: Optional[str],
    format: Optional[str],
    embed: bool,
    apply_transform: bool,
    y_up: bool,
    textures_dir: Optional[str],
    blender_override: Optional[str] = None,
) -> Job:
    """Validate inputs and return a frozen :class:`Job`.

    Both CLI (``logging.error``) and GUI (status banner) render the resulting
    :class:`JobError` message verbatim.
    """
    if not input_str:
        raise JobError("Select an input file (FBX, OBJ, or USD).")
    if not output_str:
        raise JobError("Select an output file.")

    input_path = Path(input_str)
    if not input_path.is_file():
        raise JobError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() not in SUPPORTED_INPUTS:
        raise JobError(
            f"Unsupported input '{input_path.suffix}' "
            f"(expected one of {', '.join(SUPPORTED_INPUTS)})."
        )

    output_path = Path(output_str)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise JobError(f"Cannot create output folder: {exc}") from exc

    try:
        export_format = infer_export_format(output_path, format, embed)
    except ValueError as exc:
        raise JobError(str(exc)) from exc

    blender_bin = resolve_blender_bin(blender_override)
    if not blender_bin:
        raise JobError("Blender not found. Install Blender or set BLENDER_BIN / PATH.")

    return Job(
        blender_bin=blender_bin,
        input_path=input_path,
        output_path=output_path,
        export_format=export_format,
        apply_transform=apply_transform,
        y_up=y_up,
        textures_dir=textures_dir or None,
    )


# --------------------------------------------------------------------------------------
# Embedded Blender script
# --------------------------------------------------------------------------------------

def build_blender_script() -> str:
    # Stage banners are matched by the GUI to advance a 3-step progress bar;
    # flush=True is critical so the GUI sees lines without buffering delay.
    return """\
import bpy
import sys
import os

argv = sys.argv
argv = argv[argv.index("--") + 1:]

if len(argv) != 6:
    raise SystemExit("Expected args: <input_path> <output_path> <export_format> <apply_transform> <y_up> <textures_dir>")

input_path, output_path, export_format, apply_transform, y_up, textures_dir = argv
apply_transform = apply_transform == "1"
y_up = y_up == "1"

bpy.ops.wm.read_factory_settings(use_empty=True)

ext = os.path.splitext(input_path)[1].lower()
print("[stage] importing %s: %s" % (ext, input_path), flush=True)
if ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=input_path, use_image_search=True)
elif ext == ".obj":
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=input_path)
    else:
        try:
            bpy.ops.preferences.addon_enable(module="io_scene_obj")
        except Exception:
            pass
        bpy.ops.import_scene.obj(filepath=input_path)
elif ext in (".usd", ".usda", ".usdc", ".usdz"):
    if not hasattr(bpy.ops.wm, "usd_import"):
        raise SystemExit(
            "This Blender build has no wm.usd_import (USD I/O). "
            "Use Blender 3.2+ with USD enabled."
        )
    # Default Blender USD import uses import_all_materials=False; many stages then get no Blender materials.
    try:
        bpy.ops.wm.usd_import(
            filepath=input_path,
            import_materials=True,
            import_all_materials=True,
        )
    except TypeError:
        bpy.ops.wm.usd_import(filepath=input_path)
else:
    raise SystemExit("Unsupported input format: " + ext)

def try_find_textures(path):
    if path and os.path.isdir(path):
        try:
            bpy.ops.file.find_missing_files(directory=path)
        except Exception:
            pass

input_dir = os.path.dirname(input_path)
try_find_textures(input_dir)

textures_dir = textures_dir.strip()
if textures_dir:
    try_find_textures(textures_dir)
else:
    parent_textures = os.path.join(os.path.dirname(input_dir), "textures")
    try_find_textures(parent_textures)

print("[stage] ensuring materials", flush=True)
def ensure_mesh_materials():
    # Importers (especially USD) may leave meshes with no material slots; glTF then has no materials[].
    shared = None
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.data is None:
            continue
        mesh = obj.data
        needs = len(mesh.materials) == 0 or all(m is None for m in mesh.materials)
        if not needs:
            continue
        if shared is None:
            shared = bpy.data.materials.new(name="ImportedDefaultMaterial")
            shared.use_nodes = True
            principled = shared.node_tree.nodes.get("Principled BSDF")
            if principled:
                principled.inputs["Base Color"].default_value = (0.72, 0.72, 0.74, 1.0)
                principled.inputs["Roughness"].default_value = 0.45
        mesh.materials.clear()
        mesh.materials.append(shared)

ensure_mesh_materials()

print("[stage] exporting glTF: %s" % output_path, flush=True)
bpy.ops.export_scene.gltf(
    filepath=output_path,
    export_format=export_format,
    export_yup=y_up,
    export_apply=apply_transform,
    export_texcoords=True,
    export_normals=True,
    export_tangents=True,
    export_materials="EXPORT",
)
print("[stage] complete", flush=True)
"""


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------

LineCallback = Callable[[str], None]
ProcCallback = Callable[["subprocess.Popen[str]"], None]


class BlenderRunError(RuntimeError):
    """Non-zero Blender exit. Carries returncode and the tail of merged output."""

    def __init__(self, returncode: int, output_tail: str = "") -> None:
        msg = f"Blender exited with code {returncode}"
        if output_tail:
            msg += f"\n--- Blender output (tail) ---\n{output_tail}"
        super().__init__(msg)
        self.returncode = returncode
        self.output_tail = output_tail


@contextmanager
def _blender_script() -> Iterator[Path]:
    """Write the embedded Blender script to a temp file; remove it on exit."""
    fd, name = tempfile.mkstemp(suffix=".py", text=True)
    os.close(fd)
    path = Path(name)
    try:
        path.write_text(build_blender_script(), encoding="utf-8")
        yield path
    finally:
        try:
            path.unlink()
        except OSError:
            pass


def _build_command(script_path: Path, job: Job) -> list[str]:
    return [
        str(job.blender_bin),
        "--background",
        "--factory-startup",
        "--python",
        str(script_path),
        "--",
        str(job.input_path),
        str(job.output_path),
        job.export_format,
        "1" if job.apply_transform else "0",
        "1" if job.y_up else "0",
        str(job.textures_dir or ""),
    ]


def _blender_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_blender_streaming(
    job: Job,
    on_line: LineCallback,
    on_started: Optional[ProcCallback] = None,
) -> int:
    """Spawn Blender, stream merged stdout/stderr line-by-line through ``on_line``.

    ``on_started(proc)`` is called once the subprocess is spawned so the caller
    can keep a reference for cancellation. Returns the process exit code.
    """
    with _blender_script() as script_path:
        command = _build_command(script_path, job)
        logging.debug("Running: %s", " ".join(command))
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_blender_env(),
        )
        if on_started:
            on_started(proc)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                on_line(line.rstrip("\r\n"))
        finally:
            proc.wait()
        return proc.returncode


def run_blender(job: Job) -> None:
    """Synchronous wrapper for CLI mode; raises :class:`BlenderRunError` on failure."""
    captured: list[str] = []
    rc = run_blender_streaming(job, on_line=captured.append)
    if rc != 0:
        raise BlenderRunError(rc, "\n".join(captured[-30:]))


# --------------------------------------------------------------------------------------
# OS integration + Theme
# --------------------------------------------------------------------------------------

def open_in_file_manager(path: str | os.PathLike[str]) -> None:
    """Reveal ``path`` (file or directory) in the system file manager."""
    p = Path(path)
    target = str(p if p.is_dir() else p.parent)
    try:
        if sys.platform == "win32":
            os.startfile(target)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])
    except Exception as exc:  # noqa: BLE001 - UX-only; must never crash
        logging.warning("Could not open file manager for %s: %s", target, exc)


def apply_theme(theme: str) -> bool:
    """Apply sv-ttk theme if installed. Returns True iff applied."""
    if theme == "system":
        return False
    try:
        import sv_ttk  # type: ignore
    except ImportError:
        logging.warning(
            "Theme '%s' requested but the optional package 'sv-ttk' is not installed. "
            "Falling back to the system Tk theme. To enable dark/light themes, run:\n"
            "    pip install sv-ttk\n"
            "Or pass --theme system to silence this warning.",
            theme,
        )
        return False
    try:
        sv_ttk.set_theme("dark" if theme == "dark" else "light")
        return True
    except Exception as exc:  # noqa: BLE001 - sv_ttk may raise tkinter.TclError; UX-only
        logging.warning("Failed to apply '%s' theme via sv-ttk: %s", theme, exc)
        return False


# --------------------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------------------

class State(Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE_OK = "done_ok"
    DONE_ERR = "done_err"
    CANCELLED = "cancelled"


class ConverterGui:
    """Tk window with a live Blender log, progress bar, and Cancel / Open Folder."""

    POLL_MS = 50
    PROGRESS_INTERVAL_MS = 12
    CANCEL_GRACE_MS = 2000

    def __init__(self, theme: str = "dark", blender_override: Optional[str] = None) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox, ttk
        except ImportError as exc:
            logging.error("Tkinter unavailable: %s", exc)
            raise

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox

        self.root = tk.Tk()
        self.root.title("3D scene to glTF")
        self.root.minsize(680, 560)
        self.theme_applied = apply_theme(theme)

        self.blender_override = blender_override

        self.queue: queue.Queue = queue.Queue()
        self.proc: Optional[subprocess.Popen] = None
        self.worker: Optional[threading.Thread] = None
        self.state: State = State.IDLE
        self.last_output_path: Optional[Path] = None
        self.cancelling = False

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.textures_var = tk.StringVar()
        self.format_var = tk.StringVar(value="gltf")
        self.embed_var = tk.BooleanVar(value=False)
        self.apply_var = tk.BooleanVar(value=True)
        self.yup_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready.")
        self.blender_status_var = tk.StringVar()
        self.stage_var = tk.StringVar(value="")

        self._build_ui()
        self._refresh_blender_status()

    # ----- UI construction -----

    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        self.root.columnconfigure(0, weight=1)

        pad = {"padx": 6, "pady": 4}

        src = ttk.LabelFrame(self.root, text="Source", padding=8)
        src.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        src.columnconfigure(1, weight=1)

        ttk.Label(src, text="Input:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(src, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(src, text="Browse...", command=self._choose_input).grid(row=0, column=2, **pad)

        ttk.Label(src, text="Textures (optional):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(src, textvariable=self.textures_var).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(src, text="Browse...", command=self._choose_textures).grid(row=1, column=2, **pad)

        out = ttk.LabelFrame(self.root, text="Output", padding=8)
        out.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        out.columnconfigure(1, weight=1)

        ttk.Label(out, text="Path:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(out, textvariable=self.output_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(out, text="Browse...", command=self._choose_output).grid(row=0, column=2, **pad)

        ttk.Label(out, text="Format:").grid(row=1, column=0, sticky="w", **pad)
        fmt_frame = ttk.Frame(out)
        fmt_frame.grid(row=1, column=1, columnspan=2, sticky="w")
        fmt = ttk.Combobox(
            fmt_frame, textvariable=self.format_var,
            values=["gltf", "glb"], state="readonly", width=8,
        )
        fmt.pack(side="left", padx=(6, 12))
        fmt.bind("<<ComboboxSelected>>", self._on_format_change)
        ttk.Checkbutton(
            fmt_frame, text="Embed buffers/textures (.gltf only)",
            variable=self.embed_var,
        ).pack(side="left")

        opt = ttk.LabelFrame(self.root, text="Options", padding=8)
        opt.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        ttk.Checkbutton(opt, text="Apply transforms", variable=self.apply_var).grid(
            row=0, column=0, sticky="w", padx=6, pady=2
        )
        ttk.Checkbutton(opt, text="Export Y-up", variable=self.yup_var).grid(
            row=0, column=1, sticky="w", padx=18, pady=2
        )

        self.blender_status_label = tk.Label(
            self.root, textvariable=self.blender_status_var, anchor="w",
        )
        self.blender_status_label.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 2))

        self.status_label = tk.Label(
            self.root, textvariable=self.status_var, anchor="w",
            font=("TkDefaultFont", 10, "bold"),
        )
        self.status_label.grid(row=4, column=0, sticky="ew", padx=12, pady=(2, 4))

        prog_frame = ttk.Frame(self.root)
        prog_frame.grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 4))
        prog_frame.columnconfigure(1, weight=1)
        self.stage_label = ttk.Label(prog_frame, textvariable=self.stage_var, width=24)
        self.stage_label.grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(prog_frame, mode="indeterminate", length=240)
        self.progress.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        log_frame = ttk.LabelFrame(self.root, text="Log", padding=4)
        log_frame.grid(row=6, column=0, sticky="nsew", padx=8, pady=4)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.root.rowconfigure(6, weight=1)

        log_font = ("Consolas", 9) if sys.platform == "win32" else ("Monospace", 9)
        log_bg = "#1e1e1e" if self.theme_applied else "#ffffff"
        log_fg = "#e6e6e6" if self.theme_applied else "#1a1a1a"
        self.log = tk.Text(
            log_frame, height=10, wrap="none", state="disabled",
            font=log_font, background=log_bg, foreground=log_fg,
            insertbackground=log_fg,
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        sb_y = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        sb_y.grid(row=0, column=1, sticky="ns")
        sb_x = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log.xview)
        sb_x.grid(row=1, column=0, sticky="ew")
        self.log.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        self.log.tag_configure("stage", foreground="#5aa6ff")
        self.log.tag_configure("err", foreground="#ff7b72")
        self.log.tag_configure("meta", foreground="#9aa0a6")

        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(row=7, column=0, sticky="e", padx=8, pady=(4, 8))
        self.convert_btn = ttk.Button(btn_frame, text="Convert", command=self._on_convert)
        self.convert_btn.pack(side="left", padx=4)
        self.cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self._on_cancel)
        self.cancel_btn.state(["disabled"])
        self.cancel_btn.pack(side="left", padx=4)
        self.open_folder_btn = ttk.Button(
            btn_frame, text="Open folder", command=self._on_open_folder
        )
        self.open_folder_btn.state(["disabled"])
        self.open_folder_btn.pack(side="left", padx=4)
        self.close_btn = ttk.Button(btn_frame, text="Close", command=self._on_close)
        self.close_btn.pack(side="left", padx=4)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Control-Return>", lambda _e: self._on_convert())

    # ----- File pickers -----

    def _choose_input(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Select FBX / OBJ / USD file",
            filetypes=[
                ("FBX / OBJ / USD", "*.fbx *.obj *.usd *.usda *.usdc *.usdz"),
                ("FBX", "*.fbx"),
                ("OBJ", "*.obj"),
                ("USD", "*.usd *.usda *.usdc *.usdz"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.input_var.set(path)
        if not self.output_var.get():
            self.output_var.set(str(Path(path).with_suffix(".gltf")))
        if not self.textures_var.get():
            candidate = Path(path).parent.parent / "textures"
            if candidate.is_dir():
                self.textures_var.set(str(candidate))

    def _choose_output(self) -> None:
        ext = ".glb" if self.format_var.get() == "glb" else ".gltf"
        path = self.filedialog.asksaveasfilename(
            title="Save glTF",
            defaultextension=ext,
            filetypes=[("glTF", "*.gltf"), ("glTF Binary", "*.glb"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)

    def _choose_textures(self) -> None:
        path = self.filedialog.askdirectory(title="Select textures folder")
        if path:
            self.textures_var.set(path)

    def _on_format_change(self, *_args) -> None:
        out = self.output_var.get()
        if not out:
            return
        suffix = Path(out).suffix.lower()
        target = ".glb" if self.format_var.get() == "glb" else ".gltf"
        if suffix in GLTF_OUTPUTS:
            self.output_var.set(str(Path(out).with_suffix(target)))

    # ----- Status / log helpers -----

    def _refresh_blender_status(self) -> None:
        blender_bin = resolve_blender_bin(self.blender_override)
        if blender_bin:
            ver = get_blender_version(str(blender_bin))
            suffix = f"  ({ver})" if ver else ""
            self.blender_status_var.set(f"Blender: {blender_bin}{suffix}")
            self.blender_status_label.config(fg="#39c552")
        else:
            self.blender_status_var.set(
                "Blender not found. Set BLENDER_BIN or add Blender to PATH."
            )
            self.blender_status_label.config(fg="#ff7b72")

    def _set_status(self, message: str, level: str = "info") -> None:
        self.status_var.set(message)
        color = {
            "ok": "#39c552",
            "warn": "#e3b341",
            "error": "#ff7b72",
            "info": "#9aa0a6" if self.theme_applied else "#1a1a1a",
        }.get(level, "")
        if color:
            self.status_label.config(fg=color)

    def _append_log(self, text: str, tag: Optional[str] = None) -> None:
        self.log.config(state="normal")
        if tag:
            self.log.insert("end", text, tag)
        else:
            self.log.insert("end", text)
        self.log.see("end")
        self.log.config(state="disabled")

    def _clear_log(self) -> None:
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    @staticmethod
    def _classify_line(line: str) -> Optional[str]:
        lower = line.lower()
        if "[stage]" in lower:
            return "stage"
        if "error" in lower or "traceback" in lower or "exception" in lower:
            return "err"
        return None

    def _maybe_advance_stage(self, line: str) -> None:
        for marker, step, label in STAGE_MARKERS:
            if marker in line:
                if str(self.progress.cget("mode")) == "indeterminate":
                    self.progress.stop()
                    self.progress.config(mode="determinate", maximum=3.0, value=0.0)
                self.progress.config(value=float(step))
                self.stage_var.set(f"Step {step}/3: {label}")
                return

    # ----- Convert flow -----

    def _on_convert(self) -> None:
        if self.state is State.RUNNING:
            return
        try:
            job = prepare_job(
                input_str=self.input_var.get(),
                output_str=self.output_var.get(),
                format=self.format_var.get(),
                embed=bool(self.embed_var.get()),
                apply_transform=bool(self.apply_var.get()),
                y_up=bool(self.yup_var.get()),
                textures_dir=self.textures_var.get() or None,
                blender_override=self.blender_override,
            )
        except JobError as exc:
            self._set_status(str(exc), "error")
            return

        self.last_output_path = job.output_path
        self.cancelling = False
        self.proc = None
        self._clear_log()
        self._append_log(
            f"Blender:    {job.blender_bin}\n"
            f"Input:      {job.input_path}\n"
            f"Output:     {job.output_path}  ({job.export_format})\n"
            f"Transforms: {'apply' if job.apply_transform else 'keep'}    "
            f"Y-up: {job.y_up}    Textures: {job.textures_dir or '(default)'}\n"
            "----------------------------------------\n",
            tag="meta",
        )
        self._enter_state(State.RUNNING)

        self.worker = threading.Thread(
            target=self._worker_run,
            args=(job,),
            daemon=True,
        )
        self.worker.start()
        self.root.after(self.POLL_MS, self._drain)

    def _worker_run(self, job: Job) -> None:
        try:
            rc = run_blender_streaming(
                job,
                on_line=lambda line: self.queue.put(("LINE", line)),
                on_started=lambda proc: self.queue.put(("PROC", proc)),
            )
            self.queue.put(("DONE", rc))
        except Exception as exc:  # noqa: BLE001 - any failure must surface to GUI
            self.queue.put(("ERROR", str(exc)))

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "LINE":
                    self._append_log(payload + "\n", self._classify_line(payload))
                    self._maybe_advance_stage(payload)
                elif kind == "PROC":
                    self.proc = payload
                elif kind == "DONE":
                    self._finish(returncode=payload)
                    return
                elif kind == "ERROR":
                    self._finish(error=payload)
                    return
        except queue.Empty:
            pass
        if self.state is State.RUNNING:
            self.root.after(self.POLL_MS, self._drain)

    def _finish(self, returncode: Optional[int] = None, error: Optional[str] = None) -> None:
        if self.cancelling:
            self._enter_state(State.CANCELLED)
            self._set_status("Cancelled.", "warn")
            return
        if error is not None:
            self._enter_state(State.DONE_ERR)
            self._set_status(f"Conversion failed: {error}", "error")
            return
        if returncode == 0 and self.last_output_path and self.last_output_path.is_file():
            size = self.last_output_path.stat().st_size
            self._enter_state(State.DONE_OK)
            self._set_status(
                f"Done: {self.last_output_path.name}  ({size:,} bytes)",
                "ok",
            )
            return
        if returncode == 0:
            self._enter_state(State.DONE_ERR)
            self._set_status(
                f"Blender exited cleanly but no output file was produced: "
                f"{self.last_output_path}",
                "error",
            )
            return
        self._enter_state(State.DONE_ERR)
        self._set_status(f"Blender exited with code {returncode}.", "error")

    def _on_cancel(self) -> None:
        if self.state is not State.RUNNING:
            return
        self.cancelling = True
        self._set_status("Cancelling...", "warn")
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except (OSError, ProcessLookupError):
                pass

            def _force_kill() -> None:
                if self.proc and self.proc.poll() is None:
                    try:
                        self.proc.kill()
                    except (OSError, ProcessLookupError):
                        pass

            self.root.after(self.CANCEL_GRACE_MS, _force_kill)

    def _on_open_folder(self) -> None:
        if self.last_output_path:
            open_in_file_manager(self.last_output_path)

    def _on_close(self) -> None:
        if self.state is State.RUNNING:
            if not self.messagebox.askyesno(
                "Cancel conversion?",
                "A conversion is running. Cancel and exit?",
            ):
                return
            self._on_cancel()
        try:
            self.root.quit()
        except Exception:  # noqa: BLE001 - shutting down anyway
            pass

    # ----- State machine -----

    def _enter_state(self, state: State) -> None:
        self.state = state
        if state is State.RUNNING:
            self.convert_btn.state(["disabled"])
            self.cancel_btn.state(["!disabled"])
            self.open_folder_btn.state(["disabled"])
            self.stage_var.set("Starting Blender...")
            self.progress.config(mode="indeterminate")
            self.progress.start(self.PROGRESS_INTERVAL_MS)
            self._set_status("Running...", "info")
            return

        self.progress.stop()
        self.convert_btn.state(["!disabled"])
        self.cancel_btn.state(["disabled"])

        if state is State.DONE_OK:
            self.progress.config(mode="determinate", maximum=3.0, value=3.0)
            self.stage_var.set("Step 3/3: Done")
            if self.last_output_path:
                self.open_folder_btn.state(["!disabled"])
        elif state is State.CANCELLED:
            self.progress.config(mode="determinate", maximum=3.0, value=0.0)
            self.stage_var.set("Cancelled")
        elif state is State.DONE_ERR:
            self.stage_var.set("Failed")

    # ----- Lifecycle -----

    def run(self) -> None:
        self.root.mainloop()
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001 - already torn down
            pass


def show_gui(theme: str = "dark", blender_override: Optional[str] = None) -> bool:
    """Launch the GUI; runs until the user closes the window. Returns False on init failure."""
    try:
        gui = ConverterGui(theme=theme, blender_override=blender_override)
    except Exception:  # noqa: BLE001 - log already emitted by ConverterGui
        return False
    gui.run()
    return True


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert FBX, OBJ, or USD to glTF using Blender (free, no license fee). "
            "USD requires Blender 3.2+ with USD import (wm.usd_import). "
            "Requires Blender installed or BLENDER_BIN set."
        ),
        epilog=(
            "Examples:\n"
            "  %(prog)s model.fbx out/model.glb --format glb\n"
            "  %(prog)s Rover.usdc Rover.gltf --textures path/to/textures\n"
            "  %(prog)s   # launch GUI (live Blender log) when args omitted\n"
            "\n"
            "Optional theme dependency for the GUI:\n"
            "  pip install sv-ttk   # then run with --theme dark or --theme light\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to the input FBX, OBJ, or USD (.usd/.usda/.usdc/.usdz)",
    )
    parser.add_argument("output_path", nargs="?", help="Path to the output .gltf or .glb file")
    parser.add_argument(
        "--format",
        choices=["gltf", "glb"],
        help="Force output format regardless of file extension",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Embed buffers/textures in .gltf (ignored for .glb)",
    )
    parser.add_argument(
        "--textures",
        help="Optional textures directory to relink missing images",
    )
    parser.add_argument(
        "--apply-transform",
        action="store_true",
        default=True,
        help="Apply object transforms during export (default: on)",
    )
    parser.add_argument(
        "--no-apply-transform",
        action="store_false",
        dest="apply_transform",
        help="Disable applying transforms during export",
    )
    parser.add_argument(
        "--y-up",
        action="store_true",
        default=True,
        help="Export Y-up (default: on)",
    )
    parser.add_argument(
        "--no-y-up",
        action="store_false",
        dest="y_up",
        help="Disable Y-up conversion",
    )
    parser.add_argument(
        "--blender",
        help="Path to Blender executable (or set BLENDER_BIN)",
    )
    parser.add_argument(
        "--theme",
        choices=["dark", "light", "system"],
        default="system",
        help="GUI theme (sv-ttk; falls back to system Tk if sv-ttk is not installed)",
    )
    parser.add_argument(
        "--log",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log), format="%(levelname)s: %(message)s")

    if not args.input_path or not args.output_path:
        if not show_gui(theme=args.theme, blender_override=args.blender):
            sys.exit(1)
        return

    try:
        job = prepare_job(
            input_str=args.input_path,
            output_str=args.output_path,
            format=args.format,
            embed=args.embed,
            apply_transform=args.apply_transform,
            y_up=args.y_up,
            textures_dir=args.textures,
            blender_override=args.blender,
        )
    except JobError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    logging.info("Using Blender: %s", job.blender_bin)
    logging.info(
        "Converting %s -> %s (%s)", job.input_path, job.output_path, job.export_format,
    )
    try:
        run_blender(job)
    except BlenderRunError as exc:
        logging.error("Conversion failed: %s", exc)
        sys.exit(exc.returncode if exc.returncode > 0 else 1)

    if not job.output_path.is_file():
        logging.error("Expected output file was not created: %s", job.output_path)
        sys.exit(1)

    logging.info("Done: %s (%s bytes)", job.output_path, job.output_path.stat().st_size)


if __name__ == "__main__":
    main()
