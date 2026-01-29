import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def normalize_blender_path(candidate):
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


def resolve_blender_bin(user_path):
    candidates = []
    if user_path:
        candidates.append(user_path)
    env_path = os.environ.get("BLENDER_BIN")
    if env_path:
        candidates.append(env_path)
    which_path = shutil.which("blender") or shutil.which("blender.exe")
    if which_path:
        candidates.append(which_path)

    if sys.platform == "win32":
        candidates.extend(
            [
                r"C:\Program Files\Blender Foundation",
                r"C:\Program Files (x86)\Blender Foundation",
            ]
        )

    for candidate in candidates:
        resolved = normalize_blender_path(candidate)
        if resolved:
            return resolved

    if sys.platform == "win32":
        for root in (Path(r"C:\Program Files\Blender Foundation"), Path(r"C:\Program Files (x86)\Blender Foundation")):
            if not root.exists():
                continue
            for exe in root.glob(r"Blender*\blender.exe"):
                return exe

    return None


def infer_export_format(output_path, format_arg, embed):
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


def build_blender_script():
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
"""


def run_blender(blender_bin, input_path, output_path, export_format, apply_transform, y_up, textures_dir):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as temp_script:
        temp_script.write(build_blender_script())
        script_path = temp_script.name

    try:
        command = [
            str(blender_bin),
            "--background",
            "--factory-startup",
            "--python",
            script_path,
            "--",
            str(input_path),
            str(output_path),
            export_format,
            "1" if apply_transform else "0",
            "1" if y_up else "0",
            str(textures_dir or ""),
        ]
        logging.debug("Running: %s", " ".join(command))
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Blender exited with code {result.returncode}")
    finally:
        try:
            os.remove(script_path)
        except OSError:
            pass


def show_gui_and_collect():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:
        logging.error("Tkinter unavailable: %s", exc)
        return None

    root = tk.Tk()
    root.title("FBX to glTF Converter")
    root.resizable(False, False)

    input_var = tk.StringVar()
    output_var = tk.StringVar()
    textures_var = tk.StringVar()
    format_var = tk.StringVar(value="gltf")
    embed_var = tk.BooleanVar(value=False)
    apply_var = tk.BooleanVar(value=True)
    yup_var = tk.BooleanVar(value=True)

    def choose_input():
        path = filedialog.askopenfilename(
            title="Select FBX/OBJ file",
            filetypes=[("FBX/OBJ files", "*.fbx *.obj"), ("FBX files", "*.fbx"), ("OBJ files", "*.obj"), ("All files", "*.*")],
        )
        if path:
            input_var.set(path)
            if not output_var.get():
                base = os.path.splitext(path)[0]
                output_var.set(base + ".gltf")
            if not textures_var.get():
                candidate = Path(path).parent.parent / "textures"
                if candidate.is_dir():
                    textures_var.set(str(candidate))

    def choose_output():
        ext = ".glb" if format_var.get() == "glb" else ".gltf"
        path = filedialog.asksaveasfilename(
            title="Save glTF",
            defaultextension=ext,
            filetypes=[("glTF", "*.gltf"), ("glTF Binary", "*.glb"), ("All files", "*.*")],
        )
        if path:
            output_var.set(path)

    def choose_textures():
        path = filedialog.askdirectory(title="Select textures folder")
        if path:
            textures_var.set(path)

    def on_format_change(*_):
        out = output_var.get()
        if not out:
            return
        suffix = Path(out).suffix.lower()
        target = ".glb" if format_var.get() == "glb" else ".gltf"
        if suffix in (".gltf", ".glb"):
            output_var.set(str(Path(out).with_suffix(target)))

    def update_blender_status():
        blender_bin = resolve_blender_bin(None)
        if blender_bin:
            blender_status_var.set(f"Blender found: {blender_bin}")
            blender_status_label.config(fg="#1a7f37")
        else:
            blender_status_var.set("Blender not found. Set BLENDER_BIN or add Blender to PATH.")
            blender_status_label.config(fg="#b91c1c")
        return blender_bin

    def run_and_close():
        if not input_var.get():
            messagebox.showerror("Missing input", "Select an FBX file.")
            return
        if not output_var.get():
            messagebox.showerror("Missing output", "Select an output file.")
            return
        if not update_blender_status():
            messagebox.showerror(
                "Blender missing",
                "Blender not found. Install Blender or set BLENDER_BIN / PATH.",
            )
            return
        root.quit()

    ttk.Label(root, text="Input FBX/OBJ:").grid(row=0, column=0, padx=8, pady=6, sticky="w")
    ttk.Entry(root, textvariable=input_var, width=46).grid(row=0, column=1, padx=4, pady=6)
    ttk.Button(root, text="Browse...", command=choose_input).grid(row=0, column=2, padx=8, pady=6)

    ttk.Label(root, text="Output:").grid(row=1, column=0, padx=8, pady=6, sticky="w")
    ttk.Entry(root, textvariable=output_var, width=46).grid(row=1, column=1, padx=4, pady=6)
    ttk.Button(root, text="Browse...", command=choose_output).grid(row=1, column=2, padx=8, pady=6)

    ttk.Label(root, text="Format:").grid(row=2, column=0, padx=8, pady=6, sticky="w")
    fmt = ttk.Combobox(root, textvariable=format_var, values=["gltf", "glb"], state="readonly", width=8)
    fmt.grid(row=2, column=1, padx=4, pady=6, sticky="w")
    fmt.bind("<<ComboboxSelected>>", on_format_change)

    ttk.Label(root, text="Textures folder (optional):").grid(row=3, column=0, padx=8, pady=6, sticky="w")
    ttk.Entry(root, textvariable=textures_var, width=46).grid(row=3, column=1, padx=4, pady=6)
    ttk.Button(root, text="Browse...", command=choose_textures).grid(row=3, column=2, padx=8, pady=6)

    embed_chk = ttk.Checkbutton(root, text="Embed buffers/textures (.gltf)", variable=embed_var)
    embed_chk.grid(row=4, column=1, padx=4, pady=2, sticky="w")

    apply_chk = ttk.Checkbutton(root, text="Apply transforms", variable=apply_var)
    apply_chk.grid(row=5, column=1, padx=4, pady=2, sticky="w")

    yup_chk = ttk.Checkbutton(root, text="Export Y-up", variable=yup_var)
    yup_chk.grid(row=6, column=1, padx=4, pady=2, sticky="w")

    blender_status_var = tk.StringVar()
    blender_status_label = tk.Label(root, textvariable=blender_status_var, anchor="w")
    blender_status_label.grid(row=7, column=0, columnspan=3, padx=8, pady=4, sticky="w")
    update_blender_status()

    btn_frame = ttk.Frame(root)
    btn_frame.grid(row=8, column=1, padx=4, pady=8, sticky="e")
    ttk.Button(btn_frame, text="Cancel", command=root.quit).grid(row=0, column=0, padx=4)
    ttk.Button(btn_frame, text="Convert", command=run_and_close).grid(row=0, column=1, padx=4)

    root.mainloop()
    root.destroy()

    if not input_var.get() or not output_var.get():
        return None

    return {
        "input_fbx": input_var.get(),
        "output_path": output_var.get(),
        "format": format_var.get(),
        "embed": bool(embed_var.get()),
        "apply_transform": bool(apply_var.get()),
        "y_up": bool(yup_var.get()),
        "textures_dir": textures_var.get(),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert FBX to glTF using Blender (free, no license fee). "
            "Requires Blender installed or BLENDER_BIN set."
        )
    )
    parser.add_argument("input_fbx", nargs="?", help="Path to the input FBX or OBJ file")
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
        "--log",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log), format="%(levelname)s: %(message)s")

    if not args.input_fbx or not args.output_path:
        gui_result = show_gui_and_collect()
        if gui_result is None:
            logging.info("No conversion performed.")
            return
        args.input_fbx = gui_result["input_fbx"]
        args.output_path = gui_result["output_path"]
        args.format = gui_result["format"]
        args.embed = gui_result["embed"]
        args.apply_transform = gui_result["apply_transform"]
        args.y_up = gui_result["y_up"]
        args.textures = gui_result["textures_dir"] or None

    input_path = Path(args.input_fbx)
    if not input_path.is_file():
        logging.error("Input file not found: %s", input_path)
        sys.exit(1)

    if input_path.suffix.lower() not in (".fbx", ".obj"):
        logging.error("Unsupported input format: %s", input_path.suffix)
        sys.exit(1)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        export_format = infer_export_format(output_path, args.format, args.embed)
    except ValueError as exc:
        logging.error(str(exc))
        sys.exit(1)

    blender_bin = resolve_blender_bin(args.blender)
    if not blender_bin or not blender_bin.exists():
        logging.error(
            "Blender not found. Install Blender or set BLENDER_BIN / --blender."
        )
        sys.exit(1)

    logging.info("Converting %s -> %s (%s)", input_path, output_path, export_format)
    try:
        run_blender(
            blender_bin,
            input_path,
            output_path,
            export_format,
            args.apply_transform,
            args.y_up,
            args.textures,
        )
    except Exception as exc:
        logging.error("Conversion failed: %s", exc)
        sys.exit(1)

    logging.info("Done.")


if __name__ == "__main__":
    main()
