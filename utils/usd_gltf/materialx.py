"""MaterialX surfaces (OpenPBR, Autodesk Standard Surface) read into OmniSurface's parameters, which
``convert.py`` already turns into glTF (OmniSurface is Standard Surface under other names).

Two sources, one translation:

- a shader in the USD stage (``info:id`` ``ND_open_pbr_surface_surfaceshader``...), with its inputs
  wired to MaterialX node graphs;
- a ``.mtlx`` document the stage references: USD can't compose it without the usdMtlx plugin
  (not in usd-core), so the binding points at a material that isn't there. The document is read as
  XML instead.

A connected input becomes a texture when the graph behind it leads to an image node, through the
nodes that pass an image along (convert, extract a channel, normal map, multiply...). Other graphs
keep the input's value.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from functools import lru_cache

from pxr import Sdf, Usd, UsdShade

# MaterialX node -> its family: what the translation below knows.
FAMILIES = {"open_pbr_surface": "open_pbr_surface", "standard_surface": "standard_surface"}

# Inputs that carry an image along a graph, tried in order (a multiply's first operand, a mix's
# foreground...): the texture found at the end stands in for the graph.
_PASS = ("in", "in1", "fg", "bg", "in2")
_CHANNELS = "rgba"

# (MaterialX input, OmniSurface parameter), per family.
_OPENPBR = {
    "base_weight": "diffuse_reflection_weight", "base_color": "diffuse_reflection_color",
    "base_metalness": "metalness", "specular_weight": "specular_reflection_weight",
    "specular_color": "specular_reflection_color", "specular_roughness": "specular_reflection_roughness",
    "specular_ior": "specular_reflection_ior", "transmission_weight": "specular_transmission_weight",
    "transmission_color": "specular_transmission_color", "subsurface_weight": "subsurface_weight",
    "subsurface_color": "subsurface_transmission_color", "coat_weight": "coat_weight",
    "coat_roughness": "coat_roughness", "fuzz_weight": "sheen_weight", "fuzz_color": "sheen_color",
    "fuzz_roughness": "sheen_roughness", "emission_color": "emission_color",
    "geometry_opacity": "geometry_opacity", "geometry_thin_walled": "thin_walled", "geometry_normal": "geometry_normal",
}
_STANDARD = {
    "base": "diffuse_reflection_weight", "base_color": "diffuse_reflection_color", "metalness": "metalness",
    "specular": "specular_reflection_weight", "specular_color": "specular_reflection_color",
    "specular_roughness": "specular_reflection_roughness", "specular_IOR": "specular_reflection_ior",
    "transmission": "specular_transmission_weight", "transmission_color": "specular_transmission_color",
    "subsurface": "subsurface_weight", "subsurface_color": "subsurface_transmission_color",
    "coat": "coat_weight", "coat_roughness": "coat_roughness", "sheen": "sheen_weight", "sheen_color": "sheen_color",
    "sheen_roughness": "sheen_roughness", "emission": "emission_weight", "emission_color": "emission_color",
    "opacity": "geometry_opacity", "thin_walled": "thin_walled", "normal": "geometry_normal",
}
# Each family's defaults, in OmniSurface's names (OmniSurface's own differ from OpenPBR's).
_DEFAULTS = {
    "open_pbr_surface": {"diffuse_reflection_weight": 1.0, "diffuse_reflection_color": (0.8, 0.8, 0.8),
                         "specular_reflection_roughness": 0.3, "coat_roughness": 0.0, "sheen_roughness": 0.5},
    "standard_surface": {"diffuse_reflection_weight": 0.8, "diffuse_reflection_color": (1.0, 1.0, 1.0),
                         "specular_reflection_roughness": 0.2, "coat_roughness": 0.1, "sheen_roughness": 0.3},
}
# OmniSurface parameters whose texture goes to an ``_image`` input (and a channel to ``_image_mode``).
_TEXTURED = ("diffuse_reflection_color", "specular_reflection_roughness", "metalness", "emission_color",
             "geometry_opacity", "geometry_normal")


def omnisurface_params(family: str, inputs: dict) -> dict:
    """OmniSurface parameters for a MaterialX surface's ``inputs``: {name: value, or ("texture",
    path, channel)}."""
    names = _OPENPBR if family == "open_pbr_surface" else _STANDARD
    p = dict(_DEFAULTS[family])
    for name, value in inputs.items():
        target = names.get(name)
        if target is None:
            if family == "open_pbr_surface" and name == "emission_luminance" and not isinstance(value, tuple):
                p["emission_weight"] = 1.0 if float(value) > 0 else 0.0
                p["emission_intensity"] = float(value)  # nits, as glTF emission (KHR_materials_emissive_strength)
            continue
        if isinstance(value, tuple) and value[:1] == ("texture",):
            if target in _TEXTURED:
                key = "geometry_normal_image" if target == "geometry_normal" else f"{target}_image"
                p[key], p[f"{target}_image_mode"] = value[1], value[2]
                if target == "diffuse_reflection_color":
                    p[target] = (1.0, 1.0, 1.0)  # the texture is the color; the weight still scales it
        else:
            p[target] = value
    sub = p.get("subsurface_weight", 0.0)
    if not isinstance(sub, tuple) and float(sub) > 0 and not p.get("diffuse_reflection_color_image"):
        # Subsurface replaces the diffuse base by its weight: the surface takes its color.
        base, inside = _color(p.get("diffuse_reflection_color")), _color(p.get("subsurface_transmission_color"), (0.8, 0.8, 0.8))
        p["diffuse_reflection_color"] = tuple(b + (s - b) * float(sub) for b, s in zip(base, inside))
    if isinstance(p.get("geometry_opacity"), (tuple, list)):  # Standard Surface's opacity is a color
        p["geometry_opacity"] = sum(p["geometry_opacity"][:3]) / 3
    if p.get("geometry_opacity", 1.0) < 1.0 or p.get("geometry_opacity_image"):
        p["enable_opacity"] = True
    if float(p.get("specular_transmission_weight", 0.0)) > 0:
        p["enable_specular_transmission"] = True
        if _scalar(inputs.get("transmission_depth"), 0.0) > 0:
            p["specular_transmission_color"] = (1.0, 1.0, 1.0)  # the color is the medium's (extra_extensions)
    if p.get("thin_walled") and float(p.get("subsurface_weight", 0.0)) > 0:
        p["enable_diffuse_transmission"] = True
    return p


def _scalar(v, default=0.0) -> float:
    if v is None or (isinstance(v, tuple) and v[:1] == ("texture",)):
        return default
    if isinstance(v, (tuple, list)):
        return sum(float(x) for x in v[:3]) / len(v[:3])
    return float(v)


def _color(v, default=(1.0, 1.0, 1.0)) -> list[float]:
    if v is None or (isinstance(v, tuple) and v[:1] == ("texture",)):
        return list(default)
    return [float(x) for x in v[:3]] if isinstance(v, (tuple, list)) else [float(v)] * 3


# Per family: transmission (weight, depth, color, scatter, scatter anisotropy), subsurface (weight,
# color, radius, radius scale, anisotropy), thin-walled, specular anisotropy and rotation, thin film
# (weight, thickness, ior) and the thin film thickness unit in nanometres.
_EXTRA = {
    "open_pbr_surface": ("transmission_weight", "transmission_depth", "transmission_color", "transmission_scatter",
                         "transmission_scatter_anisotropy", "subsurface_weight", "subsurface_color", "subsurface_radius",
                         "subsurface_radius_scale", "subsurface_scatter_anisotropy", "geometry_thin_walled",
                         "specular_anisotropy", "specular_rotation", "thin_film_weight", "thin_film_thickness",
                         "thin_film_ior", 1000.0),
    "standard_surface": ("transmission", "transmission_depth", "transmission_color", "transmission_scatter",
                         "transmission_scatter_anisotropy", "subsurface", "subsurface_color", "subsurface_scale",
                         "subsurface_radius", "subsurface_anisotropy", "thin_walled",
                         "specular_anisotropy", "specular_rotation", None, "thin_film_thickness", "thin_film_IOR", 1.0),
}


def extra_extensions(family: str, inputs: dict, meters_per_unit: float) -> dict:
    """The glTF extensions OmniSurface's translation doesn't make: what is under the surface (a
    transmissive medium's KHR_materials_volume attenuation and KHR_materials_scatter, or subsurface
    scattering as KHR_materials_diffuse_transmission into a scattering volume), dispersion,
    specular anisotropy and thin-film iridescence. Distances are in scene units in MaterialX and in
    metres in glTF (KHR_materials_volume: world space)."""
    tw, td, tc, ts, ta, sw, sc, sr, srs, sa, thin, aniso, rot, fw, ft, fior, nm = _EXTRA[family]
    ext = {}
    if family == "open_pbr_surface":
        abbe = _scalar(inputs.get("transmission_dispersion_abbe_number"), 20.0)
        scale = _scalar(inputs.get("transmission_dispersion_scale"), 0.0)
        dispersion = 20.0 / abbe * scale if abbe > 0 else 0.0
    else:  # Standard Surface: the Abbe number itself, 0 for none
        abbe = _scalar(inputs.get("transmission_dispersion"), 0.0)
        dispersion = 20.0 / abbe if abbe > 0 else 0.0
    if dispersion > 0:
        ext["KHR_materials_dispersion"] = {"dispersion": round(dispersion, 4)}
    strength = _scalar(inputs.get(aniso))
    if strength > 0:  # rotation: a fraction of a turn in both families
        ext["KHR_materials_anisotropy"] = {"anisotropyStrength": strength,
                                           "anisotropyRotation": _scalar(inputs.get(rot)) * 2.0 * 3.141592653589793}
    thickness = _scalar(inputs.get(ft), 0.5 if family == "open_pbr_surface" else 0.0) * nm
    film = _scalar(inputs.get(fw)) if fw else (1.0 if thickness > 0 else 0.0)
    if film > 0 and thickness > 0:
        ext["KHR_materials_iridescence"] = {"iridescenceFactor": film,
                                           "iridescenceIor": _scalar(inputs.get(fior), 1.4 if family == "open_pbr_surface" else 1.5),
                                           "iridescenceThicknessMinimum": thickness, "iridescenceThicknessMaximum": thickness}
    if inputs.get(thin) is True or _scalar(inputs.get(thin), 0.0) > 0:
        return ext  # a thin wall has no medium
    if _scalar(inputs.get(tw)) > 0:
        volume = {"thicknessFactor": 0.01}
        depth = _scalar(inputs.get(td))
        if depth > 0:
            volume["attenuationDistance"] = depth * meters_per_unit
            volume["attenuationColor"] = _color(inputs.get(tc))
        ext["KHR_materials_volume"] = volume
        scatter = _color(inputs.get(ts), (0.0, 0.0, 0.0))
        if max(scatter) > 0:
            ext["KHR_materials_scatter"] = {"multiscatterColorFactor": scatter, "scatterStrengthFactor": 1.0,
                                            "scatterAnisotropy": _scalar(inputs.get(ta))}
    elif _scalar(inputs.get(sw)) > 0:
        color = _color(inputs.get(sc), (0.8, 0.8, 0.8))
        # The mean free path per channel: OpenPBR's radius (a distance) times its radius scale (a
        # color); Standard Surface's radius (a color) times its scale (a distance).
        if family == "open_pbr_surface":
            radius = [_scalar(inputs.get(sr), 1.0) * c for c in _color(inputs.get(srs), (1.0, 0.5, 0.25))]
        else:
            radius = [_scalar(inputs.get(sr), 1.0) * c for c in _color(inputs.get(srs), (1.0, 1.0, 1.0))]
        far = max(max(radius), 1e-6)
        ext["KHR_materials_diffuse_transmission"] = {"diffuseTransmissionFactor": _scalar(inputs.get(sw)),
                                                     "diffuseTransmissionColorFactor": color}
        # The radius is a mean free path: extinction 1 / radius per channel, so a short radius
        # scatters right under the surface (and the scattering albedo keeps its color), not
        # absorbs. glTF's attenuation color is the transmittance over attenuationDistance.
        ext["KHR_materials_volume"] = {"thicknessFactor": 0.01, "attenuationDistance": far * meters_per_unit,
                                       "attenuationColor": [math.exp(-far / max(r, far * 1e-3)) for r in radius]}
        ext["KHR_materials_scatter"] = {"multiscatterColorFactor": color, "scatterStrengthFactor": 1.0,
                                        "scatterAnisotropy": _scalar(inputs.get(sa))}
    return ext


# ---- shaders in the stage ----
def surface_shader(material: UsdShade.Material) -> UsdShade.Shader | None:
    """The material's MaterialX surface (``outputs:mtlx:surface``, through node graphs), if one of
    the families above."""
    source = material.ComputeSurfaceSource("mtlx")[0]
    return source if source and family_of(source) else None


def family_of(shader: UsdShade.Shader) -> str | None:
    shader_id = str(shader.GetIdAttr().Get() or "")
    for node, family in FAMILIES.items():
        if shader_id.startswith(f"ND_{node}"):
            return family
    return None


def stage_inputs(shader: UsdShade.Shader, resolve) -> dict:
    """The shader's inputs as ``omnisurface_params`` takes them; ``resolve(value, attr)`` makes an
    asset path absolute."""
    out = {}
    for inp in shader.GetInputs():
        found = _stage_texture(inp, resolve, None, 0) if inp.HasConnectedSource() else None
        if found:
            out[inp.GetBaseName()] = found
        elif inp.GetAttr().HasAuthoredValue():  # also behind a graph without an image
            out[inp.GetBaseName()] = _plain(inp.Get())
    return out


def _stage_texture(inp: UsdShade.Input, resolve, channel, depth):
    if depth > 16:
        return None
    for source in inp.GetConnectedSources()[0]:
        prim, name = source.source.GetPrim(), source.sourceName
        for _ in range(16):  # through graph outputs, which may lead to another graph's
            if not prim.IsA(UsdShade.NodeGraph) or prim.IsA(UsdShade.Shader):
                break
            output = UsdShade.NodeGraph(prim).GetOutput(name)
            if not (output and output.HasConnectedSource()):
                prim = None
                break
            nxt = output.GetConnectedSources()[0][0]
            prim, name = nxt.source.GetPrim(), nxt.sourceName
        if prim is None:
            continue
        node = UsdShade.Shader(prim)
        node_id = str(node.GetIdAttr().Get() or "")
        if node_id.startswith(("ND_image_", "ND_tiledimage_", "ND_gltf_image_")):
            file_in = node.GetInput("file")
            path = resolve(file_in.Get(), file_in.GetAttr()) if file_in else None
            return ("texture", path, channel or "average") if path else None
        if node_id.startswith("ND_extract_"):
            index = node.GetInput("index")
            channel = _CHANNELS[int(index.Get() or 0)] if index else "r"
        for name in _PASS:
            nxt = node.GetInput(name)
            if nxt and nxt.HasConnectedSource():
                found = _stage_texture(nxt, resolve, channel, depth + 1)
                if found:
                    return found
    return None


def _plain(v):
    if isinstance(v, Sdf.AssetPath):
        return v.path
    if hasattr(v, "__len__") and not isinstance(v, str):
        return tuple(float(x) for x in v)
    return v


# ---- .mtlx documents ----
def _note_read(file: str) -> None:
    from .convert import note_read  # convert imports this module
    note_read(file)


def missing_binding(prim: Usd.Prim) -> tuple[str, str] | None:
    """(``.mtlx`` file, material name) when ``prim`` is bound to a material that should come from
    a MaterialX document the stage references (and USD couldn't read)."""
    material, rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if material or not rel:
        return None
    targets = rel.GetTargets()
    if not targets:
        return None
    path, stage = targets[0], prim.GetStage()
    # The nearest ancestor that exists holds the reference: what's below it is the path in the document.
    anchor = path
    while anchor != Sdf.Path.absoluteRootPath and not stage.GetPrimAtPath(anchor):
        anchor = anchor.GetParentPath()
    holder = stage.GetPrimAtPath(anchor)
    if not holder:
        return None
    rest = path.MakeRelativePath(anchor)
    for spec in holder.GetPrimStack():
        refs = spec.referenceList
        for ref in (*refs.GetAddedOrExplicitItems(), *refs.prependedItems, *refs.appendedItems):
            if not ref.assetPath.lower().endswith(".mtlx"):
                continue
            file = spec.layer.ComputeAbsolutePath(ref.assetPath)
            name = rest.name
            if os.path.isfile(file) and name in _document(file):
                _note_read(file)
                return file, name
    return None


def document_material(file: str, name: str) -> tuple[str, dict] | None:
    """(family, inputs) of the material ``name`` of a ``.mtlx`` document, or None."""
    _note_read(file)  # the document is parsed once per process, the reads count per conversion
    return _document(file).get(name)


@lru_cache(None)
def _document(file: str) -> dict:
    """{material name: (family, inputs)} of a ``.mtlx`` document."""
    root = ET.parse(file).getroot()
    nodes: dict[str, ET.Element] = {}
    graphs: dict[str, ET.Element] = {}
    for el in root:
        if el.tag == "nodegraph":
            graphs[el.get("name")] = el
        elif el.get("name"):
            nodes[el.get("name")] = el
    base = os.path.dirname(file)
    out = {}
    for mat in root.iter("surfacematerial"):
        surface = next((i for i in mat.iter("input") if i.get("name") == "surfaceshader"), None)
        shader = nodes.get(surface.get("nodename")) if surface is not None else None
        if shader is None or shader.tag not in FAMILIES:
            continue
        inputs = {}
        for inp in shader.iter("input"):
            if inp.get("value") is not None:
                inputs[inp.get("name")] = _xml_value(inp.get("value"), inp.get("type"))
            else:
                found = _xml_texture(inp, nodes, graphs, root.get("fileprefix", ""), base, None, 0)
                if found:
                    inputs[inp.get("name")] = found
        out[mat.get("name")] = (FAMILIES[shader.tag], inputs)
    return out


def _xml_value(text: str, kind: str | None):
    parts = [p.strip() for p in text.split(",")]
    if kind == "boolean":
        return text.strip().lower() == "true"
    if kind in ("filename", "string"):
        return text
    try:
        values = tuple(float(p) for p in parts)
    except ValueError:
        return text
    return values[0] if len(values) == 1 else values


def _xml_texture(inp, nodes, graphs, prefix, base, channel, depth):
    """The texture behind a document's input: its node (``nodename``), or a node graph's output."""
    if depth > 16:
        return None
    scope = nodes
    node = None
    if inp.get("nodegraph"):
        graph = graphs.get(inp.get("nodegraph"))
        if graph is None:
            return None
        scope = {el.get("name"): el for el in graph if el.get("name")}
        prefix = graph.get("fileprefix") or prefix
        output = next((o for o in graph.iter("output") if o.get("name") == (inp.get("output") or "out")), None)
        node = scope.get(output.get("nodename")) if output is not None else None
    elif inp.get("nodename"):
        node = scope.get(inp.get("nodename"))
    if node is None:
        return None
    if node.tag in ("image", "tiledimage", "gltf_image"):
        f = next((i for i in node.iter("input") if i.get("name") == "file"), None)
        if f is None or not f.get("value"):
            return None
        path = os.path.normpath(os.path.join(base, (node.get("fileprefix") or prefix) + f.get("value")))
        return ("texture", path, channel or "average") if os.path.isfile(path) else None
    if node.tag == "extract":
        index = next((i for i in node.iter("input") if i.get("name") == "index"), None)
        channel = _CHANNELS[int(float(index.get("value", 0)))] if index is not None else "r"
    for name in _PASS:
        nxt = next((i for i in node.iter("input") if i.get("name") == name), None)
        if nxt is not None and (nxt.get("nodename") or nxt.get("nodegraph")):
            found = _xml_texture(nxt, scope, graphs, prefix, base, channel, depth + 1)
            if found:
                return found
    return None
