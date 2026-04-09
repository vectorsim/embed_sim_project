"""
embedsim_gui.py
===============
EmbedSim Visual Block Diagram Designer
---------------------------------------
A Dear PyGui node-editor GUI for designing EmbedSim block diagrams
and generating Python simulation files.

Usage:
    python embedsim_gui.py

Features:
    • Block palette — double-click to add blocks to canvas
    • Node editor  — drag output pins to input pins to wire blocks
    • Parameters   — click any node to edit its parameters in the sidebar
    • Validate     — checks for algebraic loops (no VectorDelay in cycle)
    • Generate     — exports a ready-to-run EmbedSim .py file

Author: EmbedSim GUI Prototype
"""

import dearpygui.dearpygui as dpg
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Block registry — pure Python inspect-based, no .pyx needed
# ---------------------------------------------------------------------------

@dataclass
class ParamDef:
    name: str
    default: Any
    annotation: str = ""

@dataclass
class BlockDef:
    class_name: str
    module: str
    import_line: str
    color: Tuple[int,int,int,int]
    params: List[ParamDef]
    is_source: bool = False        # no input pins
    is_sink: bool   = False        # no output pin
    is_loop_breaker: bool = False  # VectorDelay etc.
    description: str = ""

def _make_params(defaults: Dict[str, Any]) -> List[ParamDef]:
    return [ParamDef(k, v) for k, v in defaults.items()]

# Hand-register the EmbedSim blocks visible in the GUI
# (inspect fallback would read __init__ signatures at runtime
#  once the real embedsim package is on sys.path)

BLOCK_REGISTRY: List[BlockDef] = [
    # ── Sources ─────────────────────────────────────────────────────────────
    BlockDef(
        class_name   = "SinusoidalGenerator",
        module       = "embedsim.source_blocks",
        import_line  = "from embedsim.source_blocks import SinusoidalGenerator",
        color        = (34, 139, 87, 255),
        params       = _make_params({"amplitude": 1.0, "freq": 50.0, "phase": 0.0}),
        is_source    = True,
        description  = "Sinusoidal signal source",
    ),
    BlockDef(
        class_name   = "ThreePhaseGenerator",
        module       = "embedsim.source_blocks",
        import_line  = "from embedsim.source_blocks import ThreePhaseGenerator",
        color        = (34, 139, 87, 255),
        params       = _make_params({"amplitude": 1.0, "freq": 50.0, "phase": 0.0}),
        is_source    = True,
        description  = "3-phase sinusoidal generator",
    ),
    BlockDef(
        class_name   = "VectorConstant",
        module       = "embedsim.source_blocks",
        import_line  = "from embedsim.source_blocks import VectorConstant",
        color        = (34, 139, 87, 255),
        params       = _make_params({"value": "[0.0]"}),
        is_source    = True,
        description  = "Constant vector output",
    ),
    BlockDef(
        class_name   = "VectorStep",
        module       = "embedsim.source_blocks",
        import_line  = "from embedsim.source_blocks import VectorStep",
        color        = (34, 139, 87, 255),
        params       = _make_params({"step_time": 0.0, "before_value": 0.0,
                                     "after_value": 1.0, "dim": 3}),
        is_source    = True,
        description  = "Step signal at t=step_time",
    ),
    BlockDef(
        class_name   = "VectorRamp",
        module       = "embedsim.source_blocks",
        import_line  = "from embedsim.source_blocks import VectorRamp",
        color        = (34, 139, 87, 255),
        params       = _make_params({"slope": 1.0, "initial_value": 0.0,
                                     "start_time": 0.0, "dim": 3}),
        is_source    = True,
        description  = "Linear ramp signal",
    ),
    BlockDef(
        class_name   = "GaussianNoiseBlock",
        module       = "embedsim.source_blocks",
        import_line  = "from embedsim.source_blocks import GaussianNoiseBlock",
        color        = (34, 139, 87, 255),
        params       = _make_params({"mean": 0.0, "std": 1.0, "dim": 1, "seed": "None"}),
        is_source    = True,
        description  = "Gaussian white noise",
    ),
    # ── Processing ──────────────────────────────────────────────────────────
    BlockDef(
        class_name   = "VectorGain",
        module       = "embedsim.processing_blocks",
        import_line  = "from embedsim.processing_blocks import VectorGain",
        color        = (37, 99, 235, 255),
        params       = _make_params({"gain": 1.0}),
        description  = "Scalar gain  y = K·u",
    ),
    BlockDef(
        class_name   = "VectorSum",
        module       = "embedsim.processing_blocks",
        import_line  = "from embedsim.processing_blocks import VectorSum",
        color        = (37, 99, 235, 255),
        params       = _make_params({"signs": "[1, 1]", "n_inputs": 2}),
        description  = "Sum of n inputs",
    ),
    BlockDef(
        class_name   = "ScriptBlock",
        module       = "embedsim.script_blocks",
        import_line  = "from embedsim.script_blocks import ScriptBlock",
        color        = (124, 58, 237, 255),
        params       = _make_params({"script": "output = u[0]",
                                     "output_dim": 3, "mode": "python"}),
        description  = "Custom Python / C script block",
    ),
    # ── Dynamic / loop breakers ─────────────────────────────────────────────
    BlockDef(
        class_name      = "VectorDelay",
        module          = "embedsim.simulation_engine",
        import_line     = "from embedsim.simulation_engine import VectorDelay",
        color           = (234, 88, 12, 255),
        params          = _make_params({"initial": "[0.0]"}),
        is_loop_breaker = True,
        description     = "z⁻¹ delay — loop breaker",
    ),
    BlockDef(
        class_name   = "VectorIntegrator",
        module       = "embedsim.dynamic_blocks",
        import_line  = "from embedsim.dynamic_blocks import VectorIntegrator",
        color        = (220, 38, 38, 255),
        params       = _make_params({"initial_state": "[0.0]", "dim": 1}),
        description  = "Continuous integrator  ẋ = u",
    ),
    # ── Sink ────────────────────────────────────────────────────────────────
    BlockDef(
        class_name   = "VectorEnd",
        module       = "embedsim.dynamic_blocks",
        import_line  = "from embedsim.dynamic_blocks import VectorEnd",
        color        = (100, 116, 139, 255),
        params       = [],
        is_sink      = True,
        description  = "Simulation sink (output terminal)",
    ),
]

BLOCK_BY_CLASS = {b.class_name: b for b in BLOCK_REGISTRY}

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

@dataclass
class NodeInst:
    """One placed block node on the canvas."""
    uid: int
    class_name: str
    var_name: str
    params: Dict[str, Any]          # current param values
    node_tag: int  = 0
    attr_in: int   = 0              # input attribute tag
    attr_out: int  = 0              # output attribute tag

_node_counter   = 0
_attr_counter   = 1000
_link_counter   = 5000
_var_counter: Dict[str, int] = {}  # class_name → count for unique var names

nodes:  Dict[int, NodeInst] = {}   # uid → NodeInst
links:  Dict[int, Tuple[int,int]] = {}   # link_uid → (attr_out, attr_in)

# attr_tag → node uid
attr_to_node: Dict[int, int] = {}

selected_node_uid: Optional[int] = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_uid() -> int:
    global _node_counter
    _node_counter += 1
    return _node_counter

def _next_attr() -> int:
    global _attr_counter
    _attr_counter += 1
    return _attr_counter

def _next_link() -> int:
    global _link_counter
    _link_counter += 1
    return _link_counter

def _make_var_name(class_name: str) -> str:
    count = _var_counter.get(class_name, 0) + 1
    _var_counter[class_name] = count
    short = {
        "SinusoidalGenerator": "sin_src",
        "ThreePhaseGenerator":  "three_phase",
        "VectorConstant":       "const",
        "VectorStep":           "step",
        "VectorRamp":           "ramp",
        "GaussianNoiseBlock":   "noise",
        "VectorGain":           "gain",
        "VectorSum":            "summer",
        "ScriptBlock":          "script",
        "VectorDelay":          "delay",
        "VectorIntegrator":     "integr",
        "VectorEnd":            "sink",
    }.get(class_name, class_name.lower())
    return f"{short}_{count}" if count > 1 else short

def _param_str(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)

# ---------------------------------------------------------------------------
# Code generator
# ---------------------------------------------------------------------------

def generate_code(sim_T: float, sim_dt: float, solver: str) -> str:
    """Generate a complete EmbedSim .py file from the current canvas."""

    if not nodes:
        return "# No blocks on canvas."

    # Collect needed imports
    imports_needed = set()
    imports_needed.add("from embedsim.simulation_engine import EmbedSim, ODESolver")
    for nd in nodes.values():
        bdef = BLOCK_BY_CLASS.get(nd.class_name)
        if bdef:
            imports_needed.add(bdef.import_line)
        if bdef and bdef.is_loop_breaker:
            imports_needed.add("from embedsim.simulation_engine import VectorDelay")

    # Build adjacency: attr_out → attr_in
    # We need: for each node, which nodes feed into it
    # link: (attr_out_tag, attr_in_tag)
    in_edges: Dict[int, List[int]] = {uid: [] for uid in nodes}   # uid → [src_uid]
    for attr_out, attr_in in links.values():
        src_uid = attr_to_node.get(attr_out)
        dst_uid = attr_to_node.get(attr_in)
        if src_uid and dst_uid:
            in_edges[dst_uid].append(src_uid)

    # Build lines
    lines: List[str] = []
    lines.append('"""')
    lines.append('Generated by EmbedSim GUI')
    lines.append('Run:  python <this_file>.py')
    lines.append('"""')
    lines.append("")

    # imports
    for imp in sorted(imports_needed):
        lines.append(imp)
    lines.append("")
    lines.append("# --- Simulation parameters ---")
    lines.append(f"T_SIM = {sim_T}")
    lines.append(f"DT    = {sim_dt}")
    lines.append("")
    lines.append("# --- Block instantiation ---")

    for nd in nodes.values():
        bdef = BLOCK_BY_CLASS.get(nd.class_name)
        args = [f'"{nd.var_name}"']
        for p in (bdef.params if bdef else []):
            val = nd.params.get(p.name, p.default)
            vs  = _param_str(val)
            # quote strings except known list/None literals
            if isinstance(val, str) and not (vs.startswith('[') or vs == 'None'
                                              or vs in ('python','c')):
                vs = f'"{vs}"'
            args.append(f"{p.name}={vs}")
        lines.append(f"{nd.var_name} = {nd.class_name}({', '.join(args)})")

    lines.append("")
    lines.append("# --- Wiring (>> connections) ---")

    # Emit A >> B for each link
    emitted = set()
    for attr_out, attr_in in links.values():
        src_uid = attr_to_node.get(attr_out)
        dst_uid = attr_to_node.get(attr_in)
        if src_uid and dst_uid:
            key = (src_uid, dst_uid)
            if key not in emitted:
                src_name = nodes[src_uid].var_name
                dst_name = nodes[dst_uid].var_name
                lines.append(f"{src_name} >> {dst_name}")
                emitted.add(key)

    lines.append("")
    lines.append("# --- Simulation ---")

    # Find sinks
    sink_names = [nd.var_name for nd in nodes.values()
                  if BLOCK_BY_CLASS.get(nd.class_name, BlockDef("","","",
                     (0,0,0,0),[],False,False,False,"")).is_sink]
    sinks_str = ", ".join(sink_names) if sink_names else "# TODO: add a VectorEnd sink"

    lines.append(f"sim = EmbedSim(")
    lines.append(f"    sinks  = [{sinks_str}],")
    lines.append(f"    T      = T_SIM,")
    lines.append(f"    dt     = DT,")
    lines.append(f"    solver = ODESolver.{solver},")
    lines.append(f")")
    lines.append("")
    lines.append("# --- Add scope channels ---")
    for nd in nodes.values():
        bdef = BLOCK_BY_CLASS.get(nd.class_name)
        if bdef and not bdef.is_sink:
            lines.append(f'sim.scope.add({nd.var_name}, label="{nd.var_name}")')

    lines.append("")
    lines.append("sim.run(verbose=False, progress_bar=True)")
    lines.append("")
    lines.append("# --- Plot ---")
    lines.append("from embedsim.plot_helper import create_plotter")
    lines.append("ph = create_plotter(sim)")
    signal_list = [f'"{nd.var_name}[0]"' for nd in nodes.values()
                   if not BLOCK_BY_CLASS.get(nd.class_name,
                      BlockDef("","","", (0,0,0,0),[],False,False,False,"")).is_sink]
    lines.append(f"ph.easyplot(signals=[{', '.join(signal_list)}],")
    lines.append( "           title='EmbedSim GUI — Generated Diagram')")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Algebraic loop validator (simple DFS — no actual EmbedSim import needed)
# ---------------------------------------------------------------------------

def validate_loops() -> List[str]:
    """Return list of error strings, empty if OK."""
    errors = []

    # Build adjacency: uid → [dst_uid]  (directed graph following signals)
    adj: Dict[int, List[int]] = {uid: [] for uid in nodes}
    for attr_out, attr_in in links.values():
        src = attr_to_node.get(attr_out)
        dst = attr_to_node.get(attr_in)
        if src and dst:
            adj[src].append(dst)

    # DFS cycle detection — skip edges that pass through a LoopBreaker
    WHITE, GREY, BLACK = 0, 1, 2
    color_map = {uid: WHITE for uid in nodes}

    def dfs(uid: int, path: List[int]) -> bool:
        bdef = BLOCK_BY_CLASS.get(nodes[uid].class_name)
        if bdef and bdef.is_loop_breaker:
            return False   # cuts the cycle
        color_map[uid] = GREY
        for nb in adj.get(uid, []):
            if color_map[nb] == GREY:
                cycle_names = [nodes[u].var_name for u in path] + [nodes[nb].var_name]
                errors.append("Algebraic loop: " + " → ".join(cycle_names)
                               + "  (insert VectorDelay to break it)")
                return True
            if color_map[nb] == WHITE:
                if dfs(nb, path + [uid]):
                    return True
        color_map[uid] = BLACK
        return False

    for uid in nodes:
        if color_map[uid] == WHITE:
            dfs(uid, [])

    if not any(BLOCK_BY_CLASS.get(nd.class_name,
               BlockDef("","","", (0,0,0,0),[],False,False,False,"")).is_sink
               for nd in nodes.values()):
        errors.append("No VectorEnd sink block found — add one to define simulation output.")

    return errors

# ---------------------------------------------------------------------------
# GUI — add a node to the canvas
# ---------------------------------------------------------------------------

def add_node(class_name: str, pos: Tuple[int,int] = (200, 200)) -> None:
    global selected_node_uid

    bdef = BLOCK_BY_CLASS.get(class_name)
    if not bdef:
        return

    uid      = _next_uid()
    var_name = _make_var_name(class_name)
    params   = {p.name: p.default for p in bdef.params}

    nd = NodeInst(uid=uid, class_name=class_name,
                  var_name=var_name, params=params)

    # -- Dear PyGui node --
    node_tag  = dpg.generate_uuid()
    nd.node_tag = node_tag

    r, g, b, a = bdef.color

    with dpg.node(label=f"{class_name}\n{var_name}",
                  tag=node_tag,
                  parent="node_editor",
                  pos=pos):

        # Title bar colour
        dpg.bind_item_theme(node_tag, _make_node_theme(r, g, b))

        # Input pin (not for sources)
        if not bdef.is_source:
            attr_in = _next_attr()
            nd.attr_in = attr_in
            with dpg.node_attribute(tag=attr_in,
                                    attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text("in", color=(200, 200, 200, 200))
            attr_to_node[attr_in] = uid

        # Parameter display (static text — click node to edit in sidebar)
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_text(f"[{class_name}]",
                         color=(r, g, b, 220))
            if bdef.description:
                dpg.add_text(bdef.description,
                             color=(160, 160, 160, 200))
            for p in bdef.params[:4]:          # show up to 4 params inline
                val = params[p.name]
                dpg.add_text(f"  {p.name} = {_param_str(val)}",
                             color=(210, 210, 180, 230),
                             tag=_make_param_label_tag(uid, p.name))

        # Output pin (not for sinks)
        if not bdef.is_sink:
            attr_out = _next_attr()
            nd.attr_out = attr_out
            with dpg.node_attribute(tag=attr_out,
                                    attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_text("out", color=(200, 200, 200, 200))
            attr_to_node[attr_out] = uid

    nodes[uid] = nd
    _refresh_status()

_param_label_tags: Dict[Tuple[int,str], int] = {}

def _make_param_label_tag(uid: int, pname: str) -> int:
    key = (uid, pname)
    if key not in _param_label_tags:
        _param_label_tags[key] = dpg.generate_uuid()
    return _param_label_tags[key]

# ---------------------------------------------------------------------------
# Node themes
# ---------------------------------------------------------------------------
_theme_cache: Dict[Tuple[int,int,int], int] = {}

def _make_node_theme(r: int, g: int, b: int) -> int:
    key = (r, g, b)
    if key in _theme_cache:
        return _theme_cache[key]
    theme = dpg.generate_uuid()
    with dpg.theme(tag=theme):
        with dpg.theme_component(dpg.mvNode):
            dpg.add_theme_color(dpg.mvNodeCol_TitleBar,
                                (r, g, b, 200), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered,
                                (min(r+30,255), min(g+30,255), min(b+30,255), 220),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected,
                                (min(r+50,255), min(g+50,255), min(b+50,255), 255),
                                category=dpg.mvThemeCat_Nodes)
    _theme_cache[key] = theme
    return theme

# ---------------------------------------------------------------------------
# Sidebar: show / edit selected node parameters
# ---------------------------------------------------------------------------

_sidebar_widgets: List[int] = []

def show_sidebar(uid: int) -> None:
    global selected_node_uid
    selected_node_uid = uid

    # Clear previous widgets
    for tag in _sidebar_widgets:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    _sidebar_widgets.clear()

    nd   = nodes.get(uid)
    bdef = BLOCK_BY_CLASS.get(nd.class_name) if nd else None
    if not nd or not bdef:
        return

    parent = "sidebar_content"

    def add(tag, fn, *a, **kw):
        _sidebar_widgets.append(tag)
        fn(*a, tag=tag, parent=parent, **kw)

    t = dpg.generate_uuid()
    dpg.add_text(f"Block: {nd.class_name}", tag=t, parent=parent,
                 color=(255, 220, 100, 255))
    _sidebar_widgets.append(t)

    t = dpg.generate_uuid()
    dpg.add_text(bdef.description, tag=t, parent=parent,
                 color=(160, 160, 160, 220))
    _sidebar_widgets.append(t)

    t = dpg.generate_uuid()
    dpg.add_separator(tag=t, parent=parent)
    _sidebar_widgets.append(t)

    # Var name editor
    t = dpg.generate_uuid()
    dpg.add_text("Variable name:", tag=t, parent=parent)
    _sidebar_widgets.append(t)

    t_inp = dpg.generate_uuid()
    dpg.add_input_text(tag=t_inp, parent=parent,
                       default_value=nd.var_name, width=180,
                       callback=lambda s, d: _update_varname(uid, d))
    _sidebar_widgets.append(t_inp)

    t = dpg.generate_uuid()
    dpg.add_separator(tag=t, parent=parent)
    _sidebar_widgets.append(t)

    # Parameter editors
    for p in bdef.params:
        t = dpg.generate_uuid()
        dpg.add_text(f"{p.name}:", tag=t, parent=parent,
                     color=(200, 200, 200, 220))
        _sidebar_widgets.append(t)

        val = nd.params.get(p.name, p.default)
        t_w = dpg.generate_uuid()

        if isinstance(val, bool):
            dpg.add_checkbox(tag=t_w, parent=parent, default_value=val,
                             callback=lambda s, d, pn=p.name: _update_param(uid, pn, d))
        elif isinstance(val, int):
            dpg.add_input_int(tag=t_w, parent=parent, default_value=val,
                              width=160,
                              callback=lambda s, d, pn=p.name: _update_param(uid, pn, d))
        elif isinstance(val, float):
            dpg.add_input_float(tag=t_w, parent=parent, default_value=val,
                                width=160, format="%.6g",
                                callback=lambda s, d, pn=p.name: _update_param(uid, pn, d))
        else:
            # string / list / None
            dpg.add_input_text(tag=t_w, parent=parent,
                               default_value=str(val), width=180,
                               callback=lambda s, d, pn=p.name: _update_param(uid, pn, d))
        _sidebar_widgets.append(t_w)

def _update_varname(uid: int, new_name: str) -> None:
    nd = nodes.get(uid)
    if nd and new_name.strip():
        nd.var_name = new_name.strip()
        if dpg.does_item_exist(nd.node_tag):
            dpg.set_item_label(nd.node_tag,
                               f"{nd.class_name}\n{nd.var_name}")

def _update_param(uid: int, pname: str, value: Any) -> None:
    nd = nodes.get(uid)
    if nd:
        nd.params[pname] = value
        # refresh inline label on node
        label_tag = _param_label_tags.get((uid, pname))
        if label_tag and dpg.does_item_exist(label_tag):
            dpg.set_value(label_tag, f"  {pname} = {_param_str(value)}")

# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------

def _refresh_status() -> None:
    n_nodes = len(nodes)
    n_links = len(links)
    dpg.set_value("status_text",
                  f"  Blocks: {n_nodes}   Wires: {n_links}   "
                  f"| Double-click palette to add  |  "
                  f"Drag output→input to wire  |  Click node to edit")

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def cb_link_created(sender, app_data):
    a_out, a_in = app_data
    luid = _next_link()
    links[luid] = (a_out, a_in)
    dpg.add_node_link(a_out, a_in, parent="node_editor", tag=luid)
    _refresh_status()

def cb_link_deleted(sender, app_data):
    luid = app_data
    if luid in links:
        del links[luid]
    if dpg.does_item_exist(luid):
        dpg.delete_item(luid)
    _refresh_status()

def _on_viewport_click(sender, app_data):
    # DPG 2.x: poll get_selected_nodes() after any mouse click.
    # This fires for all clicks in the viewport; we only act when
    # exactly one node is selected in our editor.
    if not dpg.does_item_exist("node_editor"):
        return
    selected = dpg.get_selected_nodes("node_editor")
    if len(selected) == 1:
        node_tag = selected[0]
        for uid, nd in nodes.items():
            if nd.node_tag == node_tag:
                show_sidebar(uid)
                return

def cb_palette_double_click(sender, app_data):
    # sender is the selectable, its label is the class_name
    class_name = dpg.get_item_label(sender)
    # place at a slightly random offset so nodes don't stack
    n = len(nodes)
    x = 250 + (n % 4) * 220
    y = 100 + (n // 4) * 180
    add_node(class_name, pos=(x, y))

def cb_validate(sender, app_data):
    errors = validate_loops()
    dpg.set_value("validate_output", "")
    if errors:
        msg = "\n".join(f"⚠  {e}" for e in errors)
        dpg.set_value("validate_output", msg)
        dpg.configure_item("validate_output", color=(255, 100, 80, 255))
    else:
        dpg.set_value("validate_output", "✓  Diagram looks valid — no algebraic loops detected.")
        dpg.configure_item("validate_output", color=(80, 220, 120, 255))

def cb_generate(sender, app_data):
    T    = dpg.get_value("gen_T")
    dt   = dpg.get_value("gen_dt")
    solver = dpg.get_value("gen_solver")
    code = generate_code(T, dt, solver)
    dpg.set_value("code_output", code)

def cb_save(sender, app_data):
    path = dpg.get_value("save_path").strip()
    if not path:
        path = "embedsim_generated.py"
    T      = dpg.get_value("gen_T")
    dt     = dpg.get_value("gen_dt")
    solver = dpg.get_value("gen_solver")
    code   = generate_code(T, dt, solver)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    dpg.set_value("validate_output",
                  f"✓  Saved to: {path}")
    dpg.configure_item("validate_output", color=(80, 220, 120, 255))

def cb_clear_canvas(sender, app_data):
    global nodes, links, attr_to_node, _var_counter, selected_node_uid

    # Delete links FIRST — deleting a node while links are still attached
    # causes a segfault in DPG 2.x because the link items reference the
    # node attribute items which are being destroyed.
    for luid in list(links.keys()):
        if dpg.does_item_exist(luid):
            dpg.delete_item(luid)
    links.clear()

    # Now safe to delete nodes
    for nd in list(nodes.values()):
        if dpg.does_item_exist(nd.node_tag):
            dpg.delete_item(nd.node_tag)
    nodes.clear()

    attr_to_node.clear()
    _var_counter.clear()
    _param_label_tags.clear()
    selected_node_uid = None
    for tag in _sidebar_widgets:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    _sidebar_widgets.clear()
    dpg.set_value("validate_output", "")
    dpg.set_value("code_output", "")
    _refresh_status()

def cb_load_example(sender, app_data):
    """Load the algebraic loop example from the docstring."""
    cb_clear_canvas(None, None)

    add_node("SinusoidalGenerator", pos=(80, 200))
    add_node("VectorDelay",         pos=(80, 380))
    add_node("VectorSum",           pos=(340, 280))
    add_node("VectorGain",          pos=(580, 280))
    add_node("VectorEnd",           pos=(800, 280))

    # Set sensible defaults for the example
    sin_uid   = list(nodes.keys())[0]
    delay_uid = list(nodes.keys())[1]
    sum_uid   = list(nodes.keys())[2]
    gain_uid  = list(nodes.keys())[3]

    nodes[sin_uid].params.update({"amplitude": 1.0, "freq": 2.0, "phase": 0.0})
    nodes[delay_uid].params.update({"initial": "[0.0]"})
    nodes[gain_uid].params.update({"gain": 0.5})

    # Wire: sin >> sum, delay >> sum, sum >> gain, gain >> sink, gain >> delay
    def wire(src_uid, dst_uid):
        a_out = nodes[src_uid].attr_out
        a_in  = nodes[dst_uid].attr_in
        luid  = _next_link()
        links[luid] = (a_out, a_in)
        dpg.add_node_link(a_out, a_in, parent="node_editor", tag=luid)

    wire(sin_uid,   sum_uid)
    wire(delay_uid, sum_uid)
    wire(sum_uid,   gain_uid)
    wire(gain_uid,  list(nodes.keys())[4])   # sink
    wire(gain_uid,  delay_uid)

    _refresh_status()
    dpg.set_value("validate_output",
                  "Example loaded: algebraic loop diagram from example_algebraic_loop.py")
    dpg.configure_item("validate_output", color=(100, 180, 255, 255))

# ---------------------------------------------------------------------------
# Global app theme
# ---------------------------------------------------------------------------

def _build_global_theme() -> int:
    theme = dpg.generate_uuid()
    with dpg.theme(tag=theme):
        with dpg.theme_component(dpg.mvAll):
            # Dark background
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg,     (18, 20, 26, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg,      (22, 25, 32, 255))
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg,      (28, 32, 42, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg,      (35, 40, 55, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,(45,52,70,255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg,      (15, 17, 22, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,(25, 28, 38, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button,       (37, 99, 180, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,(50,130,220,255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (20, 70, 150, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Header,       (37, 99, 180, 180))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,(50,130,220,200))
            dpg.add_theme_color(dpg.mvThemeCol_Text,         (220, 220, 220, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Separator,    (60, 65, 80, 255))
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,   6)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,    4)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,    4)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,      6, 5)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,   10, 8)
        with dpg.theme_component(dpg.mvNodeEditor):
            dpg.add_theme_color(dpg.mvNodeCol_GridBackground,  (14, 16, 22, 255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_GridLine,         (35, 40, 55, 255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackground,   (28, 32, 44, 230),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered,(35,40,54,240),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected,(40,46,62,255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeOutline,      (60, 70, 90, 200),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_Link,             (100, 160, 240, 200),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_LinkHovered,      (140, 200, 255, 255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_LinkSelected,     (200, 230, 255, 255),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_Pin,              (120, 180, 255, 220),
                                category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_PinHovered,       (180, 220, 255, 255),
                                category=dpg.mvThemeCat_Nodes)
    return theme

# ---------------------------------------------------------------------------
# Main window layout
# ---------------------------------------------------------------------------

PALETTE_W  = 200
SIDEBAR_W  = 240
TOOLBAR_H  = 36
STATUS_H   = 28
WIN_W      = 1400
WIN_H      = 860
CANVAS_W   = WIN_W - PALETTE_W - SIDEBAR_W
CANVAS_H   = WIN_H - TOOLBAR_H - STATUS_H - 60
CODEBOX_H  = 200

def build_ui():
    dpg.create_context()
    dpg.create_viewport(title="EmbedSim GUI — Block Diagram Designer",
                        width=WIN_W, height=WIN_H,
                        min_width=900, min_height=600)

    global_theme = _build_global_theme()
    dpg.bind_theme(global_theme)

    with dpg.window(label="EmbedSim GUI", tag="main_window",
                    no_title_bar=True, no_move=True, no_resize=True,
                    no_scrollbar=True,
                    pos=(0, 0), width=WIN_W, height=WIN_H):

        # ── Toolbar ──────────────────────────────────────────────────────────
        with dpg.child_window(tag="toolbar", height=TOOLBAR_H,
                              no_scrollbar=True, border=False):
            with dpg.group(horizontal=True):
                dpg.add_text("EmbedSim GUI", color=(100, 180, 255, 255))
                dpg.add_button(label="Load Example",
                               callback=cb_load_example, width=120)
                dpg.add_button(label="Clear Canvas",
                               callback=cb_clear_canvas, width=110)
                dpg.add_button(label="Validate",
                               callback=cb_validate, width=90)
                dpg.add_button(label="Generate Code",
                               callback=cb_generate, width=120)
                dpg.add_text("T=", color=(180, 180, 180, 255))
                dpg.add_input_float(tag="gen_T", default_value=3.0,
                                    width=70, format="%.2f", step=0)
                dpg.add_text("dt=", color=(180, 180, 180, 255))
                dpg.add_input_float(tag="gen_dt", default_value=0.01,
                                    width=70, format="%.4f", step=0)
                dpg.add_combo(tag="gen_solver",
                              items=["EULER", "HEUN", "RK4"],
                              default_value="HEUN", width=80)
                dpg.add_input_text(tag="save_path",
                                   default_value="embedsim_generated.py",
                                   width=200, hint="output path")
                dpg.add_button(label="Save .py",
                               callback=cb_save, width=80)

        # ── Main row ─────────────────────────────────────────────────────────
        with dpg.group(horizontal=True):

            # ── Palette ──────────────────────────────────────────────────────
            with dpg.child_window(tag="palette_window",
                                  width=PALETTE_W,
                                  height=WIN_H - TOOLBAR_H - STATUS_H - 20,
                                  border=True):
                dpg.add_text("BLOCK PALETTE",
                             color=(100, 180, 255, 255))
                dpg.add_text("double-click to place",
                             color=(120, 120, 120, 200))
                dpg.add_separator()

                categories = [
                    ("SOURCES",    (34, 139, 87,  255)),
                    ("PROCESSING", (37, 99,  235, 255)),
                    ("DYNAMIC",    (220, 38, 38,  255)),
                    ("SINK",       (100,116,139,  255)),
                ]
                cat_map = {
                    "SOURCES":    ["SinusoidalGenerator","ThreePhaseGenerator",
                                   "VectorConstant","VectorStep",
                                   "VectorRamp","GaussianNoiseBlock"],
                    "PROCESSING": ["VectorGain","VectorSum","ScriptBlock"],
                    "DYNAMIC":    ["VectorDelay","VectorIntegrator"],
                    "SINK":       ["VectorEnd"],
                }

                for cat_name, (cr, cg, cb_, ca) in categories:
                    dpg.add_spacer(height=4)
                    dpg.add_text(cat_name, color=(cr, cg, cb_, ca))
                    dpg.add_separator()
                    for cname in cat_map[cat_name]:
                        bdef = BLOCK_BY_CLASS.get(cname)
                        if not bdef:
                            continue
                        sel_tag = dpg.generate_uuid()
                        dpg.add_selectable(
                            label=cname,
                            tag=sel_tag,
                            callback=cb_palette_double_click,
                        )
                        if bdef.description:
                            dpg.add_text(f"  {bdef.description}",
                                         color=(100,100,100,200))

            # ── Node editor canvas ────────────────────────────────────────────
            with dpg.child_window(tag="canvas_window",
                                  width=CANVAS_W,
                                  height=WIN_H - TOOLBAR_H - STATUS_H - 20,
                                  border=True, no_scrollbar=False):

                # Validate / status feedback
                dpg.add_text("", tag="validate_output",
                             color=(80, 220, 120, 255), wrap=CANVAS_W - 20)
                dpg.add_separator()

                with dpg.node_editor(
                        tag="node_editor",
                        callback=cb_link_created,
                        delink_callback=cb_link_deleted,
                        minimap=True,
                        minimap_location=dpg.mvNodeMiniMap_Location_BottomRight):
                    pass   # nodes added dynamically

                dpg.add_separator()
                dpg.add_text("Generated Code:", color=(100, 180, 255, 255))
                dpg.add_input_text(tag="code_output",
                                   multiline=True,
                                   readonly=True,
                                   width=CANVAS_W - 20,
                                   height=CODEBOX_H,
                                   default_value="# Press ⬇ Generate Code to see output here")

            # ── Sidebar ───────────────────────────────────────────────────────
            with dpg.child_window(tag="sidebar_window",
                                  width=SIDEBAR_W,
                                  height=WIN_H - TOOLBAR_H - STATUS_H - 20,
                                  border=True):
                dpg.add_text("PARAMETERS",
                             color=(100, 180, 255, 255))
                dpg.add_text("click a node to edit",
                             color=(120, 120, 120, 200))
                dpg.add_separator()
                dpg.add_child_window(tag="sidebar_content",
                                     border=False,
                                     height=WIN_H - TOOLBAR_H - STATUS_H - 80)

        # ── Status bar ───────────────────────────────────────────────────────
        with dpg.child_window(tag="statusbar",
                              height=STATUS_H,
                              no_scrollbar=True, border=False):
            dpg.add_text("", tag="status_text",
                         color=(140, 140, 140, 220))

    _refresh_status()

    # Node selection: mvClickedHandler is not applicable to mvNodeEditor in DPG 2.x.
    # Use a global mouse-click handler that polls get_selected_nodes() instead.
    with dpg.handler_registry():
        dpg.add_mouse_click_handler(callback=_on_viewport_click)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)

    # Load the example automatically on startup
    cb_load_example(None, None)

    dpg.start_dearpygui()
    dpg.destroy_context()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_ui()