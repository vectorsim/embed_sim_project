"""
topology_printer.py
===================

EmbedSim Block Diagram Topology Visualizer
Dual-mode: importable module + standalone GUI launcher.

USAGE AS MODULE
---------------
    from topology_printer import TopologyPrinter

    sim = build_sim()
    printer = TopologyPrinter(sim)
    printer.print_console()          # Clean multi-lane ASCII diagram
    printer.show_gui()               # Opens browser interactive SVG
    printer.export_html("topo.html") # Save standalone HTML file

INTEGRATION WITH EmbedSim
--------------------------
    # Auto-attached by EmbedSim.__init__ — no extra code needed:
    sim.topo.print_console()
    sim.topo.show_gui()
    sim.topo.export_html("diagram.html")

STANDALONE DEMO
---------------
    python topology_printer.py      # Opens PMSM FOC demo in browser

Author: EmbedSim project
"""

from __future__ import annotations

import os
import json
import textwrap
import tempfile
import webbrowser
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Block metadata extraction
# =============================================================================

# Parameters worth showing in the inspector (no internal boolean flags)
_PARAM_KEYS = (
    'Kp', 'Ki', 'i_max',
    'lambda_d', 'lambda_q', 'K_sw_d', 'K_sw_q',
    'phi_d', 'phi_q', 'out_min', 'out_max',
    'step_time', 'before_value', 'after_value', 'dim',
    'value', 'vector_size',
    'fmu_path',
    'R', 'L_d', 'L_q', 'lambda_pm', 'J', 'B', 'p',
)


def _block_info(block) -> Dict[str, Any]:
    """Extract display metadata from any VectorBlock-like object."""
    name = getattr(block, 'name', str(block))
    cls  = getattr(block, '_cls', None) or type(block).__name__
    cat  = _classify(cls, name)

    # _classify works on names/strings and misses FMU subclasses with arbitrary
    # names (e.g. BuckConverterBlock).  Override via isinstance then fmu_path.
    if cat == 'generic':
        try:
            from embedsim.fmu_blocks import FMUBlock
            if isinstance(block, FMUBlock):
                cat = 'plant'
        except ImportError:
            pass
        if cat == 'generic' and getattr(block, 'fmu_path', None):
            cat = 'plant'

    stub_attrs = getattr(block, '_attrs', None)
    if stub_attrs:
        attrs = dict(stub_attrs)
    else:
        attrs = {}
        for key in _PARAM_KEYS:
            v = getattr(block, key, None)
            if v is None:
                continue
            if isinstance(v, float):
                attrs[key] = f"{v:.4g}"
            elif isinstance(v, (list, tuple)) and len(v) <= 6:
                attrs[key] = str(list(v))
            elif isinstance(v, str) and len(v) < 80:
                attrs[key] = os.path.basename(v) if key == 'fmu_path' else v
            elif isinstance(v, (int,)):
                attrs[key] = str(v)
            # skip booleans — they are not informative parameters

    return {
        'name':         name,
        'cls':          cls,
        'cat':          cat,
        'is_dynamic':   bool(getattr(block, 'is_dynamic',     False)),
        'is_lb':        bool(getattr(block, 'is_loop_breaker', False)),
        'use_c':        bool(getattr(block, 'use_c_backend',   False)),
        'output_label': str(getattr(block, 'output_label', '') or ''),
        'attrs':        attrs,
    }


def _classify(cls: str, name: str) -> str:
    token = (cls + ' ' + name).lower()
    # codegen MUST be checked before sink — 'codegenend' contains 'end'
    if any(x in token for x in ('codegen', 'codegenstart', 'codegenend')):
        return 'codegen'
    if any(x in token for x in ('step', 'constant', 'source', 'ramp',
                                  'vectorconstant', 'vectorstep')):
        return 'source'
    if any(x in token for x in ('end', 'sink', 'capture',
                                  'vectorend', 'motorcapturesink')):
        return 'sink'
    if any(x in token for x in ('delay', 'loopbreaker', 'regdelay',
                                  'unitdelay', 'vectordelay')):
        return 'delay'
    if any(x in token for x in ('motor', 'fmu', 'plant', 'threephase')):
        return 'plant'
    if any(x in token for x in ('smc', 'speedpi', 'speed_pi', 'pid',
                                  'controller', 'mrac', 'pi_buck', 'pi_',
                                  'regulator', 'observer')):
        return 'control'
    if any(x in token for x in ('park', 'clarke', 'transform',
                                  'invpark', 'invclarke')):
        return 'transform'
    if any(x in token for x in ('gain', 'sum', 'integrat', 'filter',
                                  'split', 'mux')):
        return 'processing'
    return 'generic'


# =============================================================================
# Graph extraction  (works with real EmbedSim AND demo stubs)
# =============================================================================

def _get_inputs(block) -> list:
    """
    Return the upstream input blocks.
    Real EmbedSim uses the public ``inputs`` property.
    Demo stubs use ``_inputs``.  Try both.
    """
    v = getattr(block, 'inputs', None)
    if v is None:
        v = getattr(block, '_inputs', None)
    return list(v) if v else []


def _extract_graph(sim) -> Tuple[List[Any], Dict[str, List[str]]]:
    """
    Return (nodes, edges) from a live EmbedSim or demo stub.

    Prefers ``sim.blocks`` (already in topological order) when available,
    otherwise falls back to BFS from ``sim.sinks``.

    edges: forward map  src_name -> [dst_name, ...]
    """
    nodes: List[Any] = []
    seen:  Set[str]  = set()

    def _reg(b) -> bool:
        n = getattr(b, 'name', None)
        if n and n not in seen:
            seen.add(n)
            nodes.append(b)
            return True
        return False

    sim_blocks = getattr(sim, 'blocks', None)
    if sim_blocks:
        for b in sim_blocks:
            _reg(b)
    else:
        sinks  = getattr(sim, 'sinks', []) or []
        queued: Set[str] = set()
        queue:  deque    = deque()
        for s in sinks:
            n = getattr(s, 'name', None)
            if n and n not in queued:
                queued.add(n)
                queue.append(s)
        while queue:
            b = queue.popleft()
            _reg(b)
            for inp in _get_inputs(b):
                if inp is None:
                    continue
                inp_name = getattr(inp, 'name', None)
                if inp_name and inp_name not in queued:
                    queued.add(inp_name)
                    queue.append(inp)

    # Build forward edge map from inputs
    edges: Dict[str, List[str]] = defaultdict(list)
    for b in nodes:
        dst = getattr(b, 'name', None)
        if not dst:
            continue
        for inp in _get_inputs(b):
            src = getattr(inp, 'name', None) if inp else None
            if src and dst not in edges[src]:
                edges[src].append(dst)

    return nodes, dict(edges)


def _topo_sort(nodes: List[Any], edges: Dict[str, List[str]]) -> List[Any]:
    """Kahn's algorithm — loop-breaker outgoing edges ignored for in-degree."""
    n2b: Dict[str, Any] = {getattr(b, 'name', ''): b for b in nodes}
    ind: Dict[str, int]  = {getattr(b, 'name', ''): 0 for b in nodes}

    for src, dsts in edges.items():
        sb = n2b.get(src)
        if sb and getattr(sb, 'is_loop_breaker', False):
            continue
        for dst in dsts:
            if dst in ind:
                ind[dst] += 1

    queue = deque(b for b in nodes if ind.get(getattr(b, 'name', ''), 0) == 0)
    result: List[Any] = []
    while queue:
        b = queue.popleft()
        result.append(b)
        for dst_name in edges.get(getattr(b, 'name', ''), []):
            if dst_name in ind:
                ind[dst_name] -= 1
                if ind[dst_name] == 0:
                    queue.append(n2b[dst_name])

    added = {getattr(b, 'name', '') for b in result}
    for b in nodes:
        if getattr(b, 'name', '') not in added:
            result.append(b)
    return result


# =============================================================================
# Console renderer
# =============================================================================

_CAT_ICONS = {
    'source':     '◈',
    'sink':       '■',
    'codegen':    '⬡',
    'delay':      'z⁻¹',
    'plant':      '⚙',
    'control':    '⚡',
    'transform':  '↻',
    'processing': '⊕',
    'generic':    '○',
}


def _is_plant_only(block, infos: dict, edges: dict, all_blocks: list) -> bool:
    """
    Return True if this block's outputs go exclusively to plant/delay blocks
    (no sinks, no forward-path blocks).
    Used to move source constants (like T_load) into the PLANT lane.
    """
    cat = infos.get(getattr(block, 'name', ''), {}).get('cat', '')
    if cat not in ('source', 'generic', 'processing'):
        return False
    name = getattr(block, 'name', '')
    dsts = edges.get(name, [])
    if not dsts:
        return False
    name_to_info = {getattr(b,'name',''): infos[getattr(b,'name','')] for b in all_blocks}
    return all(name_to_info.get(d, {}).get('cat', '') in ('plant', 'delay')
               for d in dsts)


def _render_console(nodes: List[Any], edges: Dict[str, List[str]]) -> str:
    """
    Render a wide, unwrapped ASCII block diagram.

    Layout
    ------
    Each lane (FORWARD PATH / PLANT / FEEDBACK DELAYS) is rendered as a
    single horizontal flow:

        [◈ omega_ref] ──► [⬡ cg_start] ──► [⚡C speed_pi] ──► ...

    The separator bar width is auto-fitted to the longest rendered row so
    the diagram expands naturally with the number of blocks — no hard wrap.
    Fan-in blocks (multiple inputs) show a compact multi-line merge:

        [◈ src_a] ──►┐
                     ├──► [⊕ sum]
        [◈ src_b] ──►┘
    """
    sn    = _topo_sort(nodes, edges)
    infos = {getattr(b, 'name', ''): _block_info(b) for b in sn}

    def _has_sink_output(b) -> bool:
        """True if this block drives at least one sink/forward block directly."""
        name = getattr(b, 'name', '')
        return any(infos.get(d, {}).get('cat', '') in ('sink', 'codegen')
                   for d in edges.get(name, []))

    forward = [b for b in sn if infos[getattr(b,'name','')]['cat']
               not in ('delay', 'plant') and not _is_plant_only(b, infos, edges, sn)
               or (infos[getattr(b,'name','')]['cat'] == 'plant'
                   and _has_sink_output(b))]
    plant   = [b for b in sn if (infos[getattr(b,'name','')]['cat'] == 'plant'
               and not _has_sink_output(b))
               or _is_plant_only(b, infos, edges, sn)]
    delays  = [b for b in sn if infos[getattr(b,'name','')]['cat'] == 'delay']

    ARROW  = ' ──► '
    MERGE  = ' ──►'    # right side of a fan-in arm (no trailing space)

    def _fmt(b) -> str:
        info = infos[getattr(b, 'name', '')]
        icon = _CAT_ICONS[info['cat']]
        c_tag   = 'C' if info['use_c'] else ''
        dyn_tag = '⚡' if (info['is_dynamic'] and info['cat'] != 'control') else ''
        tag = (dyn_tag + c_tag).strip()
        return f"[{icon}{' '+tag if tag else ''} {info['name']} ({info['cls']})]"

    # map name → block object so we can read output_label at stitch time
    b_by_name: Dict[str, Any] = {}

    def _arrow_for(bname: str) -> str:
        """Return arrow string embedding output_label if the block has one."""
        lbl = infos.get(bname, {}).get('output_label', '')
        if lbl:
            return f' ──[{lbl}]──► '
        return ARROW

    def _render_lane(blocks: List[Any]) -> List[str]:
        """
        Render one lane as a list of text rows.

        Simple chain  (no fan-in):
            [A] ──► [B] ──► [C]

        2-input fan-in:
            [A] ──►┐
                   ├──► [C] ──► [D]
            [B] ──►┘

        3-input fan-in:
            [A] ──►┐
                   │
            [B] ──►┼──► [C]
                   │
            [C2]──►┘

        Key: when we encounter a block with ≥2 lane inputs we retroactively
        *absorb* those inputs' existing single-block segments into a fanin
        segment (tombstoning the originals), so sources that were already
        registered don't appear twice.
        """
        if not blocks:
            return []

        fmts        = {getattr(b,'name',''): _fmt(b) for b in blocks}
        block_names = {getattr(b,'name','') for b in blocks}
        # populate the outer b_by_name so _arrow_for can read output_label
        for b in blocks:
            b_by_name[getattr(b,'name','')] = b

        # segments: list of ('single', name) | ('fanin', [inp_names], dst_name)
        # None entries are tombstones (absorbed into a later fanin)
        segments: List[Any] = []
        seg_idx: Dict[str, int] = {}   # block_name → index in segments

        for b in blocks:
            bname     = getattr(b, 'name', '')
            lane_inps = [getattr(i,'name','') for i in _get_inputs(b)
                         if getattr(i,'name','') in block_names
                         and not getattr(i,'is_loop_breaker', False)]

            if len(lane_inps) >= 2:
                # Absorb all lane inputs into this fanin segment
                for inp in lane_inps:
                    if inp in seg_idx:
                        segments[seg_idx[inp]] = None   # tombstone
                idx = len(segments)
                segments.append(('fanin', lane_inps, bname))
                for inp in lane_inps:
                    seg_idx[inp] = idx
                seg_idx[bname] = idx
            else:
                if bname not in seg_idx:
                    idx = len(segments)
                    segments.append(('single', bname))
                    seg_idx[bname] = idx

        # Remove tombstones
        segments = [s for s in segments if s is not None]
        if not segments:
            return []

        # Render each segment into a list of rows
        def _seg_rows(seg) -> List[str]:
            if seg[0] == 'single':
                return [fmts[seg[1]]]
            # fanin
            _, inp_names, dst_name = seg
            inp_strs = [fmts[n] for n in inp_names]
            n_arms   = len(inp_strs)
            max_src  = max(len(s) for s in inp_strs)
            suffix   = ARROW[1:] + fmts[dst_name]

            if n_arms == 2:
                pad0 = ' ' * (max_src - len(inp_strs[0]))
                pad1 = ' ' * (max_src - len(inp_strs[1]))
                bar  = ' ' * (max_src + len(MERGE))
                return [
                    f"{inp_strs[0]}{pad0}{MERGE}┐",
                    f"{bar}├{suffix}",
                    f"{inp_strs[1]}{pad1}{MERGE}┘",
                ]
            else:
                rows = []
                mid  = n_arms // 2
                for k, s in enumerate(inp_strs):
                    pad = ' ' * (max_src - len(s))
                    bar = '┐' if k == 0 else ('┘' if k == n_arms-1 else '┤')
                    row = f"{s}{pad}{MERGE}{bar}"
                    if k == mid:
                        row = row[:-1] + '┼' + suffix
                    rows.append(row)
                    if k < n_arms - 1:
                        rows.append(' ' * (max_src + len(MERGE)) + '│')
                return rows

        seg_rows = [_seg_rows(s) for s in segments]

        # Stitch horizontally — all rows padded to same height,
        # main (signal) row aligned at index max_h//2
        max_h  = max(len(r) for r in seg_rows)
        main_r = max_h // 2

        def _pad(rows: List[str]) -> List[str]:
            h       = len(rows)
            own_mid = h // 2
            top     = main_r - own_mid
            bot     = max_h - top - h
            w       = max(len(r) for r in rows)
            return ([' ' * w] * top +
                    [r.ljust(w) for r in rows] +
                    [' ' * w] * bot)

        padded = [_pad(r) for r in seg_rows]
        result = ['' for _ in range(max_h)]
        for si, p in enumerate(padded):
            for ri in range(max_h):
                result[ri] += p[ri]
            if si < len(padded) - 1:
                # Arrow carries the output_label of the left segment's output block
                seg = segments[si]
                src_bname = seg[1] if seg[0] == 'single' else seg[2]
                arr = _arrow_for(src_bname)
                for ri in range(max_h):
                    result[ri] += arr if ri == main_r else ' ' * len(arr)

        return [r.rstrip() for r in result]

    # Build all lane row-lists
    fwd_rows   = _render_lane(forward)
    plant_rows = _render_lane(plant)
    dly_rows   = _render_lane(delays)

    # Auto-fit box width to widest lane (min 78)
    all_rows = fwd_rows + plant_rows + dly_rows
    W = max(78, max(len(r) + 6 for r in all_rows) if all_rows else 78)

    def _sep(ch='·') -> str:
        return '  ' + ch * (W - 4)

    def _section(label: str, lane_rows: List[str]) -> List[str]:
        if not lane_rows:
            return []
        out = [f'  {label}', _sep()]
        for r in lane_rows:
            out.append('    ' + r)
        out.append('')
        return out

    lines: List[str] = [
        '',
        '╔' + '═' * (W - 2) + '╗',
        '║' + ' EmbedSim — Block Topology '.center(W - 2) + '║',
        '╚' + '═' * (W - 2) + '╝',
        '',
    ]
    lines += _section('FORWARD PATH  (Sources → Sinks)', fwd_rows)
    lines += _section('PLANT', plant_rows)
    lines += _section('FEEDBACK DELAYS  (z⁻¹ LoopBreakers)', dly_rows)

    # Feedback wiring table
    fb_rows = []
    for b in delays:
        src = getattr(b, 'name', '')
        for dst in edges.get(src, []):
            fb_rows.append(f'    {src:22s}  z⁻¹──►  {dst}')
    if fb_rows:
        lines += [_sep(), '  FEEDBACK WIRING', _sep()] + fb_rows + ['']

    # All connections table
    lines += [_sep(), '  ALL CONNECTIONS', _sep()]
    seen_e: Set[str] = set()
    for b in sn:
        src = getattr(b, 'name', '')
        for dst in edges.get(src, []):
            k = f"{src}→{dst}"
            if k not in seen_e:
                seen_e.add(k)
                lb = '  [z⁻¹]' if infos.get(src, {}).get('is_lb') else ''
                lines.append(f'    {src:22s}  ──►  {dst}{lb}')

    n_d = sum(1 for i in infos.values() if i['is_dynamic'])
    n_l = sum(1 for i in infos.values() if i['is_lb'])
    n_c = sum(1 for i in infos.values() if i['use_c'])
    lines += [
        '', _sep(),
        f'  Blocks: {len(nodes)}  |  Dynamic: {n_d}  |  '
        f'LoopBreakers: {n_l}  |  C-backend: {n_c}',
        '',
    ]
    return '\n'.join(lines)


# =============================================================================
# HTML / GUI renderer
# =============================================================================

# Lane Y centres for the three horizontal bands
_LANE_Y = {'forward': 130, 'delay': 340, 'plant': 530}
_NODE_W  = 160
_NODE_H  = 60
_H_GAP   = 40
_MARGIN  = 40


def _layout_nodes(nodes: List[Any],
                  edges: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Assign pixel positions to every node.

    Lane 0  forward path  (source, codegen, control, transform, sink)
    Lane 1  delays        (z⁻¹ loop-breakers) — centred under receivers
    Lane 2  plant blocks  — aligned under their primary driver
    """
    sn    = _topo_sort(nodes, edges)
    infos = {getattr(b, 'name', ''): _block_info(b) for b in sn}

    forward = [b for b in sn if infos[getattr(b,'name','')]['cat']
               not in ('delay','plant')]
    delays  = [b for b in sn if infos[getattr(b,'name','')]['cat'] == 'delay']
    plant   = [b for b in sn if infos[getattr(b,'name','')]['cat'] == 'plant']

    result: Dict[str, Dict] = {}

    def _node_dict(name, x, lane_key):
        info = infos[name]
        y = _LANE_Y[lane_key] - _NODE_H // 2
        return {
            'id':    name,
            'x':     x,
            'y':     y,
            'w':     _NODE_W,
            'h':     _NODE_H,
            'cat':   info['cat'],
            'cls':   info['cls'],
            'dyn':   info['is_dynamic'],
            'lb':    info['is_lb'],
            'c':     info['use_c'],
            'lbl':   info.get('output_label', ''),
            'attrs': info['attrs'],
        }

    # Forward path — simple left-to-right
    x = _MARGIN
    for b in forward:
        n = getattr(b, 'name', '')
        result[n] = _node_dict(n, x, 'forward')
        x += _NODE_W + _H_GAP

    # Delays — try to centre under the forward-path block they feed into
    delay_fallback_x = _MARGIN
    used_delay_x: Set[int] = set()
    for b in delays:
        n = getattr(b, 'name', '')
        recv_x = None
        for dst in edges.get(n, []):
            if dst in result:
                recv_x = result[dst]['x']
                break
        if recv_x is None:
            recv_x = delay_fallback_x
        # Avoid collision between delay nodes
        while recv_x in used_delay_x:
            recv_x += _NODE_W + _H_GAP
        used_delay_x.add(recv_x)
        result[n] = _node_dict(n, recv_x, 'delay')
        delay_fallback_x = recv_x + _NODE_W + _H_GAP

    # Plant — align under their primary forward-path driver, no collisions
    plant_x = _MARGIN
    used_plant_x: Set[int] = set()
    for b in plant:
        n = getattr(b, 'name', '')
        driver_names = [src for src, dsts in edges.items() if n in dsts]
        drv_x = None
        for d in driver_names:
            if d in result:
                drv_x = result[d]['x']
                break
        x = drv_x if drv_x is not None else plant_x
        # Avoid plant node collisions
        while x in used_plant_x:
            x += _NODE_W + _H_GAP
        used_plant_x.add(x)
        result[n] = _node_dict(n, x, 'plant')
        plant_x = x + _NODE_W + _H_GAP

    return list(result.values())


def _build_edges(nodes_data: List[Dict],
                 edges: Dict[str, List[str]],
                 infos: Dict[str, Dict],
                 wire_labels: Optional[Dict[Tuple[str,str], str]] = None) -> List[Dict]:
    """Build JS edge list with lb flag and optional signal label per arc."""
    ids = {n['id'] for n in nodes_data}
    wl  = wire_labels or {}
    out = []
    for src, dsts in edges.items():
        if src not in ids:
            continue
        lb = infos.get(src, {}).get('is_lb', False)
        for dst in dsts:
            if dst in ids:
                out.append({
                    'src': src,
                    'dst': dst,
                    'lb':  bool(lb),
                    'lbl': wl.get((src, dst), ''),
                })
    return out


def _render_html(nodes: List[Any],
                 edges: Dict[str, List[str]],
                 title: str = "EmbedSim Topology",
                 wire_labels: Optional[Dict[Tuple[str,str], str]] = None) -> str:

    infos      = {getattr(b,'name',''): _block_info(b) for b in nodes}
    nodes_data = _layout_nodes(nodes, edges)
    edges_data = _build_edges(nodes_data, edges, infos, wire_labels)

    canvas_w = max((n['x'] + n['w'] for n in nodes_data), default=800) + 80
    canvas_h = max((n['y'] + n['h'] for n in nodes_data), default=500) + 100

    # ── Compute CodeGen region bounding box ─────────────────────────────────
    cg_nodes = [n for n in nodes_data if n['cat'] == 'codegen']
    cg_region = None
    if len(cg_nodes) >= 2:
        cg_start_n = min(cg_nodes, key=lambda n: n['x'])
        cg_end_n   = max(cg_nodes, key=lambda n: n['x'])
        fwd_y = _LANE_Y['forward'] - _NODE_H // 2
        pad = 14
        cg_region = {
            'x': cg_start_n['x'] - pad,
            'y': fwd_y - pad - 22,
            'w': (cg_end_n['x'] + cg_end_n['w']) - cg_start_n['x'] + pad * 2,
            'h': _NODE_H + pad * 2 + 22,
        }
    rj = json.dumps(cg_region, ensure_ascii=False)

    nj = json.dumps(nodes_data, ensure_ascii=False)
    ej = json.dumps(edges_data, ensure_ascii=False)

    # ── Vivid light theme — all colours defined here in Python ───────────────
    # Change these constants to restyle all future exports. No HTML editing.
    C = dict(
        bg          = "#ffffff",        # pure white page background
        panel       = "#ffffff",        # header / inspector panel
        border      = "#7090c0",        # panel borders — clearly visible
        text        = "#0a1020",        # near-black primary text
        text_dim    = "#1a3060",        # dark secondary text
        canvas_bg   = "#f0f6ff",        # diagram background — faint blue tint
        # Block category colours — deep, fully saturated
        col_source  = "#0055cc",        # strong blue    — VectorConstant
        col_sink    = "#0055cc",        # strong blue    — VectorEnd
        col_control = "#007020",        # deep green     — PI, SMC
        col_xform   = "#b03800",        # deep orange    — Clarke, Park
        col_plant   = "#6600aa",        # deep purple    — PMSM FMU
        col_delay   = "#cc0000",        # pure red       — z⁻¹ loop breaker
        col_codegen = "#806000",        # deep gold      — CodeGen markers
        col_generic = "#1a3a5c",        # deep slate     — everything else
        col_proc    = "#1a3a5c",        # deep slate     — processing
        col_noise   = "#005588",        # deep teal-blue — NoisySensorBlock
        # Band background tints
        band_fwd    = "rgba(0,60,200,0.08)",
        band_fb     = "rgba(180,0,0,0.08)",
        band_plant  = "rgba(80,0,160,0.08)",
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
/* ═══════════════════════════════════════════════════════════════
   EmbedSim Topology Viewer — Vivid light theme
   All colours come from topology_printer.py Python constants.
   Change them there; re-run; new HTML is automatically updated.
   ═══════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root{{
  --bg:{C['bg']};
  --panel:{C['panel']};
  --border:{C['border']};
  --text:{C['text']};
  --dim:{C['text_dim']};
  --source:{C['col_source']};
  --sink:{C['col_sink']};
  --control:{C['col_control']};
  --transform:{C['col_xform']};
  --plant:{C['col_plant']};
  --delay:{C['col_delay']};
  --codegen:{C['col_codegen']};
  --processing:{C['col_proc']};
  --generic:{C['col_generic']};
}}
*{{box-sizing:border-box;margin:0;padding:0;}}
html,body{{width:100%;height:100%;background:var(--bg);color:var(--text);
  font-family:'Inter',sans-serif;font-size:16px;overflow:hidden;}}
header{{display:flex;align-items:center;justify-content:space-between;
  padding:14px 24px;border-bottom:3px solid var(--border);background:var(--panel);
  box-shadow:0 2px 8px rgba(0,0,0,0.10);}}
header h1{{font-size:1.3rem;font-weight:700;color:var(--text);letter-spacing:.02em;}}
header p{{font-size:14px;color:var(--dim);margin-top:3px;}}
.hdr-r{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;}}
.badge{{font-family:'JetBrains Mono',monospace;font-size:13px;padding:4px 12px;
  border-radius:4px;background:var(--bg);border:2px solid var(--border);
  color:var(--dim);font-weight:700;}}
/* Zoom buttons */
.zoom-bar{{display:flex;gap:6px;align-items:center;margin-left:16px;}}
.zbtn{{width:36px;height:36px;border:2px solid var(--border);border-radius:6px;
  background:var(--panel);color:var(--text);font-size:1.1rem;font-weight:700;
  cursor:pointer;display:flex;align-items:center;justify-content:center;
  transition:background .15s;user-select:none;}}
.zbtn:hover{{background:#e8f0ff;border-color:var(--control);}}
#zoom-label{{font-family:'JetBrains Mono',monospace;font-size:13px;
  color:var(--dim);min-width:44px;text-align:center;font-weight:700;}}
#main{{display:flex;width:100%;height:calc(100vh - 64px);}}
#cw{{flex:1;overflow:hidden;position:relative;cursor:grab;background:{C['canvas_bg']};}}
#cw:active{{cursor:grabbing;}}
canvas{{display:block;}}
#insp{{width:300px;min-width:300px;background:var(--panel);
  border-left:3px solid var(--border);display:flex;flex-direction:column;
  transition:width .2s,min-width .2s;box-shadow:-2px 0 8px rgba(0,0,0,0.08);}}
#insp.collapsed{{width:0;min-width:0;overflow:hidden;}}
#ih{{padding:16px 18px 12px;border-bottom:2px solid var(--border);
  font-size:15px;font-weight:700;color:var(--text);letter-spacing:.07em;
  display:flex;justify-content:space-between;align-items:center;}}
#ic{{padding:16px 18px;overflow-y:auto;flex:1;font-size:14px;line-height:1.9;}}
.ir{{display:flex;justify-content:space-between;padding:5px 0;
  border-bottom:1px solid #d0dff0;font-size:14px;}}
.ik{{color:var(--dim);font-weight:700;}}
.iv{{color:var(--text);text-align:right;max-width:170px;word-break:break-all;
  font-family:'JetBrains Mono',monospace;font-size:13px;}}
.is{{font-size:12px;color:var(--dim);font-weight:700;letter-spacing:.12em;
  text-transform:uppercase;margin:12px 0 4px;}}
#leg{{position:absolute;bottom:16px;left:16px;background:rgba(255,255,255,0.97);
  border:2px solid var(--border);border-radius:10px;padding:12px 18px;font-size:13px;
  display:grid;grid-template-columns:1fr 1fr;gap:7px 22px;
  box-shadow:0 2px 12px rgba(0,0,0,0.12);}}
.lg{{display:flex;align-items:center;gap:8px;color:var(--dim);font-weight:700;}}
.ld{{width:16px;height:16px;border-radius:4px;flex-shrink:0;}}
#tip{{position:fixed;background:var(--panel);border:2px solid var(--border);
  border-radius:8px;padding:10px 16px;font-size:14px;line-height:1.7;
  pointer-events:none;opacity:0;transition:opacity .12s;z-index:999;max-width:280px;
  box-shadow:0 4px 16px rgba(0,0,0,0.14);color:var(--text);}}
#tip.show{{opacity:1;}}
</style>
</head>
<body>
<header>
  <div>
    <h1>⬡ EmbedSim — Block Topology</h1>
    <p>Scroll to zoom · drag to pan · click any block for details</p>
  </div>
  <div class="hdr-r">
    <span class="badge" id="bt">— blocks</span>
    <span class="badge" id="bd">— dynamic</span>
    <span class="badge" id="bl">— loop-breakers</span>
    <span class="badge" id="bc">— C-backend</span>
    <div class="zoom-bar">
      <button class="zbtn" id="z-out" title="Zoom out">−</button>
      <span id="zoom-label">100%</span>
      <button class="zbtn" id="z-in" title="Zoom in">+</button>
      <button class="zbtn" id="z-fit" style="width:auto;padding:0 10px;font-size:.8rem" title="Fit all">Fit</button>
    </div>
  </div>
</header>
<div id="main">
  <div id="cw">
    <canvas id="c"></canvas>
    <div id="leg">
      <div class="lg"><div class="ld" style="background:var(--source)"></div>Source / Sink</div>
      <div class="lg"><div class="ld" style="background:var(--control)"></div>Controller</div>
      <div class="lg"><div class="ld" style="background:var(--codegen);border:2px dashed var(--codegen)"></div>CodeGen Region</div>
      <div class="lg"><div class="ld" style="background:var(--transform)"></div>Transform</div>
      <div class="lg"><div class="ld" style="background:var(--plant)"></div>Plant / FMU</div>
      <div class="lg"><div class="ld" style="background:var(--delay)"></div>z⁻¹ Delay</div>
    </div>
  </div>
  <div id="insp">
    <div id="ih">BLOCK INSPECTOR
      <span style="cursor:pointer;color:var(--dim);font-size:1.2rem"
            onclick="document.getElementById('insp').classList.toggle('collapsed')">✕</span>
    </div>
    <div id="ic">
      <div style="color:var(--dim);font-size:14px;padding-top:32px;text-align:center;line-height:2.2">
        Click any block<br>to inspect connections<br>and parameters
      </div>
    </div>
  </div>
</div>
<div id="tip"></div>
<script>
/* ── Data ── */
const NODES={nj};
const EDGES={ej};
const CG_REGION={rj};

/* ── Category colours — vivid, fully saturated ── */
const CC={{
  source:'{C['col_source']}',sink:'{C['col_sink']}',
  codegen:'{C['col_codegen']}',control:'{C['col_control']}',
  transform:'{C['col_xform']}',plant:'{C['col_plant']}',
  delay:'{C['col_delay']}',processing:'{C['col_proc']}',
  generic:'{C['col_generic']}',noise:'{C['col_noise']}',
}};

/* Mark NoisySensorBlock as noise category */
NODES.forEach(n=>{{ if(n.cls==='NoisySensorBlock')n.cat='noise'; }});

/* ── Swim-lane bands ── */
const BANDS=[
  {{y:84,  h:112, lbl:'FORWARD PATH', col:'{C['band_fwd']}'}},
  {{y:305, h:100, lbl:'FEEDBACK  z\u207B\u00B9', col:'{C['band_fb']}'}},
  {{y:494, h:112, lbl:'PLANT', col:'{C['band_plant']}'}},
];

const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
const wrap=document.getElementById('cw');
const nm={{}};
NODES.forEach(n=>nm[n.id]=n);
const BW={canvas_w}, BH={canvas_h};

/* ── Viewport state ── */
let scale=1, originX=0, originY=0;
let isPanning=false, panX=0, panY=0, panOX=0, panOY=0;
let hov=null;

function toDiag(sx,sy){{return{{x:(sx-originX)/scale,y:(sy-originY)/scale}};}}

function fitToView(){{
  const vw=wrap.clientWidth-40, vh=wrap.clientHeight-40;
  scale=Math.min(vw/BW,vh/BH,1.5);
  originX=(wrap.clientWidth-BW*scale)/2;
  originY=(wrap.clientHeight-BH*scale)/2;
  document.getElementById('zoom-label').textContent=Math.round(scale*100)+'%';
  draw();
}}

function zoom(factor,cx,cy){{
  cx=cx??wrap.clientWidth/2; cy=cy??wrap.clientHeight/2;
  originX=cx-(cx-originX)*factor;
  originY=cy-(cy-originY)*factor;
  scale=Math.max(0.15,Math.min(scale*factor,5));
  document.getElementById('zoom-label').textContent=Math.round(scale*100)+'%';
  draw();
}}

/* ── Helpers ── */
function rr(x,y,w,h,r){{
  ctx.beginPath();ctx.moveTo(x+r,y);
  ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();
}}
function ah(x,y,ang,sz,col){{
  ctx.save();ctx.fillStyle=col;ctx.translate(x,y);ctx.rotate(ang);
  ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(-sz,-sz*.42);ctx.lineTo(-sz,sz*.42);
  ctx.closePath();ctx.fill();ctx.restore();
}}

/* ── Draw edge ── */
function drawEdge(e){{
  const s=nm[e.src],d=nm[e.dst];if(!s||!d)return;
  const col=e.lb?'{C['col_delay']}':'#2255bb';
  const lw=e.lb?2.5:2.0;
  ctx.save();
  ctx.strokeStyle=col;ctx.lineWidth=lw;ctx.globalAlpha=e.lb?1.0:0.85;
  ctx.setLineDash(e.lb?[8,5]:[]);ctx.lineJoin='round';
  const sx=s.x+s.w,sy=s.y+s.h/2,dx=d.x,dy=d.y+d.h/2;
  let mlx=0,mly=0;
  if(sx>dx+20){{
    const by=Math.max(s.y+s.h,d.y+d.h)+30;
    const x1=s.x+s.w/2,x2=d.x+d.w/2;
    ctx.beginPath();ctx.moveTo(x1,s.y+s.h);ctx.lineTo(x1,by);
    ctx.lineTo(x2,by);ctx.lineTo(x2,d.y+d.h);ctx.stroke();
    ah(x2,d.y+d.h,Math.PI/2,9,col);
    mlx=(x1+x2)/2;mly=by-10;
  }}else if(Math.abs(sy-dy)<6){{
    ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo(dx,dy);ctx.stroke();
    ah(dx,dy,0,9,col);mlx=(sx+dx)/2;mly=sy-28;
  }}else{{
    const mx=(sx+dx)/2;
    ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo(mx,sy);
    ctx.lineTo(mx,dy);ctx.lineTo(dx,dy);ctx.stroke();
    ah(dx,dy,0,9,col);mlx=(mx+dx)/2;mly=dy-28;
  }}
  /* Signal label — rotated 90° CCW, above the arrow midpoint */
  const lbl=e.lbl||'';
  if(lbl){{
    ctx.save();
    ctx.globalAlpha=1.0;
    ctx.font='bold 11px "JetBrains Mono",monospace';
    const tw=ctx.measureText(lbl).width+12;  /* text length + padding */
    const th=15;                               /* pill narrow width     */
    const lx=mlx;
    const ly=mly-tw/2-8;   /* push centre above arrow line by half pill height + gap */
    ctx.translate(lx,ly);
    ctx.rotate(-Math.PI/2);  /* 90° counter-clockwise */
    ctx.fillStyle='rgba(255,255,255,0.97)';
    ctx.strokeStyle=col;ctx.lineWidth=1.2;ctx.setLineDash([]);
    ctx.beginPath();ctx.roundRect(-tw/2,-th/2,tw,th,4);
    ctx.fill();ctx.stroke();
    ctx.fillStyle=col;
    ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.fillText(lbl,0,0);
    ctx.restore();
  }}
  ctx.restore();
}}

/* ── Draw node ── */
function drawNode(n,hvr){{
  const col=CC[n.cat]||CC.generic;
  const{{x,y,w,h}}=n;
  /* Glow for hover/dynamic */
  if(hvr||n.dyn){{
    ctx.save();ctx.shadowColor=col;ctx.shadowBlur=hvr?18:8;
    rr(x,y,w,h,8);ctx.strokeStyle=col;ctx.lineWidth=.5;ctx.stroke();ctx.restore();
  }}
  /* Fill — white with category tint strip at top */
  ctx.save();rr(x,y,w,h,8);
  ctx.fillStyle=hvr?'#ddeeff':'#ffffff';ctx.fill();
  /* Top colour stripe */
  ctx.beginPath();ctx.rect(x+2,y+2,w-4,13);
  ctx.fillStyle=col;ctx.globalAlpha=0.22;ctx.fill();ctx.globalAlpha=1;
  /* Border — thick, vivid */
  ctx.setLineDash(n.lb?[6,4]:[]);
  ctx.strokeStyle=col;ctx.lineWidth=n.dyn?4.5:3.5;ctx.stroke();ctx.restore();
  /* Block name + class — proportional vertical positioning */
  const icon=n.dyn?'\u26A1 ':n.lb?'z\u207B\u00B9 ':'';
  const cBadge=n.c?' \u00B7C':'';
  const nameStr=icon+n.id+cBadge;
  const mw=w-16;
  ctx.save();
  ctx.textAlign='center';ctx.textBaseline='middle';
  /* Name row — 38% down */
  ctx.fillStyle=col;ctx.font='bold 13px "JetBrains Mono",monospace';
  let disp=nameStr;
  while(ctx.measureText(disp).width>mw&&disp.length>4)disp=disp.slice(0,-1);
  if(disp!==nameStr)disp=disp.slice(0,-1)+'\u2026';
  ctx.fillText(disp,x+w/2,y+h*0.37);
  /* Divider line */
  ctx.strokeStyle=col;ctx.globalAlpha=0.18;ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(x+10,y+h*0.55);ctx.lineTo(x+w-10,y+h*0.55);ctx.stroke();
  ctx.globalAlpha=1;
  /* Class row — 72% down */
  ctx.fillStyle='{C['text_dim']}';ctx.font='10px "JetBrains Mono",monospace';
  let cls=n.cls;
  while(ctx.measureText(cls).width>mw&&cls.length>4)cls=cls.slice(0,-1);
  if(cls!==n.cls)cls=cls.slice(0,-1)+'\u2026';
  ctx.fillText(cls,x+w/2,y+h*0.72);
  ctx.restore();
}}

/* ── Draw CodeGen region highlight ── */
function drawCodeGenRegion(){{
  if(!CG_REGION)return;
  const{{x,y,w,h}}=CG_REGION;
  const col='{C['col_codegen']}';
  ctx.save();
  /* Filled background */
  ctx.fillStyle='rgba(184,120,0,0.07)';
  ctx.beginPath();ctx.roundRect(x,y,w,h,12);ctx.fill();
  /* Dashed border */
  ctx.strokeStyle=col;ctx.lineWidth=2.5;ctx.setLineDash([10,6]);
  ctx.beginPath();ctx.roundRect(x,y,w,h,12);ctx.stroke();
  ctx.setLineDash([]);
  /* Label pill top-left */
  ctx.font='bold 11px "JetBrains Mono",monospace';
  const lbl='\u2B21 CodeGen Region \u2014 Aurix TriCore';
  const tw=ctx.measureText(lbl).width+16;
  ctx.fillStyle='rgba(255,248,220,0.97)';
  ctx.beginPath();ctx.roundRect(x+10,y+5,tw,20,6);ctx.fill();
  ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.setLineDash([]);
  ctx.beginPath();ctx.roundRect(x+10,y+5,tw,20,6);ctx.stroke();
  ctx.fillStyle=col;ctx.textAlign='left';ctx.textBaseline='middle';
  ctx.fillText(lbl,x+18,y+15);
  ctx.restore();
}}

/* ── Draw swim-lane bands ── */
function drawBands(){{
  BANDS.forEach(b=>{{
    ctx.save();ctx.fillStyle=b.col;ctx.fillRect(0,b.y,BW+80,b.h);
    ctx.fillStyle='rgba(10,20,60,0.55)';
    ctx.font='bold 14px "Inter",sans-serif';
    ctx.textBaseline='top';ctx.fillText(b.lbl,14,b.y+6);ctx.restore();
  }});
}}

/* ── Main draw ── */
function draw(){{
  canvas.width=wrap.clientWidth;canvas.height=wrap.clientHeight;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle='{C['canvas_bg']}';ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.save();ctx.translate(originX,originY);ctx.scale(scale,scale);
  drawBands();
  drawCodeGenRegion();
  EDGES.forEach(drawEdge);
  NODES.forEach(n=>drawNode(n,n.id===hov));
  ctx.restore();
}}

/* ── Hit test ── */
function hit(mx,my){{
  const d=toDiag(mx,my);
  return NODES.find(n=>d.x>=n.x&&d.x<=n.x+n.w&&d.y>=n.y&&d.y<=n.y+n.h)||null;
}}

/* ── Tooltip ── */
const tip=document.getElementById('tip');
canvas.addEventListener('mousemove',e=>{{
  const r=canvas.getBoundingClientRect();
  const h=isPanning?null:hit(e.clientX-r.left,e.clientY-r.top);
  if(h){{
    canvas.style.cursor='pointer';
    if(hov!==h.id){{hov=h.id;draw();}}
    const inc=EDGES.filter(ev=>ev.dst===h.id).map(ev=>ev.src);
    const out=EDGES.filter(ev=>ev.src===h.id).map(ev=>ev.dst);
    const col=CC[h.cat]||CC.generic;
    const L=[`<strong style="color:${{col}};font-size:15px">${{h.id}}</strong>`,
             `<span style="color:{C['text_dim']};font-size:13px">${{h.cls}}</span>`];
    if(h.lbl)L.push(`<span style="color:${{col}}">&#8611; ${{h.lbl}}</span>`);
    if(inc.length)L.push(`<span style="color:{C['text_dim']}">&#8592; ${{inc.join(', ')}}</span>`);
    if(out.length)L.push(`<span style="color:{C['text_dim']}">&#8594; ${{out.join(', ')}}</span>`);
    if(h.dyn)L.push('\u26A1 dynamic');
    if(h.lb)L.push('z\u207B\u00B9 loop-breaker');
    if(h.c)L.push('C backend');
    tip.innerHTML=L.join('<br>');
    tip.style.left=(e.clientX+16)+'px';tip.style.top=(e.clientY+10)+'px';
    tip.classList.add('show');
  }}else{{
    canvas.style.cursor=isPanning?'grabbing':'grab';
    if(hov!==null){{hov=null;draw();}}
    tip.classList.remove('show');
  }}
}});

/* ── Inspector panel (click) ── */
canvas.addEventListener('click',e=>{{
  const r=canvas.getBoundingClientRect();
  const h=hit(e.clientX-r.left,e.clientY-r.top);if(!h)return;
  const col=CC[h.cat]||CC.generic;
  const inc=EDGES.filter(ev=>ev.dst===h.id).map(ev=>ev.src);
  const out=EDGES.filter(ev=>ev.src===h.id).map(ev=>ev.dst);
  let html=`<div style="font-size:1.15rem;font-weight:700;color:${{col}};margin-bottom:4px">${{h.id}}</div>`;
  html+=`<div style="color:{C['text_dim']};margin-bottom:8px;font-family:'JetBrains Mono',monospace;font-size:13px">${{h.cls}}</div>`;
  if(h.lbl)html+=`<div style="color:${{col}};font-family:'JetBrains Mono',monospace;font-size:13px;margin-bottom:10px">&#8611; ${{h.lbl}}</div>`;
  const fl=[];if(h.dyn)fl.push('\u26A1 dynamic');if(h.lb)fl.push('z\u207B\u00B9 loop-breaker');if(h.c)fl.push('C backend');
  if(fl.length)html+=`<div class="is">${{fl.join(' &middot; ')}}</div>`;
  html+='<div class="is">Connections</div>';
  if(inc.length)html+=`<div class="ir"><span class="ik">inputs from</span><span class="iv">${{inc.join('<br>')}}</span></div>`;
  if(out.length)html+=`<div class="ir"><span class="ik">outputs to</span><span class="iv">${{out.join('<br>')}}</span></div>`;
  if(Object.keys(h.attrs).length){{
    html+='<div class="is">Parameters</div>';
    for(const[k,v]of Object.entries(h.attrs))
      html+=`<div class="ir"><span class="ik">${{k}}</span><span class="iv">${{v}}</span></div>`;
  }}
  document.getElementById('ic').innerHTML=html;
  document.getElementById('insp').classList.remove('collapsed');
}});

canvas.addEventListener('mouseleave',()=>{{hov=null;tip.classList.remove('show');draw();}});

/* ── Pan ── */
canvas.addEventListener('mousedown',e=>{{
  isPanning=true;panX=e.clientX;panY=e.clientY;panOX=originX;panOY=originY;
  canvas.style.cursor='grabbing';
}});
window.addEventListener('mousemove',e=>{{
  if(!isPanning)return;
  originX=panOX+(e.clientX-panX);originY=panOY+(e.clientY-panY);draw();
}});
window.addEventListener('mouseup',()=>{{isPanning=false;canvas.style.cursor='grab';}});

/* ── Scroll zoom ── */
wrap.addEventListener('wheel',e=>{{
  e.preventDefault();
  const r=canvas.getBoundingClientRect();
  zoom(e.deltaY<0?1.12:0.89,e.clientX-r.left,e.clientY-r.top);
}},{{passive:false}});

/* ── Zoom buttons ── */
document.getElementById('z-in').addEventListener('click',()=>zoom(1.25));
document.getElementById('z-out').addEventListener('click',()=>zoom(0.80));
document.getElementById('z-fit').addEventListener('click',fitToView);

/* ── Badge counters ── */
document.getElementById('bt').textContent=NODES.length+' blocks';
document.getElementById('bd').textContent=NODES.filter(n=>n.dyn).length+' dynamic';
document.getElementById('bl').textContent=NODES.filter(n=>n.lb).length+' loop-breakers';
document.getElementById('bc').textContent=NODES.filter(n=>n.c).length+' C-backend';

window.addEventListener('resize',fitToView);
fitToView();
</script>
</body>
</html>"""


# =============================================================================
# Public API
# =============================================================================

class TopologyPrinter:
    """
    EmbedSim topology visualizer — dual-mode console + browser GUI.

    Parameters
    ----------
    sim : EmbedSim
        A built simulation object (or demo stub).
    title : str
        Title shown in the browser tab.
    wire_labels : dict, optional
        Maps (src_block_name, dst_block_name) -> signal label string.
        Declared in the simulation script (the only place that knows
        signal semantics). Example::

            wire_labels = {
                ("vref",    "pi_ctrl_start"): "[V_ref]",
                ("pi_buck", "pi_ctrl_end"):   "[duty]",
                ("buck",    "fb_delay"):      "[V_out]",
            }
    """

    def __init__(self, sim, title: str = "EmbedSim Topology",
                 wire_labels: Optional[Dict[Tuple[str,str], str]] = None):
        self._sim         = sim
        self._title       = title
        self._wire_labels = wire_labels or {}
        self._nodes, self._edges = _extract_graph(sim)

    def print_console(self) -> None:
        """Print a clean multi-lane topology to stdout."""
        print(_render_console(self._nodes, self._edges))

    def get_console_str(self) -> str:
        return _render_console(self._nodes, self._edges)

    def get_html(self) -> str:
        return _render_html(self._nodes, self._edges,
                            title=self._title,
                            wire_labels=self._wire_labels)

    def export_html(self, path: str,
                    wire_labels: Optional[Dict[Tuple[str,str], str]] = None) -> str:
        """
        Write the topology HTML to *path*.

        wire_labels here overrides the instance-level wire_labels if supplied,
        so  sim.topo.export_html("out.html", wire_labels=WIRE_LABELS)  works
        even when TopologyPrinter was constructed without labels.
        """
        wl = wire_labels if wire_labels is not None else self._wire_labels
        abs_path = os.path.abspath(path)
        html = _render_html(self._nodes, self._edges,
                            title=self._title, wire_labels=wl)
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"[TopologyPrinter] Saved: {abs_path}")
        return abs_path

    def show_gui(self, path: Optional[str] = None) -> str:
        if path is None:
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix='.html',
                prefix='embedsim_topo_', mode='w', encoding='utf-8')
            tmp.write(self.get_html())
            tmp.close()
            path = tmp.name
        else:
            self.export_html(path)
        url = 'file://' + os.path.abspath(path)
        webbrowser.open(url)
        print(f"[TopologyPrinter] GUI opened: {url}")
        return path


def attach(sim, title: str = "EmbedSim Topology") -> TopologyPrinter:
    """Attach TopologyPrinter to sim as sim.topo."""
    printer = TopologyPrinter(sim, title=title)
    sim.topo = printer
    sim.print_topology_sources2sink = printer.print_console
    return printer


# =============================================================================
# Standalone demo
# =============================================================================

class _DemoBlock:
    def __init__(self, name, cls, dyn=False, lb=False, c=False, attrs=None):
        self.name            = name
        self._cls            = cls
        self.is_dynamic      = dyn
        self.is_loop_breaker = lb
        self.use_c_backend   = c
        self.inputs          = []     # public 'inputs' — same as real EmbedSim
        self._attrs          = attrs or {}


class _DemoSim:
    def __init__(self):
        B = _DemoBlock
        self.omega_ref  = B('omega_ref',  'VectorStep',
                            attrs={'step_time':'0.05 s','after_value':'100 rad/s'})
        self.cg_start   = B('cg_start',   'CodeGenStart')
        self.speed_pi   = B('speed_pi',   'SpeedPIBlock',   dyn=True,  c=True,
                            attrs={'Kp':'1.0','Ki':'20.0','i_max':'20 A'})
        self.smc        = B('smc',        'SMCBlock',       dyn=False, c=True,
                            attrs={'lambda_d':'83','K_sw_d':'40',
                                   'phi_d':'1','out_min':'-48 V','out_max':'48 V'})
        self.inv_park   = B('inv_park',   'InvParkFOC')
        self.inv_clarke = B('inv_clarke', 'InvClarkeTransformBlock')
        self.cg_end     = B('cg_end',     'CodeGenEnd')
        self.sink       = B('sink',       'VectorEnd')
        self.motor      = B('motor',      'ThreePhaseMotorBlock', dyn=True,
                            attrs={'R':'0.5 Ω','L_d':'5 mH','L_q':'6 mH',
                                   'lambda_pm':'0.175 Wb','p':'2'})
        self.motor_sink = B('motor_sink', '_MotorCaptureSink')
        self.t_load     = B('T_load',     'VectorConstant',
                            attrs={'value':'[0.2 N·m]'})
        self.d_omega    = B('delay_omega','_RegDelay',  lb=True)
        self.d_idiq     = B('delay_idiq', '_RegDelay',  lb=True)
        self.d_theta    = B('delay_theta','_RegDelay',  lb=True)

        self.cg_start.inputs   = [self.omega_ref]
        self.speed_pi.inputs   = [self.cg_start,  self.d_omega]
        self.smc.inputs        = [self.speed_pi,  self.d_idiq]
        self.inv_park.inputs   = [self.smc,        self.d_theta]
        self.inv_clarke.inputs = [self.inv_park]
        self.cg_end.inputs     = [self.inv_clarke]
        self.sink.inputs       = [self.cg_end]
        self.motor.inputs      = [self.smc,        self.t_load]
        self.motor_sink.inputs = [self.motor]
        self.d_omega.inputs    = [self.motor_sink]
        self.d_idiq.inputs     = [self.motor_sink]
        self.d_theta.inputs    = [self.motor_sink]

        # sim.blocks in topological order as EmbedSim would produce
        self.blocks = [
            self.omega_ref, self.t_load,
            self.d_omega, self.d_idiq, self.d_theta,
            self.cg_start, self.speed_pi, self.smc,
            self.inv_park, self.inv_clarke, self.cg_end, self.sink,
            self.motor, self.motor_sink,
        ]
        self.sinks = [self.sink, self.motor_sink]


def _demo():
    print("\n[topology_printer] Standalone demo — PMSM FOC (14 blocks)\n")
    sim     = _DemoSim()
    printer = TopologyPrinter(sim, title="EmbedSim — PMSM FOC Demo")
    printer.print_console()
    out = printer.export_html("embedsim_topology_demo.html")
    printer.show_gui(path=out)
    print("\nDone.")


if __name__ == '__main__':
    _demo()
