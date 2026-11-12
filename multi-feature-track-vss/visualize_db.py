"""
Management & Operator Visual Dashboard for LanceDB Video Search & Surveillance Data.
Launches a local FastAPI web GUI to visually inspect Global Person Tracks,
Tracked Objects & Abandoned Bag Alerts (with linked Owner crops), Scene Keyframes,
and Phase 2 Spatial-Windowed Video Event Summaries.
"""

import argparse
import json
from pathlib import Path
from typing import List
from urllib.parse import quote

import lancedb
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse


PROJECT_ROOT = Path(__file__).resolve().parent


def _discover_dbs(default_db: Path) -> List[str]:
    candidates = []
    seen = set()
    def _add(p: Path):
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            candidates.append(str(p))

    if default_db.exists():
        _add(default_db)
    data_dir = PROJECT_ROOT / "data"
    if data_dir.exists():
        for p in sorted(data_dir.iterdir()):
            if p.is_dir() and p.name.startswith("lancedb"):
                _add(p)
    return candidates or [str(default_db)]


def _safe_media_url(raw_path: str) -> str:
    if not raw_path:
        return ""
    p = Path(raw_path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()
    if not p.exists() or not p.is_file():
        return ""
    try:
        p.relative_to(PROJECT_ROOT)
    except ValueError:
        return ""
    return f"/media?path={quote(str(p))}"


def _read_db_snapshot(db_path: str) -> dict:
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"LanceDB directory not found: {db_path}")

    db = lancedb.connect(str(p))
    tables = set(db.table_names())

    def _rows(name: str) -> List[dict]:
        if name not in tables:
            return []
        df = db.open_table(name).to_pandas()
        # Drop large float vector columns before JSON serialization
        drop_cols = [c for c in df.columns if "vector" in c]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        return df.to_dict(orient="records")

    persons = _rows("person_tracks")
    scenes = _rows("scene_frames")
    objects = _rows("object_tracks")
    events = _rows("video_events")

    for rec in persons:
        rec["best_crop_url"] = _safe_media_url(str(rec.get("best_crop_path", "")))
        rec["face_crop_url"] = _safe_media_url(str(rec.get("face_crop_path", "")))
        try:
            rec["intervals"] = json.loads(rec.get("intervals_json", "[]") or "[]")
        except Exception:
            rec["intervals"] = []

    for rec in objects:
        rec["best_crop_url"] = _safe_media_url(str(rec.get("best_crop_path", "")))
        rec["owner_crop_url"] = _safe_media_url(str(rec.get("owner_crop_path", "")))
        rec["abandoned_frame_url"] = _safe_media_url(str(rec.get("abandoned_frame_path", "")))

    for rec in scenes:
        rec["frame_url"] = _safe_media_url(str(rec.get("frame_path", "")))

    for rec in events:
        rec["keyframe_url"] = _safe_media_url(str(rec.get("keyframe_path", "")))
        try:
            rec["referenced_track_ids"] = json.loads(rec.get("referenced_track_ids_json", "[]") or "[]")
        except Exception:
            rec["referenced_track_ids"] = []
        try:
            rec["referenced_object_ids"] = json.loads(rec.get("referenced_object_ids_json", "[]") or "[]")
        except Exception:
            rec["referenced_object_ids"] = []

    events.sort(key=lambda r: float(r.get("start_sec", 0.0)))
    scenes.sort(key=lambda r: float(r.get("timestamp_sec", 0.0)))
    objects.sort(key=lambda r: (0 if r.get("status") == "abandoned" else 1, float(r.get("start_sec", 0.0))))

    abandoned_count = sum(1 for o in objects if o.get("status") in ("abandoned", "unattended_at_end"))
    peak_people = max((int(s.get("person_count", 0)) for s in scenes), default=0)

    return {
        "db_path": str(p),
        "stats": {
            "persons": len(persons),
            "objects": len(objects),
            "abandoned": abandoned_count,
            "scenes": len(scenes),
            "events": len(events),
            "peak_people": peak_people,
            "standing": sum(1 for r in persons if r.get("pose_label") == "standing"),
            "sitting": sum(1 for r in persons if r.get("pose_label") == "sitting"),
            "faces": sum(1 for r in persons if r.get("has_face")),
        },
        "persons": persons,
        "objects": objects,
        "scenes": scenes,
        "events": events,
    }


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Auditorium VSS — LanceDB Visual Explorer</title>
<style>
  :root {
    --bg: #0f172a; --panel: #1e293b; --card: #162033; --border: #334155;
    --text: #f1f5f9; --muted: #94a3b8; --accent: #38bdf8;
    --green: #22c55e; --amber: #f59e0b; --red: #ef4444;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--text); }
  header {
    position: sticky; top: 0; z-index: 20; background: rgba(15,23,42,0.94); backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between;
  }
  .brand { font-size: 18px; font-weight: 700; display: flex; align-items: center; gap: 10px; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
  select, input {
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 12px; font-size: 14px;
  }
  input { min-width: 240px; }
  .container { max-width: 1400px; margin: 0 auto; padding: 20px 24px 60px; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin-bottom: 20px; }
  .kpi { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
  .kpi-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .kpi-val { font-size: 26px; font-weight: 700; margin-top: 4px; }
  .kpi.alert { border-color: rgba(239,68,68,0.6); background: rgba(127,29,29,0.22); }
  .kpi.alert .kpi-val { color: #fca5a5; }
  .tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
  .tab {
    background: transparent; color: var(--muted); border: 1px solid transparent; border-radius: 8px;
    padding: 8px 16px; font-size: 14px; font-weight: 600; cursor: pointer;
  }
  .tab.active { background: var(--panel); color: var(--accent); border-color: var(--border); }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 16px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; }
  .card.abandoned { border-color: var(--red); box-shadow: 0 0 0 1px rgba(239,68,68,0.35); }
  .card-body { padding: 14px; display: flex; flex-direction: column; gap: 8px; font-size: 13px; }
  .card-title { font-size: 15px; font-weight: 700; display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .badge { font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 999px; text-transform: uppercase; }
  .b-red { background: rgba(239,68,68,0.2); color: #fca5a5; border: 1px solid rgba(239,68,68,0.45); }
  .b-green { background: rgba(34,197,94,0.18); color: #86efac; border: 1px solid rgba(34,197,94,0.4); }
  .b-amber { background: rgba(245,158,11,0.18); color: #fcd34d; border: 1px solid rgba(245,158,11,0.4); }
  .b-blue { background: rgba(56,189,248,0.16); color: #7dd3fc; border: 1px solid rgba(56,189,248,0.4); }
  .media-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; background: #0b1220; padding: 8px; }
  .media-single { background: #0b1220; padding: 8px; text-align: center; }
  .thumb { width: 100%; height: 180px; object-fit: contain; border-radius: 6px; cursor: pointer; background: #020617; }
  .thumb-cap { font-size: 11px; color: var(--muted); text-align: center; margin-top: 4px; }
  .meta-row { color: var(--muted); line-height: 1.45; }
  .meta-row strong { color: var(--text); }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip { background: var(--card); border: 1px solid var(--border); border-radius: 6px; padding: 2px 7px; font-size: 11px; color: var(--accent); }
  .empty { padding: 40px; text-align: center; color: var(--muted); background: var(--panel); border-radius: 12px; border: 1px dashed var(--border); }
  #lightbox {
    display: none; position: fixed; inset: 0; z-index: 50; background: rgba(2,6,23,0.9);
    align-items: center; justify-content: center; padding: 24px; cursor: zoom-out;
  }
  #lightbox img { max-width: 95vw; max-height: 90vh; border-radius: 10px; border: 1px solid var(--border); }
</style>
</head>
<body>
<header>
  <div class="brand">Auditorium VSS &mdash; LanceDB Visual Explorer</div>
  <div class="controls">
    <select id="dbSelect"></select>
    <input id="searchInput" type="text" placeholder="Filter by ID, label, status, summary..." />
  </div>
</header>

<div class="container">
  <div class="kpis" id="kpis"></div>
  <div class="tabs">
    <button class="tab active" data-tab="objects">Objects &amp; Abandonment Alerts</button>
    <button class="tab" data-tab="persons">Global Persons</button>
    <button class="tab" data-tab="events">Video Event Timeline</button>
    <button class="tab" data-tab="scenes">Scene Keyframes</button>
  </div>
  <div id="content"></div>
</div>

<div id="lightbox" onclick="this.style.display='none'"><img id="lbImg" src="" /></div>

<script>
let state = { dbs: [], currentDb: '', data: null, tab: 'objects', query: '' };

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const rem = (s % 60).toFixed(1).padStart(4, '0');
  return `${String(m).padStart(2,'0')}:${rem}`;
}

function zoom(url) {
  if (!url) return;
  document.getElementById('lbImg').src = url;
  document.getElementById('lightbox').style.display = 'flex';
}

function statusBadge(st) {
  if (st === 'abandoned') return `<span class="badge b-red">Abandoned</span>`;
  if (st === 'unattended_at_end') return `<span class="badge b-amber">Unattended @ End</span>`;
  if (st === 'stationary') return `<span class="badge b-amber">Stationary</span>`;
  return `<span class="badge b-green">${st || 'attended'}</span>`;
}

async function loadDbs() {
  const res = await fetch('/api/dbs');
  const json = await res.json();
  state.dbs = json.dbs;
  state.currentDb = json.default_db;
  const sel = document.getElementById('dbSelect');
  sel.innerHTML = state.dbs.map(d => `<option value="${d}" ${d === state.currentDb ? 'selected' : ''}>${d}</option>`).join('');
  await loadData(state.currentDb);
}

async function loadData(dbPath) {
  const res = await fetch(`/api/data?db_path=${encodeURIComponent(dbPath)}`);
  state.data = await res.json();
  render();
}

function renderKpis() {
  const s = state.data.stats;
  document.getElementById('kpis').innerHTML = `
    <div class="kpi"><div class="kpi-label">Global Persons</div><div class="kpi-val">${s.persons}</div></div>
    <div class="kpi"><div class="kpi-label">Tracked Objects</div><div class="kpi-val">${s.objects}</div></div>
    <div class="kpi ${s.abandoned > 0 ? 'alert' : ''}"><div class="kpi-label">Abandoned Alerts</div><div class="kpi-val">${s.abandoned}</div></div>
    <div class="kpi"><div class="kpi-label">Video Event Chunks</div><div class="kpi-val">${s.events}</div></div>
    <div class="kpi"><div class="kpi-label">Scene Keyframes</div><div class="kpi-val">${s.scenes}</div></div>
    <div class="kpi"><div class="kpi-label">Peak Crowd</div><div class="kpi-val">${s.peak_people}</div></div>
  `;
}

function render() {
  if (!state.data) return;
  renderKpis();
  const q = state.query.toLowerCase().trim();
  const el = document.getElementById('content');

  if (state.tab === 'objects') {
    const items = state.data.objects.filter(o => !q || JSON.stringify(o).toLowerCase().includes(q));
    if (!items.length) { el.innerHTML = `<div class="empty">No tracked objects found in this database. Run pipeline.py with <code>--track-objects</code>.</div>`; return; }
    el.innerHTML = `<div class="grid">` + items.map(o => `
      <div class="card ${(o.status === 'abandoned' || o.status === 'unattended_at_end') ? 'abandoned' : ''}">
        <div class="media-pair">
          <div>
            ${o.best_crop_url ? `<img class="thumb" src="${o.best_crop_url}" onclick="zoom('${o.best_crop_url}')" />` : `<div class="thumb"></div>`}
            <div class="thumb-cap">Object [OBJ:${o.object_id}]</div>
          </div>
          <div>
            ${o.owner_crop_url ? `<img class="thumb" src="${o.owner_crop_url}" onclick="zoom('${o.owner_crop_url}')" />` : `<div class="thumb"></div>`}
            <div class="thumb-cap">Owner: Person #${o.owner_track_id}</div>
          </div>
        </div>
        <div class="card-body">
          <div class="card-title"><span>[OBJ:${o.object_id}] ${o.label}</span>${statusBadge(o.status)}</div>
          <div class="meta-row"><strong>Bound Owner:</strong> Global Person #${o.owner_track_id}</div>
          <div class="meta-row"><strong>Visible Clip:</strong> ${fmtTime(o.start_sec)} &ndash; ${fmtTime(o.end_sec)} (${o.start_sec.toFixed(1)}s &ndash; ${o.end_sec.toFixed(1)}s)</div>
          ${(o.status === 'abandoned' || o.status === 'unattended_at_end') && o.abandoned_since_sec >= 0 ? `<div class="meta-row" style="color:${o.status === 'abandoned' ? '#fca5a5' : '#fcd34d'};"><strong>${o.status === 'abandoned' ? 'Abandoned Since' : 'Unattended Since'}:</strong> ${fmtTime(o.abandoned_since_sec)} (${o.abandoned_since_sec.toFixed(1)}s)</div>` : ''}
          ${o.abandoned_frame_url ? `<div><button class="tab active" style="padding:4px 10px;font-size:12px;" onclick="zoom('${o.abandoned_frame_url}')">View Abandonment Alert Frame</button></div>` : ''}
        </div>
      </div>
    `).join('') + `</div>`;
  }

  else if (state.tab === 'persons') {
    const items = state.data.persons.filter(p => !q || JSON.stringify(p).toLowerCase().includes(q));
    if (!items.length) { el.innerHTML = `<div class="empty">No person tracks found.</div>`; return; }
    el.innerHTML = `<div class="grid">` + items.map(p => `
      <div class="card">
        <div class="${p.face_crop_url ? 'media-pair' : 'media-single'}">
          <div>
            ${p.best_crop_url ? `<img class="thumb" src="${p.best_crop_url}" onclick="zoom('${p.best_crop_url}')" />` : ''}
            <div class="thumb-cap">Body Crop</div>
          </div>
          ${p.face_crop_url ? `<div><img class="thumb" src="${p.face_crop_url}" onclick="zoom('${p.face_crop_url}')" /><div class="thumb-cap">Face Crop</div></div>` : ''}
        </div>
        <div class="card-body">
          <div class="card-title"><span>Global Person #${p.track_id}</span><span class="badge b-blue">${p.pose_label || 'unknown'}</span></div>
          <div class="meta-row"><strong>Total Span:</strong> ${fmtTime(p.start_sec)} &ndash; ${fmtTime(p.end_sec)}</div>
          <div class="meta-row"><strong>Face Indexed:</strong> ${p.has_face ? 'Yes' : 'No'}</div>
          <div class="chips">${(p.intervals || []).map(iv => `<span class="chip">${iv.start_sec.toFixed(1)}s&ndash;${iv.end_sec.toFixed(1)}s</span>`).join('')}</div>
        </div>
      </div>
    `).join('') + `</div>`;
  }

  else if (state.tab === 'events') {
    const items = state.data.events.filter(e => !q || JSON.stringify(e).toLowerCase().includes(q));
    if (!items.length) { el.innerHTML = `<div class="empty">No video event summaries found. Run pipeline.py with <code>--summarize</code>.</div>`; return; }
    el.innerHTML = `<div class="grid">` + items.map(e => `
      <div class="card">
        ${e.keyframe_url ? `<div class="media-single"><img class="thumb" src="${e.keyframe_url}" onclick="zoom('${e.keyframe_url}')" /></div>` : ''}
        <div class="card-body">
          <div class="card-title"><span>${e.chunk_id} (${fmtTime(e.start_sec)}&ndash;${fmtTime(e.end_sec)})</span><span class="badge b-amber">${e.trigger_reason}</span></div>
          <div class="chips">
            ${(e.referenced_track_ids || []).map(id => `<span class="chip">ID:${id}</span>`).join('')}
            ${(e.referenced_object_ids || []).map(id => `<span class="chip" style="color:#fcd34d;">OBJ:${id}</span>`).join('')}
          </div>
          <div class="meta-row" style="color:var(--text);">${e.summary_text}</div>
        </div>
      </div>
    `).join('') + `</div>`;
  }

  else if (state.tab === 'scenes') {
    const items = state.data.scenes.filter(s => !q || JSON.stringify(s).toLowerCase().includes(q));
    if (!items.length) { el.innerHTML = `<div class="empty">No scene keyframes found.</div>`; return; }
    el.innerHTML = `<div class="grid">` + items.map(s => `
      <div class="card">
        <div class="media-single">
          ${s.frame_url ? `<img class="thumb" src="${s.frame_url}" onclick="zoom('${s.frame_url}')" />` : ''}
        </div>
        <div class="card-body">
          <div class="card-title"><span>Frame #${s.frame_id} @ ${fmtTime(s.timestamp_sec)}</span><span class="badge b-blue">People: ${s.person_count}</span></div>
          <div class="meta-row"><strong>Standing:</strong> ${s.standing_count} | <strong>Sitting:</strong> ${s.sitting_count} | <strong>Unknown:</strong> ${s.unknown_count}</div>
        </div>
      </div>
    `).join('') + `</div>`;
  }
}

document.getElementById('dbSelect').addEventListener('change', e => {
  state.currentDb = e.target.value;
  loadData(state.currentDb);
});
document.getElementById('searchInput').addEventListener('input', e => {
  state.query = e.target.value;
  render();
});
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.tab = btn.dataset.tab;
    render();
  });
});

loadDbs();
</script>
</body>
</html>
"""


def create_app(default_db_path: str) -> FastAPI:
    app = FastAPI(title="Auditorium VSS LanceDB Visual Explorer")
    default_db = Path(default_db_path)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return DASHBOARD_HTML

    @app.get("/api/dbs")
    def list_dbs():
        dbs = _discover_dbs(default_db)
        return {"dbs": dbs, "default_db": dbs[0]}

    @app.get("/api/data")
    def get_data(db_path: str = Query(default=str(default_db))):
        try:
            return _read_db_snapshot(db_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/media")
    def get_media(path: str = Query(...)):
        target = Path(path).resolve()
        try:
            target.relative_to(PROJECT_ROOT)
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied outside project directory")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Media file not found")
        return FileResponse(str(target))

    return app


def main():
    parser = argparse.ArgumentParser(description="Visual Web GUI for Auditorium LanceDB Vector Database")
    parser.add_argument("--db-path", type=str, default="data/lancedb", help="Default LanceDB directory to open")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind (use 0.0.0.0 for LAN/server)")
    parser.add_argument("--port", type=int, default=7860, help="Port for web GUI")
    args = parser.parse_args()

    print(f"Starting Auditorium VSS Visual Explorer at http://{args.host}:{args.port} (default DB: {args.db_path})")
    app = create_app(args.db_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
