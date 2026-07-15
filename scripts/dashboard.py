"""Browser debug dashboard for the contextual-safety stack.

Separate process; reads the telemetry directory the demos write into and
serves a live page. The robot process never depends on this.

    pip install flask                      # once, any env
    python scripts/dashboard.py --dir results/isaac_debug --port 8000
    # then open http://localhost:8000  (or http://<robot-ip>:8000 remotely)

Panels are discovered automatically: every artifact kind that appears as
results/isaac_debug/HHMMSS_<kind>.png becomes an image panel; every key in
status.json becomes a status row. Parameter edits are written to
params.json, which the demos poll and apply on the fly.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from flask import Flask, jsonify, request, send_file

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="results/isaac_debug")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--host", default="0.0.0.0")
args = parser.parse_args()

DIR = Path(args.dir).resolve()
DIR.mkdir(parents=True, exist_ok=True)
app = Flask(__name__)

# Live-tunable parameters offered by the UI (name, min, max, step).
PARAM_SPECS = [
    ("v_max", 0.0, 1.0, 0.01),
    ("max_barrier_age", 0.0, 120.0, 1.0),
    ("perception_every", 5, 300, 5),
    ("tau", 0.1, 0.9, 0.05),
]


def _latest(kind: str) -> Path | None:
    files = sorted(DIR.glob(f"*_{kind}.png"))
    return files[-1] if files else None


def _kinds() -> list[str]:
    seen = {}
    for f in DIR.glob("*_*.png"):
        seen.setdefault(f.stem.split("_", 1)[1], None)
    order = ["rgb", "masks", "costmap"]
    return sorted(seen, key=lambda k: (order.index(k) if k in order else 99, k))


def _gpu() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3).stdout.strip()
        util, used, total = [s.strip() for s in out.splitlines()[0].split(",")]
        return f"GPU {util}% | VRAM {used}/{total} MiB"
    except Exception:
        return "GPU stats unavailable"


@app.get("/api/status")
def api_status():
    status = {}
    try:
        status = json.loads((DIR / "status.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    vlm_files = sorted(DIR.glob("*_vlm.txt"))
    vlm = vlm_files[-1].read_text(encoding="utf-8", errors="replace") if vlm_files else ""
    params = {}
    try:
        params = json.loads((DIR / "params.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    return jsonify({"status": status, "kinds": _kinds(), "gpu": _gpu(),
                    "vlm": vlm[-4000:], "params": params})


@app.get("/img/<kind>")
def img(kind):
    f = _latest(kind)
    if f is None:
        return ("", 404)
    return send_file(f, mimetype="image/png", max_age=0)


@app.post("/api/params")
def set_params():
    try:
        current = json.loads((DIR / "params.json").read_text(encoding="utf-8"))
    except Exception:
        current = {}
    current.update(request.get_json(force=True))
    (DIR / "params.json").write_text(json.dumps(current), encoding="utf-8")
    return jsonify(current)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>safety dashboard</title><style>
body{background:#14171c;color:#dde;font-family:system-ui,sans-serif;margin:14px}
h1{font-size:17px} h2{font-size:13px;color:#9ab;margin:4px 0}
.grid{display:flex;flex-wrap:wrap;gap:14px}
.panel{background:#1c2129;border-radius:8px;padding:10px;min-width:300px}
img{max-width:480px;width:100%;border-radius:4px;background:#000}
pre{white-space:pre-wrap;font-size:11px;max-width:480px;max-height:260px;
    overflow:auto;background:#12151a;padding:8px;border-radius:4px}
table{font-size:12px;border-collapse:collapse} td{padding:2px 10px 2px 0}
td:first-child{color:#9ab} .gpu{color:#7c9;font-size:12px;margin-left:12px}
input[type=number]{width:80px;background:#12151a;color:#dde;border:1px solid #345;
                   border-radius:4px;padding:3px}
button{background:#2a6;border:0;color:#fff;border-radius:4px;padding:4px 12px;
       cursor:pointer;margin-left:6px}
.param{margin:4px 0;font-size:12px} .param label{display:inline-block;width:140px}
</style></head><body>
<h1>contextual-safety dashboard <span class="gpu" id="gpu"></span></h1>
<div class="grid" id="imgs"></div>
<div class="grid">
 <div class="panel"><h2>status</h2><table id="status"></table></div>
 <div class="panel"><h2>parameters (applied on next cycle)</h2>
   <div id="params"></div></div>
 <div class="panel"><h2>VLM output (latest)</h2><pre id="vlm"></pre></div>
</div>
<script>
const SPECS = __SPECS__;
let built = false;
function buildParams(cur){
  const box = document.getElementById('params'); box.innerHTML='';
  for (const [name,min,max,step] of SPECS){
    const v = cur[name] !== undefined ? cur[name] : '';
    box.insertAdjacentHTML('beforeend',
      `<div class="param"><label>${name}</label>
       <input type="number" id="p_${name}" min="${min}" max="${max}"
              step="${step}" value="${v}" placeholder="(default)"></div>`);
  }
  box.insertAdjacentHTML('beforeend',
    '<button onclick="sendParams()">apply</button>');
}
function sendParams(){
  const out = {};
  for (const [name] of SPECS){
    const v = document.getElementById('p_'+name).value;
    if (v !== '') out[name] = parseFloat(v);
  }
  fetch('/api/params',{method:'POST',headers:{'Content-Type':'application/json'},
                       body:JSON.stringify(out)});
}
async function tick(){
  try{
    const r = await fetch('/api/status'); const d = await r.json();
    document.getElementById('gpu').textContent = d.gpu;
    document.getElementById('vlm').textContent = d.vlm || '(no VLM output yet)';
    const st = document.getElementById('status'); st.innerHTML='';
    for (const [k,v] of Object.entries(d.status))
      st.insertAdjacentHTML('beforeend',
        `<tr><td>${k}</td><td>${typeof v==='number'?v.toFixed(3):v}</td></tr>`);
    const imgs = document.getElementById('imgs');
    for (const k of d.kinds){
      let el = document.getElementById('img_'+k);
      if (!el){
        imgs.insertAdjacentHTML('beforeend',
          `<div class="panel"><h2>${k}</h2>
           <img id="img_${k}" src="/img/${k}"></div>`);
        el = document.getElementById('img_'+k);
      }
      el.src = '/img/'+k+'?t='+Date.now();
    }
    if (!built){ buildParams(d.params); built = true; }
  }catch(e){}
}
setInterval(tick, 1500); tick();
</script></body></html>"""


@app.get("/")
def index():
    return PAGE.replace("__SPECS__", json.dumps(PARAM_SPECS))


if __name__ == "__main__":
    print(f"dashboard on http://{args.host}:{args.port}  (dir: {DIR})")
    app.run(host=args.host, port=args.port, debug=False)
