"""demo_server.py — Demo upload + Model performance dashboard.

Run:  python demo_server.py
Open: http://localhost:8888
"""

import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from PIL import Image

from ai_engine.image_processing.defect_detection import detect_defect_mobilenet_demo as detect_defect_mobilenet

app = FastAPI()

RESULTS_JSON = Path("ai_engine/models/results/image_baseline_results.json")
LABELED_DIR  = Path("labeled/labeled")

HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Defect Detection Demo</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', sans-serif;
  background: #0f172a; color: #e2e8f0;
  min-height: 100vh; padding: 32px 20px;
  display: flex; flex-direction: column; align-items: center;
}
h1 { font-size: 1.6rem; font-weight: 700; color: #f1f5f9; margin-bottom: 4px; text-align:center; }
.sub { color: #64748b; font-size: 0.85rem; margin-bottom: 24px; text-align:center; }

/* Tabs */
.tabs { display: flex; gap: 4px; margin-bottom: 24px; background:#1e293b;
  padding: 4px; border-radius: 12px; }
.tab { padding: 8px 22px; border-radius: 9px; border: none; cursor: pointer;
  font-size: 0.88rem; font-weight: 600; transition: all 0.2s;
  color: #64748b; background: transparent; }
.tab.active { background: #6366f1; color: white; }

.wrap { width: 100%; max-width: 800px; }
.panel { display: none; }
.panel.show { display: block; }

/* Drop zone */
.drop-zone {
  border: 2px dashed #334155; border-radius: 16px;
  padding: 40px 24px; text-align: center; cursor: pointer;
  transition: all 0.2s; background: #1e293b;
}
.drop-zone:hover, .drop-zone.drag { border-color: #6366f1; background: #1e2d45; }
.drop-zone svg { width: 44px; height: 44px; color: #475569; margin-bottom: 10px; }
.drop-zone p { color: #64748b; font-size: 0.88rem; }
.drop-zone input { display: none; }

/* Spinner */
.loading { display: none; text-align: center; color: #818cf8; padding: 12px; }
.loading.show { display: block; }
.spinner {
  display: inline-block; width: 18px; height: 18px;
  border: 3px solid #334155; border-top-color: #6366f1;
  border-radius: 50%; animation: spin 0.7s linear infinite;
  vertical-align: middle; margin-right: 6px;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Summary */
.summary { display: none; }
.summary.show { display: block; }
.card { background: #1e293b; border: 1px solid #334155; border-radius: 16px; padding: 20px; margin-bottom: 16px; }
.stat-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 16px; }
.stat-card { background: #0f172a; border-radius: 12px; padding: 14px 16px; border-left: 4px solid; }
.stat-card .count { font-size: 1.8rem; font-weight: 700; }
.stat-card .lbl { font-size: 0.78rem; color: #64748b; }
.stat-card .pct { font-size: 0.75rem; color: #94a3b8; }
.completion { display: flex; align-items: center; gap: 16px; margin-bottom: 16px; background:#0f172a; border-radius:12px; padding:16px; }
.score-ring { width: 68px; height: 68px; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; font-size: 1.2rem; font-weight: 700; flex-shrink: 0; border: 4px solid; }
.img-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 8px; }
.img-item { background: #0f172a; border-radius: 10px; overflow: hidden; border: 2px solid transparent; position: relative; }
.img-item img { width: 100%; height: 85px; object-fit: cover; display: block; }
.img-item .badge { position: absolute; bottom: 0; left: 0; right: 0; font-size: 0.68rem;
  padding: 2px 4px; text-align: center; font-weight: 600; background: rgba(0,0,0,0.75); }
.img-item .conf { font-size: 0.65rem; color: #94a3b8; text-align: center; padding: 2px 0; background: #0f172a; }
.reset-btn { margin-top: 4px; background: #6366f1; color: white; border: none;
  padding: 10px; border-radius: 8px; cursor: pointer; font-size: 0.9rem; font-weight: 600; width: 100%; }
.reset-btn:hover { background: #4f46e5; }

/* Metrics panel */
.metric-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; flex-wrap:wrap; gap:8px; }
.metric-header h2 { font-size: 1rem; font-weight: 600; color: #f1f5f9; }
.eval-btn { background: #0f172a; color: #6366f1; border: 1px solid #334155;
  padding: 7px 16px; border-radius: 8px; cursor: pointer; font-size: 0.82rem;
  font-weight: 600; transition: all 0.2s; }
.eval-btn:hover { border-color: #6366f1; }
.eval-btn:disabled { opacity:0.4; cursor:not-allowed; }

.big-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 16px; }
.big-metric { background: #0f172a; border-radius: 12px; padding: 16px; text-align: center; }
.big-metric .val { font-size: 2rem; font-weight: 700; color: #6366f1; }
.big-metric .key { font-size: 0.78rem; color: #64748b; margin-top: 4px; }

.class-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.class-table th { color: #64748b; font-weight: 600; text-align: left; padding: 8px 10px;
  border-bottom: 1px solid #334155; }
.class-table td { padding: 10px; border-bottom: 1px solid #1e293b; }
.class-table tr:last-child td { border-bottom: none; }

.bar-mini { height: 8px; border-radius: 999px; background: #1e293b; overflow: hidden; min-width: 80px; }
.bar-mini-fill { height: 100%; border-radius: 999px; }

.eval-status { font-size: 0.8rem; color: #64748b; margin-top: 8px; text-align:right; }
</style>
</head>
<body>
<h1>Defect Detection Demo</h1>
<p class="sub">MobileNetV3 — Nhan dien tinh trang hop hang tu anh review</p>

<div class="tabs">
  <button class="tab active" onclick="switchTab('upload')">Upload Test</button>
  <button class="tab" onclick="switchTab('metrics')">Model Performance</button>
</div>

<div class="wrap">
  <!-- TAB 1: UPLOAD -->
  <div class="panel show" id="panel-upload">
    <div class="drop-zone" id="dropZone">
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
          d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
      </svg>
      <p>Keo tha <strong style="color:#6366f1">nhieu anh</strong> vao day hoac click de chon</p>
      <p style="margin-top:5px">JPG, PNG — toi da 50 anh</p>
      <input type="file" id="fileInput" accept="image/*" multiple>
    </div>
    <div class="loading" id="loading">
      <span class="spinner"></span><span id="loadingText">Dang phan tich...</span>
    </div>
    <div class="summary" id="summary">
      <div class="card">
        <div class="completion" id="completionBadge"></div>
        <h2 style="font-size:0.88rem;color:#64748b;margin-bottom:12px">
          Phan bo tinh trang (<span id="totalCount">0</span> anh)
        </h2>
        <div class="stat-grid" id="statGrid"></div>
        <p style="font-size:0.82rem;color:#64748b;margin-bottom:8px">Chi tiet tung anh</p>
        <div class="img-grid" id="imgGrid"></div>
        <button class="reset-btn" style="margin-top:12px" onclick="reset()">Thu lai bo anh khac</button>
      </div>
    </div>
  </div>

  <!-- TAB 2: METRICS -->
  <div class="panel" id="panel-metrics">
    <div class="card">
      <div class="metric-header">
        <h2>Hieu suat model tren tap du lieu labeled</h2>
        <button class="eval-btn" id="evalBtn" onclick="runEval()">
          Chay danh gia lai (~2 phut)
        </button>
      </div>
      <div id="metricsContent">
        <p style="color:#64748b;font-size:0.85rem;text-align:center;padding:24px">
          Dang tai ket qua...
        </p>
      </div>
      <div class="eval-status" id="evalStatus"></div>
    </div>
  </div>
</div>

<script>
const COLORS = { intact:'#22c55e', damaged:'#ef4444', wrong_item:'#f59e0b', irrelevant:'#6b7280' };
const EMOJIS = { intact:'OK', damaged:'HU', wrong_item:'SAI', irrelevant:'KLQ' };
const VIET   = { intact:'Hop nguyen ven', damaged:'Hop hu hong', wrong_item:'Giao sai hang', irrelevant:'Khong lien quan' };

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    t.classList.toggle('active', ['upload','metrics'][i] === name);
  });
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('show'));
  document.getElementById('panel-' + name).classList.add('show');
  if (name === 'metrics') loadMetrics();
}

// ---------- UPLOAD TAB ----------
const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const loading   = document.getElementById('loading');
const summary   = document.getElementById('summary');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag');
  if (e.dataTransfer.files.length) processFiles(Array.from(e.dataTransfer.files));
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) processFiles(Array.from(fileInput.files));
});

async function processFiles(files) {
  files = files.filter(f => f.type.startsWith('image/')).slice(0, 50);
  if (!files.length) return;
  dropZone.style.display = 'none';
  summary.classList.remove('show');
  loading.classList.add('show');
  const results = [], previews = [];
  for (let i = 0; i < files.length; i++) {
    document.getElementById('loadingText').textContent = `Dang phan tich anh ${i+1}/${files.length}...`;
    previews.push(URL.createObjectURL(files[i]));
    const form = new FormData();
    form.append('file', files[i]);
    try {
      const r = await fetch('/predict', { method:'POST', body:form });
      results.push(await r.json());
    } catch { results.push({ label:'irrelevant', confidence:0, probabilities:{}, inference_ms:0 }); }
  }
  loading.classList.remove('show');
  renderUploadResults(results, previews);
}

function renderUploadResults(results, previews) {
  const total = results.length;
  const counts = { intact:0, damaged:0, wrong_item:0, irrelevant:0 };
  results.forEach(r => { if (r.label in counts) counts[r.label]++; });
  document.getElementById('totalCount').textContent = total;

  const intactRate = counts.intact / total;
  const score = Math.round(intactRate * 100);
  let grade, gc;
  if (score>=90){grade='XUAT SAC';gc='#22c55e';}
  else if(score>=70){grade='KHA TOT';gc='#84cc16';}
  else if(score>=50){grade='TRUNG BINH';gc='#f59e0b';}
  else{grade='KEM';gc='#ef4444';}

  document.getElementById('completionBadge').innerHTML = `
    <div class="score-ring" style="color:${gc};border-color:${gc}">${score}%</div>
    <div>
      <div style="font-size:1rem;font-weight:700;color:${gc}">${grade}</div>
      <div style="font-size:0.8rem;color:#64748b;margin-top:3px">
        Intact: ${score}% &nbsp;|&nbsp; Defect: ${Math.round((counts.damaged+counts.wrong_item)/total*100)}%
      </div>
    </div>`;

  const sg = document.getElementById('statGrid');
  sg.innerHTML = '';
  ['intact','damaged','wrong_item','irrelevant'].forEach(k => {
    const pct = total ? (counts[k]/total*100).toFixed(1) : '0';
    sg.innerHTML += `
      <div class="stat-card" style="border-color:${COLORS[k]}">
        <div class="count" style="color:${COLORS[k]}">${counts[k]}</div>
        <div class="lbl">${VIET[k]}</div>
        <div class="pct">${pct}% / ${total} anh</div>
      </div>`;
  });

  const ig = document.getElementById('imgGrid');
  ig.innerHTML = '';
  results.forEach((r,i) => {
    const conf = (r.confidence*100).toFixed(0);
    ig.innerHTML += `
      <div class="img-item" style="border-color:${COLORS[r.label]}">
        <img src="${previews[i]}" alt="">
        <div class="badge" style="color:${COLORS[r.label]}">${EMOJIS[r.label]}</div>
        <div class="conf">${conf}% · ${r.inference_ms}ms</div>
      </div>`;
  });
  summary.classList.add('show');
}

function reset() {
  summary.classList.remove('show');
  dropZone.style.display = '';
  fileInput.value = '';
}

// ---------- METRICS TAB ----------
async function loadMetrics() {
  try {
    const res = await fetch('/metrics');
    const data = await res.json();
    renderMetrics(data);
  } catch(e) {
    document.getElementById('metricsContent').innerHTML =
      '<p style="color:#ef4444;text-align:center;padding:24px">Chua co ket qua. Bam "Chay danh gia lai".</p>';
  }
}

function renderMetrics(data) {
  const classes = ['damaged','intact','irrelevant','wrong_item'];
  const acc  = (data.overall_accuracy * 100).toFixed(1);
  const f1   = (data.macro_f1 * 100).toFixed(1);
  const time = data.training_minutes ? data.training_minutes.toFixed(1) : '-';

  let classRows = '';
  classes.forEach(cls => {
    const m = data.per_class ? data.per_class[cls] : null;
    if (!m) return;
    const p = (m.precision*100).toFixed(1);
    const r = (m.recall*100).toFixed(1);
    const f = (m.f1*100).toFixed(1);
    classRows += `
      <tr>
        <td style="color:${COLORS[cls]};font-weight:600">${cls}</td>
        <td>${m.support}</td>
        <td>
          <div style="display:flex;align-items:center;gap:8px">
            <div class="bar-mini" style="width:80px">
              <div class="bar-mini-fill" style="width:${p}%;background:${COLORS[cls]}"></div>
            </div>
            <span>${p}%</span>
          </div>
        </td>
        <td>
          <div style="display:flex;align-items:center;gap:8px">
            <div class="bar-mini" style="width:80px">
              <div class="bar-mini-fill" style="width:${r}%;background:${COLORS[cls]}"></div>
            </div>
            <span>${r}%</span>
          </div>
        </td>
        <td style="font-weight:700;color:${parseFloat(f)>=80?'#22c55e':'#f59e0b'}">${f}%</td>
      </tr>`;
  });

  document.getElementById('metricsContent').innerHTML = `
    <div class="big-metrics">
      <div class="big-metric">
        <div class="val">${acc}%</div>
        <div class="key">Overall Accuracy</div>
      </div>
      <div class="big-metric">
        <div class="val">${f1}%</div>
        <div class="key">Macro F1</div>
      </div>
      <div class="big-metric">
        <div class="val">${time}m</div>
        <div class="key">Train Time</div>
      </div>
    </div>
    <table class="class-table">
      <thead>
        <tr>
          <th>Class</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th>
        </tr>
      </thead>
      <tbody>${classRows}</tbody>
    </table>`;

  if (data.evaluated_at) {
    document.getElementById('evalStatus').textContent =
      'Danh gia luc: ' + new Date(data.evaluated_at).toLocaleString('vi-VN');
  }
}

async function runEval() {
  const btn = document.getElementById('evalBtn');
  btn.disabled = true;
  btn.textContent = 'Dang chay... (~2 phut)';
  document.getElementById('evalStatus').textContent = 'Dang danh gia toan bo dataset...';
  try {
    const res = await fetch('/evaluate', { method:'POST' });
    const data = await res.json();
    renderMetrics(data);
    btn.textContent = 'Chay danh gia lai (~2 phut)';
  } catch(e) {
    document.getElementById('evalStatus').textContent = 'Loi: ' + e;
    btn.textContent = 'Chay danh gia lai (~2 phut)';
  }
  btn.disabled = false;
}

// Auto load metrics khi click tab
window.addEventListener('load', () => {});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tmp = os.path.join(tempfile.gettempdir(), "demo_upload.jpg")
    img.save(tmp)
    return detect_defect_mobilenet(tmp)


@app.get("/metrics")
async def get_metrics():
    """Tra ve ket qua da luu san trong JSON."""
    if not RESULTS_JSON.exists():
        return JSONResponse(status_code=404, content={"error": "Chua co ket qua. Hay chay evaluate."})
    data = json.loads(RESULTS_JSON.read_text())
    # RESULTS_JSON la list, lay item dau tien
    return data[0] if isinstance(data, list) else data


@app.post("/evaluate")
async def run_evaluate():
    """Chay danh gia chinh thuc tren toan bo labeled dataset."""
    import datetime
    from sklearn.metrics import classification_report
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import torch

    if not LABELED_DIR.exists():
        return JSONResponse(status_code=404, content={"error": "Khong tim thay thu muc labeled"})

    from ai_engine.models.image_baseline import ImageBaselineModel, _build_transforms

    model = ImageBaselineModel.load("ai_engine/models/weights/mobilenet_v3_defect.pt")

    ds = datasets.ImageFolder(str(LABELED_DIR), transform=_build_transforms(is_train=False))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model.model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(model.device)
            preds = model.model(imgs).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    report = classification_report(all_labels, all_preds,
                                   target_names=ds.classes, output_dict=True)
    result = {
        "overall_accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "training_minutes": None,
        "evaluated_at": datetime.datetime.now().isoformat(),
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall":    report[cls]["recall"],
                "f1":        report[cls]["f1-score"],
                "support":   int(report[cls]["support"]),
            }
            for cls in ds.classes if cls in report
        }
    }

    # Luu ket qua vao JSON de lan sau load nhanh
    if RESULTS_JSON.exists():
        existing = json.loads(RESULTS_JSON.read_text())
        if isinstance(existing, list) and existing:
            result["training_minutes"] = existing[0].get("training_minutes")
    RESULTS_JSON.write_text(json.dumps([result], indent=2))

    return result


if __name__ == "__main__":
    print("\nDemo: http://localhost:8888\n")
    uvicorn.run(app, host="0.0.0.0", port=8888)
