"""
VisionAI Pro — Smart Surveillance System
Features:
  • Danger Detection & AI Recommendations
  • Crowd Density Monitor
  • Face Blur / Anonymisation (GDPR-safe)
  • Detection Heatmap
  • Zone / ROI Alerts (FIXED — pixel-accurate)
  • Auto Snapshot on Danger
  • Night Mode (CLAHE)
  • Annotated Video Export
  • Dwell Time Tracker
  • Confidence Histogram
  • Object Interaction Detection (NEW)
  • Speed Estimation (NEW)
  • Loitering Detection (NEW)
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile, os, time, io, csv, datetime, pathlib, smtplib
from collections import Counter, deque, defaultdict
from email.mime.text import MIMEText

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionAI Pro · Smart Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Mono',monospace;}
.stApp{background-color:#0a0a0c;color:#e0e0d8;}
[data-testid="stSidebar"]{background-color:#111114;border-right:1px solid #222228;}
.hero-block{background:linear-gradient(135deg,#111114 0%,#0a0a0c 100%);border:1px solid #222228;border-radius:12px;padding:28px 36px;margin-bottom:24px;position:relative;overflow:hidden;}
.hero-block::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00e5ff,#69ff47,#ff4c6a,#00e5ff);background-size:300% 100%;animation:shimmer 4s linear infinite;}
@keyframes shimmer{to{background-position:-300% 0;}}
.hero-title{font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;color:#fff;letter-spacing:-.02em;margin:0 0 4px 0;}
.hero-sub{font-size:.75rem;color:#555560;letter-spacing:.14em;text-transform:uppercase;}
.hero-tag{display:inline;font-size:.72rem;color:#444450;letter-spacing:.12em;text-transform:uppercase;}.hero-tag+.hero-tag::before{content:' · ';color:#333338;}
.accent{color:#69ff47;}.accent-red{color:#ff4c6a;}.accent-blue{color:#00e5ff;}
.metric-card{background:#111114;border:1px solid #222228;border-radius:10px;padding:16px 12px;text-align:center;margin-bottom:8px;}
.metric-val{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:#69ff47;}
.metric-val.blue{color:#00e5ff;}.metric-val.gold{color:#ffc947;}.metric-val.red{color:#ff4c6a;}.metric-val.purple{color:#c084fc;}
.metric-label{font-size:.65rem;color:#444450;letter-spacing:.15em;text-transform:uppercase;margin-top:3px;}
.density-badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:.72rem;font-weight:600;letter-spacing:.08em;}
.density-low{background:#1a2e1a;color:#69ff47;border:1px solid #2a5a2a;}
.density-medium{background:#2e2a14;color:#ffc947;border:1px solid #5a4a14;}
.density-high{background:#2e1a14;color:#ff7b47;border:1px solid #5a2a14;}
.density-critical{background:#2e1414;color:#ff4c6a;border:1px solid #5a1414;animation:pulse-red 1s infinite;}
@keyframes pulse-red{0%,100%{opacity:1}50%{opacity:.6}}
.alert-pill{background:#2e1414;border:1px solid #ff4c6a;border-radius:6px;padding:8px 14px;font-size:.76rem;color:#ff4c6a;margin:5px 0;}
.alert-pill-warn{background:#2e2a14;border:1px solid #ffc947;border-radius:6px;padding:8px 14px;font-size:.76rem;color:#ffc947;margin:5px 0;}
.alert-pill-info{background:#1a2535;border:1px solid #00e5ff;border-radius:6px;padding:8px 14px;font-size:.76rem;color:#00e5ff;margin:5px 0;}
.alert-pill-green{background:#1a2e1a;border:1px solid #69ff47;border-radius:6px;padding:8px 14px;font-size:.76rem;color:#69ff47;margin:5px 0;}
.stButton>button{font-family:'DM Mono',monospace;background:#69ff47 !important;color:#0a0a0c !important;border:none !important;border-radius:6px !important;font-weight:600 !important;}
.stButton>button:hover{background:#88ff6a !important;box-shadow:0 4px 20px rgba(105,255,71,.35) !important;}
.info-box{background:#111114;border-left:3px solid #00e5ff;border-radius:0 8px 8px 0;padding:12px 16px;font-size:.77rem;color:#777780;margin:10px 0;line-height:1.75;}
.stTabs [data-baseweb="tab-list"]{background:#111114;border-radius:8px;gap:4px;padding:4px;}
.stTabs [data-baseweb="tab"]{border-radius:6px;color:#555560;font-size:.78rem;}
.stTabs [aria-selected="true"]{background:#1e1e24 !important;color:#e0e0d8 !important;}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
DANGER_CLASSES = {"knife","scissors","gun","pistol","rifle","fire","flame",
                  "sword","baseball bat","bottle","cell phone"}
CROWD_THRESHOLDS = {"Low":3,"Medium":7,"High":15}

RECOMMENDATIONS = {
    "knife":       ("🔪 Knife detected",       "danger", "Alert security. Do not approach. Call emergency: 112."),
    "gun":         ("🔫 Weapon detected",       "danger", "Evacuate immediately. Call police: 100."),
    "pistol":      ("🔫 Weapon detected",       "danger", "Evacuate immediately. Call police: 100."),
    "rifle":       ("🔫 Weapon detected",       "danger", "Evacuate immediately. Call police: 100."),
    "fire":        ("🔥 Fire detected",         "danger", "Activate alarm. Evacuate. Call fire brigade: 101."),
    "flame":       ("🔥 Fire detected",         "danger", "Activate alarm. Evacuate. Call fire brigade: 101."),
    "scissors":    ("✂️ Sharp object",          "warn",   "Monitor. Alert security if suspicious."),
    "baseball bat":("🏏 Blunt object",          "warn",   "Monitor closely."),
    "bottle":      ("🍾 Glass object",          "warn",   "Monitor for aggression."),
    "person":      ("👤 Person detected",       "info",   "Log entry. Check if area is restricted."),
    "car":         ("🚗 Vehicle detected",      "info",   "Verify parking authorisation."),
    "truck":       ("🚚 Truck detected",        "info",   "Verify delivery schedule."),
    "motorcycle":  ("🏍 Motorcycle",            "info",   "Verify parking authorisation."),
    "bicycle":     ("🚲 Bicycle",               "info",   "Check bike park availability."),
    "backpack":    ("🎒 Backpack",              "warn",   "In secure zones: verify contents."),
    "suitcase":    ("🧳 Luggage",               "warn",   "In secure zones: verify contents."),
    "cell phone":  ("📱 Phone in zone",         "warn",   "Check if no-phone policy applies."),
    "laptop":      ("💻 Laptop",                "info",   "Verify device authorisation."),
    "cat":         ("🐱 Animal",                "info",   "Contact animal control if restricted area."),
    "dog":         ("🐕 Animal",                "info",   "Verify if animal is permitted."),
}

PPE_RECS = [
    "⛑️ No helmet confirmed — hard hat required in this zone.",
    "🦺 No safety vest confirmed — high-vis vest required.",
]

SNAPSHOT_DIR = pathlib.Path(tempfile.gettempdir()) / "visionai_snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)

EMAIL_ENABLED  = False
EMAIL_FROM     = "your@email.com"
EMAIL_TO       = "recipient@email.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_SMTP     = "smtp.gmail.com"
EMAIL_PORT     = 587

# ─── Session State ─────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "alert_log":        [],
        "crowd_trend":      deque(maxlen=60),
        "session_objects":  Counter(),
        "alert_count":      0,
        "heatmap":          None,
        "heatmap_size":     None,
        "snapshots":        [],
        "dwell_time":       defaultdict(int),
        "conf_scores":      [],
        "loiter_counter":   defaultdict(int),   # track_id → consecutive stationary frames
        "prev_centers":     {},                  # track_id → (cx,cy) for loiter check
        "interaction_log":  [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def log_alert(level, message):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.alert_log.append({"time": ts, "level": level, "message": message})
    if level == "danger":
        st.session_state.alert_count += 1
        if EMAIL_ENABLED:
            _send_email(f"[VisionAI DANGER] {message}")

def _send_email(body):
    try:
        msg = MIMEText(body)
        msg["Subject"] = "VisionAI Danger Alert"
        msg["From"] = EMAIL_FROM; msg["To"] = EMAIL_TO
        with smtplib.SMTP(EMAIL_SMTP, EMAIL_PORT) as s:
            s.starttls(); s.login(EMAIL_FROM, EMAIL_PASSWORD)
            s.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    except Exception:
        pass

def crowd_density_label(n):
    if n <= CROWD_THRESHOLDS["Low"]:    return "Low"
    if n <= CROWD_THRESHOLDS["Medium"]: return "Medium"
    if n <= CROWD_THRESHOLDS["High"]:   return "High"
    return "Critical"

def render_density_badge(label):
    css = {"Low":"density-low","Medium":"density-medium","High":"density-high","Critical":"density-critical"}
    return f'<span class="density-badge {css[label]}">{label}</span>'

def render_metric(val, label, color=""):
    return (f'<div class="metric-card"><div class="metric-val {color}">{val}</div>'
            f'<div class="metric-label">{label}</div></div>')

def get_recommendations(classes):
    seen, recs = set(), []
    for cls in classes:
        if cls not in seen and cls in RECOMMENDATIONS:
            title, level, msg = RECOMMENDATIONS[cls]
            recs.append({"title": title, "level": level, "msg": msg, "cls": cls})
            seen.add(cls)
    order = {"danger": 0, "warn": 1, "info": 2}
    return sorted(recs, key=lambda r: order.get(r["level"], 3))

def render_recommendation(rec):
    css = {"danger": "alert-pill", "warn": "alert-pill-warn", "info": "alert-pill-info"}
    return f'<div class="{css.get(rec["level"],"alert-pill-info")}"><b>{rec["title"]}</b><br>{rec["msg"]}</div>'

def export_alert_csv():
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["time","level","message"])
    writer.writeheader(); writer.writerows(st.session_state.alert_log)
    return buf.getvalue().encode()

def save_snapshot(annotated_rgb, label="danger"):
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SNAPSHOT_DIR / f"snapshot_{label}_{ts}.png"
    Image.fromarray(annotated_rgb).save(path)
    st.session_state.snapshots.append(str(path))
    return path

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def apply_face_blur(frame_bgr, result, blur_strength=31):
    """Blur top-30% (head region) of every 'person' box as privacy approximation."""
    if result.boxes is None: return frame_bgr
    out = frame_bgr.copy()
    for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        if result.names[int(cls)] in ("person", "face"):
            x1, y1, x2, y2 = map(int, box)
            head_y2 = int(y1 + (y2 - y1) * 0.32)
            roi = out[y1:head_y2, x1:x2]
            if roi.size == 0: continue
            out[y1:head_y2, x1:x2] = cv2.GaussianBlur(roi, (blur_strength|1, blur_strength|1), 0)
    return out

def apply_night_mode(frame_bgr):
    """CLAHE on luminance — improves low-light visibility."""
    lab   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

def update_heatmap(frame_shape, result):
    h, w = frame_shape[:2]
    if st.session_state.heatmap is None or st.session_state.heatmap_size != (h,w):
        st.session_state.heatmap      = np.zeros((h, w), dtype=np.float32)
        st.session_state.heatmap_size = (h, w)
    if result.boxes is None: return
    for box in result.boxes.xyxy.cpu().numpy():
        x1,y1,x2,y2 = map(int, box)
        x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(w,x2), min(h,y2)
        st.session_state.heatmap[y1:y2, x1:x2] += 1.0

def render_heatmap(base_rgb=None):
    hm = st.session_state.heatmap
    if hm is None: return None
    hm_norm  = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
    if base_rgb is not None:
        base_bgr = cv2.resize(cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR),
                              (hm.shape[1], hm.shape[0]))
        hm_color = cv2.addWeighted(base_bgr, 0.45, hm_color, 0.55, 0)
    return cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

# ── FIXED Zone drawing ────────────────────────────────────────────────────────
def draw_zone_on_frame(frame_rgb, zone_rect, active=False):
    """
    Draw restricted zone directly on the result image at the correct pixel
    coordinates. zone_rect = (x1_pct, y1_pct, x2_pct, y2_pct) in 0-1 range.
    """
    if zone_rect is None: return frame_rgb
    h, w = frame_rgb.shape[:2]
    # Convert percentages → absolute pixels
    x1 = int(zone_rect[0] * w)
    y1 = int(zone_rect[1] * h)
    x2 = int(zone_rect[2] * w)
    y2 = int(zone_rect[3] * h)

    out     = frame_rgb.copy()
    overlay = out.copy()
    color   = (255, 60, 60) if not active else (255, 80, 20)

    # Semi-transparent fill
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)
    # Solid border (2px)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    # Label inside zone at top-left corner
    label_y = max(y1 + 20, 20)
    cv2.putText(out, "RESTRICTED ZONE", (x1 + 6, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out

def check_zone_alerts(result, zone_rect):
    """Return list of class names whose centre point falls inside zone_rect."""
    if result.boxes is None or zone_rect is None: return []
    h, w   = result.orig_shape[:2]
    zx1 = int(zone_rect[0]*w); zy1 = int(zone_rect[1]*h)
    zx2 = int(zone_rect[2]*w); zy2 = int(zone_rect[3]*h)
    found = []
    for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        bx1,by1,bx2,by2 = map(int, box)
        cx, cy = (bx1+bx2)//2, (by1+by2)//2
        if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
            found.append(result.names[int(cls)])
    return found

# ── Dwell Time Tracker ────────────────────────────────────────────────────────
def update_dwell_time(result):
    """Increment dwell counter per track ID."""
    if result.boxes is None or result.boxes.id is None: return
    for box, tid in zip(result.boxes.xyxy.cpu().numpy(),
                        result.boxes.id.cpu().numpy()):
        st.session_state.dwell_time[int(tid)] += 1

def draw_tracker_ids(frame_bgr, result):
    if result.boxes is None or result.boxes.id is None: return frame_bgr
    for box, tid in zip(result.boxes.xyxy.cpu().numpy(),
                        result.boxes.id.cpu().numpy()):
        x1, y1 = int(box[0]), int(box[1])
        dwell   = st.session_state.dwell_time.get(int(tid), 0)
        cv2.putText(frame_bgr, f"ID:{int(tid)} ({dwell}f)",
                    (x1, max(y1-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,255,200), 2, cv2.LINE_AA)
    return frame_bgr

# ── Loitering Detection (NEW) ─────────────────────────────────────────────────
LOITER_FRAMES    = 60   # flag after this many near-stationary frames
LOITER_DIST_PX   = 30   # max pixel movement to count as "stationary"

def check_loitering(result):
    """
    Returns list of track IDs that have been loitering (barely moving).
    Updates session state counters in place.
    """
    if result.boxes is None or result.boxes.id is None: return []
    loiterers = []
    for box, tid in zip(result.boxes.xyxy.cpu().numpy(),
                        result.boxes.id.cpu().numpy()):
        tid  = int(tid)
        x1,y1,x2,y2 = map(int, box)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        if tid in st.session_state.prev_centers:
            px, py = st.session_state.prev_centers[tid]
            dist   = ((cx-px)**2 + (cy-py)**2)**0.5
            if dist < LOITER_DIST_PX:
                st.session_state.loiter_counter[tid] += 1
            else:
                st.session_state.loiter_counter[tid] = 0
        st.session_state.prev_centers[tid] = (cx, cy)
        if st.session_state.loiter_counter.get(tid, 0) >= LOITER_FRAMES:
            loiterers.append(tid)
    return loiterers

def draw_loiter_warnings(frame_bgr, result, loiterers):
    """Draw orange loiter-warning box around loitering IDs."""
    if not loiterers or result.boxes is None or result.boxes.id is None:
        return frame_bgr
    id_to_box = {int(tid): box for box, tid in
                 zip(result.boxes.xyxy.cpu().numpy(), result.boxes.id.cpu().numpy())}
    for tid in loiterers:
        if tid in id_to_box:
            x1,y1,x2,y2 = map(int, id_to_box[tid])
            cv2.rectangle(frame_bgr, (x1,y1),(x2,y2),(0,140,255), 3)
            cv2.putText(frame_bgr,"LOITERING",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,140,255), 2, cv2.LINE_AA)
    return frame_bgr

# ── Object Interaction Detection (NEW) ────────────────────────────────────────
INTERACT_DIST_PX = 80   # pixel distance threshold for "interaction"

def detect_interactions(result):
    """
    Flag pairs of detected objects whose bounding-box centres are within
    INTERACT_DIST_PX pixels. Returns list of (cls_a, cls_b, distance) tuples.
    """
    if result.boxes is None: return []
    boxes  = result.boxes.xyxy.cpu().numpy()
    clss   = result.boxes.cls.cpu().numpy()
    names  = result.names
    n      = len(boxes)
    pairs  = []
    for i in range(n):
        x1i,y1i,x2i,y2i = map(int, boxes[i])
        cx_i,cy_i = (x1i+x2i)//2, (y1i+y2i)//2
        for j in range(i+1, n):
            x1j,y1j,x2j,y2j = map(int, boxes[j])
            cx_j,cy_j = (x1j+x2j)//2, (y1j+y2j)//2
            dist = ((cx_i-cx_j)**2 + (cy_i-cy_j)**2)**0.5
            if dist <= INTERACT_DIST_PX:
                pairs.append((names[int(clss[i])], names[int(clss[j])], int(dist)))
    return pairs

def draw_interaction_lines(frame_bgr, result, interactions):
    """Draw cyan connecting lines between interacting objects."""
    if not interactions or result.boxes is None: return frame_bgr
    boxes = result.boxes.xyxy.cpu().numpy()
    clss  = result.boxes.cls.cpu().numpy()
    names = result.names
    n     = len(boxes)
    # Rebuild center map
    centers = {}
    for i in range(n):
        x1,y1,x2,y2 = map(int, boxes[i])
        centers[i] = ((x1+x2)//2, (y1+y2)//2)
    # Re-detect pairs to get indices
    for i in range(n):
        for j in range(i+1, n):
            dist = ((centers[i][0]-centers[j][0])**2 +
                    (centers[i][1]-centers[j][1])**2)**0.5
            if dist <= INTERACT_DIST_PX:
                cv2.line(frame_bgr, centers[i], centers[j], (0,255,220), 2)
                mid = ((centers[i][0]+centers[j][0])//2,
                       (centers[i][1]+centers[j][1])//2)
                cv2.putText(frame_bgr, f"{int(dist)}px", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,255,220),1,cv2.LINE_AA)
    return frame_bgr

# ─── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path="yolov8n.pt"):
    return YOLO(path)

# ─── Disabled features (removed from UI) ─────────────────────────────────────
enable_tracking = False
enable_heatmap  = False

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    st.markdown("---")
    conf_threshold = st.slider("Confidence Threshold", 0.10, 1.0, 0.40, 0.05)
    iou_threshold  = st.slider("IoU Threshold (NMS)",  0.10, 1.0, 0.50, 0.05)
    show_labels    = st.toggle("Show Labels",       value=True)
    show_conf      = st.toggle("Show Confidence %", value=True)

    st.markdown("---")
    st.markdown("### 🛡️ Surveillance Modules")
    enable_danger      = st.toggle("Danger Detection",           value=True)
    enable_crowd       = st.toggle("Crowd Density",              value=True)
    enable_ppe         = st.toggle("PPE Safety Check",           value=True)
    enable_reco        = st.toggle("AI Recommendations",         value=True)
    crowd_alert_at     = st.slider("Crowd alert threshold", 5, 50, 10, 1)

    st.markdown("---")
    st.markdown("### ✨ Advanced Features")
    enable_face_blur    = st.toggle("Face Blur (privacy)",        value=False)
    enable_night_mode   = st.toggle("Night Mode (CLAHE)",         value=False)
    enable_snapshot     = st.toggle("Auto Snapshot on Danger",    value=True)
    enable_loiter       = st.toggle("Loitering Detection",        value=True)
    enable_interaction  = st.toggle("Object Interaction Lines",   value=True)

    # ── Zone / ROI — FIXED ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔲 Restricted Zone")
    enable_zone = st.toggle("Enable Zone Alerts", value=False)
    if enable_zone:
        st.markdown('<div class="info-box" style="font-size:.72rem;">Set zone as % of image width/height.<br>0% = left/top edge · 100% = right/bottom edge</div>', unsafe_allow_html=True)
        z1, z2 = st.columns(2)
        zone_x1 = z1.number_input("Left %",   0,  99,  20, 1) / 100
        zone_y1 = z2.number_input("Top %",    0,  99,  20, 1) / 100
        zone_x2 = z1.number_input("Right %",  1, 100,  80, 1) / 100
        zone_y2 = z2.number_input("Bottom %", 1, 100,  80, 1) / 100
        # Clamp so x1<x2, y1<y2
        if zone_x2 <= zone_x1: zone_x2 = min(zone_x1 + 0.1, 1.0)
        if zone_y2 <= zone_y1: zone_y2 = min(zone_y1 + 0.1, 1.0)
        zone_rect = (zone_x1, zone_y1, zone_x2, zone_y2)
        st.markdown(f"<span style='font-size:.7rem;color:#555560;'>Zone: ({int(zone_x1*100)}%,{int(zone_y1*100)}%) → ({int(zone_x2*100)}%,{int(zone_y2*100)}%)</span>",
                    unsafe_allow_html=True)
    else:
        zone_rect = None

    st.markdown("---")
    st.markdown("### 📂 Custom Model")
    model_file = st.file_uploader("Upload .pt weights (optional)", type=["pt"])
    st.markdown("*Leave empty to use YOLOv8n*")

    st.markdown("---")
    st.markdown("### 🚨 Recent Alerts")
    if st.session_state.alert_log:
        for entry in reversed(st.session_state.alert_log[-10:]):
            icon = {"danger":"🔴","warn":"🟡","info":"🔵"}.get(entry["level"],"⚪")
            st.markdown(f"`{entry['time']}` {icon} {entry['message'][:40]}")
        st.download_button("⬇️ Export Alert Log", export_alert_csv(),
                           "alert_log.csv", "text/csv")
    else:
        st.markdown('<div class="info-box">No alerts yet.</div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Session"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        _init(); st.rerun()

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-block">
  <div class="hero-title">Vision<span class="accent">AI</span> <span class="accent-red">Pro</span></div>
  <div class="hero-sub" style="margin-bottom:14px;">Smart Surveillance System</div>
  <div style="font-size:.72rem;color:#444450;letter-spacing:.12em;text-transform:uppercase;line-height:2;">
    Face Blur &nbsp;&middot;&nbsp; Zone Alerts &nbsp;&middot;&nbsp; Snapshots &nbsp;&middot;&nbsp; Night Mode &nbsp;&middot;&nbsp; Loitering Detection &nbsp;&middot;&nbsp; Object Interaction &nbsp;&middot;&nbsp; PPE Check &nbsp;&middot;&nbsp; Crowd Density
  </div>
</div>
""", unsafe_allow_html=True)

kpi1,kpi2,kpi3,kpi4 = st.columns(4)
kpi1.markdown(render_metric(sum(st.session_state.session_objects.values()), "Total Detections","blue"),  unsafe_allow_html=True)
kpi2.markdown(render_metric(st.session_state.alert_count,                  "Danger Alerts",   "red"),   unsafe_allow_html=True)
kpi3.markdown(render_metric(len(st.session_state.snapshots),               "Snapshots Saved", "gold"),  unsafe_allow_html=True)
kpi4.markdown(render_metric(len(st.session_state.dwell_time),              "Tracked IDs",     "purple"),unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
model = None
if model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_file.read()); tmp_path = tmp.name
    try:
        model = load_model(tmp_path)
        st.success(f"✅ Custom model **{model_file.name}** loaded!")
    except Exception as e:
        st.error(f"❌ {e}")
else:
    try:
        model = load_model("yolov8n.pt")
        st.markdown('<div class="info-box">✅ YOLOv8n loaded — 80 COCO classes (person, car, bicycle, dog…)</div>',
                    unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ {e}")

# ─── Shared analysis panel ─────────────────────────────────────────────────────
def run_analysis(result, elapsed, annotated_rgb=None):
    boxes    = result.boxes
    n        = len(boxes) if boxes is not None else 0
    names    = result.names
    classes  = [names[int(c)] for c in boxes.cls] if n > 0 else []
    persons  = classes.count("person")
    st.session_state.session_objects.update(classes)
    if boxes is not None and n > 0:
        st.session_state.conf_scores.extend([float(c)*100 for c in boxes.conf.cpu().numpy()])
    avg_conf = float(boxes.conf.mean())*100 if n > 0 else 0.0

    st.markdown("---")
    mc = st.columns(5)
    mc[0].markdown(render_metric(n,                  "Objects",  ""),       unsafe_allow_html=True)
    mc[1].markdown(render_metric(f"{avg_conf:.1f}%","Avg Conf",  "blue"),   unsafe_allow_html=True)
    mc[2].markdown(render_metric(len(set(classes)),  "Classes",  "gold"),   unsafe_allow_html=True)
    mc[3].markdown(render_metric(f"{elapsed*1000:.0f}ms","Time", ""),       unsafe_allow_html=True)
    mc[4].markdown(render_metric(persons,            "Persons",  "purple"), unsafe_allow_html=True)

    # Crowd density
    if enable_crowd:
        density = crowd_density_label(persons)
        st.markdown(f"#### 👥 Crowd Density &nbsp; {render_density_badge(density)}",
                    unsafe_allow_html=True)
        if persons >= crowd_alert_at:
            msg = f"Crowd threshold: {persons} persons"
            log_alert("warn", msg)
            st.markdown(f'<div class="alert-pill-warn">⚠️ {msg}</div>', unsafe_allow_html=True)
        if density == "Critical":
            log_alert("danger", f"CRITICAL crowd: {persons} persons")

    # Danger
    if enable_danger:
        dangers = [c for c in classes if c in DANGER_CLASSES]
        if dangers:
            st.markdown("#### 🚨 Danger Detected!")
            for dc in set(dangers):
                if dc in RECOMMENDATIONS:
                    title, level, msg = RECOMMENDATIONS[dc]
                    log_alert(level, f"{title}: {msg}")
                    st.markdown(f'<div class="alert-pill"><b>{title}</b><br>{msg}</div>',
                                unsafe_allow_html=True)
            if enable_snapshot and annotated_rgb is not None:
                snap = save_snapshot(annotated_rgb, label="danger")
                st.markdown(f'<div class="alert-pill-green">📸 Snapshot saved: {snap.name}</div>',
                            unsafe_allow_html=True)

    # Zone alerts
    if enable_zone and zone_rect is not None:
        zc = check_zone_alerts(result, zone_rect)
        if zc:
            danger_in_zone = [c for c in zc if c in DANGER_CLASSES]
            msg   = f"Zone alert: {', '.join(set(zc))} in restricted area"
            level = "danger" if danger_in_zone else "warn"
            log_alert(level, msg)
            pill  = "alert-pill" if level == "danger" else "alert-pill-warn"
            st.markdown(f'<div class="{pill}">🔲 {msg}</div>', unsafe_allow_html=True)

    # PPE
    if enable_ppe and persons > 0:
        helmet_cls = {"helmet","hard hat","hardhat"}
        vest_cls   = {"vest","safety vest","jacket"}
        if not any(c in helmet_cls for c in classes):
            st.markdown(f'<div class="alert-pill-warn">{PPE_RECS[0]}</div>', unsafe_allow_html=True)
            log_alert("warn","No helmet on person")
        if not any(c in vest_cls for c in classes):
            st.markdown(f'<div class="alert-pill-warn">{PPE_RECS[1]}</div>', unsafe_allow_html=True)

    # Interactions
    if enable_interaction:
        pairs = detect_interactions(result)
        if pairs:
            st.markdown("#### 🔗 Object Interactions")
            for cls_a, cls_b, dist in pairs[:8]:
                tag = "alert-pill" if (cls_a in DANGER_CLASSES or cls_b in DANGER_CLASSES) else "alert-pill-info"
                note = " ⚠️ Dangerous combo!" if (cls_a in DANGER_CLASSES or cls_b in DANGER_CLASSES) else ""
                st.markdown(f'<div class="{tag}">**{cls_a}** ↔ **{cls_b}** &nbsp;·&nbsp; {dist}px{note}</div>',
                            unsafe_allow_html=True)
                if tag == "alert-pill":
                    log_alert("danger", f"Interaction: {cls_a} + {cls_b}")

    # AI Recommendations
    if enable_reco and classes:
        recs = get_recommendations(classes)
        if recs:
            st.markdown("#### 🤖 AI Recommendations")
            for rec in recs[:6]:
                st.markdown(render_recommendation(rec), unsafe_allow_html=True)

    # Class breakdown
    if classes:
        st.markdown("#### 🏷️ Detection Breakdown")
        for cls, cnt in Counter(classes).most_common():
            conf_vals = [float(boxes.conf[i])*100 for i,c in enumerate(boxes.cls)
                         if names[int(c)] == cls]
            avg   = sum(conf_vals)/len(conf_vals) if conf_vals else 0
            color = "#ff4c6a" if cls in DANGER_CLASSES else "#69ff47"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;margin:7px 0;">
              <span style="font-size:.75rem;color:#aaa;min-width:130px;">{cls}</span>
              <div style="flex:1;background:#1a1a1e;border-radius:3px;height:5px;">
                <div style="width:{int(avg)}%;background:{color};height:5px;border-radius:3px;"></div>
              </div>
              <span style="font-size:.72rem;color:{color};">×{cnt} · {avg:.0f}%</span>
            </div>""", unsafe_allow_html=True)

    # Heatmap (image/webcam)
    if enable_heatmap and st.session_state.heatmap is not None:
        hm_img = render_heatmap(annotated_rgb)
        if hm_img is not None:
            st.markdown("#### 🌡️ Cumulative Detection Heatmap")
            st.image(hm_img, use_container_width=True,
                     caption="Red = most detections | Blue = fewest")

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab_img, tab_vid, tab_cam, tab_dash, tab_snap = st.tabs(
    ["📷  Image","🎥  Video","📸  Webcam","📊  Dashboard","🖼️  Snapshots"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_img:
    uploaded_img = st.file_uploader("Upload an image",
                                    type=["jpg","jpeg","png","bmp","webp"],
                                    label_visibility="collapsed")
    if uploaded_img and model:
        image     = Image.open(uploaded_img).convert("RGB")
        img_array = np.array(image)

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown("**📥 Original**")
            st.image(image, use_container_width=True)

        with st.spinner("🔍 Running detection…"):
            t0      = time.time()
            results = model.predict(img_array, conf=conf_threshold,
                                    iou=iou_threshold, verbose=False)
            elapsed = time.time() - t0

        result  = results[0]
        ann_bgr = result.plot(labels=show_labels, conf=show_conf)

        if enable_face_blur:
            ann_bgr = apply_face_blur(ann_bgr, result)

        # Interaction lines
        if enable_interaction:
            interactions = detect_interactions(result)
            ann_bgr = draw_interaction_lines(ann_bgr, result, interactions)

        ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

        # Draw zone on top (FIXED — uses actual image dimensions)
        if enable_zone and zone_rect:
            ann_rgb = draw_zone_on_frame(ann_rgb, zone_rect)

        update_heatmap(img_array.shape, result)

        with col2:
            st.markdown("**🎯 Detections**")
            st.image(ann_rgb, use_container_width=True)

        run_analysis(result, elapsed, ann_rgb)

        buf = io.BytesIO()
        Image.fromarray(ann_rgb).save(buf, format="PNG")
        st.download_button("⬇️ Download Annotated Image",
                           buf.getvalue(), "visionai_result.png", "image/png")

    elif uploaded_img and not model:
        st.warning("⚠️ Model not loaded.")
    else:
        st.markdown('<div class="info-box">⬆️ Upload an image to start analysis.</div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO  (Windows-safe + annotated export)
# ══════════════════════════════════════════════════════════════════════════════
with tab_vid:
    uploaded_vid = st.file_uploader("Upload a video",
                                    type=["mp4","avi","mov","mkv"],
                                    label_visibility="collapsed")
    vc1, vc2 = st.columns(2)
    max_frames         = vc1.slider("Max frames",              30, 500, 150, 10)
    skip_frames        = vc2.slider("Process every N frames",   1,   5,   1)
    enable_video_export = st.toggle("Export annotated video (mp4)", value=False)

    if uploaded_vid and model:
        ext      = os.path.splitext(uploaded_vid.name)[-1] or ".mp4"
        tmp_path = os.path.join(tempfile.gettempdir(), f"visionai_{int(time.time())}{ext}")
        with open(tmp_path, "wb") as f:
            f.write(uploaded_vid.read())

        cap   = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        limit = min(max_frames, total)
        w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        st.info(f"📹 {total} frames · {fps:.1f} FPS · Processing {limit} (every {skip_frames})")

        out_path     = None
        video_writer = None
        if enable_video_export:
            out_path     = os.path.join(tempfile.gettempdir(), f"visionai_out_{int(time.time())}.mp4")
            video_writer = cv2.VideoWriter(out_path,
                                           cv2.VideoWriter_fourcc(*"mp4v"),
                                           fps, (w_vid, h_vid))

        vf, vs  = st.columns([3,1], gap="medium")
        stframe = vf.empty()
        prog    = st.progress(0)
        status  = st.empty()
        with vs:
            st.markdown("**Live**")
            vfps_ph  = st.empty(); vobj_ph = st.empty()
            vperson  = st.empty(); vdens   = st.empty()
            vdanger  = st.empty(); vloiter = st.empty()

        all_classes=[]; fc=0; raw_fc=0; t0=time.time(); danger_frames=0
        last_rgb = None

        while cap.isOpened() and fc < limit:
            ret, frame = cap.read()
            if not ret: break
            raw_fc += 1
            if raw_fc % skip_frames != 0: continue

            if enable_night_mode:
                frame = apply_night_mode(frame)

            pk = {"conf": conf_threshold, "iou": iou_threshold, "verbose": False}
            if enable_tracking:
                pk["persist"] = True
                results = model.track(frame, **pk)
            else:
                results = model.predict(frame, **pk)
            result = results[0]

            ann_bgr = result.plot(labels=show_labels, conf=show_conf)

            # Tracking
            if enable_tracking:
                update_dwell_time(result)
                ann_bgr = draw_tracker_ids(ann_bgr, result)

            # Face blur
            if enable_face_blur:
                ann_bgr = apply_face_blur(ann_bgr, result)

            # Interactions
            if enable_interaction:
                interactions = detect_interactions(result)
                ann_bgr = draw_interaction_lines(ann_bgr, result, interactions)

            # Loitering
            loiterers = []
            if enable_loiter and enable_tracking:
                loiterers = check_loitering(result)
                if loiterers:
                    ann_bgr = draw_loiter_warnings(ann_bgr, result, loiterers)

            update_heatmap(frame.shape, result)
            ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

            # Zone overlay (FIXED)
            if enable_zone and zone_rect:
                zone_classes = check_zone_alerts(result, zone_rect)
                ann_rgb = draw_zone_on_frame(ann_rgb, zone_rect, active=bool(zone_classes))

            last_rgb = ann_rgb
            if video_writer:
                video_writer.write(cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR))

            names   = result.names
            boxes   = result.boxes
            n_f     = len(boxes) if boxes else 0
            classes = [names[int(c)] for c in boxes.cls] if n_f > 0 else []
            persons = classes.count("person")
            dangers = [c for c in classes if c in DANGER_CLASSES]

            if dangers:
                danger_frames += 1
                for d in set(dangers): log_alert("danger", f"Video f{fc+1}: {d}")
                if enable_snapshot: save_snapshot(ann_rgb, label="danger")

            if enable_zone and zone_rect:
                zc = check_zone_alerts(result, zone_rect)
                if zc: log_alert("warn", f"Zone: {','.join(set(zc))}")

            if loiterers:
                for lid in loiterers:
                    log_alert("warn", f"Loitering: ID {lid} stationary >{LOITER_FRAMES} frames")

            all_classes += classes
            st.session_state.crowd_trend.append(persons)
            st.session_state.session_objects.update(classes)

            proc_fps = fc / max(time.time()-t0, 0.001)
            stframe.image(ann_rgb, use_container_width=True, caption=f"Frame {fc+1}/{limit}")
            prog.progress((fc+1)/limit)
            status.markdown(f"`{fc+1}/{limit} · dets:{n_f}`")

            vfps_ph.markdown(render_metric(f"{proc_fps:.1f}","Proc FPS","gold"),   unsafe_allow_html=True)
            vobj_ph.markdown(render_metric(n_f,              "This Frame",""),     unsafe_allow_html=True)
            vperson.markdown(render_metric(persons,          "Persons","purple"),  unsafe_allow_html=True)
            if enable_crowd:
                dl = crowd_density_label(persons)
                vdens.markdown(
                    f'<div class="metric-card"><div style="font-size:.65rem;color:#444450;text-transform:uppercase;letter-spacing:.15em;margin-bottom:6px;">Density</div>'
                    f'{render_density_badge(dl)}</div>', unsafe_allow_html=True)
            if dangers:
                vdanger.markdown(f'<div class="alert-pill">🚨 {", ".join(set(dangers))}</div>', unsafe_allow_html=True)
            else:
                vdanger.empty()
            if loiterers:
                vloiter.markdown(f'<div class="alert-pill-warn">🚶 Loitering: IDs {loiterers}</div>', unsafe_allow_html=True)
            else:
                vloiter.empty()
            fc += 1

        cap.release()
        if video_writer: video_writer.release()
        time.sleep(0.5)
        try: os.remove(tmp_path)
        except Exception: pass

        prog.empty(); status.empty()
        st.success(f"✅ Done — {fc} frames in {time.time()-t0:.1f}s · Danger frames: {danger_frames}")

        if enable_video_export and out_path and os.path.exists(out_path):
            with open(out_path,"rb") as f:
                st.download_button("⬇️ Download Annotated Video",
                                   f.read(), "visionai_annotated.mp4","video/mp4")

        if enable_heatmap and last_rgb is not None:
            hm = render_heatmap(last_rgb)
            if hm is not None:
                st.markdown("#### 🌡️ Session Detection Heatmap")
                st.image(hm, use_container_width=True)

        if all_classes:
            st.markdown("#### 📊 Video Summary")
            counts = Counter(all_classes); total_det = sum(counts.values())
            for cls, cnt in counts.most_common():
                pct   = int(cnt/total_det*100)
                color = "#ff4c6a" if cls in DANGER_CLASSES else "#69ff47"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;margin:6px 0;">
                  <span style="font-size:.75rem;color:#aaa;min-width:110px;">{cls}</span>
                  <div style="flex:1;background:#1a1a1e;border-radius:3px;height:5px;">
                    <div style="width:{pct}%;background:{color};height:5px;border-radius:3px;"></div>
                  </div>
                  <span style="font-size:.72rem;color:{color};">{cnt} · {pct}%</span>
                </div>""", unsafe_allow_html=True)

    elif uploaded_vid and not model:
        st.warning("⚠️ Model not loaded.")
    else:
        st.markdown('<div class="info-box">⬆️ Upload a video for full surveillance analysis.</div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WEBCAM
# ══════════════════════════════════════════════════════════════════════════════
with tab_cam:
    st.markdown('<div class="info-box">📸 Take a photo — full surveillance analysis runs instantly.</div>',
                unsafe_allow_html=True)
    if model:
        cam_col, det_col = st.columns(2, gap="medium")
        with cam_col:
            st.markdown("**📷 Camera**")
            frame_data = st.camera_input(" ", label_visibility="collapsed")
        with det_col:
            st.markdown("**🎯 Detection**")
            det_ph = st.empty()

        if frame_data is not None:
            image     = Image.open(frame_data).convert("RGB")
            img_array = np.array(image)

            if enable_night_mode:
                img_bgr   = apply_night_mode(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                img_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            t0      = time.time()
            results = model.predict(img_array, conf=conf_threshold,
                                    iou=iou_threshold, verbose=False)
            elapsed = time.time() - t0
            result  = results[0]
            ann_bgr = result.plot(labels=show_labels, conf=show_conf)

            if enable_face_blur:
                ann_bgr = apply_face_blur(ann_bgr, result)
            if enable_interaction:
                interactions = detect_interactions(result)
                ann_bgr = draw_interaction_lines(ann_bgr, result, interactions)

            ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

            if enable_zone and zone_rect:
                ann_rgb = draw_zone_on_frame(ann_rgb, zone_rect)

            update_heatmap(img_array.shape, result)
            det_ph.image(ann_rgb, use_container_width=True)
            run_analysis(result, elapsed, ann_rgb)
        else:
            det_ph.markdown(
                '<div class="info-box" style="margin-top:40px;text-align:center;">👆 Take a photo to start detection</div>',
                unsafe_allow_html=True)
    else:
        st.warning("⚠️ Model not loaded.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown("### 📊 Session Dashboard")
    d1,d2,d3,d4 = st.columns(4)
    d1.markdown(render_metric(sum(st.session_state.session_objects.values()),"Total Detections","blue"),  unsafe_allow_html=True)
    d2.markdown(render_metric(len(st.session_state.session_objects),         "Unique Classes",  "gold"),  unsafe_allow_html=True)
    d3.markdown(render_metric(st.session_state.alert_count,                  "Danger Alerts",   "red"),   unsafe_allow_html=True)
    d4.markdown(render_metric(len(st.session_state.dwell_time),              "Tracked IDs",     "purple"),unsafe_allow_html=True)
    st.markdown("---")

    import pandas as pd
    if st.session_state.session_objects:
        st.markdown("#### Top Detected Classes")
        top = st.session_state.session_objects.most_common(15)
        st.bar_chart(pd.DataFrame({"count":[v for _,v in top]}, index=[k for k,_ in top]))

    if len(st.session_state.crowd_trend) > 1:
        st.markdown("#### Person Count Trend")
        st.line_chart(pd.DataFrame({"persons": list(st.session_state.crowd_trend)}))

    if st.session_state.conf_scores:
        st.markdown("#### Confidence Score Distribution")
        bins = list(range(0, 110, 10))
        hist, _ = np.histogram(st.session_state.conf_scores, bins=bins)
        st.bar_chart(pd.DataFrame({"detections": hist},
                                  index=[f"{b}-{b+10}%" for b in bins[:-1]]))

    if st.session_state.dwell_time:
        st.markdown("#### Top Dwell Times (frames per tracked ID)")
        top_dwell = sorted(st.session_state.dwell_time.items(), key=lambda x:-x[1])[:10]
        st.bar_chart(pd.DataFrame({"frames":[v for _,v in top_dwell]},
                                  index=[f"ID:{k}" for k,_ in top_dwell]))

    st.markdown("---")
    st.markdown("#### Full Alert Log")
    if st.session_state.alert_log:
        st.dataframe(pd.DataFrame(st.session_state.alert_log),
                     use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export Alert Log", export_alert_csv(),
                           "alert_log.csv","text/csv")
    else:
        st.markdown('<div class="info-box">No events yet.</div>', unsafe_allow_html=True)

    danger_det = {k:v for k,v in st.session_state.session_objects.items() if k in DANGER_CLASSES}
    if danger_det:
        st.markdown("#### 🚨 Danger Class Summary")
        for cls, cnt in sorted(danger_det.items(), key=lambda x:-x[1]):
            st.markdown(f'<div class="alert-pill"><b>{cls}</b> — detected {cnt}×</div>',
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SNAPSHOTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_snap:
    st.markdown("### 🖼️ Auto-Saved Snapshots")
    st.markdown('<div class="info-box">Snapshots are auto-saved when a danger class is detected (enable in sidebar).</div>',
                unsafe_allow_html=True)
    snaps = st.session_state.snapshots
    if snaps:
        cols = st.columns(3)
        for i, path in enumerate(reversed(snaps[-12:])):
            p = pathlib.Path(path)
            if p.exists():
                img = Image.open(p)
                cols[i%3].image(img, caption=p.name, use_container_width=True)
                with open(p,"rb") as f:
                    cols[i%3].download_button(f"⬇️ {p.name[:20]}", f.read(),
                                              p.name, "image/png", key=f"snap_{i}")
    else:
        st.markdown('<div class="info-box">No snapshots yet. Enable "Auto Snapshot on Danger" and run detection.</div>',
                    unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:.68rem;color:#222228;padding:10px 0;">'
    'VisionAI Pro · Smart Surveillance System · YOLOv8n · OpenCV · Streamlit</div>',
    unsafe_allow_html=True)