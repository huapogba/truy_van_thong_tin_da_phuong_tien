import os
from pathlib import Path
import torch
import clip
import open_clip
import faiss
import numpy as np
import json
import re
import requests

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any, Dict, List

from googletrans import Translator
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# ============================
# CACHE (PERFORMANCE)
# ============================
TRANSLATE_CACHE: Dict[str, str] = {}
TEXT_EMB_CACHE: Dict[str, np.ndarray] = {}

# ============================
# FASTAPI CONFIG
# ============================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FRAME_DIR = DATA_DIR / "frames"
FAISS_DIR = DATA_DIR / "faiss"

app.mount("/frames", StaticFiles(directory=FRAME_DIR), name="frames")
translator = Translator()

# ============================
# LOAD MODELS
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# -------- ViT-B/16 (OpenAI CLIP) --------
#clip_b16, _ = clip.load("ViT-B/16", device=device, jit=False)
#clip_b16.eval()

# -------- ViT-L/14 (OpenCLIP) --------
model_L14, _, preprocess_L14 = open_clip.create_model_and_transforms(
    "ViT-L-14-quickgelu",
    pretrained="openai",
    device=device
)
tokenizer_L14 = open_clip.get_tokenizer("ViT-L-14")
model_L14.eval()

# ============================
# LOAD FAISS + METADATA
# ============================
index_frame = faiss.read_index(str(FAISS_DIR / "frames_ivf_L14.index"))
index_frame.nprobe = 16  # ⭐ QUAN TRỌNG: tăng tốc + ổn định

#index_frame2 = faiss.read_index(str(FAISS_DIR / "frames.index"))
index_L14 = faiss.read_index(str(FAISS_DIR / "frames_clipL14.index"))

with open(DATA_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("[INFO] Loaded FAISS indices and metadata")

# ============================
# API MODELS
# ============================
class QueryText(BaseModel):
    queries: List[str]

class SimilarQuery(BaseModel):
    frame_name: str
    model: str

class SubmitPayload(BaseModel):
    videoId: str
    timestamp: str
    question: Optional[str] = None

class SubmitVideoPayload(BaseModel):
    videoId: str
    timestamp_ms: int          # ✅ milliseconds từ YouTube
    question: Optional[str] = None


# ============================
# DRES CONFIG
# ============================
DRES_BASE_URL = "http://192.168.20.156:5601"
SESSION_ID = "_M0uOfhUacUfVWNSAYo2omm_GU1-YKH3"
USERNAME: Optional[str] = "team008"
PASSWORD: Optional[str] = "1234556"
DEFAULT_FPS = 2

# ============================
# UTILS
# ============================
def get_session_id() -> str:
    return SESSION_ID

def get_active_evaluation_id(session_id: str) -> str:
    print(f"\n{'='*60}")
    print(f"[GET EVALUATION] Requesting evaluation list...")
    print(f"[GET EVALUATION] URL: {DRES_BASE_URL}/api/v2/client/evaluation/list")
    print(f"[GET EVALUATION] Session: {session_id}")
    
    resp = requests.get(
        f"{DRES_BASE_URL}/api/v2/client/evaluation/list",
        params={"session": session_id},
        timeout=10
    )
    
    print(f"[GET EVALUATION] Status: {resp.status_code}")
    print(f"[GET EVALUATION] Raw Response ({len(resp.text)} chars): {resp.text[:500]}")
    
    resp.raise_for_status()
    
    evaluations = resp.json()
    print(f"[GET EVALUATION] Total evaluations: {len(evaluations)}")
    
    active = next(
        (e for e in evaluations if str(e.get("status")).upper() == "ACTIVE"),
        None
    )
    if not active:
        print(f"[GET EVALUATION] ❌ ERROR: No active evaluation found!")
        print(f"[GET EVALUATION] Available evaluations: {[e.get('status') for e in evaluations]}")
        raise RuntimeError("No active evaluation found")
    
    eval_id = str(active.get("id"))
    print(f"[GET EVALUATION] ✅ Active Evaluation ID: {eval_id}")
    print(f"{'='*60}\n")
    return eval_id

def ms_from_frame_index(frame_value: Any, fps: float = DEFAULT_FPS) -> int:
    frame_index = int(frame_value)
    # ⚠️ Metadata dùng: timestamp = (frame_number - 1) / fps
    # Vì frame đánh số từ 1, không phải 0
    ms = int(((frame_index - 1) / fps) * 1000)
    print(f"[TIMESTAMP] frame_index={frame_index}, fps={fps} → ms={ms}")
    return ms

def submit_result_to_dres(result: Dict[str, Any], question: Optional[str] = None):
    session_id = get_session_id()
    evaluation_id = get_active_evaluation_id(session_id)

    ms = ms_from_frame_index(result["timestamp"])
    video_id = str(result["videoId"])

    if question:
        body = {"answerSets": [{"answers": [{"text": f"QA-{question}-{video_id}-{ms}"}]}]}
    else:
        body = {"answerSets": [{"answers": [{"mediaItemName": video_id, "start": ms, "end": ms}]}]}

    print(f"\n{'='*60}")
    print(f"[DRES SUBMIT] Submitting result...")
    print(f"[DRES SUBMIT] URL: {DRES_BASE_URL}/api/v2/submit/{evaluation_id}")
    print(f"[DRES SUBMIT] Session: {session_id}")
    print(f"[DRES SUBMIT] Video: {video_id}, Time: {ms}ms")
    print(f"[DRES SUBMIT] Question: {question}")
    print(f"[DRES SUBMIT] Body: {json.dumps(body, indent=2)}")
    
    resp = requests.post(
        f"{DRES_BASE_URL}/api/v2/submit/{evaluation_id}",
        params={"session": session_id},
        json=body,
        timeout=15
    )
    
    print(f"[DRES SUBMIT] Response Status: {resp.status_code}")
    print(f"[DRES SUBMIT] Response Headers: {dict(resp.headers)}")
    print(f"[DRES SUBMIT] Raw Response ({len(resp.text)} chars): {resp.text[:1000]}")
    
    resp.raise_for_status()
    
    result_data = resp.json()
    print(f"[DRES SUBMIT] ✅ SUCCESS: {result_data}")
    print(f"{'='*60}\n")
    return result_data
    
def submit_video_time_to_dres(video_id: str, ms: int, question: Optional[str] = None):
    session_id = get_session_id()
    evaluation_id = get_active_evaluation_id(session_id)
    
    if question:
        body = {
            "answerSets": [{
                "answers": [{
                    "text": f"QA-{question}-{video_id}-{ms}"
                }]
            }]
        }
    else:
        body = {
            "answerSets": [{
                "answers": [{
                    "mediaItemName": video_id,
                    "start": ms,
                    "end": ms
                }]
            }]
        }

    print(f"\n{'='*60}")
    print(f"[DRES VIDEO SUBMIT] Submitting video time...")
    print(f"[DRES VIDEO SUBMIT] URL: {DRES_BASE_URL}/api/v2/submit/{evaluation_id}")
    print(f"[DRES VIDEO SUBMIT] Session: {session_id}")
    print(f"[DRES VIDEO SUBMIT] Video: {video_id}, Time: {ms}ms")
    print(f"[DRES VIDEO SUBMIT] Question: {question}")
    print(f"[DRES VIDEO SUBMIT] Body: {json.dumps(body, indent=2)}")
    
    resp = requests.post(
        f"{DRES_BASE_URL}/api/v2/submit/{evaluation_id}",
        params={"session": session_id},
        json=body,
        timeout=15
    )
    
    print(f"[DRES VIDEO SUBMIT] Response Status: {resp.status_code}")
    print(f"[DRES VIDEO SUBMIT] Response Headers: {dict(resp.headers)}")
    print(f"[DRES VIDEO SUBMIT] Raw Response ({len(resp.text)} chars): {resp.text[:1000]}")
    
    resp.raise_for_status()
    
    result_data = resp.json()
    print(f"[DRES VIDEO SUBMIT] ✅ SUCCESS: {result_data}")
    print(f"{'='*60}\n")
    return result_data

def get_team_id(session_id: str) -> str:
    print(f"\n{'='*60}")
    print(f"[GET TEAM] Requesting team list...")
    print(f"[GET TEAM] URL: {DRES_BASE_URL}/api/v2/client/team/list")
    print(f"[GET TEAM] Session: {session_id}")
    
    resp = requests.get(
        f"{DRES_BASE_URL}/api/v2/client/team/list",
        params={"session": session_id},
        timeout=10
    )
    
    print(f"[GET TEAM] Status: {resp.status_code}")
    print(f"[GET TEAM] Raw Response ({len(resp.text)} chars): {resp.text[:500]}")
    
    resp.raise_for_status()

    teams = resp.json()
    print(f"[GET TEAM] Total teams: {len(teams)}")
    
    if not teams:
        print(f"[GET TEAM] ❌ ERROR: No team found for this session!")
        raise RuntimeError("No team found for this session")

    team_id = str(teams[0]["id"])
    team_name = teams[0].get("name", "Unknown")
    print(f"[GET TEAM] ✅ Team ID: {team_id} ({team_name})")
    print(f"{'='*60}\n")
    return team_id


# ============================
# TEXT PROCESSING (CACHED)
# ============================
def translate_keep_quotes(text: str) -> str:
    if text in TRANSLATE_CACHE:
        return TRANSLATE_CACHE[text]

    parts = re.split(r'(".*?")', text)
    output = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith('"') and part.endswith('"'):
            output.append(part[1:-1])
        else:
            try:
                if detect(part) == "vi":
                    part = translator.translate(part, dest="en").text
            except:
                pass
            output.append(part)

    final = " ".join(output)
    TRANSLATE_CACHE[text] = final
    return final

def fuse_text_embeddings(texts: List[str]) -> np.ndarray:
    key = "||".join(texts)
    if key in TEXT_EMB_CACHE:
        return TEXT_EMB_CACHE[key]

    tokens = tokenizer_L14(texts).to(device)
    with torch.no_grad():
        feats = model_L14.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        fused = feats.mean(dim=0, keepdim=True)
        fused = fused / fused.norm(dim=-1, keepdim=True)

    out = fused.cpu().numpy().astype("float32")
    TEXT_EMB_CACHE[key] = out
    return out

# ============================
# SEARCH TEXT
# ============================
@app.post("/search_text")
async def search_text(data: QueryText):
    queries = [q.strip() for q in data.queries if q.strip()]
    if not queries:
        return {"error": "Empty query list"}

    # ✅ THÊM Ở ĐÂY
    #for q in queries:
     #   print(f"[QUERY] {q}")

    translated = [translate_keep_quotes(q) for q in queries]

    with torch.no_grad():
        # ----- L14 -----
        fused_vec = fuse_text_embeddings(translated)

        _, ids_ivf = index_frame.search(fused_vec, 50)
        results_b16 = [
            {"video": metadata[i]["video"], "frame_name": metadata[i]["path"], "timestamp": metadata[i]["timestamp"]}
            for i in ids_ivf[0] if 0 <= i < len(metadata)
        ]

        _, ids_l14 = index_L14.search(fused_vec, 50)
        results_L14 = [
            {"video": metadata[i]["video"], "frame_name": metadata[i]["path"], "timestamp": metadata[i]["timestamp"]}
            for i in ids_l14[0] if 0 <= i < len(metadata)
        ]

        # ----- B16 -----
       # tokens_b16 = clip.tokenize(translated).to(device)
       # emb = clip_b16.encode_text(tokens_b16)
       # emb = emb / emb.norm(dim=-1, keepdim=True)

        #fused_b16 = emb.mean(dim=0, keepdim=True)
        #fused_b16 = fused_b16 / fused_b16.norm(dim=-1, keepdim=True)

        #_, ids_b16 = index_frame2.search(
        #    fused_b16.cpu().numpy().astype("float32"), 50
        #)
        results_b32 = [
            #{"video": metadata[i]["video"], "frame_name": metadata[i]["path"], "timestamp": metadata[i]["timestamp"]}
            #for i in ids_b16[0] if 0 <= i < len(metadata)
        ]

    return {
        "frame_results_b16": results_b16,
        "frame_results_b32": results_b32,
        "frame_results_L14": results_L14
    }

# ============================
# SEARCH SIMILAR
# ============================
@app.post("/search_similar")
async def search_similar(data: SimilarQuery):
    index = {
        "b16": index_frame,
        #"b32": index_frame2,
        "l14": index_L14
    }.get(data.model.lower())

    if not index:
        return {"similar_frames": []}

    idx = next(
        (i for i, m in enumerate(metadata) if m["path"] == data.frame_name),
        None
    )

    if idx is None:
        return {"similar_frames": []}

    vec = index.reconstruct(idx).reshape(1, -1)
    _, I = index.search(vec, 20)

    return {
        "similar_frames": [
            {
                "video": metadata[i]["video"],
                "frame_name": metadata[i]["path"],
                "timestamp": metadata[i]["timestamp"]
            }
            for i in I[0]
            if 0 <= i < len(metadata)
        ]
    }

#============================
# GET NEIGHBOR FRAMES
# ============================
@app.post("/get_neighbor_frames")
async def get_neighbor_frames(req: Dict[str, Any]):
    video = req.get("video")
    frame_path = req.get("frame_name")
    r = int(req.get("range", 20))

    if not video or not frame_path:
        return {"frames": []}

    # Lấy index của frame hiện tại
    m = re.search(r"frame_(\d+)\.jpg$", frame_path)
    if not m:
        return {"frames": []}

    cur_idx = int(m.group(1))
    frames = []

    for item in metadata:
        if item.get("video") != video:
            continue

        path = item.get("path", "")
        m2 = re.search(r"frame_(\d+)\.jpg$", path)
        if not m2:
            continue

        idx = int(m2.group(1))
        if abs(idx - cur_idx) <= r:
            frames.append({
                "video": video,
                "frame_name": path,
                "timestamp": item.get("timestamp"),
                "frame_index": idx
            })

    frames.sort(key=lambda x: x["frame_index"])
    return {"frames": frames}


# ============================
# SUBMIT RESULT
# ============================
@app.post("/submit_result")
async def submit_result(payload: SubmitPayload):
    try:
        data = submit_result_to_dres(
            {"videoId": payload.videoId, "timestamp": payload.timestamp},
            payload.question
        )
        return {"status": "ok", "data": data}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

@app.post("/submit_video")
async def submit_video(payload: SubmitVideoPayload):
    try:
        data = submit_video_time_to_dres(
            video_id=payload.videoId,
            ms=payload.timestamp_ms,
            question=payload.question
        )
        return {"status": "ok", "data": data}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )


# ============================
# RUN
# ============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
