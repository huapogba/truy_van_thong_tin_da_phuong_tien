import faiss
import json
import torch
import clip
from googletrans import Translator
import re
import math
from collections import defaultdict
import threading

# ======================
# DEVICE
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# LOAD CLIP MODEL
# ======================
model, _ = clip.load("ViT-L/14", device=device)
model.eval()

# ======================
# LOAD DATA
# ======================
index = faiss.read_index("data/bin/all_videos.bin")

with open("data/bin/all_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

with open("data/bin/ocr.json", "r", encoding="utf-8") as f:
    ocr_metadata = json.load(f)

with open("data/bin/audio.json", "r", encoding="utf-8") as f:
    audio_metadata = json.load(f)

translator = Translator()

# ======================
# CONFIG PAGINATION
# ======================
PAGE_SIZE = 50
MAX_PAGES = 40


# ======================
# PAGINATION (FIXED)
# ======================
def paginate(results, page=1, page_size=50, max_pages=40):

    total = len(results)

    total_pages = math.ceil(total / page_size)

    if total_pages == 0:
        total_pages = 1

    total_pages = min(total_pages, max_pages)

    page = max(1, min(page, total_pages))

    start = (page - 1) * page_size
    end = start + page_size

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "results": results[start:end]
    }

# ======================
# TRANSLATE + CLIP
# ======================
translate_cache = {}

def translate_vi_to_en(text):
    if text in translate_cache:
        return translate_cache[text]

    try:
        en = translator.translate(text, src='vi', dest='en').text
    except:
        en = text

    translate_cache[text] = en
    return en

def encode_text(text):
    text_en = translate_vi_to_en(text)

    with torch.no_grad():
        token = clip.tokenize([text_en]).to(device)
        feat = model.encode_text(token)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.cpu().numpy().astype("float32")


# ======================
# CLIP SEARCH
# ======================
def search_clip(query, k=216):

    qvec = encode_text(query)
    D, I = index.search(qvec, k)

    results = []

    for idx in I[0]:
        if idx < len(metadata):
            item = metadata[idx]
            #if "frame" in item:
            results.append(item)

    return results


def search_clip_api(query, page=1, page_size=50):

    results = search_clip(query, k=216)
    paged = paginate(results, page, page_size)

    output = []

    for item in paged["results"]:
        output.append({
            "frame": "/" + item["frame"].replace("\\", "/"),
            "timestamp": item.get("timestamp"),
            "video": item.get("video")
        })

    return {
        "query": query,
        "page": paged["page"],
        "page_size": paged["page_size"],
        "total": paged["total"],
        "total_pages": paged["total_pages"],
        "results": output
    }

# ======================
# FRAME LÂN CẬN 
# ======================
video_timeline = defaultdict(list)

for item in metadata:
    if "video" in item and "timestamp" in item:
        video_timeline[item["video"]].append(item)

for v in video_timeline:
    video_timeline[v].sort(key=lambda x: x["timestamp"])

def get_context(video, timestamp, window=20):

    timeline = video_timeline.get(video, [])
    if not timeline:
        return []

    pos = min(
        range(len(timeline)),
        key=lambda i: abs(timeline[i]["timestamp"] - timestamp)
    )

    start = max(0, pos - window)
    end = min(len(timeline), pos + window + 1)

    return timeline[start:end]

def get_clip_context_api(video, timestamp, window=20):

    if video not in video_timeline:
        return {
            "video": video,
            "center_timestamp": timestamp,
            "frames": []
        }

    if timestamp is None:
        return {
            "video": video,
            "center_timestamp": timestamp,
            "frames": []
        }

    frames = get_context(video, timestamp, window)

    return {
        "video": video,
        "center_timestamp": timestamp,
        "frames": [
            {
                "frame": "/" + f["frame"].replace("\\", "/"),
                "timestamp": f.get("timestamp")
            }
            for f in frames
        ]
    }
# ======================
# OCR INDEX
# ======================
class HashOCRIndex:

    def __init__(self):
        self.index = defaultdict(set)
        self.meta = {}

    def clean(self, text):
        text = text.lower()
        text = re.sub(r"[^\wÀ-ỹ\s]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def add(self, item):
        frame = item["frame"]
        self.meta[frame] = item

        words = self.clean(item.get("ocr", "")).split()

        for w in words:
            self.index[w].add(frame)

    def build(self, data):
        for item in data:
            self.add(item)

    def search(self, query, k=2000):

        words = self.clean(query).split()
        if not words:
            return []

        scores = defaultdict(int)

        for w in words:
            for frame in self.index.get(w, []):
                scores[frame] += 1

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []

        for frame, _ in ranked[:k]:
            if frame in self.meta:
                results.append(self.meta[frame])

        return results


# build OCR index
hash_index = HashOCRIndex()
hash_index.build(ocr_metadata)


def search_ocr(query, page=1, page_size=50):

    results = hash_index.search(query, k=200)
    return paginate(results, page, page_size)


def search_ocr_api(query, page=1, page_size=50):

    paged = search_ocr(query, page, page_size)

    output = []

    for item in paged["results"]:
        output.append({
            "frame": "/" + item["frame"].replace("\\", "/"),
            "timestamp": item.get("timestamp"),
            "video": item.get("video"),
            "ocr": item.get("ocr")
        })

    return {
        "query": query,
        "page": paged["page"],
        "page_size": paged["page_size"],
        "total": paged["total"],
        "total_pages": paged["total_pages"],
        "results": output
    }


# ======================
# AUDIO SEARCH
# ======================
class PhraseAudioIndex:

    def __init__(self, ngram=3):
        self.ngram = ngram
        self.index = defaultdict(set)
        self.meta = {}

    def clean(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9À-ỹ\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def add(self, item):
        text = self.clean(item.get("text", ""))
        tokens = text.split()

        if not tokens:
            return

        clip_id = item["clip_path"]
        if not clip_id:
           return
        self.meta[clip_id] = item

        # word + phrase index
        for n in range(1, self.ngram + 1):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i:i+n])
                self.index[phrase].add(clip_id)

    def build(self, data):
        for item in data:
            self.add(item)

audio_index = PhraseAudioIndex(ngram=3)
audio_index.build(audio_metadata)

def search_audio(query, k=2000):

    query = query.lower().strip()
    query = re.sub(r"\s+", " ", query)

    tokens = query.split()

    scores = defaultdict(int)

    # ưu tiên phrase dài trước
    for n in range(min(3, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i+n])

            for clip_id in audio_index.index.get(phrase, []):
                scores[clip_id] += n  # phrase dài điểm cao hơn

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []

    for clip_id, _ in ranked[:k]:
        item = audio_index.meta.get(clip_id)
        if not item:
            continue

        results.append({
            "video": item.get("video"),
            "clip_path": "/data/clips/" + clip_id.split("data/clips")[-1].replace("\\", "/"),
            "start": float(item.get("start", 0)),
            "end": float(item.get("end", 0)),
            "text": item.get("text")
        })

    return results

def search_audio_api(query, page=1, page_size=50):

    results = search_audio(query, k=2000)
    paged = paginate(results, page, page_size)

    return {
        "query": query,
        "page": paged["page"],
        "page_size": paged["page_size"],
        "total": paged["total"],
        "total_pages": paged["total_pages"],
        "results": paged["results"]
    }