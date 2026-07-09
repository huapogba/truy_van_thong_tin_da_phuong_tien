from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from search import (
    search_clip_api,
    search_ocr_api,
    search_audio_api,
    get_clip_context_api,
    search_clip_in_one_video
)

app = Flask(__name__)
CORS(app)

# ==========================
# CLIP IN ONE VIDEO SEARCH
# ==========================
@app.route("/clip/in-video")
def clip_in_video_route():

    query = request.args.get("q", "")
    video_id = request.args.get("video", "")
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("topk", 50))

    if not query or not video_id:
        return jsonify({
            "error": "q and video are required"
        }), 400

    result = search_clip_in_one_video(
        query=query,
        video_id=video_id,
        page=page,
        page_size=page_size
    )

    return jsonify(result)

# ==========================
# CLIP SEARCH
# ==========================
@app.route("/search")
def search_route():

    query = request.args.get("q", "")
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("topk", 50))

    result = search_clip_api(
        query=query,
        page=page,
        page_size=page_size
    )

    return jsonify(result)

# ==========================
# CLIP CONTEXT (CLICK FRAME)
# ==========================
@app.route("/clip/context")
def clip_context_route():

    video = request.args.get("video", "")
    timestamp = request.args.get("timestamp", type=float)
    window = int(request.args.get("window", 20))

    if not video or timestamp is None:
        return jsonify({
            "error": "video and timestamp are required"
        }), 400

    result = get_clip_context_api(
        video=video,
        timestamp=timestamp,
        window=window
    )

    return jsonify(result)

# ==========================
# OCR SEARCH
# ==========================
@app.route("/ocr")
def ocr_route():

    query = request.args.get("q", "")
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("topk", 50))

    result = search_ocr_api(
        query=query,
        page=page,
        page_size=page_size
    )

    return jsonify(result)


# ==========================
# AUDIO SEARCH
# ==========================
@app.route("/audio")
def audio_route():

    query = request.args.get("q", "")
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("topk", 50))

    result = search_audio_api(
        query=query,
        page=page,
        page_size=page_size
    )

    return jsonify(result)


# ==========================
# SERVE STATIC FILES
# ==========================
@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory("data", filename)


@app.route("/")
def home():
    return "Video Retrieval API"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True
    )