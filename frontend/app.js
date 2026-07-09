let currentPage = 1;
let totalPages = 1;

let currentContext = [];
let currentCenterTime = null;

/* =========================
   SEARCH MAIN
========================= */

async function search(page = 1) {

    document.getElementById("backBtn").style.display = "none";
    currentContext = [];
    currentCenterTime = null;

    const query = document.getElementById("query").value.trim();
    const mode = document.getElementById("searchMode").value;
    const topk = document.getElementById("topk").value;
    const videoId = document.getElementById("videoId").value.trim();

    let url = "";
    let params = new URLSearchParams();

    params.append("q", query);
    params.append("topk", topk);
    params.append("page", page);

    if (mode === "clip") {
        url = "/search";
    }

    else if (mode === "clip_in_video") {

        if (!videoId) {
            alert("Please enter Video ID");
            return;
        }

        url = "/clip/in-video";
        params.append("video", videoId);
    }

    else if (mode === "ocr") {
        url = "/ocr";
    }

    else if (mode === "audio") {
        url = "/audio";
    }

    const res = await fetch(`http://127.0.0.1:5000${url}?${params}`);
    const data = await res.json();

    currentPage = data.page || 1;
    totalPages = data.total_pages || 1;

    renderResults(data.results || []);
    updatePagination();
}

/* =========================
   PAGINATION
========================= */

function updatePagination() {
    document.getElementById("pageInfo").innerText =
        `Page ${currentPage} / ${totalPages}`;
}

function changePage(step) {
    const next = currentPage + step;
    if (next < 1 || next > totalPages) return;
    search(next);
}

/* =========================
   RENDER RESULTS
========================= */

function renderResults(results) {

    const gallery = document.getElementById("results");
    gallery.innerHTML = "";

    const mode = document.getElementById("searchMode").value;

    if (!results.length) {
        gallery.innerHTML = "<p>No results</p>";
        return;
    }

    results.forEach(item => {

        const card = document.createElement("div");
        card.className = "result";

        if (mode === "audio") {

            card.innerHTML = `
                <div class="info">
                    <p><b>${item.video}</b></p>
                    <p>${item.start.toFixed(2)} - ${item.end.toFixed(2)}</p>
                    <p>${item.text}</p>
                </div>
            `;

            card.onclick = () => playAudio(item);
        }

        else {

            const frame = item.frame
                ? `http://127.0.0.1:5000${item.frame}`
                : "";

            card.innerHTML = `
                <img src="${frame}">
                <div class="info">
                    <p><b>${item.video}</b></p>
                    <p>Time: ${item.timestamp}s</p>
                </div>
            `;

            card.onclick = () => {
                jumpToTime(item);
                loadContext(item);
            };
        }

        gallery.appendChild(card);
    });
}

/* =========================
   AUDIO
========================= */

function playAudio(item) {

    const player = document.getElementById("videoPlayer");
    player.src = `http://127.0.0.1:5000${item.clip_path}`;
    player.load();
    player.play();
}

/* =========================
   CONTEXT
========================= */

async function loadContext(item) {

    document.getElementById("backBtn").style.display = "block";

    const url =
        `http://127.0.0.1:5000/clip/context?video=${item.video}&timestamp=${item.timestamp}&window=20`;

    const res = await fetch(url);
    const data = await res.json();

    currentContext = data.frames || [];
    currentCenterTime = data.center_timestamp;

    renderContext(currentContext);
}

function renderContext(frames) {

    const gallery = document.getElementById("results");
    gallery.innerHTML = "";

    frames.forEach(item => {

        const frame = `http://127.0.0.1:5000${item.frame}`;
        const isCenter = item.timestamp === currentCenterTime;

        const card = document.createElement("div");
        card.className = "result";

        if (isCenter) {
            card.style.border = "2px solid #22c55e";
        }

        card.innerHTML = `
            <img src="${frame}">
            <div class="info">
                <p>${item.video}</p>
                <p>${item.timestamp}</p>
            </div>
        `;

        card.onclick = () => {
            jumpToTime(item);
            loadContext(item);
        };

        gallery.appendChild(card);
    });
}

/* =========================
   BACK
========================= */

function backToSearch() {
    document.getElementById("backBtn").style.display = "none";
    search(currentPage);
}

/* =========================
   VIDEO SEEK
========================= */

function jumpToTime(item) {

    const player = document.getElementById("videoPlayer");

    player.src = `http://127.0.0.1:5000/data/clips/${item.video}.mp4`;

    player.onloadedmetadata = () => {
        player.currentTime = Number(item.timestamp);
        player.play();
    };
}