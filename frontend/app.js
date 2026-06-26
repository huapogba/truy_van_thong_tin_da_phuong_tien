let currentPage = 1;
let totalPages = 1;

// context state
let currentContext = [];
let currentCenterTime = null;

/* =========================
   SEARCH MODE
========================= */
async function search(page = 1) {

    // reset context UI
    document.getElementById("backBtn").style.display = "none";
    currentContext = [];
    currentCenterTime = null;

    const query = document.getElementById("query").value.trim();
    const mode = document.getElementById("searchMode").value;
    const topk = document.getElementById("topk").value;

    const endpoint = mode === "clip" ? "search" : mode;

    const url =
        `http://127.0.0.1:5000/${endpoint}` +
        `?q=${encodeURIComponent(query)}` +
        `&topk=${topk}` +
        `&page=${page}`;

    const res = await fetch(url);
    const data = await res.json();

    currentPage = data.page || page;
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
    const newPage = currentPage + step;
    if (newPage < 1 || newPage > totalPages) return;
    search(newPage);
}

/* =========================
   RENDER SEARCH RESULTS
========================= */
function renderResults(results) {

    const gallery = document.getElementById("results");
    gallery.innerHTML = "";

    if (!results.length) {
        gallery.innerHTML = "<p>No results</p>";
        return;
    }

    results.forEach(item => {

        const frame = item.frame
            ? `http://127.0.0.1:5000${item.frame}`
            : "";

        const card = document.createElement("div");
        card.className = "result";

        card.innerHTML = `
            <img src="${frame}">
            <div class="info">
                <p><b>${item.video || ""}</b></p>
                <p>⏱ ${item.timestamp ?? ""}</p>
            </div>
        `;

        // CLICK → LOAD CONTEXT
        card.onclick = () => loadContext(item);

        gallery.appendChild(card);
    });
}

/* =========================
   LOAD CONTEXT API
========================= */
async function loadContext(item) {

    if (!item.video || item.timestamp === undefined) return;

    // show back button
    document.getElementById("backBtn").style.display = "block";

    const url =
        `http://127.0.0.1:5000/clip/context` +
        `?video=${encodeURIComponent(item.video)}` +
        `&timestamp=${item.timestamp}` +
        `&window=20`;

    const res = await fetch(url);
    const data = await res.json();

    currentContext = data.frames || [];
    currentCenterTime = data.center_timestamp;

    renderContext(currentContext);
}

/* =========================
   RENDER CONTEXT VIEW
========================= */
function renderContext(frames) {

    const gallery = document.getElementById("results");
    gallery.innerHTML = "";

    if (!frames.length) {
        gallery.innerHTML = "<p>No context</p>";
        return;
    }

    frames.forEach(item => {

        const frame = item.frame
            ? `http://127.0.0.1:5000${item.frame}`
            : "";

        const isCenter = item.timestamp === currentCenterTime;

        const card = document.createElement("div");
        card.className = "result";

        if (isCenter) {
            card.style.border = "2px solid #22c55e";
            card.style.transform = "scale(1.05)";
        }

        card.innerHTML = `
            <img src="${frame}">
            <div class="info">
                <p><b>${item.video || ""}</b></p>
                <p>⏱ ${item.timestamp}</p>
            </div>
        `;

        // click → recenter context
        card.onclick = () => loadContext({
            video: item.video,
            timestamp: item.timestamp
        });

        gallery.appendChild(card);
    });
}

/* =========================
   BACK TO SEARCH MODE
========================= */
function backToSearch() {

    document.getElementById("backBtn").style.display = "none";

    currentContext = [];
    currentCenterTime = null;

    search(currentPage);
}

/* =========================
   VIDEO SEEK (optional)
========================= */
function jumpToTime(item) {

    const player = document.getElementById("videoPlayer");

    const videoPath = (item.video || "").replace(/\\/g, "/");
    if (!videoPath) return;

    player.src = `http://127.0.0.1:5000/${videoPath}`;

    player.onloadedmetadata = () => {

        const t = item.timestamp ?? 0;

        player.currentTime = Number(t);
        player.play();
    };
}