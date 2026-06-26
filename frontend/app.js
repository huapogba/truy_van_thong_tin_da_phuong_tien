let currentPage = 1;
let totalPages = 1;

let currentContext = [];
let currentCenterTime = null;

/* =========================
   SEARCH
========================= */

async function search(page = 1) {

    document.getElementById("backBtn").style.display = "none";
    currentContext = [];
    currentCenterTime = null;

    const query = document.getElementById("query").value.trim();
    const mode = document.getElementById("searchMode").value;
    const topk = document.getElementById("topk").value;

    const endpoint = mode === "clip" ? "search" : mode;

    const url =
        `http://127.0.0.1:5000/${endpoint}?q=${encodeURIComponent(query)}&topk=${topk}&page=${page}`;

    const res = await fetch(url);
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

    if (next < 1 || next > totalPages)
        return;

    search(next);
}

/* =========================
   RENDER SEARCH RESULTS
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

        const frame = item.frame
            ? `http://127.0.0.1:5000${item.frame}`
            : "";

        if (mode === "audio") {

            card.innerHTML = `
                <div class="info" style="padding:15px">
                    <p><b>${item.video}</b></p>
                    <p>⏱ ${item.start.toFixed(2)} - ${item.end.toFixed(2)}</p>
                    <p>${item.text}</p>
                </div>
            `;

            card.onclick = () => playAudio(item);

        }
        else {

            card.innerHTML = `
                <img src="${frame}">
                <div class="info">
                    <p><b>${item.video}</b></p>
                    <p>⏱ ${item.timestamp}</p>
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
   PLAY AUDIO CLIP
========================= */

function playAudio(item) {

    const player = document.getElementById("videoPlayer");

    player.src = `http://127.0.0.1:5000${item.clip_path}`;

    player.load();

    player.play();

}

/* =========================
   LOAD CONTEXT
========================= */

async function loadContext(item) {

    if (!item.video || item.timestamp === undefined)
        return;

    document.getElementById("backBtn").style.display = "block";

    const url =
        `http://127.0.0.1:5000/clip/context?video=${encodeURIComponent(item.video)}&timestamp=${item.timestamp}&window=20`;

    const res = await fetch(url);
    const data = await res.json();

    currentContext = data.frames || [];
    currentCenterTime = data.center_timestamp;

    renderContext(currentContext);

}

/* =========================
   CONTEXT VIEW
========================= */

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
            card.style.transform = "scale(1.05)";

        }

        card.innerHTML = `
            <img src="${frame}">
            <div class="info">
                <p><b>${item.video}</b></p>
                <p>⏱ ${item.timestamp}</p>
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

    currentContext = [];
    currentCenterTime = null;

    search(currentPage);

}

/* =========================
   PLAY FULL VIDEO
========================= */

function jumpToTime(item) {

    const player = document.getElementById("videoPlayer");

    player.src = `http://127.0.0.1:5000/data/clips/${item.video}.mp4`;

    player.onloadedmetadata = () => {

        player.currentTime = Number(item.timestamp);

        player.play();

    };

}
