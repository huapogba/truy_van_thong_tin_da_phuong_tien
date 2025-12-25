/*************************************************
 * GLOBAL STATE
 *************************************************/
let searchData = null;
let currentResults = [];
let currentView = "l14";
window._neighborFrames = [];
window._similarFrames = [];
window.currentResults = currentResults; // Expose for video player

/*************************************************
 * 1. SEARCH TEXT
 *************************************************/
async function search() {
  const q1 = document.getElementById("query1").value.trim();
  const q2 = document.getElementById("query2").value.trim();
  const q3 = document.getElementById("query3").value.trim();

  const queries = [q1, q2, q3].filter(Boolean);
  if (!queries.length) return alert("Please enter at least 1 context!");

  try {
    const res = await fetch("http://192.168.28.98:8000/search_text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ queries })
    });

    if (!res.ok) throw new Error();
    searchData = await res.json();

    currentView = "l14";
    currentResults = getResultsByView(currentView);
    window.currentResults = currentResults; // Update global reference

    document.getElementById("results").innerHTML = "";
    renderResults();
  } catch {
    alert("Cannot connect to server!");
  }
}

/*************************************************
 * 2. GET RESULT BY MODEL
 *************************************************/
function getResultsByView(view) {
  if (!searchData) return [];
  if (view === "l14") return searchData.frame_results_L14 || [];
  if (view === "b32") return searchData.frame_results_b32 || [];
  if (view === "b16") return searchData.frame_results_b16 || [];
  return [];
}

/*************************************************
 * 3. RENDER SEARCH RESULTS
 *************************************************/
function renderResults() {
  const container = document.getElementById("results");
  container.innerHTML = "";

  const grid = document.createElement("div");
  grid.className = "frame-grid";

  currentResults.forEach(f => {
    const item = document.createElement("div");
    item.className = "frame-item";

    const img = document.createElement("img");
    img.src = `http://192.168.28.98:8000/frames/${f.frame_name}`;
    img.alt = f.frame_name;
    img.loading = "lazy";
    img.onclick = () => openModalByTimeline(f);

    const cap = document.createElement("p");
    cap.className = "caption";
    cap.textContent = `${f.frame_name} - ${Math.round((f.timestamp || 0) * 1000)} ms`;

    const btnSimilar = document.createElement("button");
    btnSimilar.className = "similar-btn";
    btnSimilar.textContent = "Find Similar";

    const btnVideo = document.createElement("button");
    btnVideo.className = "video-btn";
    btnVideo.textContent = "Video";

    const btnSubmit = document.createElement("button");
    btnSubmit.className = "submit-btn";
    btnSubmit.textContent = "Submit";

    item.append(img, cap, btnSimilar, btnVideo, btnSubmit);
    grid.appendChild(item);
  });

  container.appendChild(grid);
}

/*************************************************
 * 4. OPEN MODAL – FRAME LÂN C?N (GRID)
 *************************************************/
async function openModalByTimeline(frameData) {
  const modal = document.getElementById("imageModal");
  const grid = document.getElementById("neighborGrid");

  modal.style.display = "block";
  grid.innerHTML = "";

  try {
    const res = await fetch("http://192.168.28.98:8000/get_neighbor_frames", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video: frameData.video,
        frame_name: frameData.frame_name,
        range: 20
      })
    });

    if (!res.ok) throw new Error();
    const { frames = [] } = await res.json();

    // ?? luu d? submit dùng
    window._neighborFrames = frames;

    frames.forEach(f => {
      const wrap = document.createElement("div");
      wrap.className = "frame-item";

      if (f.frame_name === frameData.frame_name) {
        wrap.classList.add("active-frame");
      }

      const img = document.createElement("img");
      img.src = `http://192.168.28.98:8000/frames/${f.frame_name}`;
      img.alt = f.frame_name;
      img.onclick = () => openModalByTimeline(f);

      const cap = document.createElement("p");
      cap.className = "caption";
      cap.textContent = `${f.video} - ${f.frame_name}`;

      // Buttons container
      const btnContainer = document.createElement("div");
      btnContainer.className = "frame-buttons";

      const btnVideo = document.createElement("button");
      btnVideo.className = "video-btn";
      btnVideo.textContent = "Video";

      const btnSubmit = document.createElement("button");
      btnSubmit.className = "submit-btn";
      btnSubmit.textContent = "Submit";

      btnContainer.append(btnVideo, btnSubmit);
      wrap.append(img, cap, btnContainer);
      grid.appendChild(wrap);
    });

  } catch (err) {
    console.error(err);
    alert("Cannot get neighbor frames!");
  }
}

/*************************************************
 * 5. CLOSE MODAL
 *************************************************/
document.querySelector(".close")?.addEventListener("click", () => {
  document.getElementById("imageModal").style.display = "none";
});

/*************************************************
 * 6. SWITCH MODEL VIEW
 *************************************************/
function switchView(view) {
  currentView = view;
  currentResults = getResultsByView(view);
  window.currentResults = currentResults; // Update global reference
  document.getElementById("results").innerHTML = "";
  renderResults();
}

document.getElementById("searchBtn").onclick = search;
document.getElementById("l14Btn").onclick = () => switchView("l14");
document.getElementById("b32Btn").onclick = () => switchView("b32");
document.getElementById("b16Btn").onclick = () => switchView("b16");

/*************************************************
 * 7. SEARCH SIMILAR  ? ÐÃ S?A – CÓ VIDEO + FRAME
 *************************************************/
document.addEventListener("click", async e => {
  if (!e.target.classList.contains("similar-btn")) return;

  const frameName = e.target.closest(".frame-item")
    ?.querySelector("img")?.alt;

  if (!frameName) return;

  try {
    const res = await fetch("http://192.168.28.98:8000/search_similar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame_name: frameName, model: currentView })
    });

    if (!res.ok) throw new Error();
    const { similar_frames = [] } = await res.json();

    // ?? luu d? submit
    window._similarFrames = similar_frames.slice(0, 20);

    const grid = document.getElementById("similarGrid");
    grid.innerHTML = "";

    window._similarFrames.forEach(f => {
      const wrap = document.createElement("div");
      wrap.className = "frame-item";

      const img = document.createElement("img");
      img.src = `http://192.168.28.98:8000/frames/${f.frame_name}`;
      img.alt = f.frame_name;
      img.onclick = () => openModalByTimeline(f);

      const cap = document.createElement("p");
      cap.className = "caption";
      cap.textContent = `${f.video} - ${f.frame_name}`;

      // Buttons container
      const btnContainer = document.createElement("div");
      btnContainer.className = "frame-buttons";

      const btnVideo = document.createElement("button");
      btnVideo.className = "video-btn";
      btnVideo.textContent = "Video";

      const btnSubmit = document.createElement("button");
      btnSubmit.className = "submit-btn";
      btnSubmit.textContent = "Submit";

      btnContainer.append(btnVideo, btnSubmit);
      wrap.append(img, cap, btnContainer);
      grid.appendChild(wrap);
    });

    document.getElementById("similarModal").style.display = "block";
  } catch {
    alert("Cannot find similar images!");
  }
});

/*************************************************
 * 8. CLOSE SIMILAR MODAL
 *************************************************/
document.querySelector(".close-similar")?.addEventListener("click", () => {
  document.getElementById("similarModal").style.display = "none";
});

document.getElementById("similarModal")?.addEventListener("click", e => {
  if (e.target.id === "similarModal") {
    e.currentTarget.style.display = "none";
  }
});

/*************************************************
 * 9. SUBMIT FRAME – CHUNG CHO C? 3 LO?I
 *************************************************/
document.addEventListener("click", async e => {
  if (!e.target.classList.contains("submit-btn")) return;

  const img = e.target.closest(".frame-item")?.querySelector("img");
  if (!img) return;

  // ?? Tìm frame trong 3 ngu?n
  let frameData = currentResults.find(f => f.frame_name === img.alt) ||
                  (window._neighborFrames || []).find(f => f.frame_name === img.alt) ||
                  (window._similarFrames || []).find(f => f.frame_name === img.alt);

  if (!frameData) return alert("Cannot identify frame!");

  // ⚠️ Lấy frame number từ tên file, KHÔNG phải từ folder path
  // Ví dụ: "L17_V003/frame_1816.jpg" → lấy "1816"
  const match = img.alt.match(/frame_(\d+)/);
  if (!match) return alert("Cannot extract frame number from: " + img.alt);

  // ✅ LOG REQUEST
  const payload = {
    videoId: String(frameData.video),
    timestamp: match[1],  // frame number từ "frame_1816" → "1816"
    question: frameData.question || null
  };
  
  console.log("============================================================");
  console.log("[SUBMIT REQUEST]");
  console.log("URL:", "http://192.168.28.98:8000/submit_result");
  console.log("Payload:", JSON.stringify(payload, null, 2));
  console.log("Frame Data:", frameData);

  try {
    const res = await fetch("http://192.168.28.98:8000/submit_result", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    // ✅ LOG RESPONSE HEADERS
    console.log("\n[SUBMIT RESPONSE]");
    console.log("Status:", res.status);
    console.log("Status Text:", res.statusText);
    console.log("Headers:", Object.fromEntries(res.headers.entries()));

    // ✅ LOG RAW TEXT
    const text = await res.text();
    console.log("Raw Response Length:", text.length, "chars");
    console.log("Raw Response:", text);

    let data;
    try { 
      data = JSON.parse(text);
      console.log("Parsed JSON:", data);
    } catch (parseError) { 
      console.error("❌ JSON Parse Error:", parseError);
      console.error("Cannot parse response as JSON!");
      data = text; 
    }

    console.log("============================================================\n");

    if (!res.ok) {
      return alert(
        `Submit failed!\nStatus: ${res.status}\n` +
        `Response:\n${typeof data === "string" ? data : JSON.stringify(data, null, 2)}`
      );
    }

    alert(
      `Submit successful!\n` +
      `Response:\n${typeof data === "string" ? data : JSON.stringify(data, null, 2)}`
    );

  } catch (err) {
    console.error("============================================================");
    console.error("[SUBMIT ERROR]");
    console.error("Error Type:", err.name);
    console.error("Error Message:", err.message);
    console.error("Full Error:", err);
    console.error("============================================================\n");
    alert("Cannot connect to server!\n" + err.message);
  }
});

// Nút đóng modal frame lân cận
document.getElementById("closeNeighborBtn")?.addEventListener("click", () => {
  document.getElementById("imageModal").style.display = "none";
});

/*************************************************
 * ENTER TO SEARCH
 *************************************************/
["query1", "query2", "query3"].forEach(id => {
  const input = document.getElementById(id);
  if (!input) return;

  input.addEventListener("keydown", e => {
    if (e.key === "Enter") {
      e.preventDefault(); // tránh reload form
      search();
    }
  });
});

