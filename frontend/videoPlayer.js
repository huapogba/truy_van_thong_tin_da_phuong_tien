/*************************************************
 * VIDEO PLAYER MODULE
 * Handles YouTube video playback with timestamp seeking
 *************************************************/

let youtubeData = [];
let player = null;
let currentVideoId = null;
let timeUpdateInterval = null;
let currentFrameData = null; // Store current frame data for submit

// Load YouTube URL mapping
async function loadYouTubeMapping() {
  try {
    const res = await fetch('youtube_url.json');
    youtubeData = await res.json();
    console.log('YouTube URL mapping loaded:', youtubeData.length, 'videos');
  } catch (err) {
    console.error('Failed to load YouTube URL mapping:', err);
  }
}

// Get YouTube URL from video folder name
function getYouTubeUrl(videoFolder) {
  const entry = youtubeData.find(item => item.folder === videoFolder);
  return entry ? entry.watch_url : null;
}

// Extract video ID from YouTube URL
function extractVideoId(url) {
  const match = url.match(/[?&]v=([^&]+)/);
  return match ? match[1] : null;
}

// Open video modal with YouTube player
function openVideoModal(frameData, timestamp) {
  const videoFolder = frameData.video;
  const youtubeUrl = getYouTubeUrl(videoFolder);
  
  // Store frame data for submit
  currentFrameData = frameData;
  
  if (!youtubeUrl) {
    alert('YouTube URL not found for video: ' + videoFolder);
    return;
  }

  const videoId = extractVideoId(youtubeUrl);
  if (!videoId) {
    alert('Invalid YouTube URL format');
    return;
  }

  // Create modal if doesn't exist
  let modal = document.getElementById('videoModal');
  if (!modal) {
    modal = createVideoModal();
  }

  // Show modal
  modal.style.display = 'flex';
  
  // Load YouTube API if not loaded
  if (typeof YT === 'undefined') {
    loadYouTubeAPI(() => initPlayer(videoId, timestamp));
  } else if (player && currentVideoId === videoId) {
    // Same video, just seek
    player.seekTo(timestamp / 1000, true);
    startTimeUpdate();
  } else {
    // Different video or no player yet
    initPlayer(videoId, timestamp);
  }
  
  currentVideoId = videoId;
}

// Create video modal HTML
function createVideoModal() {
  const modal = document.createElement('div');
  modal.id = 'videoModal';
  modal.className = 'video-modal';
  
  modal.innerHTML = `
    <div class="video-modal-content">
      <span class="video-close">&times;</span>
      <h3>Video Player</h3>
      <div id="ytPlayer"></div>
      <div class="video-controls">
        <div class="timestamp-display">
          <span>Current Time: </span>
          <span id="currentTimestamp">0</span>
          <span> ms</span>
        </div>
        <button id="videoSubmitBtn" class="video-submit-btn">Submit</button>
      </div>
    </div>
  `;
  
  document.body.appendChild(modal);
  
  // Close button handler
  modal.querySelector('.video-close').onclick = () => {
    closeVideoModal();
  };
  
  // Click outside to close
  modal.onclick = (e) => {
    if (e.target === modal) {
      closeVideoModal();
    }
  };
  
  // Submit button handler
  modal.querySelector('#videoSubmitBtn').onclick = () => {
    submitVideoTimestamp();
  };
  
  return modal;
}

// Close video modal
function closeVideoModal() {
  const modal = document.getElementById('videoModal');
  if (modal) {
    modal.style.display = 'none';
  }
  
  if (timeUpdateInterval) {
    clearInterval(timeUpdateInterval);
    timeUpdateInterval = null;
  }
  
  if (player) {
    player.pauseVideo();
  }
}

// Load YouTube IFrame API
function loadYouTubeAPI(callback) {
  if (window.YT && window.YT.Player) {
    callback();
    return;
  }
  
  const tag = document.createElement('script');
  tag.src = 'https://www.youtube.com/iframe_api';
  
  window.onYouTubeIframeAPIReady = () => {
    console.log('YouTube API Ready');
    callback();
  };
  
  const firstScriptTag = document.getElementsByTagName('script')[0];
  firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
}

// Initialize YouTube player
function initPlayer(videoId, startTime) {
  const playerDiv = document.getElementById('ytPlayer');
  if (!playerDiv) return;
  
  // Destroy existing player
  if (player) {
    player.destroy();
  }
  
  // Create new player
  player = new YT.Player('ytPlayer', {
    height: '480',
    width: '854',
    videoId: videoId,
    playerVars: {
      'autoplay': 1,
      'start': Math.floor(startTime / 1000)
    },
    events: {
      'onReady': (event) => {
        console.log('Player ready, seeking to:', startTime, 'ms');
        event.target.seekTo(startTime / 1000, true);
        startTimeUpdate();
      },
      'onStateChange': onPlayerStateChange
    }
  });
}

// Handle player state changes
function onPlayerStateChange(event) {
  if (event.data === YT.PlayerState.PLAYING) {
    startTimeUpdate();
  } else if (event.data === YT.PlayerState.PAUSED || event.data === YT.PlayerState.ENDED) {
    if (timeUpdateInterval) {
      clearInterval(timeUpdateInterval);
      timeUpdateInterval = null;
    }
  }
}

// Start updating timestamp display
function startTimeUpdate() {
  if (timeUpdateInterval) {
    clearInterval(timeUpdateInterval);
  }
  
  timeUpdateInterval = setInterval(() => {
    if (player && player.getCurrentTime) {
      const currentTime = Math.round(player.getCurrentTime() * 1000);
      const display = document.getElementById('currentTimestamp');
      if (display) {
        display.textContent = currentTime;
      }
    }
  }, 100); // Update every 100ms for smooth display
}

// Add Video button click handler
function handleVideoButtonClick(e) {
  if (!e.target.classList.contains('video-btn')) return;
  
  const frameItem = e.target.closest('.frame-item');
  const img = frameItem?.querySelector('img');
  if (!img) return;
  
  const frameName = img.alt;
  
  // Find frame data
  let frameData = window.currentResults?.find(f => f.frame_name === frameName) ||
                  (window._neighborFrames || []).find(f => f.frame_name === frameName) ||
                  (window._similarFrames || []).find(f => f.frame_name === frameName);
  
  if (!frameData) {
    alert('Cannot identify frame data');
    return;
  }
  
  // Extract timestamp from caption or frame name
  const caption = frameItem.querySelector('.caption');
  let timestamp = 0;
  
  if (caption) {
    const match = caption.textContent.match(/(\d+)\s*ms/);
    if (match) {
      timestamp = parseInt(match[1]);
    }
  }
  
  // Fallback: extract from frame name (frame number * some interval)
  if (timestamp === 0 && frameData.timestamp) {
    timestamp = Math.round(frameData.timestamp * 1000);
  }
  
  openVideoModal(frameData, timestamp);
}

// Submit current video timestamp
async function submitVideoTimestamp() {
  if (!player || !currentFrameData) {
    console.error('Submit failed: No player or frame data');
    alert('Cannot submit: Video not loaded');
    return;
  }
  
  // Get current timestamp in milliseconds
  const currentTimeMs = Math.round(player.getCurrentTime() * 1000);
  
  const submitData = {
  videoId: currentFrameData.video,
  timestamp_ms: currentTimeMs,   // ✅ đúng tên, đúng kiểu
  question: currentFrameData.question || null
   };

  
  console.log('=== VIDEO SUBMIT DEBUG ===');
  console.log('Frame Data:', currentFrameData);
  console.log('Current Time (ms):', currentTimeMs);
  console.log('Submit Payload:', JSON.stringify(submitData, null, 2));
  console.log('Destination:', 'http://192.168.28.98:8000/submit_video');
  
  try {
    const res = await fetch("http://192.168.28.98:8000/submit_video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(submitData)
    });

    console.log('Response Status:', res.status, res.statusText);
    
    const text = await res.text();
    let data;
    try { 
      data = JSON.parse(text); 
      console.log('Response Data (parsed):', data);
    } catch { 
      data = text; 
      console.log('Response Data (text):', data);
    }

    if (!res.ok) {
      console.error('Submit Error Response:', data);
      return alert(
        `Submit failed!\nStatus: ${res.status}\n` +
        `Response:\n${typeof data === "string" ? data : JSON.stringify(data, null, 2)}`
      );
    }

    console.log('✅ Submit successful!');
    alert(
      `Submit successful!\n` +
      `Video: ${currentFrameData.video}\n` +
      `Timestamp: ${currentTimeMs} ms\n` +
      `Response:\n${typeof data === "string" ? data : JSON.stringify(data, null, 2)}`
    );

  } catch (err) {
    console.error('Submit Exception:', err);
    alert("Cannot connect to server!");
  }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  loadYouTubeMapping();
  document.addEventListener('click', handleVideoButtonClick);
});

// Export for use in other modules
window.VideoPlayer = {
  openVideoModal,
  closeVideoModal
};
