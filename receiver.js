const lastWordEl  = document.getElementById("lastWord");
const historyList = document.getElementById("historyList");
const cameraFeed  = document.getElementById("cameraFeed");
const noSignal    = document.getElementById("noSignal");
const statusDot   = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");
const clearBtn    = document.getElementById("clearBtn");

let words       = [];
let frameTimeout = null;

function setOnline() {
    statusDot.classList.remove("offline");
    statusDot.classList.add("online");
    statusLabel.textContent = "Intérprete conectado";
}

function setOffline() {
    statusDot.classList.remove("online");
    statusDot.classList.add("offline");
    statusLabel.textContent = "Sin conexión";
    noSignal.classList.remove("hidden");
}

function showFrame(dataUrl) {
    cameraFeed.src = dataUrl;
    noSignal.classList.add("hidden");
    setOnline();
    clearTimeout(frameTimeout);
    frameTimeout = setTimeout(setOffline, 3000);
}

function flashWord(word) {
    lastWordEl.classList.remove("flash");
    void lastWordEl.offsetWidth;
    lastWordEl.textContent = word;
    lastWordEl.classList.add("flash");
}

function renderHistory() {
    historyList.innerHTML = "";
    words.forEach(({ word, time }) => {
        const li = document.createElement("li");
        li.innerHTML = `
            <span class="historyDot"></span>
            <span class="historyWord">${word}</span>
            <span class="historyTime">${time}</span>
        `;
        historyList.appendChild(li);
    });
}

function initFirebase() {
    const { initializeApp } = window.firebaseApp;
    const { getDatabase, ref, onValue } = window.firebaseDb;

    const app = initializeApp({
        apiKey:            "AIzaSyCvOY0ViyK7SEIt5Ym3EeLtVMN7CfA7MZY",
        authDomain:        "cafe-central-2a4f3.firebaseapp.com",
        databaseURL:       "https://cafe-central-2a4f3-default-rtdb.firebaseio.com",
        projectId:         "cafe-central-2a4f3",
        storageBucket:     "cafe-central-2a4f3.firebasestorage.app",
        messagingSenderId: "334981865406",
        appId:             "1:334981865406:web:0fe64f912092a961f423ef"
    });

    const db = getDatabase(app);

    // Listen for camera frames
    onValue(ref(db, "session/frame"), (snapshot) => {
        const data = snapshot.val();
        if (data && data.dataUrl) showFrame(data.dataUrl);
    });

    // Listen for new words
    onValue(ref(db, "session/words"), (snapshot) => {
        const data = snapshot.val();
        if (!data) return;
        words = data.allWords || [];
        flashWord(data.word);
        renderHistory();
    });

    console.log("Firebase receiver initialized");
}

clearBtn.addEventListener("click", () => {
    words = [];
    lastWordEl.textContent = "—";
    renderHistory();
});

// Fullscreen
const fsBtn = document.getElementById("fullscreenBtn");
fsBtn.addEventListener("click", () => {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
        fsBtn.textContent = "✕";
        fsBtn.title = "Salir de pantalla completa";
    } else {
        document.exitFullscreen();
    }
});
document.addEventListener("fullscreenchange", () => {
    if (!document.fullscreenElement) {
        fsBtn.textContent = "⛶";
        fsBtn.title = "Pantalla completa";
    }
});

// Firebase loads via module script in HTML, wait for it
window.addEventListener("load", initFirebase);
