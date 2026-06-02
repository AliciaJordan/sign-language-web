const channel     = new BroadcastChannel("sign_language_receiver");
const lastWordEl  = document.getElementById("lastWord");
const historyList = document.getElementById("historyList");
const cameraFeed  = document.getElementById("cameraFeed");
const noSignal    = document.getElementById("noSignal");
const statusDot   = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");
const clearBtn    = document.getElementById("clearBtn");

let words         = [];
let frameTimeout  = null;

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

    // If no frame arrives for 2s, show no-signal
    clearTimeout(frameTimeout);
    frameTimeout = setTimeout(setOffline, 2000);
}

function addWord(word, time) {
    words.unshift({ word, time });
    if (words.length > 15) words.pop();

    // Flash the last word display
    lastWordEl.classList.remove("flash");
    void lastWordEl.offsetWidth; // reflow to restart animation
    lastWordEl.textContent = word;
    lastWordEl.classList.add("flash");

    renderHistory();
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

channel.onmessage = (e) => {
    const { type } = e.data;
    if (type === "frame") {
        showFrame(e.data.dataUrl);
    } else if (type === "word") {
        // Sync full history from interpreter tab
        words = e.data.allWords;
        lastWordEl.classList.remove("flash");
        void lastWordEl.offsetWidth;
        lastWordEl.textContent = e.data.word;
        lastWordEl.classList.add("flash");
        renderHistory();
    }
};

clearBtn.addEventListener("click", () => {
    words = [];
    lastWordEl.textContent = "—";
    renderHistory();
});
