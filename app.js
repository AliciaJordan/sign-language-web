console.log("app.js loaded");

// =========================
// CONFIG
// =========================

const LABELS = ["before", "thin", "cool", "drink", "go"];

const NUM_FRAMES        = 64;
const NUM_FEATURES      = 96;
const MODEL_INPUT_SHAPE = [1, NUM_FRAMES, NUM_FEATURES];

const CONFIDENCE_THRESHOLD  = 60;
const HISTORY_COOLDOWN_MS   = 500;
const INFERENCE_COOLDOWN_MS = 1000; // min ms between inferences

// Gesture segmentation
const NO_HAND_TRIGGER    = 15;  // consecutive no-hand frames before inferring (~500ms)
const MIN_GESTURE_FRAMES = 8;   // minimum frames to bother inferring

// DEBUG MODE
const DEBUG_MODE           = true;
const DEBUG_CAPTURE_FRAMES = 64;

// =========================
// STATE
// =========================

let session         = null;
let modelInputName  = "input";
let modelOutputName = "output";

let currentPrediction  = "Waiting...";
let currentConfidence  = 0;
let historyWords       = [];
let lastHistoryWordTime = 0;
let lastInferenceTime   = 0;

// Gesture buffer
const gestureFrames  = [];
let noHandFrameCount = 0;
let gestureActive    = false;
let inferenceRunning = false;

// Mirror canvas for webcam (matches Python cv2.flip)
let mirrorCanvas = null;
let mirrorCtx    = null;

// DEBUG
let debugFrameVectors    = [];
let debugCaptureActive   = false;
let debugInferenceData   = [];
let debugLandmarkData    = [];
let debugSequenceTensors = [];

// =========================
// DEBUG FUNCTIONS
// =========================

function startDebugCapture() {
    debugFrameVectors    = [];
    debugSequenceTensors = [];
    debugInferenceData   = [];
    debugLandmarkData    = [];
    debugCaptureActive   = true;
    console.log("[DEBUG] Capture started");
}

function stopDebugCapture() {
    debugCaptureActive = false;
    console.log(`[DEBUG] Capture stopped. Collected ${debugFrameVectors.length} frames`);
    return debugFrameVectors;
}

function logLandmarks(handResult, poseResult, featureVector) {
    if (!DEBUG_MODE) return;
    const info = {
        timestamp: performance.now(),
        hands: [],
        pose: [],
        rawFeatures: Array.from(featureVector)
    };
    if (handResult && handResult.landmarks) {
        handResult.landmarks.forEach((hand, idx) => {
            info.hands.push({
                handedness: handResult.handedness[idx][0].categoryName,
                firstFive: hand.slice(0, 5).map(lm => ({
                    x: lm.x.toFixed(4), y: lm.y.toFixed(4)
                }))
            });
        });
    }
    if (poseResult && poseResult.landmarks && poseResult.landmarks.length) {
        const pose = poseResult.landmarks[0];
        [11,12,13,14,15,16].forEach(idx => {
            info.pose.push({ index: idx, x: pose[idx].x.toFixed(4), y: pose[idx].y.toFixed(4) });
        });
    }
    debugLandmarkData.push(info);
}

function logInference(logits, probs, best) {
    if (!DEBUG_MODE) return;
    debugInferenceData.push({
        timestamp: performance.now(),
        rawLogits: Array.from(logits).map(x => x.toFixed(6)),
        softmaxProbs: Array.from(probs).map(x => x.toFixed(6)),
        topPrediction: { label: best.label, confidence: best.conf.toFixed(2) }
    });
}

function exportDebugData() {
    const data = {
        config: {
            NUM_FRAMES, NUM_FEATURES, LABELS,
            POSE_INDICES: [11,12,13,14,15,16],
            modelInputName, modelOutputName,
            modelInputShape: MODEL_INPUT_SHAPE,
            padding: "post (zeros)",
            truncation: "linear_resample_to_64",
            normalization: "subtract shoulder center, divide by shoulder distance; undetected hand slots reset to 0"
        },
        frameVectors:    debugFrameVectors,
        sequenceTensors: debugSequenceTensors,
        inferences:      debugInferenceData,
        landmarks:       debugLandmarkData
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `debug_data_${Date.now()}.json`;
    a.click();
    console.log("[DEBUG] Exported debug_data_*.json");
    return data;
}

function captureSequenceTensor(tensor) {
    if (!DEBUG_MODE) return;
    debugSequenceTensors.push({
        timestamp: performance.now(),
        shape: tensor.dims,
        dtype: tensor.type,
        data: Array.from(tensor.data)
    });
}

function captureFeatureVector(featureVector) {
    if (!debugCaptureActive) return;
    debugFrameVectors.push({ frameIndex: debugFrameVectors.length, data: Array.from(featureVector) });
    if (debugFrameVectors.length === DEBUG_CAPTURE_FRAMES) {
        console.log(`[DEBUG] Full sequence captured: ${DEBUG_CAPTURE_FRAMES} frames`);
        stopDebugCapture();
    }
}

function auditFeatureExtraction() {
    console.log("\n=== FEATURE EXTRACTION AUDIT ===");
    console.log("Layout: [0-41] Right hand | [42-83] Left hand | [84-95] Pose");
    console.log("Norm:   center=(LS+RS)/2, scale=dist(LS,RS)");
    console.log("Fix:    undetected hand slots reset to 0 after normalizing");
    if (debugLandmarkData.length > 0) {
        const last = debugLandmarkData[debugLandmarkData.length - 1];
        console.log(`\nLast frame — hands detected: ${last.hands.length}`);
        last.hands.forEach((h, i) => console.log(`  Hand ${i}: ${h.handedness}`));
        const f = last.rawFeatures;
        console.log(`Right[0-3]:    [${f[0].toFixed(3)}, ${f[1].toFixed(3)}, ${f[2].toFixed(3)}, ${f[3].toFixed(3)}]`);
        console.log(`Left[42-45]:   [${f[42].toFixed(3)}, ${f[43].toFixed(3)}, ${f[44].toFixed(3)}, ${f[45].toFixed(3)}]`);
        console.log(`Shoulders[84-87]: [${f[84].toFixed(3)}, ${f[85].toFixed(3)}, ${f[86].toFixed(3)}, ${f[87].toFixed(3)}]`);
    }
}

function analyzeFeatureStatistics() {
    if (debugLandmarkData.length === 0) {
        console.log("[STATS] No data. Run debugCapture() first.");
        return;
    }
    console.log("\n=== FEATURE STATISTICS ===");
    const byIdx = Array.from({length: 96}, () => []);
    debugLandmarkData.forEach(l => l.rawFeatures.forEach((v, i) => byIdx[i].push(v)));
    const regions = {
        "RightHand[0-41]":   [0,41],  "LeftHand[42-83]":  [42,83],
        "LShoulder[84-85]":  [84,85], "RShoulder[86-87]": [86,87],
        "LElbow[88-89]":     [88,89], "RElbow[90-91]":    [90,91],
        "LWrist[92-93]":     [92,93], "RWrist[94-95]":    [94,95]
    };
    for (const [name, [s,e]] of Object.entries(regions)) {
        const vals = [];
        for (let i=s;i<=e;i++) vals.push(...byIdx[i]);
        const mn   = Math.min(...vals).toFixed(2);
        const mx   = Math.max(...vals).toFixed(2);
        const mean = (vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(2);
        const zeros = (vals.filter(v=>v===0).length/vals.length*100).toFixed(1);
        console.log(`${name.padEnd(18)} min:${mn.padStart(7)} max:${mx.padStart(7)} mean:${mean.padStart(7)} zeros:${zeros}%`);
    }
}

// =========================
// UI ELEMENTS
// =========================

let video, canvas, ctx;
let predictionText, confidenceText, historyList, topPredictionsContainer;
let recordedVideoInput, recordedVideoButton, recordedVideo, recordedVideoResult;
let recordedStartFrame, recordedEndFrame;
let recordedProcessingCanvas, recordedProcessingCtx;

function getRequiredElement(id) {
    const el = document.getElementById(id);
    if (!el) throw new Error(`Required element '${id}' not found.`);
    return el;
}

function initializeUI() {
    video                    = getRequiredElement("video");
    canvas                   = getRequiredElement("overlay");
    ctx                      = canvas.getContext("2d");
    predictionText           = getRequiredElement("predictionText");
    confidenceText           = getRequiredElement("confidenceText");
    historyList              = getRequiredElement("historyList");
    topPredictionsContainer  = getRequiredElement("topPredictionsContainer");
    recordedVideoInput       = getRequiredElement("recordedVideoInput");
    recordedVideoButton      = getRequiredElement("recordedVideoBtn");
    recordedVideo            = getRequiredElement("recordedVideo");
    recordedVideoResult      = getRequiredElement("recordedVideoResult");
    recordedStartFrame       = getRequiredElement("recordedStartFrame");
    recordedEndFrame         = getRequiredElement("recordedEndFrame");
    recordedProcessingCanvas = document.createElement("canvas");
    recordedProcessingCtx    = null;
}

// =========================
// SOFTMAX
// =========================

function softmax(logits) {
    const max  = Math.max(...logits);
    const exps = logits.map(v => Math.exp(v - max));
    const sum  = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
}

// =========================
// CAMERA
// =========================

function showError(message) {
    console.error(message);
    predictionText.textContent = message;
    confidenceText.textContent = "";
    topPredictionsContainer.innerHTML = `<div class="errorMessage">${message}</div>`;
}

async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
        throw new Error("Camera not supported. Use HTTPS or localhost.");
    if (!window.isSecureContext)
        throw new Error("Camera requires secure context (HTTPS or localhost).");

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: 1280, height: 720 }
        });
        video.srcObject   = stream;
        video.muted       = true;
        video.playsInline = true;
        await new Promise(resolve => { video.onloadedmetadata = resolve; });
        await video.play();
        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;

        // Mirror canvas — matches Python cv2.flip(frame, 1)
        mirrorCanvas        = document.createElement("canvas");
        mirrorCanvas.width  = video.videoWidth;
        mirrorCanvas.height = video.videoHeight;
        mirrorCtx           = mirrorCanvas.getContext("2d");

        console.log("camera ready");
    } catch (error) {
        showError("Unable to access camera. Allow camera permission and use HTTPS or localhost.");
        throw error;
    }
}

// =========================
// LOAD ONNX
// =========================

async function loadModel() {
    session = await ort.InferenceSession.create("./model/sign_model.onnx");
    modelInputName  = session.inputNames[0];
    modelOutputName = session.outputNames[0];
    console.log(`ONNX loaded (${modelInputName} -> ${modelOutputName})`);
}

// =========================
// HISTORY
// =========================

function addToHistory(word) {
    historyWords.unshift(word);
    if (historyWords.length > 15) historyWords.pop();
    renderHistory();
}

function renderHistory() {
    historyList.innerHTML = "";
    historyWords.forEach(word => {
        const li = document.createElement("li");
        li.textContent = word;
        historyList.appendChild(li);
    });
}

// =========================
// SPEECH
// =========================

function speakPrediction() {
    if (currentPrediction === "Waiting...") return;
    speechSynthesis.speak(new SpeechSynthesisUtterance(currentPrediction));
}

// =========================
// PREDICTION UI
// =========================

function updatePredictionUI(label, confidence, probs) {
    currentPrediction = label;
    currentConfidence = confidence;
    predictionText.textContent = label;
    confidenceText.textContent = `${confidence.toFixed(1)}%`;
    topPredictionsContainer.innerHTML = probs.map(item =>
        `<div class="predRow"><span>${item.label}</span><span>${item.conf.toFixed(1)}%</span></div>`
    ).join("");
}

// =========================
// HEURISTIC CLASSIFIER
// =========================

// Returns the z-component of the cross product (a→b) × (a→c).
// Positive = counter-clockwise turn, negative = clockwise.
function cross2D(ax, ay, bx, by, cx, cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

// Classify a single detected hand from raw MediaPipe landmarks.
// Returns { palmFacingCamera, fingersTipY } where:
//   palmFacingCamera: true  → palm faces the camera (toward face)
//                    false → back of hand faces camera (away from face)
//   fingersTipY: average y of fingertips (landmarks 8,12,16,20); smaller = higher on screen
function classifyHandOrientation(landmarks, mirrored = true) {
    // Wrist = 0, Index MCP = 5, Pinky MCP = 17
    // Cross product of wrist→index vs wrist→pinky tells us palm facing direction.
    const w  = landmarks[0];
    const im = landmarks[5];
    const pm = landmarks[17];
    const c  = cross2D(w.x, w.y, im.x, im.y, pm.x, pm.y);
    const palmFacingCamera = mirrored ? c > 0 : c < 0;

    const fingersTipY = (landmarks[8].y + landmarks[12].y + landmarks[16].y + landmarks[20].y) / 4;

    // Finger extension: compare tip-to-wrist distance vs MCP-to-wrist distance.
    // Extended fingers have tips far from wrist; curled fingers have tips close.
    function dist(a, b) { return Math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2); }
    const tipDist = (dist(landmarks[8], w) + dist(landmarks[12], w) + dist(landmarks[16], w) + dist(landmarks[20], w)) / 4;
    const mcpDist = (dist(landmarks[5], w) + dist(landmarks[9], w) + dist(landmarks[13], w) + dist(landmarks[17], w)) / 4;
    const tipMcpRatio = mcpDist > 0 ? tipDist / mcpDist : 1;
    const fingersExtended = tipMcpRatio >= 1.4;

    console.log(`[hand] cross=${c.toFixed(4)} palmFacing=${palmFacingCamera} ratio=${tipMcpRatio.toFixed(3)} extended=${fingersExtended}`);

    return { palmFacingCamera, fingersTipY, fingersExtended, tipMcpRatio };
}

// Classify orientation across all detected hands in a frame sequence.
// Uses the most recent frame that has hand landmarks.
function classifyGestureHeuristic(handResultFrames) {
    // Find the last frame that has landmarks
    let lastHandResult = null;
    for (let i = handResultFrames.length - 1; i >= 0; i--) {
        if (handResultFrames[i] && handResultFrames[i].landmarks && handResultFrames[i].landmarks.length > 0) {
            lastHandResult = handResultFrames[i];
            break;
        }
    }
    if (!lastHandResult) return null;

    const count = lastHandResult.landmarks.length;

    if (count === 1) {
        const { tipMcpRatio } = classifyHandOrientation(lastHandResult.landmarks[0], true);

        // Basado en valores reales observados:
        // drink  → ratio ~1.8  (dedos muy extendidos)
        // cool   → ratio ~0.93 (dedos medios)
        // before → ratio ~0.85 (dedos más cerrados)
        if (tipMcpRatio >= 1.4) {
            return { label: "drink", conf: 90 };
        }
        // Zona ambigua entre before y cool (0.845 vs 0.933, muy cercanos)
        if (tipMcpRatio >= 0.87 && tipMcpRatio < 1.4) {
            return { label: "cool", conf: 70, ambiguous: true, top2: ["cool", "before"] };
        }
        return { label: "before", conf: 90 };
    }

    if (count >= 2) {
        // go   → 2 manos, ambas palmas hacia cámara
        // thin → 2 manos, orientaciones mixtas
        const h0 = classifyHandOrientation(lastHandResult.landmarks[0], true);
        const h1 = classifyHandOrientation(lastHandResult.landmarks[1], true);
        const bothFacingCamera = h0.palmFacingCamera && h1.palmFacingCamera;
        if (bothFacingCamera) {
            return { label: "go", conf: 90 };
        }
        return { label: "thin", conf: 90 };
    }

    return null;
}

// =========================
// RUN MODEL
// =========================

async function inferFlattenedSequence(flattened) {
    const tensor = new ort.Tensor("float32", flattened, MODEL_INPUT_SHAPE);
    if (DEBUG_MODE) captureSequenceTensor(tensor);

    const result       = await session.run({ [modelInputName]: tensor });
    const outputTensor = result[modelOutputName];
    const logits       = Array.from(outputTensor.cpuData || outputTensor.data);
    const probs        = softmax(logits);
    const ranked       = probs
        .map((p, i) => ({ label: LABELS[i], conf: p * 100, index: i }))
        .sort((a, b) => b.conf - a.conf);

    return { logits, probs, ranked, best: ranked[0] };
}

// Buffer to store raw hand results per frame for heuristic classification
const gestureHandResults = [];

// =========================
// AMBIGUITY MODAL
// =========================

let ambiguityOverlay, ambiguityBtn0, ambiguityBtn1, ambiguityDismiss;

function initAmbiguityModal() {
    ambiguityOverlay  = document.getElementById("ambiguityOverlay");
    ambiguityBtn0     = document.getElementById("ambiguityBtn0");
    ambiguityBtn1     = document.getElementById("ambiguityBtn1");
    ambiguityDismiss  = document.getElementById("ambiguityDismiss");
    ambiguityDismiss.addEventListener("click", hideAmbiguityModal);
}

function showAmbiguityModal(top2, onChoice) {
    ambiguityBtn0.textContent = top2[0];
    ambiguityBtn1.textContent = top2[1];

    const pick = (label) => {
        hideAmbiguityModal();
        onChoice(label);
    };

    ambiguityBtn0.onclick = () => pick(top2[0]);
    ambiguityBtn1.onclick = () => pick(top2[1]);

    ambiguityOverlay.classList.remove("hidden");
}

function hideAmbiguityModal() {
    ambiguityOverlay.classList.add("hidden");
}

function commitPrediction(label) {
    const now = performance.now();
    updatePredictionUI(label, 90, [{ label, conf: 90 }]);
    if (
        (historyWords.length === 0 || historyWords[0] !== label) &&
        (now - lastHistoryWordTime) >= HISTORY_COOLDOWN_MS
    ) {
        addToHistory(label);
        lastHistoryWordTime = now;
    }
}

function runHeuristicModel(handResultFrames) {
    const result = classifyGestureHeuristic(handResultFrames);
    if (!result) return;

    updatePredictionUI(result.label, result.conf, [{ label: result.label, conf: result.conf }]);

    if (result.ambiguous) {
        showAmbiguityModal(result.top2, (chosen) => {
            commitPrediction(chosen);
        });
        return;
    }

    commitPrediction(result.label);
}

async function runModel(flattened) {
    const { logits, probs, ranked, best } = await inferFlattenedSequence(flattened);

    if (DEBUG_MODE) logInference(logits, probs, best);

    updatePredictionUI(best.label, best.conf, ranked.slice(0, 3));

    const now = performance.now();
    if (
        best.conf >= CONFIDENCE_THRESHOLD &&
        (historyWords.length === 0 || historyWords[0] !== best.label) &&
        (now - lastHistoryWordTime) >= HISTORY_COOLDOWN_MS
    ) {
        addToHistory(best.label);
        lastHistoryWordTime = now;
    }
}

// =========================
// MEDIAPIPE
// =========================

let handLandmarker, poseLandmarker;
let recordedHandLandmarker, recordedPoseLandmarker;
let drawingUtils;

const POSE_INDICES = [11, 12, 13, 14, 15, 16];

async function createMediaPipe() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    const handOpts = {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        },
        numHands: 2,
        minHandDetectionConfidence: 0.3,
        minHandPresenceConfidence:  0.3,
        minTrackingConfidence:      0.3
    };
    const poseOpts = {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        },
        numPoses: 1
    };

    handLandmarker = await HandLandmarker.createFromOptions(vision, { ...handOpts, runningMode: "VIDEO" });
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, { ...poseOpts, runningMode: "VIDEO" });

    recordedHandLandmarker = await HandLandmarker.createFromOptions(vision, { ...handOpts, runningMode: "IMAGE" });
    recordedPoseLandmarker = await PoseLandmarker.createFromOptions(vision, { ...poseOpts, runningMode: "IMAGE" });

    drawingUtils = new DrawingUtils(ctx);
    console.log("MediaPipe ready");
}

// =========================
// BUILD FEATURE VECTOR
// =========================

function buildFeatureVector(handResult, poseResult, mirroredSource = false) {
    const features = new Array(96).fill(0);

    let rightHandDetected = false;
    let leftHandDetected  = false;

    if (handResult && handResult.landmarks) {
        for (let handIdx = 0; handIdx < handResult.landmarks.length; handIdx++) {
            const hand       = handResult.landmarks[handIdx];
            const handedness = handResult.handedness[handIdx][0].categoryName;

            // mirroredSource=true (flipped canvas like Python cv2.flip):
            //   "Left" from MediaPipe = subject's left hand → offset 42
            // mirroredSource=false (raw video, not flipped):
            //   "Right" from MediaPipe = subject's left hand → offset 42
            const isSubjectLeft = mirroredSource
                ? handedness === "Left"
                : handedness === "Right";
            const offset = isSubjectLeft ? 42 : 0;

            if (isSubjectLeft) leftHandDetected  = true;
            else               rightHandDetected = true;

            for (let i = 0; i < 21; i++) {
                features[offset + i * 2]     = hand[i].x;
                features[offset + i * 2 + 1] = hand[i].y;
            }
        }
    }

    if (poseResult && poseResult.landmarks && poseResult.landmarks.length) {
        const pose = poseResult.landmarks[0];
        let poseOffset = 84;
        for (const idx of POSE_INDICES) {
            features[poseOffset++] = pose[idx].x;
            features[poseOffset++] = pose[idx].y;
        }
    }

    normalizeFeatures(features);

    // CRITICAL: reset undetected hand slots to 0.
    // normalizeFeatures converts undetected 0,0 → (-center/scale) which is wrong.
    // Model was trained expecting true zeros for absent hands.
    if (!rightHandDetected) {
        for (let i = 0;  i < 42; i++) features[i] = 0;
    }
    if (!leftHandDetected) {
        for (let i = 42; i < 84; i++) features[i] = 0;
    }

    if (DEBUG_MODE) logLandmarks(handResult, poseResult, features);
    captureFeatureVector(features);

    return features;
}

// =========================
// NORMALIZATION
// =========================

function normalizeFeatures(features) {
    const centerX = (features[84] + features[86]) / 2;
    const centerY = (features[85] + features[87]) / 2;

    const dx = features[84] - features[86];
    const dy = features[85] - features[87];
    let scale = Math.sqrt(dx * dx + dy * dy);
    if (scale < 0.00001) scale = 1;

    for (let i = 0; i < 96; i += 2) {
        features[i]     = (features[i]     - centerX) / scale;
        features[i + 1] = (features[i + 1] - centerY) / scale;
    }
}

// =========================
// SEQUENCE HELPERS
// =========================

function prepareSequenceFrames(frames) {
    if (frames.length <= NUM_FRAMES) {
        // Post-padding with zeros — matches Python prepare_input behavior
        return frames;
    }
    // Downsample if too many frames — matches Python linspace resampling
    const sampled = [];
    for (let i = 0; i < NUM_FRAMES; i++) {
        const idx = Math.round((i / (NUM_FRAMES - 1)) * (frames.length - 1));
        sampled.push(frames[Math.min(idx, frames.length - 1)]);
    }
    return sampled;
}

function flattenFeatureFrames(framesSource) {
    // Float32Array initializes to zeros — provides post-padding automatically
    const padded = new Float32Array(NUM_FRAMES * NUM_FEATURES);
    const frames = prepareSequenceFrames(framesSource);
    for (let i = 0; i < frames.length; i++) {
        padded.set(frames[i], i * NUM_FEATURES);
    }
    return padded;
}

// =========================
// DRAW
// =========================

function drawResults(handResult) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (handResult && handResult.landmarks) {
        for (const landmarks of handResult.landmarks) {
            drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS);
            drawingUtils.drawLandmarks(landmarks);
        }
    }
}

// =========================
// MAIN LOOP
// =========================

function getMirroredFrame() {
    // Flip horizontally — matches Python cv2.flip(frame, 1)
    mirrorCtx.save();
    mirrorCtx.translate(mirrorCanvas.width, 0);
    mirrorCtx.scale(-1, 1);
    mirrorCtx.drawImage(video, 0, 0);
    mirrorCtx.restore();
    return mirrorCanvas;
}

function predictLoop() {
    const now    = performance.now();
    const source = getMirroredFrame();

    const handResult = handLandmarker.detectForVideo(source, now);
    const poseResult = poseLandmarker.detectForVideo(source, now);

    drawResults(handResult);

    const hasHands = !!(
        handResult &&
        handResult.landmarks &&
        handResult.landmarks.length > 0
    );

    const featureVector = buildFeatureVector(
        handResult,
        poseResult,
        true
    );

    if (hasHands) {
        // Live debug: log hand orientation every 30 frames
        if (gestureFrames.length % 30 === 0 && handResult.landmarks.length > 0) {
            const { palmFacingCamera, fingersExtended, tipMcpRatio } = classifyHandOrientation(handResult.landmarks[0], true);
            console.log(`[hand] palmFacing=${palmFacingCamera} ratio=${tipMcpRatio.toFixed(3)} extended=${fingersExtended}`);
        }

        gestureFrames.push(featureVector);
        gestureHandResults.push(handResult);
        noHandFrameCount = 0;
        gestureActive    = true;

        // Cap at 3x NUM_FRAMES to avoid memory bloat on very long gestures
        if (gestureFrames.length > NUM_FRAMES * 3) {
            gestureFrames.shift();
            gestureHandResults.shift();
        }

        predictionText.textContent = `Recording... (${gestureFrames.length}f)`;
        confidenceText.textContent = "";

    } else if (gestureActive) {
        noHandFrameCount++;

        if (noHandFrameCount >= NO_HAND_TRIGGER) {
            gestureActive = false;

            const timeSinceLastInference = now - lastInferenceTime;
            const enoughFrames = gestureFrames.length >= MIN_GESTURE_FRAMES;
            const cooldownPassed = timeSinceLastInference >= INFERENCE_COOLDOWN_MS;

            if (enoughFrames && !inferenceRunning && cooldownPassed) {
                const handFrames = gestureHandResults.slice();
                gestureFrames.length = 0;
                gestureHandResults.length = 0;
                noHandFrameCount     = 0;
                lastInferenceTime    = now;

                predictionText.textContent = "Thinking...";
                confidenceText.textContent = "";

                runHeuristicModel(handFrames);
            } else {
                gestureFrames.length = 0;
                gestureHandResults.length = 0;
                noHandFrameCount     = 0;
            }
        }
    }

    requestAnimationFrame(predictLoop);
}

// =========================
// RECORDED VIDEO TEST
// =========================

function waitForVideoMetadata(videoElement) {
    if (Number.isFinite(videoElement.duration) && videoElement.duration > 0)
        return Promise.resolve();
    return new Promise((resolve, reject) => {
        videoElement.onloadedmetadata = () => resolve();
        videoElement.onerror = () => reject(new Error("Unable to load video."));
    });
}

function waitForSeek(videoElement, time) {
    return new Promise((resolve, reject) => {
        videoElement.onseeked = () => resolve();
        videoElement.onerror  = () => reject(new Error("Unable to seek video."));
        videoElement.currentTime = time;
    });
}

function getRecordedFrameRange() {
    const start     = parseInt(recordedStartFrame.value || "0", 10);
    const parsedEnd = parseInt(recordedEndFrame.value, 10);
    return {
        startFrame: Number.isFinite(start)     ? Math.max(0, start)     : 0,
        endFrame:   Number.isFinite(parsedEnd) ? Math.max(0, parsedEnd) : null
    };
}

function drawRecordedFrame(videoElement) {
    const w = videoElement.videoWidth  || 640;
    const h = videoElement.videoHeight || 480;
    if (recordedProcessingCanvas.width !== w || recordedProcessingCanvas.height !== h) {
        recordedProcessingCanvas.width  = w;
        recordedProcessingCanvas.height = h;
        recordedProcessingCtx = recordedProcessingCanvas.getContext("2d", { willReadFrequently: true });
    }
    recordedProcessingCtx.drawImage(videoElement, 0, 0, w, h);
    return recordedProcessingCanvas;
}

let _recordedStats = { hands: 0, pose: 0, total: 0 };

function extractRecordedFeatureVector(videoElement) {
    const source     = drawRecordedFrame(videoElement);
    const handResult = recordedHandLandmarker.detect(source);
    const poseResult = recordedPoseLandmarker.detect(source);

    _recordedStats.total++;
    if (handResult && handResult.landmarks && handResult.landmarks.length > 0) _recordedStats.hands++;
    if (poseResult && poseResult.landmarks && poseResult.landmarks.length > 0) _recordedStats.pose++;

    return buildFeatureVector(handResult, poseResult, false);
}

async function collectRecordedFramesBySeeking(videoElement) {
    const frames   = [];
    const duration = videoElement.duration;
    const { startFrame, endFrame } = getRecordedFrameRange();
    const startRatio = endFrame === null ? 0 : startFrame / Math.max(1, endFrame);

    for (let i = 0; i < NUM_FRAMES; i++) {
        const ratio = endFrame === null
            ? i / Math.max(1, NUM_FRAMES - 1)
            : startRatio + (i / Math.max(1, NUM_FRAMES - 1)) * (1 - startRatio);
        const time = duration <= 0 ? 0 : ratio * duration;
        await waitForSeek(videoElement, Math.min(time, Math.max(0, duration - 0.001)));
        frames.push(extractRecordedFeatureVector(videoElement));
    }
    return frames;
}

function formatRecordedResult(frameCount, inference) {
    const s = _recordedStats;
    const lines = [
        `Frames extracted: ${frameCount}`,
        `Tensor: [1, ${NUM_FRAMES}, ${NUM_FEATURES}] float32`,
        `Frame preprocessing: native resolution, 64 evenly-spaced frames, IMAGE mode`,
        `Frame range: ${recordedStartFrame.value || 0} to ${recordedEndFrame.value || "end"}`,
        `Detection: hands=${s.hands}/${s.total} (${(s.hands/s.total*100).toFixed(0)}%), pose=${s.pose}/${s.total} (${(s.pose/s.total*100).toFixed(0)}%)`
    ];
    if (s.pose  < s.total * 0.5) lines.push("  ⚠ WARNING: Pose not detected in many frames");
    if (s.hands < s.total * 0.3) lines.push("  ⚠ WARNING: Hands not detected in many frames");
    lines.push("", "Raw logits:", JSON.stringify(inference.logits), "", "Softmax probabilities:");
    inference.probs.forEach((p, i) => lines.push(`${i} ${LABELS[i]}: ${(p*100).toFixed(4)}%`));
    lines.push("", "Ranked:");
    inference.ranked.forEach(item => lines.push(`${item.index} ${item.label}: ${item.conf.toFixed(4)}%`));
    return lines.join("\n");
}

async function testRecordedVideo() {
    if (!recordedVideo.src) {
        recordedVideoResult.textContent = "Choose a video first.";
        return;
    }
    recordedVideoButton.disabled = true;
    recordedVideoResult.textContent = "Processing video...";
    _recordedStats = { hands: 0, pose: 0, total: 0 };

    try {
        await waitForVideoMetadata(recordedVideo);
        const frames    = await collectRecordedFramesBySeeking(recordedVideo);
        recordedVideo.pause();
        if (frames.length === 0) throw new Error("No frames extracted.");
        const flattened = flattenFeatureFrames(frames);
        const inference = await inferFlattenedSequence(flattened);
        recordedVideoResult.textContent = formatRecordedResult(frames.length, inference);
    } catch (error) {
        recordedVideoResult.textContent = `Error: ${error.message}`;
        console.error(error);
    } finally {
        recordedVideoButton.disabled = false;
    }
}

function handleRecordedVideoSelected(event) {
    const file = event.target.files[0];
    if (!file) return;
    if (recordedVideo.src) URL.revokeObjectURL(recordedVideo.src);
    recordedVideo.src = URL.createObjectURL(file);
    recordedVideoResult.textContent = `Loaded: ${file.name}`;
}

// =========================
// START
// =========================

async function startApp() {
    try {
        await setupCamera();
        await createMediaPipe();
        await loadModel();

        if (DEBUG_MODE) {
            window.debugCapture = startDebugCapture;
            window.debugExport  = exportDebugData;
            window.debugAudit   = auditFeatureExtraction;
            window.debugStats   = analyzeFeatureStatistics;
            console.log("[DEBUG] Mode enabled. Functions: debugCapture(), debugExport(), debugAudit(), debugStats()");
        }

        predictLoop();
    } catch (error) {
        console.error("App failed to start:", error);
        if (error && error.message) showError(error.message);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    initializeUI();
    initAmbiguityModal();
    getRequiredElement("speakBtn").addEventListener("click", speakPrediction);
    getRequiredElement("clearBtn").addEventListener("click", () => {
        historyWords = [];
        renderHistory();
    });
    recordedVideoInput.addEventListener("change", handleRecordedVideoSelected);
    recordedVideoButton.addEventListener("click", testRecordedVideo);
    startApp();
});