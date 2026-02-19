import { useEffect, useMemo, useRef, useState } from 'react';
import { FaceMesh } from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';

const CALIBRATION_MS = 3000;
const HISTORY_POINTS = 150;
const SAMPLE_INTERVAL_MS = 200;
const DEFAULT_BASELINE = 0.3;

const safeNumber = (value, fallback = 0) => (Number.isFinite(value) ? value : fallback);
const clamp = (value, min = 0, max = 100) => Math.min(max, Math.max(min, safeNumber(value, min)));
const smooth = (previous, next, alpha = 0.2) => safeNumber(previous) * (1 - alpha) + safeNumber(next) * alpha;

const distance2D = (a, b) => {
  if (!a || !b) return 0;
  const dx = safeNumber(a.x) - safeNumber(b.x);
  const dy = safeNumber(a.y) - safeNumber(b.y);
  return Math.sqrt(dx * dx + dy * dy);
};

const computeEar = (landmarks, outer, inner, upper, lower) => {
  const width = distance2D(landmarks[outer], landmarks[inner]);
  const height = distance2D(landmarks[upper], landmarks[lower]);
  if (width <= 0) return 0;
  return safeNumber(height / width);
};

const engagementLabel = (score) => {
  if (score > 70) return 'High Engagement';
  if (score >= 40) return 'Moderate Engagement';
  return 'Low Engagement';
};

function Gauge({ value }) {
  const radius = 64;
  const circumference = 2 * Math.PI * radius;
  const normalized = clamp(value);
  const dash = (normalized / 100) * circumference;

  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
      <h2 className="mb-4 text-lg font-semibold text-slate-100">Attention Gauge</h2>
      <div className="relative flex items-center justify-center">
        <svg width="180" height="180" viewBox="0 0 180 180" className="-rotate-90">
          <circle cx="90" cy="90" r={radius} stroke="#1e293b" strokeWidth="14" fill="none" />
          <circle
            cx="90"
            cy="90"
            r={radius}
            stroke="#22d3ee"
            strokeWidth="14"
            fill="none"
            strokeLinecap="round"
            strokeDasharray={`${dash} ${circumference}`}
          />
        </svg>
        <div className="absolute text-center">
          <p className="text-4xl font-bold text-cyan-300">{Math.round(normalized)}</p>
          <p className="text-xs uppercase tracking-widest text-slate-400">Score</p>
        </div>
      </div>
    </div>
  );
}

function SignalCard({ title, value }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900 p-4">
      <p className="text-sm text-slate-400">{title}</p>
      <p className="mt-2 text-2xl font-semibold text-slate-100">{Math.round(clamp(value))}</p>
    </div>
  );
}

function ScoreChart({ history }) {
  const points = useMemo(() => {
    if (!history.length) return '';
    return history
      .map((score, idx) => {
        const x = (idx / Math.max(history.length - 1, 1)) * 100;
        const y = 100 - clamp(score);
        return `${x},${y}`;
      })
      .join(' ');
  }, [history]);

  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
      <h2 className="mb-4 text-lg font-semibold text-slate-100">Attention Trend (~30s)</h2>
      <svg viewBox="0 0 100 100" className="h-48 w-full rounded-lg bg-slate-950 p-2">
        <polyline fill="none" stroke="#22d3ee" strokeWidth="1.5" points={points} />
      </svg>
    </div>
  );
}

export default function App() {
  const videoRef = useRef(null);
  const cameraRef = useRef(null);
  const faceMeshRef = useRef(null);
  const calibrationStartRef = useRef(0);
  const baselineSamplesRef = useRef([]);
  const baselineRef = useRef(DEFAULT_BASELINE);
  const lastNoseRef = useRef(null);
  const sampleTickRef = useRef(0);
  const calibratingRef = useRef(false);

  const [running, setRunning] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [eyeActivity, setEyeActivity] = useState(0);
  const [headStability, setHeadStability] = useState(100);
  const [facialMovement, setFacialMovement] = useState(0);
  const [attentionScore, setAttentionScore] = useState(0);
  const [history, setHistory] = useState([]);
  const [status, setStatus] = useState('Idle');

  useEffect(() => {
    calibratingRef.current = calibrating;
  }, [calibrating]);

  useEffect(() => {
    return () => {
      if (cameraRef.current) cameraRef.current.stop();
      if (faceMeshRef.current) faceMeshRef.current.close();
    };
  }, []);

  const processFrame = (landmarks) => {
    const leftEar = computeEar(landmarks, 33, 133, 159, 145);
    const rightEar = computeEar(landmarks, 362, 263, 386, 374);
    const ear = safeNumber((leftEar + rightEar) / 2);
    const eyeScoreRaw = ((ear - 0.14) / 0.2) * 100;

    const nose = landmarks[1];
    let headScoreRaw = 100;
    if (lastNoseRef.current && nose) {
      const movement = distance2D(nose, lastNoseRef.current);
      headScoreRaw = 100 - (movement / 0.03) * 100;
    }
    lastNoseRef.current = nose;

    const mouthHeight = distance2D(landmarks[13], landmarks[14]);
    const mouthWidth = distance2D(landmarks[78], landmarks[308]);
    const ratio = mouthWidth > 0 ? mouthHeight / mouthWidth : baselineRef.current;
    const safeRatio = safeNumber(ratio, baselineRef.current);

    if (calibratingRef.current) {
      baselineSamplesRef.current.push(safeRatio);
      const elapsed = Date.now() - calibrationStartRef.current;
      if (elapsed >= CALIBRATION_MS) {
        const samples = baselineSamplesRef.current;
        const avg = samples.length
          ? samples.reduce((sum, v) => sum + safeNumber(v), 0) / samples.length
          : DEFAULT_BASELINE;
        baselineRef.current = safeNumber(avg, DEFAULT_BASELINE);
        setCalibrating(false);
        setStatus('Tracking attention signals');
      }
      return;
    }

    const baseline = safeNumber(baselineRef.current, DEFAULT_BASELINE);
    const deviation = baseline > 0 ? Math.abs(safeRatio - baseline) / baseline : 0;
    const facialRaw = Math.min((deviation / 0.8) * 100, 100);

    setEyeActivity((prev) => clamp(smooth(prev, eyeScoreRaw)));
    setHeadStability((prev) => clamp(smooth(prev, headScoreRaw)));
    setFacialMovement((prev) => clamp(smooth(prev, facialRaw)));
  };

  useEffect(() => {
    if (!running) return;

    const interval = setInterval(() => {
      sampleTickRef.current += SAMPLE_INTERVAL_MS;
      if (sampleTickRef.current < SAMPLE_INTERVAL_MS) return;
      sampleTickRef.current = 0;
      if (!calibrating && faceDetected) {
        setAttentionScore((prev) => {
          const next = clamp((eyeActivity * 0.4) + (headStability * 0.4) + (facialMovement * 0.2));
          const smoothed = clamp(smooth(prev, next, 0.25));
          setHistory((old) => [...old, smoothed].slice(-HISTORY_POINTS));
          return smoothed;
        });
      }
    }, SAMPLE_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [running, calibrating, eyeActivity, headStability, facialMovement, faceDetected]);

  const start = async () => {
    if (!videoRef.current) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      stream.getTracks().forEach((track) => track.stop());

      const faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });

      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      faceMesh.onResults((results) => {
        const face = results.multiFaceLandmarks?.[0];
        if (!face) {
          setFaceDetected(false);
          setStatus(calibratingRef.current ? 'Calibrating baseline… waiting for face' : 'Face not detected — updates paused');
          return;
        }
        setFaceDetected(true);
        setStatus(calibratingRef.current ? 'Calibrating baseline…' : 'Tracking attention signals');
        processFrame(face);
      });

      faceMeshRef.current = faceMesh;

      cameraRef.current = new Camera(videoRef.current, {
        onFrame: async () => {
          if (faceMeshRef.current) {
            await faceMeshRef.current.send({ image: videoRef.current });
          }
        },
        width: 640,
        height: 480,
      });

      baselineSamplesRef.current = [];
      baselineRef.current = DEFAULT_BASELINE;
      lastNoseRef.current = null;
      calibrationStartRef.current = Date.now();
      setCalibrating(true);
      setFaceDetected(false);
      setEyeActivity(0);
      setHeadStability(100);
      setFacialMovement(0);
      setAttentionScore(0);
      setHistory([]);
      setStatus('Calibrating baseline…');
      await cameraRef.current.start();
      setRunning(true);
    } catch {
      setStatus('Camera access denied or unavailable');
    }
  };

  const stop = () => {
    if (cameraRef.current) cameraRef.current.stop();
    if (faceMeshRef.current) faceMeshRef.current.close();
    cameraRef.current = null;
    faceMeshRef.current = null;
    baselineSamplesRef.current = [];
    lastNoseRef.current = null;
    setRunning(false);
    setCalibrating(false);
    setFaceDetected(false);
    setStatus('Stopped');
  };

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-8 text-slate-200">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="space-y-2">
          <h1 className="text-3xl font-bold text-white">FocusLens — Behavioral Attention Index</h1>
          <p className="text-slate-400">Real-time local analysis of facial behavior signals for attention experimentation.</p>
        </header>

        <section className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
          <div className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-slate-100">Live Webcam</h2>
              <button
                type="button"
                onClick={running ? stop : start}
                className="rounded-lg bg-cyan-500 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-400"
              >
                {running ? 'Stop' : 'Start'}
              </button>
            </div>
            <div className="overflow-hidden rounded-xl border border-slate-700 bg-black">
              <video ref={videoRef} autoPlay playsInline muted className="h-full w-full -scale-x-100" />
            </div>
            <p className="mt-3 text-sm text-slate-400">Status: {status}</p>
            {!faceDetected && running && <p className="mt-1 text-sm text-amber-300">Single face required for active scoring.</p>}
          </div>

          <Gauge value={attentionScore} />
        </section>

        <section className="grid gap-4 md:grid-cols-3">
          <SignalCard title="Eye Activity" value={eyeActivity} />
          <SignalCard title="Head Stability" value={headStability} />
          <SignalCard title="Facial Movement" value={facialMovement} />
        </section>

        <section className="grid gap-6 lg:grid-cols-[1fr_auto]">
          <ScoreChart history={history} />
          <div className="rounded-2xl border border-slate-700 bg-slate-900 p-6 lg:w-72">
            <p className="text-sm uppercase tracking-wider text-slate-400">Engagement State</p>
            <p className="mt-3 text-2xl font-bold text-cyan-300">{engagementLabel(attentionScore)}</p>
          </div>
        </section>

        <footer className="rounded-xl border border-cyan-900/60 bg-cyan-950/30 p-4 text-sm text-cyan-100">
          This project is experimental and educational.
          <br />
          It does not diagnose emotions or mental states.
          <br />
          All processing happens locally in your browser.
        </footer>
      </div>
    </main>
  );
}
