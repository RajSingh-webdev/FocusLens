# FocusLens — Behavioral Attention Index

Frontend-only React + Vite web app that analyzes webcam facial behavior signals locally in the browser and computes a real-time Behavioral Attention Index (0–100).

## Tech stack
- React + Vite (JavaScript)
- Tailwind CSS
- MediaPipe FaceMesh (web)
- Browser Media APIs (`getUserMedia`)

## Run
```bash
npm install
npm run dev
```

## Folder structure
```text
FocusLens/
├── index.html
├── package.json
├── postcss.config.js
├── tailwind.config.js
├── vite.config.js
└── src/
    ├── App.jsx
    ├── index.css
    └── main.jsx
```

## Notes
- Single-face landmark processing only.
- No backend, no storage, no model training.
- Educational and experimental use only.
