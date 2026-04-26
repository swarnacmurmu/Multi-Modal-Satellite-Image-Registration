# from pathlib import Path
# from fastapi import FastAPI, Request, UploadFile, File
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates

# from backend.services.predictor import Predictor

# app = FastAPI(title="SAR Optical Registration App")

# BASE_DIR = Path(__file__).resolve().parents[1]

# templates = Jinja2Templates(directory=str(BASE_DIR / "frontend" / "templates"))

# app.mount(
#     "/static",
#     StaticFiles(directory=str(BASE_DIR / "frontend" / "static")),
#     name="static"
# )

# app.mount(
#     "/outputs",
#     StaticFiles(directory=str(BASE_DIR / "outputs")),
#     name="outputs"
# )

# predictor = None


# @app.on_event("startup")
# def startup_event():
#     global predictor
#     predictor = Predictor()


# @app.get("/", response_class=HTMLResponse)
# def home(request: Request):
#     return templates.TemplateResponse(
#         request=request,
#         name="index.html",
#         context={"request": request}
#     )


# @app.post("/predict")
# async def predict(moving_file: UploadFile = File(...), fixed_file: UploadFile = File(...)):
#     moving_bytes = await moving_file.read()
#     fixed_bytes = await fixed_file.read()
#     result = predictor.predict(moving_bytes, fixed_bytes)
#     return result











from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.services.predictor import Predictor

app = FastAPI(title="SAR Optical Registration App")

BASE_DIR = Path(__file__).resolve().parents[1]

app.mount(
    "/outputs",
    StaticFiles(directory=str(BASE_DIR / "outputs")),
    name="outputs"
)

predictor = None


@app.on_event("startup")
def startup_event():
    global predictor
    predictor = Predictor()


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SAR Optical Registration</title>

<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Arial, sans-serif;
    min-height: 100vh;
    background: linear-gradient(135deg, #020617, #0f172a, #1e3a8a, #312e81);
    color: white;
}

.container {
    width: 94%;
    max-width: 1180px;
    margin: 40px auto;
}

.hero {
    background: rgba(15, 23, 42, 0.85);
    border-radius: 28px;
    padding: 38px;
    box-shadow: 0 20px 55px rgba(0,0,0,0.35);
    margin-bottom: 28px;
    border: 1px solid rgba(147,197,253,0.25);
}

.badge {
    display: inline-block;
    background: rgba(59,130,246,0.25);
    color: #93c5fd;
    padding: 9px 16px;
    border-radius: 999px;
    font-weight: 800;
    margin-bottom: 18px;
}

h1 {
    font-size: 48px;
    margin-bottom: 16px;
}

.hero p {
    color: #dbeafe;
    font-size: 18px;
    line-height: 1.7;
}

.upload-card {
    background: rgba(15, 23, 42, 0.85);
    border-radius: 28px;
    padding: 32px;
    box-shadow: 0 20px 55px rgba(0,0,0,0.30);
    border: 1px solid rgba(147,197,253,0.25);
}

.upload-card h2 {
    margin-bottom: 24px;
}

.upload-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 22px;
    margin-bottom: 24px;
}

.upload-box {
    background: rgba(30,41,59,0.85);
    border: 2px dashed rgba(147,197,253,0.45);
    border-radius: 22px;
    padding: 28px;
    transition: 0.25s ease;
}

.upload-box:hover {
    transform: translateY(-5px);
    border-color: #60a5fa;
    background: rgba(37,99,235,0.20);
}

.icon {
    font-size: 42px;
    display: block;
    margin-bottom: 10px;
}

.upload-title {
    font-size: 22px;
    font-weight: 900;
    display: block;
}

.upload-desc {
    color: #bfdbfe;
    display: block;
    margin: 8px 0 12px;
}

input[type="file"] {
    color: #dbeafe;
}

button {
    border: none;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    padding: 15px 30px;
    border-radius: 16px;
    font-weight: 900;
    font-size: 16px;
    cursor: pointer;
    box-shadow: 0 14px 32px rgba(37,99,235,0.45);
}

button:hover {
    transform: translateY(-2px);
}

#status {
    margin-top: 18px;
    color: #93c5fd;
    font-weight: 900;
}

.result-title {
    margin: 32px 0 18px;
    font-size: 30px;
}

.result-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 22px;
}

.result-card {
    background: rgba(15,23,42,0.88);
    border-radius: 24px;
    padding: 18px;
    border: 1px solid rgba(147,197,253,0.25);
    box-shadow: 0 18px 45px rgba(0,0,0,0.28);
}

.result-card h3 {
    margin-bottom: 12px;
}

.result-card img {
    width: 100%;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.22);
}

.result-card p {
    color: #cbd5e1;
    margin-top: 10px;
}

.good {
    border-color: rgba(34,197,94,0.6);
}

.wide {
    grid-column: span 2;
}

.matrix-card {
    margin-top: 24px;
    background: rgba(15,23,42,0.88);
    border-radius: 24px;
    padding: 22px;
    border: 1px solid rgba(147,197,253,0.25);
}

pre {
    background: #020617;
    color: #e5e7eb;
    padding: 16px;
    border-radius: 14px;
    overflow-x: auto;
}

.error-box {
    margin-top: 24px;
    background: rgba(127,29,29,0.55);
    color: #fecaca;
    padding: 18px;
    border-radius: 14px;
}

@media(max-width: 850px) {
    .upload-grid,
    .result-grid {
        grid-template-columns: 1fr;
    }

    .wide {
        grid-column: span 1;
    }

    h1 {
        font-size: 34px;
    }
}
</style>
</head>

<body>
<div class="container">

    <section class="hero">
        <span class="badge">AI Powered Image Registration</span>
        <h1>SAR Optical Image Registration</h1>
        <p>
            Upload Sentinel-1 SAR as the moving image and Sentinel-2 Optical as the fixed reference.
            The trained model aligns the SAR image and shows before-after registration results.
        </p>
    </section>

    <section class="upload-card">
        <h2>Upload Image Pair</h2>

        <form id="uploadForm">
            <div class="upload-grid">
                <label class="upload-box">
                    <span class="icon">🛰️</span>
                    <span class="upload-title">Moving Image</span>
                    <span class="upload-desc">SAR / Sentinel-1</span>
                    <input type="file" name="moving_file" accept="image/*" required>
                </label>

                <label class="upload-box">
                    <span class="icon">🌍</span>
                    <span class="upload-title">Fixed Image</span>
                    <span class="upload-desc">Optical / Sentinel-2</span>
                    <input type="file" name="fixed_file" accept="image/*" required>
                </label>
            </div>

            <button type="submit">Run Registration</button>
        </form>

        <div id="status"></div>
    </section>

    <section id="result"></section>

</div>

<script>
const form = document.getElementById("uploadForm");
const statusDiv = document.getElementById("status");
const resultDiv = document.getElementById("result");

form.addEventListener("submit", async function(event) {
    event.preventDefault();

    statusDiv.innerHTML = "⏳ Running registration...";
    resultDiv.innerHTML = "";

    const formData = new FormData(form);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(await response.text());
        }

        const data = await response.json();
        const t = Date.now();

        statusDiv.innerHTML = "✅ Registration completed successfully";

        resultDiv.innerHTML = `
            <h2 class="result-title">Registration Result</h2>

            <div class="result-grid">
                <div class="result-card">
                    <h3>Input SAR</h3>
                    <img src="${data.input_sar}?t=${t}">
                </div>

                <div class="result-card">
                    <h3>Fixed Optical</h3>
                    <img src="${data.fixed_optical}?t=${t}">
                </div>

                <div class="result-card">
                    <h3>Before Alignment</h3>
                    <img src="${data.before_overlay}?t=${t}">
                    <p>SAR before registration over optical image.</p>
                </div>

                <div class="result-card good">
                    <h3>After Alignment</h3>
                    <img src="${data.after_overlay}?t=${t}">
                    <p>Warped SAR after registration over optical image.</p>
                </div>

                <div class="result-card wide">
                    <h3>Final Warped SAR Output</h3>
                    <img src="${data.warped_sar}?t=${t}">
                </div>
            </div>

            <div class="matrix-card">
                <h3>Predicted Affine Matrix</h3>
                <pre>${JSON.stringify(data.theta, null, 2)}</pre>
            </div>
        `;
    } catch (error) {
        statusDiv.innerHTML = "❌ Registration failed";
        resultDiv.innerHTML = `<pre class="error-box">${error.message}</pre>`;
    }
});
</script>

</body>
</html>
"""


@app.post("/predict")
async def predict(moving_file: UploadFile = File(...), fixed_file: UploadFile = File(...)):
    moving_bytes = await moving_file.read()
    fixed_bytes = await fixed_file.read()
    result = predictor.predict(moving_bytes, fixed_bytes)
    return result