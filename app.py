
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
import os

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_pir_api"] = "0"
from paddleocr import PaddleOCR
from pathlib import Path
from datetime import datetime
import shutil
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from services.image_preprocessor import ImagePreprocessor
app = FastAPI(
    title="PP-OCRv6 Local API",
    version="1.0.0"
)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
# -----------------------------
# Create folders automatically
# -----------------------------
UPLOAD_DIR = Path("uploads")
OUTPUT_IMAGE_DIR = Path("output/images")
OUTPUT_JSON_DIR = Path("output/json")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load OCR Model ONCE
# -----------------------------
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=True,
    use_textline_orientation=False,
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )
# -----------------------------
# OCR Endpoint
# -----------------------------
@app.post("/ocr")
async def detect(file: UploadFile = File(...)):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        extension = Path(file.filename).suffix
        image_path = UPLOAD_DIR / f"{timestamp}{extension}"

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
  
        result = ocr.predict(str(image_path))

        extracted_text = []

        for page in result:
            # 1. Check if 'page' is a dict (or dict-like) containing 'rec_texts'
            if isinstance(page, dict) and "rec_texts" in page:
                extracted_text.extend(page["rec_texts"])
            
            # 2. Check if 'page' is an object with 'rec_texts' attribute
            elif hasattr(page, "rec_texts"):
                extracted_text.extend(page.rec_texts)
                
            # 3. Handle standard PaddleOCR format list of tuples: [ [ [box], ("text", score) ], ... ]
            elif isinstance(page, list):
                for line in page:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        # Extract the text portion: line[1][0]
                        extracted_text.append(line[1][0])

        return {
            "success": True,
            "text": "\n".join(extracted_text),
            "total_lines": len(extracted_text)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )



@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        extension = Path(file.filename).suffix

        # Original upload
        original_path = UPLOAD_DIR / f"{timestamp}{extension}"

        with open(original_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Enhanced output
        enhanced_path = OUTPUT_IMAGE_DIR / f"{timestamp}_enhanced.png"

        ImagePreprocessor.process(
            image_path=original_path,
            output_path=enhanced_path,

            scale=3.0,
            denoise_strength=8,
            sharpen_strength=1.0,
            clahe_clip=2.0,
            adaptive_threshold=False,
            padding=10,
        )

        return {
            "success": True,
            "message": "Image enhanced successfully.",
            "original_image": str(original_path),
            "enhanced_image": str(enhanced_path),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )