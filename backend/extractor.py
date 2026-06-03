import os
import json
import io
import base64
from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _to_base64(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _pdf_to_bytes(file_bytes: bytes) -> bytes:
    import fitz  # pymupdf — no poppler needed
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(dpi=200)
    return pix.tobytes("jpeg")


def extract_document(file_bytes: bytes, filename: str, doc_type: str) -> dict:
    if filename.lower().endswith(".pdf"):
        file_bytes = _pdf_to_bytes(file_bytes)

    b64 = _to_base64(file_bytes)

    prompt = f"""
You are a document parser specializing in {doc_type.replace("_", " ")} documents.
Extract ALL important fields and return ONLY valid JSON, no markdown, no explanation.

Fields to extract:
- vendor_name (string)
- doc_number (string)
- date (string)
- line_items (list of objects with: description, quantity, unit_price, total)
- subtotal (number)
- tax (number)
- total_amount (number)

If a field is not found, use null.
Document type: {doc_type}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }],
        temperature=0,
        max_tokens=2000
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())