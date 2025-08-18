import os, io, re, json
from typing import List
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import fitz
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = os.getenv("MODEL_ID", "rednote-hilab/dots.ocr")
DTYPE = os.getenv("TORCH_DTYPE", "bfloat16")

app = FastAPI()
model = None
processor = None
device = "cpu"

def load_model():
    global model, processor, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and DTYPE.lower().startswith("bf") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=dtype, device_map="auto" if device=="cuda" else None
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

@app.on_event("startup")
def _startup():
    load_model()

def pdf_or_image_to_pils(data: bytes, filename: str) -> List[Image.Image]:
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        imgs = []
        for p in doc:
            pix = p.get_pixmap(matrix=fitz.Matrix(200/72, 200/72), alpha=False)
            imgs.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        return imgs
    return [Image.open(io.BytesIO(data)).convert("RGB")]

PROMPT = """
Extrage câmpurile de factură și returnează UN SINGUR obiect JSON valid, fără alt text.

Schema:
{
 "invoice_number": null,
 "issue_date": null,
 "due_date": null,
 "seller": {"name": null, "vat_id": null, "iban": null, "address": null},
 "buyer": {"name": null, "vat_id": null, "address": null},
 "currency": null,
 "line_items": [{"description": null, "quantity": null, "unit_price": null, "line_total": null, "tax_rate": null}],
 "subtotal": null, "tax": null, "total": null
}
Reguli: păstrează limba originală a textelor; nu traduce; nu inventa; dacă lipsesc date, lasă null; numere cu punct zecimal și fără separatori de mii.
"""

def infer(images: List[Image.Image]) -> dict:
    content = [{"type":"image","image":img} for img in images] + [{"type":"text","text":PROMPT}]
    messages = [{"role":"user","content":content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=4096, temperature=0.01)
    resp = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"\{.*\}", resp, flags=re.S)
    payload = m.group(0) if m else resp
    try:
        import json5
        return json5.loads(payload)
    except Exception:
        return json.loads(payload)

class ExtractResponse(BaseModel):
    pages: int
    data: dict

@app.post("/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...)):
    blob = await file.read()
    images = pdf_or_image_to_pils(blob, file.filename)
    data = infer(images)
    return {"pages": len(images), "data": data}
