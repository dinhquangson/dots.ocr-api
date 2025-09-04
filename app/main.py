import os, io, re, json
from typing import List
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import fitz
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_cached_module_file

from pathlib import Path

import flash_attn_stub  # noqa: F401 - ensure flash_attn is stubbed for CPU-only use

MODEL_ID = os.getenv("MODEL_ID", "rednote-hilab/dots.ocr")

app = FastAPI()
model = None
processor = None


def load_model():
    global model, processor

    # Work around a transformers quirk when the repo name contains dots.
    # The "dots.ocr" repository is cached under a folder with the same name,
    # while transformers tries to import it as "dots", causing a
    # ``ModuleNotFoundError``.  We pre-download the module and create an
    # alias without dots so the import machinery can locate it.
    repo_name = MODEL_ID.split("/")[-1]
    if "." in repo_name:
        try:
            # Download one of the modeling files to ensure the repo is cached
            cached = Path(get_cached_module_file(MODEL_ID, "modeling_dots_ocr.py"))
            src = cached.parents[2] / repo_name  # .../rednote-hilab/dots.ocr
            dst = cached.parents[2] / repo_name.split(".")[0]
            if src.exists() and not dst.exists():
                dst.symlink_to(src, target_is_directory=True)
        except Exception:
            # If anything goes wrong we still attempt to load the model normally
            pass

    model = AutoModelForCausalLM.from_pretrained(
        "rednote-hilab/dots.ocr",
        trust_remote_code=True,
        token="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"
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True,
        token="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"
    )

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
