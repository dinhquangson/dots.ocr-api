# dots.ocr-api

This repository contains a minimal setup for extracting invoice fields with the [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) model using a CPU-only stack.

## Running the API

Build and start the service with Docker Compose:

```bash
cd invoice-extract
docker compose up -d --build
```

The API will be available at `http://localhost:8000`.

## Example: extract from a PDF

Send a PDF file to the `/extract` endpoint using `curl`:

```bash
curl -F "file=@/path/to/invoice.pdf" http://localhost:8000/extract
```

The response is a JSON object containing the parsed invoice fields.

## Batch processing

Process all PDF or image files in a folder and save the outputs:

```bash
mkdir -p out
for f in /invoices/*.{pdf,jpg,png,jpeg}; do
  [ -e "$f" ] || continue
  b=$(basename "$f")
  curl -s -F "file=@$f" http://localhost:8000/extract > "out/${b%.*}.json"
done
```
