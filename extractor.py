# extractor.py
# Handles ALL content types from any PDF:
# → Text (pdfplumber)
# → Tables (pdfplumber)
# → Images (PyMuPDF + Vision AI)
# → Math formulas (text based extraction)

import os
import fitz  # PyMuPDF — for extracting images
import pdfplumber  # for text and tables
import base64  # for converting images to sendable format
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def describe_image_with_ai(image_bytes):
    """
    Sends image to Vision AI for description.
    Returns None if no meaningful figure found.
    Uses NO_FIGURE signal to avoid false positives.
    """
    try:
        img_base64 = base64.b64encode(image_bytes).decode('utf-8')

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": """Analyze this image carefully.

IMPORTANT RULE:
Look specifically for ANY of these:
- Architecture diagrams or flowcharts
- Charts, graphs, or data plots
- Technical figures or visualizations  
- Attention heatmaps or matrices
- Neural network diagrams
- Mathematical plots or figures

If you find ANY visual element — describe it fully:
- What type of figure is it?
- What are the main components?
- What labels or text appear in it?
- What is it demonstrating?

ONLY respond with exactly NO_FIGURE if the image
contains ONLY plain text paragraphs with absolutely
zero diagrams, charts, or visual elements."""
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        description = response.choices[0].message.content

        # reject if Vision AI found no meaningful figure
        if "NO_FIGURE" in description.upper():
            print(f"    ↳ No meaningful figure detected — skipping")
            return None

        return description

    except Exception as e:
        print(f"    ↳ Vision AI error: {e}")
        return None
def extract_images_from_pdf(pdf_path):
    image_descriptions = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]

        # DEBUG
        image_list = page.get_images(full=True)
        drawings = page.get_drawings()
        print(f"  Page {page_num+1}: {len(image_list)} bitmaps, {len(drawings)} drawings")

        # Method 1 — bitmap images
        found_images = False
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                if len(image_bytes) < 5000:
                    continue
                print(f"  🖼️ Processing bitmap image on page {page_num+1}...")
                description = describe_image_with_ai(image_bytes)
                image_descriptions.append({
                    "page": page_num + 1,
                    "type": "bitmap",
                    "description": description
                })
                found_images = True
            except Exception as e:
                continue

       # Method 2 — vector graphics
        if not found_images:
            try:
                if len(drawings) > 10:
                    print(f"  🖼️ Processing vector graphics on page {page_num+1}...")

            # try cropping to figure area first
                    try:
                        all_rects = [d['rect'] for d in drawings
                             if 'rect' in d]

                        if all_rects:
                            x0 = min(r[0] for r in all_rects)
                            y0 = min(r[1] for r in all_rects)
                            x1 = max(r[2] for r in all_rects)
                            y1 = max(r[3] for r in all_rects)

                            clip = fitz.Rect(
                                max(0, x0 - 10),
                                max(0, y0 - 10),
                                x1 + 10,
                                y1 + 10
                    )

                            mat = fitz.Matrix(3, 3)
                            pix = page.get_pixmap(matrix=mat, clip=clip)
                            img_bytes = pix.tobytes("jpeg")

                    except Exception:
                # fallback to full page render
                        mat = fitz.Matrix(3, 3)
                        pix = page.get_pixmap(matrix=mat)
                        img_bytes = pix.tobytes("jpeg")

                    description = describe_image_with_ai(img_bytes)

                    if description is None:
                        continue

                    image_descriptions.append({
                        "page": page_num + 1,
                        "type": "vector_graphic",
                        "description": description
            })

            except Exception as e:
                continue

    doc.close()
    return image_descriptions
def extract_full_document(pdf_path, extract_images=True):
    """
    Master extraction function
    Extracts EVERYTHING from a PDF:
    → Text
    → Tables  
    → Images (optional — slower but thorough)
    
    Returns structured content dictionary
    """
    print(f"\n📄 Extracting: {os.path.basename(pdf_path)}")

    result = {
        "filename": os.path.basename(pdf_path),
        "pages": [],
        "full_text": "",
        "table_count": 0,
        "image_count": 0
    }

    # ── EXTRACT TEXT AND TABLES WITH PDFPLUMBER ──────
    print("  📝 Extracting text and tables...")

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_content = {
                "page_number": page_num + 1,
                "text": "",
                "tables": []
            }

            # extract text
            text = page.extract_text()
            if text:
                page_content["text"] = text.strip()
                result["full_text"] += f"\n--- Page {page_num+1} ---\n{text}\n"

            # extract tables
            tables = page.extract_tables()
            if tables:
                for table_idx, table in enumerate(tables):
                    if not table:
                        continue

                    # convert table to readable text
                    table_text = f"\n[TABLE {table_idx+1} on Page {page_num+1}]\n"

                    for row in table:
                        # clean up None values
                        clean_row = [
                            str(cell).strip() if cell else ""
                            for cell in row
                        ]
                        table_text += " | ".join(clean_row) + "\n"

                    table_text += f"[END TABLE {table_idx+1}]\n"

                    page_content["tables"].append(table_text)
                    result["full_text"] += table_text
                    result["table_count"] += 1

            result["pages"].append(page_content)

    print(f"  ✅ Text extracted: {len(result['full_text'])} characters")
    print(f"  ✅ Tables found: {result['table_count']}")

    # ── EXTRACT AND DESCRIBE IMAGES ──────────────────
    if extract_images:
        print("  🖼️ Extracting and analyzing images...")
        image_descriptions = extract_images_from_pdf(pdf_path)
        result["image_count"] = len(image_descriptions)

        if image_descriptions:
            result["full_text"] += "\n--- IMAGES AND FIGURES ---\n"
            for img_data in image_descriptions:
                img_text = f"""
[IMAGE on Page {img_data['page']}]
{img_data['description']}
[END IMAGE]
"""
                result["full_text"] += img_text

        print(f"  ✅ Images processed: {result['image_count']}")

    print(f"\n✅ Extraction complete!")
    print(f"   Pages: {len(result['pages'])}")
    print(f"   Tables: {result['table_count']}")
    print(f"   Images: {result['image_count']}")
    print(f"   Total content: {len(result['full_text'])} characters")

    return result


def get_extraction_summary(result):
    """
    Returns a human readable summary of what was extracted
    Shown to user in the UI
    """
    return f"""
📊 Document Analysis Complete:
- Pages processed: {len(result['pages'])}
- Tables extracted: {result['table_count']}
- Images analyzed: {result['image_count']}
- Total content: {len(result['full_text']):,} characters
"""