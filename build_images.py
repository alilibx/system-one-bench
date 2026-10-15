"""Build the image cases: image_cases.jsonl plus the images it points to under images/.

Each case is one System One request body whose state holds an image, plus gold labels. The image is stored as
{"image": "images/<suite>/<n>.jpg"} and run_decisions.py swaps in a base64 data URL when it sends the case, so the
cases file stays small. Every model gets the same bytes.

Suites (200 cases each):
  img_sst2          the 200 SST-2 reviews from cases.jsonl, rendered as an image, noul      same items as sst2
  img_ar_sentiment  the 200 Arabic tweets from cases.jsonl, rendered as an image, 3 options  same items as ar_sentiment
  img_pets          Oxford-IIIT Pet test, breed, 37 options (choice)
  img_vqa_yesno     VQAv2 validation (first shard), yes/no questions only (noul)
  img_count         CountBenchQA, how many objects, 2 to 10 (score with 9 levels)
  img_refcoco       RefCOCO testA (people) and testB (objects), 100 each: box around the described object

Every image is rotated by its EXIF tag, converted to RGB and resized so it fits the strictest limits we know of:
Jev wants at least 78,400 pixels in all and each side at most 1,568 px, and Clef works best under ~300 KB. Jev's docs
give 256 px as the shortest side, but on 15 October 2026 it answered "Invalid image dimensions" for every image with a
side under 280 px (272 and 274 included), so we keep every side at 280 px or more.
Photos are JPEG; rendered text is PNG. The two rendered suites need cases.jsonl (build_suites.py, build_arabic.py)
and Pillow built with libraqm for Arabic shaping.

Usage: python build_images.py   -> image_cases.jsonl, images/, fonts/
"""
import io
import json
import math
import os
import random
import re
from collections import Counter

import httpx
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont, ImageOps, features

N = 200
SEED = 0
LONG = 1024              # longest side we resize photos down to
MIN_SIDE, MAX_SIDE, MIN_PIXELS = 280, 1568, 78_400  # Jev rejects sides under 280 px, though its docs say 256
MAX_BYTES = 300_000
FONTS = {"latin": "Noto Sans", "arabic": "Noto Naskh Arabic"}  # SIL Open Font License, from Google Fonts
out = []


def sample(rows, n=N, seed=SEED):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def fit(im):
    """Rotate by EXIF, drop alpha, and scale into the size window every model accepts."""
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    s = min(1.0, LONG / max(w, h))
    s = max(s, MIN_SIDE / min(w, h), math.sqrt(MIN_PIXELS / (w * h)))
    if max(w, h) * s > MAX_SIDE:
        return None  # too elongated to satisfy both bounds; skip it
    if abs(s - 1) > 1e-3:
        im = im.resize((math.ceil(w * s), math.ceil(h * s)), Image.LANCZOS)  # round up: the minimums are hard limits
    return im


def save(im, suite, i, fmt="JPEG"):
    os.makedirs(f"images/{suite}", exist_ok=True)
    path = f"images/{suite}/{i}.{'jpg' if fmt == 'JPEG' else 'png'}"
    for q in (88, 82, 75, 68, 60):
        buf = io.BytesIO()
        im.save(buf, fmt, **({"quality": q, "optimize": True} if fmt == "JPEG" else {"optimize": True}))
        if buf.tell() <= MAX_BYTES or fmt != "JPEG":
            break
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    return {"image": path}


def add(suite, i, image, qid, q, gold):
    out.append({"suite": suite, "id": f"{suite}-{i}", "state": [image], "questions": {qid: q},
                "gold": {qid: {"type": q["type"], **gold}}})


# ---------- rendered text: the same items as the text suites, shown as a picture ----------

def font(kind, size):
    path = f"fonts/{FONTS[kind].replace(' ', '')}-Regular.ttf"
    if not os.path.exists(path):
        css = httpx.get("https://fonts.googleapis.com/css2", params={"family": FONTS[kind]},
                        headers={"User-Agent": "Mozilla/4.0"}).text  # an old user agent gets a plain TTF link
        url = re.search(r"url\((https://[^)]+\.ttf)\)", css).group(1)
        os.makedirs("fonts", exist_ok=True)
        with open(path, "wb") as f:
            f.write(httpx.get(url).content)
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM if kind == "arabic" else ImageFont.Layout.BASIC)


def render(text, kind, width=768, size=30, margin=40):
    """Black text on white, wrapped at word boundaries; Arabic is shaped by raqm and set right to left."""
    rtl = kind == "arabic"
    f = font(kind, size)
    opts = {"direction": "rtl", "language": "ar"} if rtl else {}
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if cur and probe.textlength(cand, font=f, **opts) > width - 2 * margin:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    lines.append(cur)
    lh = round(size * 1.6)
    h = max(MIN_SIDE, math.ceil(MIN_PIXELS / width), 2 * margin + lh * len(lines))
    im = Image.new("RGB", (width, h), "white")
    d = ImageDraw.Draw(im)
    for j, line in enumerate(lines):
        y = margin + j * lh
        if rtl:
            d.text((width - margin, y), line, font=f, fill="black", anchor="ra", **opts)
        else:
            d.text((margin, y), line, font=f, fill="black", anchor="la")
    return im


if not features.check("raqm"):
    raise SystemExit("Pillow was built without libraqm, which Arabic shaping needs. On macOS: brew install libraqm")
text_cases = {c["id"]: c for c in (json.loads(l) for l in open("cases.jsonl"))}

for i in range(N):
    c = text_cases[f"sst2-{i}"]
    add("img_sst2", i, save(render(c["state"]["review"], "latin"), "img_sst2", i, "PNG"), "positive",
        {"type": "noul", "instructions": "Does the movie review shown in the image express a positive opinion of the movie?"},
        {"label": c["gold"]["positive"]["label"]})

for i in range(N):
    c = text_cases[f"ar_sentiment-{i}"]
    q = c["questions"]["sentiment"]
    add("img_ar_sentiment", i, save(render(c["state"]["tweet"], "arabic"), "img_ar_sentiment", i, "PNG"), "sentiment",
        {"type": "choice", "instructions": "What is the sentiment of the Arabic tweet shown in the image?", "criteria": q["criteria"]},
        {"label": c["gold"]["sentiment"]["label"]})

# ---------- photos ----------

# Oxford-IIIT Pet, 37 breeds
pets = load_dataset("timm/oxford-iiit-pet", split="test")
names = pets.features["label"].names
CATS = {"abyssinian", "bengal", "birman", "bombay", "british_shorthair", "egyptian_mau", "maine_coon", "persian",
        "ragdoll", "russian_blue", "siamese", "sphynx"}
PETS = {n: f"{n.replace('_', ' ').title()} ({'cat' if n in CATS else 'dog'} breed)" for n in names}
i = 0
for r in sample(pets, len(pets)):
    im = fit(r["image"])
    if im is None:
        continue
    add("img_pets", i, save(im, "img_pets", i), "breed",
        {"type": "choice", "instructions": "Which breed is the pet in the image?", "criteria": PETS}, {"label": names[r["label"]]})
    i += 1
    if i == N:
        break

# VQAv2, yes/no questions, one question per image. The first validation shard keeps the download near 100 MB.
vqa = load_dataset("lmms-lab/VQAv2", data_files={"validation": "data/validation-00000-of-00068.parquet"}, split="validation")
vqa = vqa.filter(lambda r: r["answer_type"] == "yes/no" and r["multiple_choice_answer"] in ("yes", "no"))
seen, i = set(), 0
for r in sample(vqa, len(vqa)):
    if r["image_id"] in seen:
        continue
    im = fit(r["image"])
    if im is None:
        continue
    seen.add(r["image_id"])
    add("img_vqa_yesno", i, save(im, "img_vqa_yesno", i), "answer_yes",
        {"type": "noul", "instructions": f"Looking at the image, is the answer to this question yes? {r['question']}"},
        {"label": int(r["multiple_choice_answer"] == "yes")})
    i += 1
    if i == N:
        break

# CountBenchQA: 2 to 10 objects, an ordered score
cb = load_dataset("vikhyatk/CountBenchQA", split="test")
WORDS = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
i = 0
for r in sample(cb, len(cb)):
    im = fit(r["image"])
    if im is None or not 2 <= r["number"] <= 10:
        continue
    add("img_count", i, save(im, "img_count", i), "count",
        {"type": "score", "instructions": r["question"], "criteria": [f"exactly {w}" for w in WORDS]},
        {"label": r["number"] - 2})
    i += 1
    if i == N:
        break

# RefCOCO testA (people) and testB (other objects): box the described object. One expression per image.
i = 0
for split in ("testA", "testB"):
    ref = load_dataset("lmms-lab/RefCOCO", data_files={split: f"data/{split}-*.parquet"}, split=split)  # skip val and test
    seen, k = set(), 0
    for r in sample(ref, len(ref)):
        if r["file_name"] in seen or not r["answer"]:
            continue
        if r["image"].getexif().get(0x0112, 1) != 1:
            continue  # the COCO box is in stored pixel order, so a rotated photo would no longer match it
        W, H = r["image"].size
        im = fit(r["image"])
        if im is None:
            continue
        seen.add(r["file_name"])
        x, y, w, h = r["bbox"]
        expr = random.Random(SEED + i).choice(r["answer"]).strip()
        add("img_refcoco", i, save(im, "img_refcoco", i), "target",
            {"type": "box", "instructions": f"the person or object in the photo described as “{expr}”"},
            {"box": {"top": y / H, "bottom": (y + h) / H, "left": x / W, "right": (x + w) / W}, "split": split})
        i += 1
        k += 1
        if k == N // 2:
            break

with open("image_cases.jsonl", "w") as f:
    for c in out:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
print(len(out), "image cases", Counter(c["suite"] for c in out))
