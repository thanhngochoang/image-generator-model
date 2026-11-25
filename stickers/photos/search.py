"""Openverse search (CC0 / public domain photos only) -> numbered contact sheets for picking sticker sources."""
import json, sys, ssl, urllib.request, urllib.parse, io, time, os
from PIL import Image, ImageDraw
CA = sys.argv[1]
ctx = ssl.create_default_context(cafile=CA)
QUERIES = {
    "sunglasses": ["sunglasses", "aviator sunglasses"],
    "heart_glasses": ["heart shaped sunglasses"],
    "crown": ["gold crown", "tiara", "royal crown"],
    "flower_crown": ["flower crown", "floral wreath head"],
    "party_hat": ["party hat"],
    "cat": ["cat face", "kitten portrait"],
    "bunny": ["rabbit portrait", "bunny ears"],
    "dog": ["puppy face", "dog tongue out"],
    "mustache": ["moustache", "fake mustache"],
    "horns": ["devil horns headband", "horns headband"],
}
def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "lumen-sticker-research"})
    for i in range(4):
        try:
            return urllib.request.urlopen(req, context=ctx, timeout=timeout).read()
        except Exception as e:
            time.sleep(3 * (i + 1)); last = e
    raise last
os.makedirs("search", exist_ok=True)
for name, qs in QUERIES.items():
    rows = []
    for q in qs:
        u = "https://api.openverse.org/v1/images/?" + urllib.parse.urlencode({"q": q, "license": "cc0,pdm", "category": "photograph", "page_size": 20, "mature": "false"})
        try:
            d = json.loads(get(u))
        except Exception as e:
            print(name, q, "search failed", e); continue
        for r in d.get("results", []):
            if (r.get("width") or 0) >= 600 and r["id"] not in {x["id"] for x in rows}:
                rows.append(r)
        time.sleep(2)
    json.dump(rows, open(f"search/{name}.json", "w"), indent=1)
    thumbs = []
    for i, r in enumerate(rows[:30]):
        try:
            b = get(r.get("thumbnail") or r["url"], 20)
            im = Image.open(io.BytesIO(b)).convert("RGB"); im.thumbnail((200, 200))
        except Exception:
            im = Image.new("RGB", (200, 200), "gray")
        t = Image.new("RGB", (200, 220), "white"); t.paste(im, ((200 - im.width) // 2, 0))
        ImageDraw.Draw(t).text((4, 204), f"{i} {r['license']} {r.get('width')}x{r.get('height')}", fill="black")
        thumbs.append(t)
    if thumbs:
        cols = 6; sheet = Image.new("RGB", (cols * 200, ((len(thumbs) + cols - 1) // cols) * 220), "white")
        for i, t in enumerate(thumbs): sheet.paste(t, ((i % cols) * 200, (i // cols) * 220))
        sheet.save(f"search/{name}.jpg", quality=80)
    print(name, len(rows))
