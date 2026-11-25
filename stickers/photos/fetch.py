"""Download chosen Openverse originals (CC0 / PDM only) into src/, keeping their credit lines in src/credits.json."""
import json, sys, ssl, urllib.request, os, time
CA = sys.argv[1]
ctx = ssl.create_default_context(cafile=CA)
picks = [p.split(":") for p in sys.argv[2:]]  # name:index
os.makedirs("src", exist_ok=True)
credits = json.load(open("src/credits.json")) if os.path.exists("src/credits.json") else {}
for name, idx in picks:
    r = json.load(open(f"search/{name}.json"))[int(idx)]
    assert r["license"] in ("cc0", "pdm"), r["license"]
    out = f"src/{name}_{idx}.jpg"
    if not os.path.exists(out):
        req = urllib.request.Request(r["url"], headers={"User-Agent": "lumen-sticker-research"})
        for i in range(4):
            try:
                open(out, "wb").write(urllib.request.urlopen(req, context=ctx, timeout=60).read()); break
            except Exception as e:
                print("retry", name, idx, e); time.sleep(4)
    credits[f"{name}_{idx}"] = {k: r.get(k) for k in ("title", "creator", "license", "license_version", "foreign_landing_url", "url", "source", "width", "height")}
    print(name, idx, os.path.getsize(out) if os.path.exists(out) else "missing", r["license"], r.get("creator"))
json.dump(credits, open("src/credits.json", "w"), indent=1, ensure_ascii=False)
