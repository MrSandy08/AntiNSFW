from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json, tempfile, os, io
from nudenet import NudeDetector
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# ── NudeNet ──────────────────────────────────────────────────────────────────
detector = NudeDetector()
print("· NudeNet cargado OK")

NSFW_LABELS = {
    'FEMALE_GENITALIA_EXPOSED',
    'MALE_GENITALIA_EXPOSED',
    'FEMALE_BREAST_EXPOSED',
    'ANUS_EXPOSED',
    'BUTTOCKS_EXPOSED',
}
NSFW_THRESHOLD = 0.55

# ── CLIP (gore) ───────────────────────────────────────────────────────────────
print("· Cargando CLIP...")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("· CLIP cargado OK")

GORE_LABELS = [
    "gore, blood and guts, extreme violence",
    "graphic injury, mutilation, dead body",
    "animal cruelty, torture",
]
SAFE_LABELS = [
    "a normal everyday photo",
    "food, landscape, people talking",
]
ALL_LABELS     = GORE_LABELS + SAFE_LABELS
GORE_THRESHOLD = 0.50

def check_gore(image_bytes: bytes) -> tuple[bool, float, str]:
    try:
        image  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = clip_processor(text=ALL_LABELS, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = clip_model(**inputs).logits_per_image
        probs      = logits.softmax(dim=-1)[0].tolist()
        gore_probs = probs[:len(GORE_LABELS)]
        max_score  = max(gore_probs)
        max_label  = GORE_LABELS[gore_probs.index(max_score)]
        return max_score >= GORE_THRESHOLD, round(max_score, 3), max_label
    except Exception as e:
        print(f"· CLIP error: {e}")
        return False, 0.0, ""

# ── Handler ───────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass  # silenciar logs de cada request

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            data   = self.rfile.read(length)
            if not data:
                self._json(400, {'error': 'empty body'})
                return

            # Intentar abrir como imagen antes de pasarla a NudeNet
            try:
                Image.open(io.BytesIO(data)).verify()
            except Exception:
                self._json(400, {'error': 'invalid image'})
                return

            # ── NudeNet ──
            detections = []
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            try:
                # Guardar como JPEG normalizado (evita errores con WebP/PNG raros)
                img = Image.open(io.BytesIO(data)).convert('RGB')
                img.save(tmp.name, 'JPEG', quality=90)
                tmp.close()
                detections = detector.detect(tmp.name)
            except Exception as e:
                print(f'· NudeNet detect error: {e}')
            finally:
                try: os.unlink(tmp.name)
                except: pass

            nsfw_hits = [
                d for d in detections
                if d.get('class') in NSFW_LABELS and d.get('score', 0) >= NSFW_THRESHOLD
            ]
            nsfw_flag = len(nsfw_hits) > 0

            # ── CLIP gore (solo si NudeNet no encontró nudidad) ──
            gore_flag, gore_score, gore_label = False, 0.0, ""
            if not nsfw_flag:
                gore_flag, gore_score, gore_label = check_gore(data)

            result = {
                'nsfw':        nsfw_flag or gore_flag,
                'nsfw_nudity': nsfw_flag,
                'nsfw_gore':   gore_flag,
                'hits': [{'label': d['class'], 'score': round(d['score'], 3)} for d in nsfw_hits],
                'gore_hit': {'label': gore_label, 'score': gore_score} if gore_flag else None,
                'all':  [{'label': d['class'], 'score': round(d['score'], 3)} for d in detections],
            }
            self._json(200, result)

        except Exception as e:
            print(f'· handler error: {e}')
            self._json(500, {'error': str(e)})

    def do_GET(self):
        # Health check — Railway espera 200 aquí
        body = b'nudenet+clip ok'
        self.send_response(200)
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

PORT = int(os.environ.get('PORT', os.environ.get('NUDENET_PORT', 5000)))
print(f'· Servidor escuchando en :{PORT}')
ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
