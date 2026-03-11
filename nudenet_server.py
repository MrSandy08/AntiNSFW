from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import tempfile
import os
import io
from nudenet import NudeDetector
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# ── NudeNet (NSFW/desnudez) ──────────────────────────────────────────────────
detector = NudeDetector()
print("· NudeNet cargado OK")

NSFW_LABELS = {
    'FEMALE_GENITALIA_EXPOSED',
    'MALE_GENITALIA_EXPOSED',
    'FEMALE_BREAST_EXPOSED',
    'ANUS_EXPOSED',
    'BUTTOCKS_EXPOSED',
    'FEMALE_GENITALIA_COVERED',
    'MALE_GENITALIA_COVERED',
}
NSFW_THRESHOLD = 0.55

# ── CLIP (gore / violencia) ───────────────────────────────────────────────────
print("· Cargando CLIP para detección de gore...")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("· CLIP cargado OK")

# Candidatos: cuanto mayor sea la probabilidad de los labels gore, se bloquea
GORE_LABELS  = [
    "gore, blood and guts, extreme violence",
    "graphic injury, mutilation, dead body",
    "animal cruelty, torture",
]
SAFE_LABELS  = [
    "a normal everyday photo",
    "food, landscape, people talking",
]
ALL_LABELS   = GORE_LABELS + SAFE_LABELS
GORE_THRESHOLD = 0.50   # probabilidad mínima de cualquier etiqueta gore para bloquear

def check_gore(image_bytes: bytes) -> tuple[bool, float, str]:
    """Retorna (es_gore, score_max, label_ganadora)"""
    try:
        image  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = clip_processor(text=ALL_LABELS, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = clip_model(**inputs).logits_per_image   # shape [1, n_labels]
        probs  = logits.softmax(dim=-1)[0].tolist()

        # Solo miramos las probs de los labels gore
        gore_probs = probs[:len(GORE_LABELS)]
        max_score  = max(gore_probs)
        max_label  = GORE_LABELS[gore_probs.index(max_score)]
        is_gore    = max_score >= GORE_THRESHOLD
        return is_gore, round(max_score, 3), max_label
    except Exception as e:
        print(f"· CLIP gore error: {e}")
        return False, 0.0, ""


# ── Handler HTTP ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            data   = self.rfile.read(length)

            # ── 1. NudeNet ──
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            tmp.write(data)
            tmp.close()
            try:
                detections = detector.detect(tmp.name)
            except Exception as e:
                detections = []
                print(f'· NudeNet detect error: {e}')
            finally:
                os.unlink(tmp.name)

            nsfw_hits = [
                d for d in detections
                if d.get('class') in NSFW_LABELS and d.get('score', 0) >= NSFW_THRESHOLD
            ]
            nsfw_flag = len(nsfw_hits) > 0

            # ── 2. CLIP gore (solo si NudeNet no encontró nada, para ahorrar tiempo) ──
            gore_flag, gore_score, gore_label = False, 0.0, ""
            if not nsfw_flag:
                gore_flag, gore_score, gore_label = check_gore(data)

            result = {
                'nsfw':       nsfw_flag or gore_flag,
                'nsfw_nudity': nsfw_flag,
                'nsfw_gore':   gore_flag,
                'hits': [{'label': d['class'], 'score': round(d['score'], 3)} for d in nsfw_hits],
                'gore_hit': {'label': gore_label, 'score': gore_score} if gore_flag else None,
                'all':  [{'label': d['class'], 'score': round(d['score'], 3)} for d in detections]
            }

            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)

        except Exception as e:
            print(f'· server error: {e}')
            self.send_response(500)
            self.end_headers()

    def do_GET(self):
        body = b'nudenet+gore ok'
        self.send_response(200)
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)


PORT = int(os.environ.get('NUDENET_PORT', 5000))
print(f'· Servidor escuchando en puerto {PORT}')
HTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
