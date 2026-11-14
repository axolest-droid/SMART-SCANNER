import os

# ১. ওয়ান-ডিএনএন (oneDNN/MKLDNN) এবং পিআইআর (PIR) শুরুতেই বন্ধ করা
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import glob
import fitz
import cv2
import numpy as np
import paddle

# ২. প্যাডেল যেন ভেতর থেকে জোর করে ওয়ান-ডিএনএন চালু করতে না পারে তা নিশ্চিত করা
try:
    paddle.set_flags({"FLAGS_use_mkldnn": False})
except Exception:
    pass

try:
    paddle.inference.Config.enable_mkldnn = lambda *args, **kwargs: None
except Exception:
    pass

from paddleocr import PaddleOCR

print("১. ফোল্ডারে পিডিএফ খোঁজা হচ্ছে...")
pdf_path = "epr10.pdf"
if not os.path.exists(pdf_path):
    dl_path = os.path.expanduser("~/Downloads/epr10.pdf")
    if os.path.exists(dl_path):
        pdf_path = dl_path
    else:
        pdfs = glob.glob("*.pdf")
        if pdfs:
            pdf_path = pdfs[0]
        else:
            print("❌ কোনো পিডিএফ ফাইল পাওয়া যায়নি!")
            exit()

print(f"✅ পিডিএফ পাওয়া গেছে: {pdf_path}")

# পিডিএফের প্রথম পৃষ্ঠা ছবিতে রূপান্তর
doc = fitz.open(pdf_path)
page = doc[0]
pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

print("\n২. PaddleOCR v3 ইঞ্জিন চালু হচ্ছে (সিপিইউ মোডে)...")
try:
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
        lang="en"
    )
except Exception:
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en"
    )

print("৩. পৃষ্ঠা স্ক্যান চলছে...")
results = list(ocr.predict(img_bgr))

vis_img = img_bgr.copy()
extracted_lines = []

for res in results:
    data = res.json.get("res", res) if hasattr(res, "json") and isinstance(res.json, dict) else res
    if isinstance(data, dict) or hasattr(data, "get"):
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        polys = data.get("dt_polys", data.get("rec_polys", []))
        for poly, txt, sc in zip(polys, texts, scores):
            extracted_lines.append(f"[{sc:.2f}] {txt}")
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 200, 0), thickness=2)

# ফলাফল ফাইলে সেভ করা
cv2.imwrite("paddle_boxes.jpg", vis_img)
with open("paddle_text.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(extracted_lines))

print(f"\n✅ স্ক্যান সম্পন্ন! মোট {len(extracted_lines)}টি লাইন শনাক্ত হয়েছে।")
print("📄 শনাক্ত করা প্রথম ১৫টি লাইন:\n" + "-" * 40)
for ln in extracted_lines[:15]:
    print(ln)

print("\n🖼 'paddle_boxes.jpg' ফাইলে সবুজ বক্স এঁকে সেভ করা হয়েছে। ছবিটি ওপেন হচ্ছে...")
try:
    os.startfile("paddle_boxes.jpg")
except Exception:
    pass