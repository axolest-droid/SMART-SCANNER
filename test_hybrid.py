import os

# ওয়ান-ডিএনএন এবং পিআইআর ক্র্যাশ বন্ধ করার সেটিংস
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import re
import glob
import fitz
import cv2
import numpy as np
import paddle

try:
    paddle.set_flags({"FLAGS_use_mkldnn": False})
    paddle.inference.Config.enable_mkldnn = lambda *args, **kwargs: None
except Exception:
    pass

from paddleocr import PaddleOCR

SC = 2.0  # রেন্ডার স্কেল

# ১. পিডিএফ ফাইল খোঁজা
pdf_files = glob.glob("*.pdf")
if not pdf_files:
    print("❌ এই ফোল্ডারে কোনো পিডিএফ ফাইল পাওয়া যায়নি! একটি পিডিএফ ফাইল রাখুন।")
    exit()

pdf_path = "expr99.pdf" if "expr99.pdf" in pdf_files else ("epr10.pdf" if "epr10.pdf" in pdf_files else pdf_files[0])
print(f"✅ পিডিএফ লোড হয়েছে: {pdf_path}")

doc = fitz.open(pdf_path)
page = doc[0]
W, H = page.rect.width, page.rect.height

pix = page.get_pixmap(matrix=fitz.Matrix(SC, SC), alpha=False)
img_rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ২. কলামের মাঝখানের বিভাজক রেখা (Divider) শনাক্তকরণ
def find_column_divider(gray_img, w_pt, h_pt):
    x0, x1 = int(w_pt * 0.35 * SC), int(w_pt * 0.65 * SC)
    blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)
    roi = thresh[:, x0:x1]
    h_px = gray_img.shape[0]
    pts_y, pts_x = [], []
    for y_start in range(int(0.08 * h_px), int(0.90 * h_px), int(25 * SC)):
        y_end = min(h_px, y_start + int(40 * SC))
        proj = roi[y_start:y_end, :].mean(axis=0)
        best_c = int(np.argmax(proj))
        if proj[best_c] > 95:
            pts_y.append((y_start + y_end) * 0.5 / SC)
            pts_x.append((x0 + best_c) / SC)
    if len(pts_x) >= 3:
        med_x = float(np.median(pts_x))
        inliers = [i for i in range(len(pts_x)) if abs(pts_x[i] - med_x) < 20.0]
        if len(inliers) >= 2:
            slope, intercept = np.polyfit([pts_y[i] for i in inliers], [pts_x[i] for i in inliers], 1)
            slope = float(np.clip(slope, -0.04, 0.04))
            return med_x, True, lambda y: float(slope * float(y) + intercept)
    return w_pt / 2.0, True, lambda _y: w_pt / 2.0


# ৩. ওপেনসিভি দিয়ে অপশনের গোল বৃত্ত (ক, খ, গ, ঘ) গণনা করা
def count_option_circles(gray_img, x0_pt, y0_pt, x1_pt, y1_pt):
    x0, y0 = max(0, int(x0_pt * SC)), max(0, int(y0_pt * SC))
    x1, y1 = min(gray_img.shape[1], int(x1_pt * SC)), min(gray_img.shape[0], int(y1_pt * SC))
    if x1 - x0 < 20 or y1 - y0 < 8:
        return 0
    roi = gray_img[y0:y1, x0:x1]
    bw = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = 0
    min_d, max_d = int(6.5 * SC), int(16.5 * SC)
    for c in cnts:
        bx, by, bw_c, bh_c = cv2.boundingRect(c)
        if min_d <= bw_c <= max_d and min_d <= bh_c <= max_d:
            aspect = bw_c / float(max(1, bh_c))
            if 0.72 <= aspect <= 1.38:
                area = cv2.contourArea(c)
                if area > 0.35 * (bw_c * bh_c):
                    circles += 1
    return circles


print("২. ভিজ্যুয়াল লেআউট ডিটেক্টর চালু হচ্ছে...")
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

results = list(ocr.predict(img_bgr))
raw_boxes = []

for res in results:
    data = res.json.get("res", res) if hasattr(res, "json") and isinstance(res.json, dict) else res
    if isinstance(data, dict) or hasattr(data, "get"):
        texts = data.get("rec_texts", [])
        polys = data.get("dt_polys", data.get("rec_polys", []))
        for poly, txt in zip(polys, texts):
            pts = np.array(poly, dtype=float).reshape(-1, 2) / SC
            bx0, by0 = float(pts[:, 0].min()), float(pts[:, 1].min())
            bx1, by1 = float(pts[:, 0].max()), float(pts[:, 1].max())
            raw_boxes.append({"x0": bx0, "y0": by0, "x1": bx1, "y1": by1, "text": str(txt).strip()})

div_x, is_two_col, div_fn = find_column_divider(gray, W, H)
col_defs = [(10.0, div_x - 2.0), (div_x + 2.0, W - 10.0)] if is_two_col else [(10.0, W - 10.0)]

# প্রশ্ন নম্বর চেনার রেগুলার এক্সপ্রেশন (যেমন: 4., 5.?, 11. -, 13.Cs)
Q_NUM_START_RE = re.compile(r"^\s*[\(\[]?([0-9]{1,3})(?:[\.\)\]\:\-\?\s…]|[A-Za-z])")
BOARD_TAG_RE = re.compile(r"[\[\(].*(?:1[0-9]|2[0-6]).*[\]\)]")

detected_questions = []

for col_idx, (cx0, cx1) in enumerate(col_defs):
    # এই কলামের সব বক্স ফিল্টার করা (উপরের হেডার ও নিচের পেজ নম্বর বাদে)
    c_boxes = [
        b for b in raw_boxes
        if (b["x0"] + b["x1"]) * 0.5 >= cx0 and (b["x0"] + b["x1"]) * 0.5 <= cx1
        and 42.0 < b["y0"] < (H - 35.0)
        and "educationalpdf" not in b["text"].lower()
        and "rhombus" not in b["text"].lower()
    ]
    if not c_boxes:
        continue

    # কলামের বাম মার্জিন নির্ণয়
    left_edges = [b["x0"] for b in c_boxes if (b["x1"] - b["x0"]) > 25.0]
    col_left = float(np.percentile(left_edges, 12)) if left_edges else cx0 + 8.0
    col_right = max(b["x1"] for b in c_boxes)
    col_width = max(100.0, col_right - col_left)

    # কাছাকাছি Y-অবস্থানের বক্সগুলোকে একই সারিতে (Row) সাজানো
    c_boxes.sort(key=lambda b: ((b["y0"] + b["y1"]) * 0.5, b["x0"]))
    rows = []
    for b in c_boxes:
        cy = (b["y0"] + b["y1"]) * 0.5
        if rows and abs(cy - rows[-1]["cy"]) <= 6.5:
            rows[-1]["boxes"].append(b)
            all_b = rows[-1]["boxes"]
            rows[-1]["cy"] = sum((x["y0"] + x["y1"]) * 0.5 for x in all_b) / len(all_b)
            rows[-1]["x0"] = min(x["x0"] for x in all_b)
            rows[-1]["x1"] = max(x["x1"] for x in all_b)
            rows[-1]["y0"] = min(x["y0"] for x in all_b)
            rows[-1]["y1"] = max(x["y1"] for x in all_b)
        else:
            rows.append({
                "cy": cy, "x0": b["x0"], "x1": b["x1"],
                "y0": b["y0"], "y1": b["y1"], "boxes": [b]
            })

    for r in rows:
        r["boxes"].sort(key=lambda x: x["x0"])
        r["text"] = " ".join(x["text"] for x in r["boxes"]).strip()

    # প্রতিটি সারির ভিজ্যুয়াল বৈশিষ্ট্য পরীক্ষা করে প্রশ্ন নম্বর খোঁজা
    q_start_indices = []
    for idx, r in enumerate(rows):
        first_b = r["boxes"][0]
        # বাম মার্জিনের কাছে শুরু হয়েছে কি না
        if first_b["x0"] <= col_left + 28.0:
            m = Q_NUM_START_RE.match(first_b["text"])
            if m:
                q_num = int(m.group(1))
                # প্রশ্ন নম্বর ১ থেকে ১৬০ এবং সারিতে যথেষ্ট লেখা বা একাধিক বক্স থাকতে হবে
                if 1 <= q_num <= 160 and ((r["x1"] - r["x0"]) > 55.0 or len(r["boxes"]) >= 2):
                    if not q_start_indices or (r["y0"] - rows[q_start_indices[-1]]["y0"]) > 18.0:
                        q_start_indices.append(idx)

    # প্রতিটি প্রশ্নের শুরু থেকে অপশন সারির শেষ পর্যন্ত বক্স নির্ধারণ (উত্তর/ব্যাখ্যা বাদ দিয়ে)
    for q_i, start_idx in enumerate(q_start_indices):
        end_limit_idx = q_start_indices[q_i + 1] if (q_i + 1) < len(q_start_indices) else len(rows)
        q_top = rows[start_idx]["y0"] - 2.5
        q_bot = rows[start_idx]["y1"] + 2.0

        seen_options = False

        for r_idx in range(start_idx + 1, end_limit_idx):
            r = rows[r_idx]
            non_tag_boxes = [b for b in r["boxes"] if not BOARD_TAG_RE.search(b["text"])]
            num_boxes = len(non_tag_boxes)

            # সারিটি কি অপশন সারি? (পাশাপাশি ২+ বক্স, অথবা বাম মার্জিন থেকে ডানে সরে শুরু, অথবা গোল অপশন বৃত্ত আছে)
            is_indented = (r["x0"] > col_left + 7.5) and (r["x0"] < col_left + col_width * 0.45)
            has_split_boxes = (num_boxes >= 2) and any(b["x0"] > col_left + col_width * 0.28 for b in non_tag_boxes[1:])
            has_circle_sym = any(c in r["text"] for c in ("④", "⑧", "⑨", "①", "②", "③"))
            circle_cnt = count_option_circles(gray, r["x0"] - 4, r["y0"] - 2, r["x1"] + 4, r["y1"] + 2)

            is_opt_row = has_split_boxes or is_indented or has_circle_sym or (circle_cnt >= 2)

            if not seen_options:
                # যদি অপশন শুরু হওয়ার আগে প্রশ্নের ২য় লাইন বা বোর্ড ট্যাগ থাকে
                if is_opt_row:
                    seen_options = True
                    q_bot = r["y1"] + 2.5
                elif (r["y0"] - q_bot) <= 14.0 and r_idx <= start_idx + 2:
                    q_bot = r["y1"] + 2.0
                else:
                    break
            else:
                # অপশন চলাকালীন পরবর্তী অপশন লাইন যোগ করা, কিন্তু বাম মার্জিনে 'উত্তর:' শুরু হলেই থেমে যাওয়া!
                Returns_to_left_margin = (r["x0"] <= col_left + 7.0) and not has_split_boxes
                if is_opt_row and not Returns_to_left_margin and (r["y0"] - q_bot) <= 16.0:
                    q_bot = r["y1"] + 2.5
                else:
                    # এখানেই 'উত্তর:' বা 'ব্যাখ্যা:' শুরু হয়েছে, তাই সাথে সাথে বক্স কাটা শেষ!
                    break

        qx0 = max(6.0, col_left - 5.0)
        qx1 = min(W - 6.0, div_fn((q_top + q_bot) * 0.5) - 3.0 if (is_two_col and col_idx == 0) else col_right + 5.0)
        detected_questions.append((qx0, q_top, qx1, q_bot))

# ৪. ফলাফল ছবিতে আঁকা এবং ইয়োলো (YOLO) ডেটাসেট হিসেবে সেভ করা
vis_out = img_bgr.copy()
os.makedirs("yolo_dataset/images", exist_ok=True)
os.makedirs("yolo_dataset/labels", exist_ok=True)

yolo_lines = []
for idx, (x0, y0, x1, y1) in enumerate(detected_questions, 1):
    px0, py0, px1, py1 = int(x0 * SC), int(y0 * SC), int(x1 * SC), int(y1 * SC)
    cv2.rectangle(vis_out, (px0, py0), (px1, py1), (0, 0, 230), 3)
    cv2.putText(vis_out, f"Q{idx}", (px0 + 4, py0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 230), 2)

    # YOLO ফরম্যাট: class_id center_x center_y width height (0 থেকে 1 এর মধ্যে)
    xc = ((x0 + x1) * 0.5) / W
    yc = ((y0 + y1) * 0.5) / H
    bw = (x1 - x0) / W
    bh = (y1 - y0) / H
    yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

cv2.imwrite("hybrid_questions.jpg", vis_out)
cv2.imwrite("yolo_dataset/images/page_1.jpg", img_bgr)
with open("yolo_dataset/labels/page_1.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(yolo_lines))

print(f"\n🎯 ম্যাজিক সম্পন্ন! মোট {len(detected_questions)}টি প্রশ্ন নিখুঁতভাবে শনাক্ত হয়েছে (উত্তর ও ব্যাখ্যা বাদে)!")
print("📁 ইয়োলো ট্রেইনিং ডেটা সেভ হয়েছে: yolo_dataset/labels/page_1.txt")
print("🖼 'hybrid_questions.jpg' ছবিটি ওপেন হচ্ছে...")
try:
    os.startfile("hybrid_questions.jpg")
except Exception:
    pass