# -*- coding: utf-8 -*-
import os

# ওয়ান-ডিএনএন এবং পিআইআর ক্র্যাশ বন্ধ করার সেটিংস
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import re
import glob
import shutil
import fitz
import cv2
import numpy as np
from ultralytics import YOLO

SC = 2.0


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


def count_option_circles(gray_img, x0_pt, y0_pt, x1_pt, y1_pt):
    x0, y0 = max(0, int(x0_pt * SC)), max(0, int(y0_pt * SC))
    x1, y1 = min(gray_img.shape[1], int(x1_pt * SC)), min(gray_img.shape[0], int(y1_pt * SC))
    if x1 - x0 < 20 or y1 - y0 < 8:
        return 0
    roi = gray_img[y0:y1, x0:x1]
    bw = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = 0
    min_d, max_d = int(6.2 * SC), int(16.8 * SC)
    for c in cnts:
        bx, by, bw_c, bh_c = cv2.boundingRect(c)
        if min_d <= bw_c <= max_d and min_d <= bh_c <= max_d:
            aspect = bw_c / float(max(1, bh_c))
            if 0.70 <= aspect <= 1.40:
                area = cv2.contourArea(c)
                if area > 0.34 * (bw_c * bh_c):
                    circles += 1
    return circles


def build_multipage_dataset_if_needed(img_dir, lbl_dir):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        return

    # যদি মাত্র ১টি বা ০টি ছবি থাকে, তবে পিডিএফের একাধিক পৃষ্ঠা স্ক্যান করে ডেটাসেট সমৃদ্ধ করা
    existing_imgs = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
    if len(existing_imgs) >= 4:
        print(f"✅ ডেটাসেটে ইতিমধ্যে {len(existing_imgs)}টি পৃষ্ঠার ডেটা আছে।")
        return

    try:
        import paddle
        try:
            paddle.set_flags({"FLAGS_use_mkldnn": False})
            paddle.inference.Config.enable_mkldnn = lambda *args, **kwargs: None
        except Exception:
            pass
        from paddleocr import PaddleOCR
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
    except Exception as e:
        print(f"⚠️ অটো-ডেটাসেট জেনারেটর স্কিপ করা হলো ({e})।")
        return

    q_num_re = re.compile(r"^\s*[\(\[]?([0-9]{1,3})(?:[\.\)\]\:\-\?\s…]|[A-Za-z])")
    board_tag_re = re.compile(r"[\[\(].*(?:1[0-9]|2[0-6]).*[\]\)]")

    page_counter = len(existing_imgs)
    for pdf_path in pdf_files:
        doc = fitz.open(pdf_path)
        max_p = min(len(doc), 8)
        print(f"📄 '{pdf_path}' থেকে {max_p}টি পৃষ্ঠা ডেটাসেটে যোগ করা হচ্ছে...")
        for pno in range(max_p):
            page = doc[pno]
            W, H = page.rect.width, page.rect.height
            pix = page.get_pixmap(matrix=fitz.Matrix(SC, SC), alpha=False)
            img_rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

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
            detected_questions = []

            for col_idx, (cx0, cx1) in enumerate(col_defs):
                c_boxes = [
                    b for b in raw_boxes
                    if (b["x0"] + b["x1"]) * 0.5 >= cx0 and (b["x0"] + b["x1"]) * 0.5 <= cx1
                    and 42.0 < b["y0"] < (H - 35.0)
                    and "educationalpdf" not in b["text"].lower()
                    and "rhombus" not in b["text"].lower()
                ]
                if not c_boxes:
                    continue

                left_edges = [b["x0"] for b in c_boxes if (b["x1"] - b["x0"]) > 25.0]
                col_left = float(np.percentile(left_edges, 12)) if left_edges else cx0 + 8.0
                col_right = max(b["x1"] for b in c_boxes)
                col_width = max(100.0, col_right - col_left)

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
                        rows.append({"cy": cy, "x0": b["x0"], "x1": b["x1"], "y0": b["y0"], "y1": b["y1"], "boxes": [b]})

                for r in rows:
                    r["boxes"].sort(key=lambda x: x["x0"])
                    r["text"] = " ".join(x["text"] for x in r["boxes"]).strip()

                q_start_indices = []
                for idx, r in enumerate(rows):
                    first_b = r["boxes"][0]
                    if first_b["x0"] <= col_left + 28.0:
                        m = q_num_re.match(first_b["text"])
                        if m:
                            q_num = int(m.group(1))
                            if 1 <= q_num <= 160 and ((r["x1"] - r["x0"]) > 55.0 or len(r["boxes"]) >= 2):
                                if not q_start_indices or (r["y0"] - rows[q_start_indices[-1]]["y0"]) > 18.0:
                                    q_start_indices.append(idx)

                for q_i, start_idx in enumerate(q_start_indices):
                    end_limit_idx = q_start_indices[q_i + 1] if (q_i + 1) < len(q_start_indices) else len(rows)
                    q_top = rows[start_idx]["y0"] - 2.5
                    q_bot = rows[start_idx]["y1"] + 2.0
                    seen_options = False
                    total_circles = 0
                    opt_rows = 0

                    for r_idx in range(start_idx + 1, end_limit_idx):
                        r = rows[r_idx]
                        non_tag_boxes = [b for b in r["boxes"] if not board_tag_re.search(b["text"])]
                        num_boxes = len(non_tag_boxes)
                        is_indented = (r["x0"] > col_left + 7.0) and (r["x0"] < col_left + col_width * 0.45)
                        has_split_boxes = (num_boxes >= 2) and any(b["x0"] > col_left + col_width * 0.28 for b in non_tag_boxes[1:])
                        has_circle_sym = any(c in r["text"] for c in ("④", "⑧", "⑨", "①", "②", "③"))
                        circle_cnt = count_option_circles(gray, r["x0"] - 4, r["y0"] - 2, r["x1"] + 4, r["y1"] + 2)
                        is_opt_row = has_split_boxes or is_indented or has_circle_sym or (circle_cnt >= 2)
                        returns_to_left = (r["x0"] <= col_left + 7.0)

                        if not seen_options:
                            if is_opt_row:
                                seen_options = True
                                opt_rows = 1
                                total_circles += max(2, circle_cnt)
                                q_bot = r["y1"] + 2.5
                            elif (r["y0"] - q_bot) <= 14.0 and r_idx <= start_idx + 2:
                                q_bot = r["y1"] + 2.0
                            else:
                                break
                        else:
                            if returns_to_left and (total_circles >= 4 or opt_rows >= 2 or (circle_cnt <= 1 and not has_split_boxes)):
                                break
                            elif is_opt_row and (r["y0"] - q_bot) <= 16.0:
                                opt_rows += 1
                                total_circles += max(2, circle_cnt)
                                q_bot = r["y1"] + 2.5
                            else:
                                break

                    qx0 = max(6.0, col_left - 5.0)
                    qx1 = min(W - 6.0, div_fn((q_top + q_bot) * 0.5) - 3.0 if (is_two_col and col_idx == 0) else col_right + 5.0)
                    detected_questions.append((qx0, q_top, qx1, q_bot))

            if len(detected_questions) >= 3:
                page_counter += 1
                yolo_lines = []
                for (x0, y0, x1, y1) in detected_questions:
                    xc = ((x0 + x1) * 0.5) / W
                    yc = ((y0 + y1) * 0.5) / H
                    bw = (x1 - x0) / W
                    bh = (y1 - y0) / H
                    yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                img_name = f"page_{page_counter}.jpg"
                lbl_name = f"page_{page_counter}.txt"
                cv2.imwrite(os.path.join(img_dir, img_name), img_bgr)
                with open(os.path.join(lbl_dir, lbl_name), "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines))
        doc.close()


def find_trained_best_pt(results):
    candidates = []
    if results is not None and hasattr(results, "save_dir"):
        candidates.append(os.path.join(str(results.save_dir), "weights", "best.pt"))
    candidates.extend([
        os.path.join("runs", "detect", "yolo_runs", "q_model", "weights", "best.pt"),
        os.path.join("yolo_runs", "q_model", "weights", "best.pt"),
    ])
    for path in glob.glob(os.path.join("runs", "**", "best.pt"), recursive=True):
        candidates.append(path)
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def clean_and_filter_yolo_boxes(raw_boxes, gray_img):
    """
    ইয়োলো মডেলের দেওয়া ঘিজিমিজি বক্সগুলো থেকে:
    ১. দুই কলাম জুড়ে যাওয়া বা সাদা জায়গার ভুয়া বক্স বাদ দেয়।
    ২. একই প্রশ্নের ওপর থাকা একাধিক ডুপ্লিকেট বক্স মুছে সেরা ১টি বক্স রাখে।
    ৩. কলামের বাম ও ডান মার্জিন অনুযায়ী বক্সের প্রস্থ সমান করে দেয়।
    """
    h_px, w_px = gray_img.shape[:2]
    w_pt, h_pt = w_px / SC, h_px / SC
    div_x_pt, is_two_col, _ = find_column_divider(gray_img, w_pt, h_pt)
    div_px = div_x_pt * SC

    # কালি নির্ণয়ের থ্রেশহোল্ড (সাদা ফাঁকা জায়গার বক্স বাদ দেওয়ার জন্য)
    blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
    ink_bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12) > 0

    col_buckets = {0: [], 1: []}
    for (x0, y0, x1, y1, conf) in raw_boxes:
        bw, bh = x1 - x0, y1 - y0
        # অস্বাভাবিক চওড়া, খুব ছোট বা উপরের/নিচের মার্জিনের বক্স বাদ দেওয়া
        if bw < w_px * 0.18 or bw > w_px * 0.54:
            continue
        if bh < h_px * 0.022 or bh > h_px * 0.35:
            continue
        if y0 < h_px * 0.05 or y1 > h_px * 0.94:
            continue

        # বক্সটি যদি মাঝখানের ডিভাইডার লাইনকে এপার-ওপার কেটে ফেলে তবে তা বাতিল
        if is_two_col and (x0 < div_px - 18) and (x1 > div_px + 18):
            continue

        # বক্সের ভেতরে আসলেই লেখা (কালি) আছে কি না তা পরীক্ষা করা
        roi_ink = ink_bw[max(0, int(y0)):min(h_px, int(y1)), max(0, int(x0)):min(w_px, int(x1))]
        if roi_ink.size == 0 or roi_ink.mean() < 0.015:
            continue

        cx = (x0 + x1) * 0.5
        col_idx = 0 if (not is_two_col or cx < div_px) else 1
        col_buckets[col_idx].append([float(x0), float(y0), float(x1), float(y1), float(conf)])

    final_boxes = []
    for col_idx in (0, 1):
        c_boxes = col_buckets[col_idx]
        if not c_boxes:
            continue

        # কনফিডেন্স (আত্মবিশ্বাস) অনুযায়ী বড় থেকে ছোট সাজানো
        c_boxes.sort(key=lambda b: b[4], reverse=True)
        kept = []
        for b in c_boxes:
            x0, y0, x1, y1, conf = b
            bh = y1 - y0
            duplicate = False
            for kb in kept:
                kx0, ky0, kx1, ky1, _ = kb
                kbh = ky1 - ky0
                # উল্লম্বভাবে (Y-অক্ষে) কতটুকু ওভারল্যাপ করেছে তা দেখা
                inter_y = max(0.0, min(y1, ky1) - max(y0, ky0))
                if inter_y / max(1.0, min(bh, kbh)) > 0.32 or abs(((y0 + y1) - (ky0 + ky1)) * 0.5) < 28.0:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(b)

        # উপর থেকে নিচে সাজানো এবং পরপর দুটি বক্স একটার ভেতর আরেকটা ঢুকে গেলে নিচের সীমানা ঠিক করা
        kept.sort(key=lambda b: b[1])
        col_x0 = int(np.median([b[0] for b in kept]))
        col_x1 = int(np.median([b[2] for b in kept]))

        for i, b in enumerate(kept):
            x0, y0, x1, y1, conf = b
            if i + 1 < len(kept) and y1 > kept[i + 1][1] - 4:
                y1 = max(y0 + 24.0, kept[i + 1][1] - 4.0)
            final_boxes.append((int(min(x0, col_x0)), int(y0), int(max(x1, col_x1)), int(y1), conf))

    return final_boxes


def main():
    dataset_dir = os.path.abspath("yolo_dataset")
    img_dir = os.path.join(dataset_dir, "images")
    lbl_dir = os.path.join(dataset_dir, "labels")

    build_multipage_dataset_if_needed(img_dir, lbl_dir)

    if not os.path.exists(img_dir) or not os.listdir(img_dir):
        print("❌ yolo_dataset/images ফোল্ডারে কোনো ছবি পাওয়া যায়নি!")
        return

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    yaml_content = (
        f"path: {dataset_dir.replace(os.sep, '/')}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: question\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # একদম ফ্রেশ yolov8n.pt থেকে ডকুমেন্ট মোডে (পাতা উল্টানো ও মোজাইক বন্ধ রেখে) ট্রেইনিং
    print("🧠 ফ্রেশ YOLOv8 মডেল ডকুমেন্ট মোডে ট্রেইনিং শুরু হচ্ছে (ফ্লিপ ও মোজাইক বন্ধ)...")
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=yaml_path,
        epochs=60,
        imgsz=1024,
        batch=2,
        workers=0,
        fliplr=0.0,      # পাতা ডানে-বামে উল্টানো সম্পূর্ণ বন্ধ
        flipud=0.0,      # পাতা উপর-নিচে উল্টানো বন্ধ
        mosaic=0.0,      # পাতা ৪ টুকরো করে জোড়া লাগানো বন্ধ
        degrees=0.0,
        scale=0.1,
        project="yolo_runs",
        name="q_model_clean",
        exist_ok=True
    )

    best_model_path = None
    if results is not None and hasattr(results, "save_dir"):
        cand = os.path.join(str(results.save_dir), "weights", "best.pt")
        if os.path.exists(cand):
            best_model_path = cand
    if not best_model_path:
        best_model_path = find_trained_best_pt(results)

    target_model = "question_detector.pt"
    if best_model_path and os.path.exists(best_model_path):
        shutil.copy(best_model_path, target_model)
        print(f"\n🎉 অভিনন্দন! নিখুঁত AI মডেল তৈরি হয়েছে: {os.path.abspath(target_model)}")
    else:
        print("❌ best.pt খুঁজে পাওয়া যায়নি!")
        return

    # নতুন মডেল ও স্মার্ট কলাম ফিল্টার দিয়ে টেস্ট স্ক্যান
    test_img_path = os.path.join(img_dir, sorted(os.listdir(img_dir))[0])
    print(f"🔍 '{target_model}' দিয়ে টেস্ট স্ক্যান চলছে ({test_img_path})...")
    trained_ai = YOLO(target_model)
    preds = trained_ai.predict(source=test_img_path, conf=0.28, iou=0.35, imgsz=1024)

    vis = cv2.imread(test_img_path)
    gray_vis = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

    raw_boxes = []
    for r in preds:
        for box in r.boxes:
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            raw_boxes.append((x0, y0, x1, y1, conf))

    clean_boxes = clean_and_filter_yolo_boxes(raw_boxes, gray_vis)

    for idx, (x0, y0, x1, y1, conf) in enumerate(clean_boxes, 1):
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 80, 0), 3)
        cv2.putText(vis, f"Q{idx} ({conf:.2f})", (x0 + 4, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 0), 2)

    out_file = "yolo_ai_prediction.jpg"
    cv2.imwrite(out_file, vis)
    print(f"✅ টেস্ট সম্পন্ন! ঘিজিমিজি মুছে মোট {len(clean_boxes)}টি পরিষ্কার প্রশ্ন শনাক্ত হয়েছে।")
    try:
        os.startfile(out_file)
    except Exception:
        pass


if __name__ == "__main__":
    main()