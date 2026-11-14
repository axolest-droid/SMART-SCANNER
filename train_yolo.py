import os
import shutil
import cv2
from ultralytics import YOLO

def main():
    dataset_dir = os.path.abspath("yolo_dataset")
    img_dir = os.path.join(dataset_dir, "images")
    lbl_dir = os.path.join(dataset_dir, "labels")

    if not os.path.exists(img_dir) or not os.listdir(img_dir):
        print("❌ yolo_dataset/images ফোল্ডারে কোনো ছবি পাওয়া যায়নি! আগে test_hybrid.py চালান।")
        return

    # ১. YOLOv8-এর জন্য data.yaml কনফিগারেশন ফাইল তৈরি করা
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

    print("✅ ১. data.yaml তৈরি সম্পন্ন!")
    print("🧠 ২. YOLOv8 ন্যানো মডেল (yolov8n.pt) ডাউনলোড ও ট্রেইনিং শুরু হচ্ছে...")

    # ২. বেস YOLOv8 মডেল লোড ও আপনার ডেটাসেটে ট্রেইন করানো
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=yaml_path,
        epochs=25,          # ২৫ বার পুরো ডেটাসেট রিভিশন দেবে
        imgsz=640,          # ইমেজ রেজোলিউশন
        batch=2,
        workers=0,          # উইন্ডোজে ক্র্যাশ এড়াতে ওয়ার্কার ০ রাখা হয়েছে
        project="yolo_runs",
        name="q_model",
        exist_ok=True
    )

    # ৩. ট্রেইন হওয়া সেরা মডেলটি (best.pt) মূল ফোল্ডারে নিয়ে আসা
    best_model_path = os.path.join("yolo_runs", "q_model", "weights", "best.pt")
    target_model = "question_detector.pt"
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, target_model)
        print(f"\n🎉 অভিনন্দন! আপনার নিজস্ব AI মডেল তৈরি হয়ে গেছে: D:\\SmartScanner\\{target_model}")
    else:
        target_model = "yolov8n.pt"

    # ৪. নতুন ট্রেইন হওয়া AI মডেল দিয়ে পৃষ্ঠাটি টেস্ট করা
    test_img_path = os.path.join(img_dir, os.listdir(img_dir)[0])
    print(f"🔍 ৩. '{target_model}' দিয়ে টেস্ট স্ক্যান চলছে...")
    trained_ai = YOLO(target_model)
    preds = trained_ai.predict(source=test_img_path, conf=0.20, imgsz=640)

    vis = cv2.imread(test_img_path)
    box_count = 0
    for r in preds:
        for box in r.boxes:
            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            box_count += 1
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 80, 0), 3)
            cv2.putText(vis, f"YOLO {conf:.2f}", (x0 + 4, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 0), 2)

    out_file = "yolo_ai_prediction.jpg"
    cv2.imwrite(out_file, vis)
    print(f"✅ টেস্ট সম্পন্ন! AI মডেল {box_count}টি প্রশ্ন শনাক্ত করেছে। '{out_file}' ওপেন হচ্ছে...")
    try:
        os.startfile(out_file)
    except Exception:
        pass

if __name__ == "__main__":
    main()