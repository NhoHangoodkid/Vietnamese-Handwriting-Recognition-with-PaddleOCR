import json
import csv
import io
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_ocr_boxes(image_input, boxes, texts=None, scores=None, color=(0, 255, 0), line_width=2):
    """
    Draw detection boxes and recognition text onto an image.
    image_input: PIL.Image or OpenCV numpy array
    boxes: list of 4-point bounding boxes [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    texts: list of predicted text strings
    scores: list of confidence scores
    """
    if isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("RGB"))
    else:
        img = image_input.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3 and isinstance(image_input, np.ndarray):
            # assume OpenCV BGR if ndarray
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    overlay = img.copy()

    for i, box in enumerate(boxes):
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(255, 60, 60), thickness=line_width)
        cv2.fillPoly(overlay, [pts], (255, 100, 100))

    # Blend polygon overlay
    cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
    return Image.fromarray(img)


def export_to_json(results_list):
    """
    results_list: list of dicts with keys: file_name, text, confidence, boxes
    """
    return json.dumps(results_list, ensure_ascii=False, indent=2)


def export_to_txt(results_list):
    """
    Export all recognized texts to a plain text file format.
    """
    lines = []
    for item in results_list:
        file_name = item.get("file_name", "")
        text = item.get("text", "")
        confidence = item.get("confidence", 0.0)
        lines.append(f"[{file_name}] (Confidence: {confidence:.2%})\n{text}\n")
    return "\n".join(lines)


def export_to_csv(results_list):
    """
    Export results to CSV string.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["File Name", "Recognized Text", "Confidence Score"])
    for item in results_list:
        writer.writerow([
            item.get("file_name", "upload"),
            item.get("text", ""),
            f"{item.get('confidence', 0.0):.4f}"
        ])
    return output.getvalue()


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) using Levenshtein distance.
    """
    r = reference.strip()
    h = hypothesis.strip()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.uint8)
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return float(d[len(r)][len(h)]) / len(r)
