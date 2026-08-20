import os
import sys
import subprocess
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from .config import AppConfig, PROJECT_ROOT
from .preprocessor import ImagePreprocessor


class OCREngine:
    """
    OCR Engine for Vietnamese Handwriting Recognition
    using fine-tuned SVTR model with PaddleOCR.
    """

    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()

    def infer_custom_svtr(self, image_input):
        """Run SVTR inference via PaddleOCR tools/infer_rec.py with UTF-8 encoding"""
        cfg_p = str(self.config.config_path)
        wt_p = str(self.config.weight_path)

        temp_img = None
        if isinstance(image_input, (str, Path)):
            img_path = str(image_input)
        else:
            if isinstance(image_input, np.ndarray):
                pil_img = ImagePreprocessor.to_pil(image_input)
            elif isinstance(image_input, Image.Image):
                pil_img = image_input
            else:
                return None, "Định dạng hình ảnh không hợp lệ"

            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            pil_img.save(temp_file.name)
            img_path = temp_file.name
            temp_img = img_path

        try:
            infer_script = self.config.root_dir / "PaddleOCR" / "tools" / "infer_rec.py"
            if not infer_script.exists():
                return None, f"Không tìm thấy file script: {infer_script}"

            command = [
                sys.executable,
                str(infer_script),
                "-c", cfg_p,
                "-o", f"Global.pretrained_model={wt_p}",
                f"Global.infer_img={img_path}"
            ]

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["FLAGS_enable_pir_api"] = "0"

            process = subprocess.Popen(
                command,
                cwd=str(self.config.root_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="ignore",
                env=env
            )
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                lines = stdout.strip().splitlines()
                texts = []
                scores = []
                for line in lines:
                    if "result:" in line:
                        res_part = line.split("result:", 1)[1].strip()
                        parts = res_part.split("\t")
                        if len(parts) >= 2:
                            text = parts[0].strip()
                            try:
                                score = float(parts[1].strip())
                            except ValueError:
                                score = 1.0
                            if text:
                                texts.append(text)
                                scores.append(score)
                return {"texts": texts, "scores": scores, "raw": stdout}, None
            else:
                return None, stderr
        finally:
            if temp_img and os.path.exists(temp_img):
                try:
                    os.remove(temp_img)
                except Exception:
                    pass

    def recognize(self, image_input):
        """
        Main recognition method.
        """
        if isinstance(image_input, (str, Path)):
            image_input = Image.open(image_input)

        res, err = self.infer_custom_svtr(image_input)
        if res is not None and res.get("texts"):
            return {
                "texts": res["texts"],
                "scores": res["scores"],
                "boxes": [],
                "method": "svtr_custom",
                "raw": res.get("raw", "")
            }

        return {
            "texts": [],
            "scores": [],
            "boxes": [],
            "method": "none",
            "raw": err or "Chưa nhận diện được chữ."
        }
