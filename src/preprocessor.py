import cv2
import numpy as np
from PIL import Image, ImageEnhance


class ImagePreprocessor:
    """Image Preprocessing suite for Handwriting OCR optimization."""

    @staticmethod
    def to_cv2(image_input):
        """Convert PIL Image or file path to OpenCV BGR image"""
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not read image from {image_input}")
            return img
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            return image_input
        else:
            raise TypeError("Unsupported image format")

    @staticmethod
    def to_pil(cv2_image):
        """Convert OpenCV BGR image to PIL Image"""
        if len(cv2_image.shape) == 2:
            return Image.fromarray(cv2_image)
        rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @classmethod
    def enhance_contrast(cls, image_input, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        img = cls.to_cv2(image_input)
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l_channel)
        
        merged = cv2.merge((cl, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @classmethod
    def binarize(cls, image_input, method="otsu", block_size=11, c=2):
        """Convert image to black and white binary mask"""
        img = cls.to_cv2(image_input)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if method.lower() == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method.lower() == "adaptive_mean":
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
        elif method.lower() == "adaptive_gaussian":
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
        else:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @classmethod
    def denoise(cls, image_input, h=10):
        """Remove noise using Non-Local Means Denoising"""
        img = cls.to_cv2(image_input)
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

    @classmethod
    def auto_deskew(cls, image_input):
        """Auto correct angle/skew of text lines"""
        img = cls.to_cv2(image_input)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        
        # Threshold to get text foreground
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        
        if len(coords) < 10:
            return img
            
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        if abs(angle) < 0.5 or abs(angle) > 45:
            return img  # Ignore minimal or extreme rotations
            
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @classmethod
    def crop_to_content(cls, image_input, padding=18, threshold=240):
        """
        Auto crop blank/white margins around text or handwritten strokes.
        Crucial for canvas drawings so small drawings aren't shrunk into oblivion.
        """
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            pil_img = cls.to_pil(cls.to_cv2(image_input))

        np_img = np.array(pil_img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        
        # Identify non-white pixels (foreground strokes)
        coords = np.argwhere(gray < threshold)
        if len(coords) < 10:
            return pil_img  # Nothing or almost empty

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        h, w = gray.shape
        y_min = max(0, int(y_min - padding))
        y_max = min(h, int(y_max + padding))
        x_min = max(0, int(x_min - padding))
        x_max = min(w, int(x_max + padding))

        return pil_img.crop((x_min, y_min, x_max, y_max))

    @classmethod
    def split_words(cls, image_input, min_gap=18, min_word_width=12, padding=12):
        """
        Split a multi-word handwriting image into separate word images based on horizontal whitespace gaps.
        Returns a list of PIL Images (one for each detected word).
        """
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            pil_img = cls.to_pil(cls.to_cv2(image_input))

        np_img = np.array(pil_img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        
        # Binary mask where strokes = 1, background = 0
        binary = (gray < 235).astype(np.uint8)
        
        # Vertical projection (sum of ink along each column)
        col_sums = np.sum(binary, axis=0)
        has_ink = col_sums > 0

        segments = []
        in_segment = False
        seg_start = 0
        gap_size = 0

        for col_idx, active in enumerate(has_ink):
            if active:
                if not in_segment:
                    in_segment = True
                    seg_start = max(0, col_idx - padding)
                gap_size = 0
            else:
                if in_segment:
                    gap_size += 1
                    if gap_size >= min_gap or col_idx == len(has_ink) - 1:
                        seg_end = min(len(has_ink), col_idx - gap_size + padding)
                        if (seg_end - seg_start) >= min_word_width:
                            segments.append((seg_start, seg_end))
                        in_segment = False
                        gap_size = 0

        if in_segment:
            seg_end = len(has_ink)
            if (seg_end - seg_start) >= min_word_width:
                segments.append((seg_start, seg_end))

        if len(segments) <= 1:
            # Only 1 word or couldn't split
            return [cls.crop_to_content(pil_img, padding=padding)]

        # Crop each word segment
        word_crops = []
        for (x1, x2) in segments:
            cropped = pil_img.crop((x1, 0, x2, pil_img.height))
            word_crops.append(cls.crop_to_content(cropped, padding=padding))

        return word_crops

    @classmethod
    def resize_pad(cls, image_input, target_shape=(48, 160)):
        """
        Resize image while maintaining aspect ratio, padding with white background
        target_shape: (height, width)
        """
        img = cls.to_cv2(image_input)
        h, w = img.shape[:2]
        target_h, target_w = target_shape
        
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Create blank canvas (white)
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        
        # Center the image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return canvas
