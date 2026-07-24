from pathlib import Path
import cv2
import numpy as np


class ImagePreprocessor:

    @staticmethod
    def process(
        image_path: str | Path,
        output_path: str | Path | None = None,

        scale: float = 1.0,
        denoise_strength: int = 0,
        sharpen_strength: float = 0.0,
        clahe_clip: float = 0.0,
        adaptive_threshold: bool = False,
        padding: int = 0,
    ):
        """
        Returns processed image path.
        """

        image_path = str(image_path)

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Cannot open image: {image_path}")

        # -----------------
        # Resize
        # -----------------
        if scale != 1.0:
            image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # -----------------
        # CLAHE
        # -----------------
        if clahe_clip > 0:

            clahe = cv2.createCLAHE(
                clipLimit=clahe_clip,
                tileGridSize=(8, 8),
            )

            gray = clahe.apply(gray)

        # -----------------
        # Denoise
        # -----------------
        if denoise_strength > 0:

            gray = cv2.fastNlMeansDenoising(
                gray,
                None,
                denoise_strength,
                7,
                21,
            )

        # -----------------
        # Sharpen
        # -----------------
        if sharpen_strength > 0:

            blur = cv2.GaussianBlur(gray, (0, 0), 3)

            gray = cv2.addWeighted(
                gray,
                1 + sharpen_strength,
                blur,
                -sharpen_strength,
                0,
            )

        # -----------------
        # Adaptive Threshold
        # -----------------
        if adaptive_threshold:

            gray = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                15,
            )

        # -----------------
        # Padding
        # -----------------
        if padding > 0:

            gray = cv2.copyMakeBorder(
                gray,
                padding,
                padding,
                padding,
                padding,
                cv2.BORDER_CONSTANT,
                value=255,
            )

        if output_path is None:

            output_path = str(
                Path(image_path).with_name(
                    Path(image_path).stem + "_enhanced.png"
                )
            )

        cv2.imwrite(str(output_path), gray)

        return str(output_path)