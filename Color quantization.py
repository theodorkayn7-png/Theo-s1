from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_IMAGE_PATH = BASE_DIR / "images" / "input.jpg"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_IMAGE_PATH = OUTPUT_DIR / "quantized_k3.jpg"

NUMBER_OF_CLUSTERS = 3
MAX_ITERATIONS = 100
EPSILON = 0.2
ATTEMPTS = 10
RANDOM_SEED = 42


# ---------------------------------------------------------
# Image loading
# ---------------------------------------------------------

def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image from disk and convert it from BGR to RGB.

    Args:
        image_path: Path to the input image.

    Returns:
        The loaded image as an RGB NumPy array.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
        ValueError: If the loaded image does not have 3 color channels.
    """

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(
            f"Could not load image from: {image_path}"
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            "The input image must have exactly 3 color channels."
        )

    return image_rgb


# ---------------------------------------------------------
# K-Means color quantization
# ---------------------------------------------------------

def quantize_image(
    image_rgb: np.ndarray,
    number_of_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce the number of image colors using K-Means clustering.

    Each pixel is treated as one sample with three features:
    Red, Green, and Blue.

    Args:
        image_rgb: Input RGB image.
        number_of_clusters: Number of color clusters.

    Returns:
        A tuple containing:
        1. The quantized RGB image.
        2. The final RGB cluster centers.
    """

    height, width, channels = image_rgb.shape

    # Convert the 3D image into a 2D matrix:
    # (height, width, 3) -> (number_of_pixels, 3)
    pixels = image_rgb.reshape(-1, channels)

    # K-Means calculations require floating-point values.
    pixels = pixels.astype(np.float32)

    # Stop K-Means when either:
    # 1. The maximum number of iterations is reached, or
    # 2. Cluster-center movement becomes smaller than EPSILON.
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        MAX_ITERATIONS,
        EPSILON,
    )

    # Use a fixed random seed to make results more reproducible.
    cv2.setRNGSeed(RANDOM_SEED)

    compactness, labels, centers = cv2.kmeans(
        data=pixels,
        K=number_of_clusters,
        bestLabels=None,
        criteria=criteria,
        attempts=ATTEMPTS,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    # Convert the final cluster centers back to valid image values.
    centers = np.clip(np.rint(centers), 0, 255).astype(np.uint8)

    # Replace each original pixel with the color of its assigned cluster center.
    quantized_pixels = centers[labels.flatten()]

    # Restore the original image shape.
    quantized_image = quantized_pixels.reshape(
        height,
        width,
        channels,
    )

    return quantized_image, centers


# ---------------------------------------------------------
# Image saving
# ---------------------------------------------------------

def save_image(image_rgb: np.ndarray, output_path: Path) -> None:
    """
    Save an RGB image to disk.

    OpenCV saves images in BGR order, so the image is converted
    from RGB back to BGR before writing.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    saved_successfully = cv2.imwrite(
        str(output_path),
        image_bgr,
    )

    if not saved_successfully:
        raise IOError(
            f"Could not save image to: {output_path}"
        )


# ---------------------------------------------------------
# Result visualization
# ---------------------------------------------------------

def show_comparison(
    original_image: np.ndarray,
    quantized_image: np.ndarray,
) -> None:
    """
    Display the original and quantized images side by side.
    """

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
    )

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(quantized_image)
    axes[1].set_title(
        f"Quantized Image  K={NUMBER_OF_CLUSTERS}"
    )
    axes[1].axis("off")

    figure.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

def main() -> None:
    """
    Run the complete image color quantization pipeline.
    """

    # 1. Load and validate the input image.
    original_image = load_image(INPUT_IMAGE_PATH)

    height, width, channels = original_image.shape

    print("Input image loaded successfully.")
    print(f"Image path: {INPUT_IMAGE_PATH}")
    print(f"Image shape: {original_image.shape}")
    print(f"Height: {height}")
    print(f"Width: {width}")
    print(f"Channels: {channels}")
    print(f"Data type: {original_image.dtype}")

    # 2. Apply K-Means color quantization.
    quantized_image, cluster_centers = quantize_image(
        image_rgb=original_image,
        number_of_clusters=NUMBER_OF_CLUSTERS,
    )

    print("\nK-Means clustering completed.")
    print(f"Number of clusters: {NUMBER_OF_CLUSTERS}")
    print("Final RGB cluster centers:")

    for index, center in enumerate(cluster_centers, start=1):
        print(f"Cluster {index}: {center.tolist()}")

    # 3. Save the quantized image.
    save_image(
        image_rgb=quantized_image,
        output_path=OUTPUT_IMAGE_PATH,
    )

    print(f"\nOutput image saved to: {OUTPUT_IMAGE_PATH}")

    # 4. Display the original and quantized images.
    show_comparison(
        original_image=original_image,
        quantized_image=quantized_image,
    )


if __name__ == "__main__":
    main()