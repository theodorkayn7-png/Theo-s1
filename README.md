# Theo-s1
An image color quantization project using K-Means clustering to reduce the color palette of an image while preserving its overall visual structure. Built as part of my hands-on journey into Computer Vision and AI Engineering.


# Image Color Quantization with K-Means

A simple and clean Computer Vision project that reduces the number of colors in an image using K-Means clustering.

## What the project does

Each image pixel is treated as one sample with three RGB features.

The pipeline is:

1. Load the image
2. Convert BGR to RGB
3. Reshape the image into a pixel matrix
4. Run K-Means clustering
5. Replace each pixel with its cluster center
6. Rebuild the image
7. Save and display the result

The default number of clusters is:

`K = 3`

## Project structure

- `main.py` — main program
- `images/` — put the input image here as `input.jpg`
- `outputs/` — generated quantized images
- `requirements.txt` — Python dependencies

## Setup

Create and activate a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Put your image here:

`images/input.jpg`

Then run:

```bash
python main.py
```

The output will be saved as:

`outputs/quantized_k3.jpg`

## Main technologies

- Python
- NumPy
- OpenCV
- Matplotlib
- K-Means clustering

## Learning goals

This project demonstrates:

- Image representation as numerical data
- RGB color channels
- Reshaping image data
- K-Means clustering
- Cluster centers
- Pixel-to-cluster assignment
- Image reconstruction
- Basic Computer Vision project structure
