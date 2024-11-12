

```markdown
# Robot and Ball Detection

This repository contains a project for detecting robots and balls in images using color classification and contour detection techniques. The project uses OpenCV for image processing and visualization.

## Features
- Classify pixels based on color thresholds.
- Detect and classify robots and balls.
- Draw rectangles around detected objects.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- Google Colab (for `cv2_imshow` patch)

You can install the required libraries using pip:
```bash
pip install opencv-python-headless numpy
```

## Usage

1. **Clone the repository:**
```bash
git clone https://github.com/USERNAME/Robot-and-Ball-Detection.git
cd Robot-and-Ball-Detection
```

2. **Prepare your image and threshold files:**
   Place your image files (e.g., `sample1.bmp`) and threshold files (e.g., `sample1.txt`) in the appropriate directory.

3. **Run the detection script:**
   Modify the `image_file` and `threshold_file` variables in the script to point to your files and run the script:
```python
python script.py
```

## Code Overview

### `classifyImage(image, thresholds)`
Classifies pixels in the image based on the provided color thresholds.

### `sslDetect(image_file, threshold_file)`
Detects robots and balls in the image using the classified colors and draws rectangles around them.

### `classifyPixel(pixel, thresholds, colors)`
Classifies a single pixel based on the color thresholds.

### `getColor(color_class)`
Returns the color value for the given color class.

## Example

### Usage of `sslDetect` function:
```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

def classifyImage(image, thresholds):
    colors = ["Yellow", "Orange", "Pink", "Green", "Blue"]
    classified_image = np.zeros_like(image)

    for i, threshold in enumerate(thresholds):
        lower = np.array(threshold[:3], dtype=np.uint8)
        upper = np.array(threshold[3:], dtype=np.uint8)
        mask = cv2.inRange(image, lower, upper)
        classified_image[mask != 0] = i + 1  # Using color ID as the assigned value
    return classified_image

def sslDetect(image_file, threshold_file):
    image = cv2.imread(image_file)
    thresholds = np.loadtxt(threshold_file, delimiter=",", dtype=np.uint8)

    classified_image = classifyImage(image, thresholds)
    gray_image = cv2.cvtColor(classified_image, cv2.COLOR_BGR2GRAY)

    # Classify robots and balls
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    robot_centers = []
    ball_centers = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        center = (int(x + w/2), int(y + h/2))

        if w > h:
            robot_centers.append(center)
        else:
            ball_centers.append(center)

        # Calculate rectangle coordinates
        rect_width = 20
        rect_height = 20
        x_min = x - rect_width // 2
        y_min = y - rect_height // 2
        x_max = x + rect_width // 2
        y_max = y + rect_height // 2

        # Draw rectangle around the object
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2_imshow(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return robot_centers, ball_centers

# Example usage of sslDetect function with the provided image and thresholds
image_file = "/content/sample1.bmp"
threshold_file = "/content/sample1.txt"

robot_centers, ball_centers = sslDetect(image_file, threshold_file)

print("Robot Centers:", robot_centers)
print("Ball Centers:", ball_centers)
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
```

