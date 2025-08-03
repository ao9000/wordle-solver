import cv2

Y_THRESH = 15  # Tolerance for y center alignment for grouping boxes into rows




def filter_non_square_contours(cnts):
    # Define temp list
    square_indexes = []

    # Contour Approximation to find squares
    for cnt_index, cnt in enumerate(cnts, start=0):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        # Check if square
        if len(approx) == 4:
            # # Get width & height of contour
            # _, _, width, height = cv2.boundingRect(approx)

            # Change to rotatedRect
            rect = cv2.minAreaRect(approx)
            width, height = rect[1]

            # Compute aspect ratio
            aspect_ratio = width / height

            # Square will have aspect ratio of around 1
            tol = 0.15  # Tolerance for aspect ratio
            if (1.0 - tol) <= aspect_ratio <= (1.0 + tol):
                # Append into list
                square_indexes.append(cnt_index)

    # Filter list to only contain square contours
    cnts = [cnt for cnt_index, cnt in enumerate(cnts) if cnt_index in square_indexes]

    return cnts


def get_wordle_grid_boxes(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding to create a binary image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

    # Find contours
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to keep only square-like shapes
    cnts = filter_non_square_contours(cnts)

    # Try to detect wordle 6x5 grid squares
    # Grid cnts attributes:
    # The grid lives in the upper part of the image.
    # It consists of exactly 6 rows of 5 boxes all aligned on the same 6 y-levels

    # Convert contours to bounding rectangles
    boxes = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        cy = y + h / 2
        boxes.append((cnt, x, y, w, h, cy))

    # Attributes of each box in boxes
    # cnt = Contour obj, x = top left corner of cnt, y = top left corner of cnt,
    # w = width of cnt, h = height of cnt, cy = y center of cnt

    # Extract the y axis center of each box, group them into rows
    # If less than 15 pixels apart, consider them in the same row

    rows = []
    # Sort boxes by their y center, group them into rows
    # First box will be placed in a new row, subsequent boxes will be mapped to the closest row via threshold
    for cnt, x, y, w, h, y_center in sorted(boxes, key=lambda b: b[5]):
        placed = False
        for row in rows:
            # row[0][5] is the y-center of the first box in that row
            if abs(row[0][5] - y_center) < Y_THRESH:
                row.append((cnt, x, y, w, h, y_center))
                placed = True
                break
        if not placed:
            rows.append([(cnt, x, y, w, h, y_center)])

    # If image is noisy, we might have more than 6 rows, so we filter them to 5x6 wordle format
    # Keep only rows which has 5 boxes in them
    rows = [r for r in rows if len(r) == 5]

    if len(rows) == 6:
        # Sort them by x pos to get the correct order of boxes in each row
        rows = [sorted(row, key=lambda b: b[1]) for row in rows]
        return rows
    else:
        return None
