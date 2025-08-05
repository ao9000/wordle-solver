from image_processing import get_wordle_grid_boxes, crop_cell_margin, detect_letter, tesseract_inference, extract_color_from_cell
import cv2
import pickle
from wordle_wordlists import answers
from solver import rank_guesses, GUESSES, GRAY_STATE, YELLOW_STATE, GREEN_STATE

image = cv2.imread('images/row 1 valid.PNG')

# Load precomputed pattern dictionary
with open('pattern_dict.pkl', 'rb') as f:
    pattern_dict = pickle.load(f)

rows = get_wordle_grid_boxes(image)

if rows:
    # Crop the cells and put them into a nested list
    wordle_board_imgs = []
    for row in rows:
        wordle_board_row = []
        for cell in row:
            x, y, w, h = cell[1:5]
            cropped_cell = image[y:y+h, x:x+w]
            cropped_cell = crop_cell_margin(cropped_cell)
            wordle_board_row.append(cropped_cell)
        wordle_board_imgs.append(wordle_board_row)

    # Run Tesseract OCR on each cell
    wordle_board_letters = [[None for _ in range(5)] for _ in range(6)]
    wordle_board_colors = [[None for _ in range(5)] for _ in range(6)]
    for i, row in enumerate(wordle_board_imgs):
        for j, cell in enumerate(row):
            if detect_letter(cell.copy()):
                # Extract letter
                letter = tesseract_inference(cell.copy())
                wordle_board_letters[i][j] = letter
                # Extract color
                color = extract_color_from_cell(cell.copy())
                wordle_board_colors[i][j] = color

                # Check if there is any missed OCR letters
                if letter is None and color != "white":
                    raise ValueError(f"Detected a color {color} without a letter in cell ({i}, {j}), please retry")

    # Check the level of completed board
    # If any color in a row is white, clear the row
    completed_rows = []

    for letters_row, colors_row in zip(wordle_board_letters, wordle_board_colors):
        # Skip any row that is having 1 empty cell, not yet submitted
        if any(letter is None for letter in letters_row):
            continue

        # Skip any row containing a 'white' cell, also not yet submitted
        if any(color == "white" for color in colors_row):
            continue

        # Keep submitted rows
        completed_rows.append({
            "letters": letters_row,
            "colors": colors_row
        })

    # Pass board to solver to get the next bext guess and possible answers
    if not completed_rows:
        print("No submitted guesses detected. Please submit at least one row in the image.")
    else:
        # Initialize the remaining answer set
        remaining = set(answers)

        for row in completed_rows:
            letters = row["letters"]
            colors = row["colors"]

            # Convert colors to feedback pattern digits
            feedback_pattern = tuple(
                GREEN_STATE if color == 'green' else
                YELLOW_STATE if color == 'yellow' else
                GRAY_STATE
                for color in colors
            )

            # Join into string
            guess = ''.join(letters).lower()
            print(f"Guess: {guess}, Feedback: {feedback_pattern}")

            # Prune the remaining set
            remaining = remaining.intersection(pattern_dict[guess][feedback_pattern])

            # Print some possible answers
            print(f"Possible answers ({len(remaining)}): {",".join(remaining)}")

            # If possible answer left 1, print the answer
            if len(remaining) == 1:
                print(f"The answer is: {next(iter(remaining))}")
                break
            elif len(remaining) == 0:
                print("No possible answers left! Please check your inputs.")
                break
            else:
                # Print the best next guess
                suggestions = rank_guesses(remaining, pattern_dict, 5)
                print("Best next guesses:")
                for word, entropy in suggestions:
                    print(f"{word:10s} → {entropy:.3f} bits")
                print()

else:
    print(f"Failed to extract wordle board, found {len(rows)} rows instead of 6")



