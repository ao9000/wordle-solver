from image_processing import get_wordle_grid_boxes, crop_cell_margin, detect_letter, extract_color_from_cell
from alphabet_classifier.helper_functions import MNISTClassifier, wordle_cell_preprocessing, transform_handwritten_alphabet_dataset
import cv2
import pickle
from wordle_wordlists import answers
from solver import rank_guesses, GUESSES, GRAY_STATE, YELLOW_STATE, GREEN_STATE, WHITE_STATE, isValid
from PIL import Image
import torch

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTClassifier().to(device)
state_dict = torch.load("alphabet_classifier/models/finetune_model_100.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

transform = transform_handwritten_alphabet_dataset()

image = cv2.imread('images/row 1.PNG')

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
                # Extract color
                color = extract_color_from_cell(cell.copy())
                wordle_board_colors[i][j] = color

                if isValid(color):
                    # Only try to extract letter if the color is valid, because not confirmed guesses are useless
                    # Better approach is to use the trained CNN model to extract letters
                    # Extract letter
                    # Pytorch model inference
                    # Convert image to binary thresholded image
                    thresh = wordle_cell_preprocessing(cell.copy())

                    alphabet = Image.fromarray(thresh)

                    # Convert to tensor and apply transformations
                    alphabet_tensor = transform(alphabet)
                    # Add batch dimension
                    alphabet_tensor = alphabet_tensor.unsqueeze(0).to(device)

                    # Inference
                    with torch.no_grad():
                        logits = model(alphabet_tensor)
                        pred = logits.argmax(dim=1).item()
                        letter = chr(ord('A') + pred)

                        # Get confidence
                        confidence = torch.max(torch.softmax(logits, dim=1)).item()

                    # Old approach using Tesseract OCR, less accurate
                    # letter = tesseract_inference(cell.copy())

                    wordle_board_letters[i][j] = letter


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

            # Join into string
            guess = ''.join(letters).lower()
            feedback_pattern = tuple(int(c) for c in colors)

            print(f"Guess: {guess}, Feedback: {feedback_pattern}")

            # Prune the remaining set
            remaining = remaining.intersection(pattern_dict[guess][feedback_pattern])

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



