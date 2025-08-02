from wordle_wordlists import allowed, answers, all_possible_words
import pickle
from solver import GUESSES, rank_guesses, GRAY_STATE, YELLOW_STATE, GREEN_STATE, WORD_LENGTH


# Load the pattern dict from a pickle file
with open('pattern_dict.pkl', 'rb') as f:
    pattern_dict = pickle.load(f)

# Define the set of all possible answers
remaining = set(answers)

# Show best starting guesses
first_choices = rank_guesses(remaining, pattern_dict, 10)
print("Top starting words:")
for word, entropy in first_choices:
    print(f"  {word:10s} → {entropy:.3f} bits")
print()

for turn_num in range(GUESSES):
    # Prompt user for a guess
    guess = input(f'Guess #{turn_num+1}: ')
    guess = guess.lower()

    # Check if the guess is valid
    if guess not in all_possible_words:
        print(f"“{guess}” isn’t a valid Wordle guess. Try again.")
        break

    # Prompt user for the feedback pattern
    feedback_pattern = input(f"What is the feedback? ({GRAY_STATE}=gray, {YELLOW_STATE}=yellow, {GREEN_STATE}=green): ").strip()
    feedback_pattern = tuple(int(digit) for digit in feedback_pattern)

    # Prune the remaining set
    # remaining &= pattern_dict[guess][feedback_pattern]
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

