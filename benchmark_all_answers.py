from wordle_wordlists import allowed, answers, all_possible_words
import pickle
from solver import GUESSES, rank_guesses, GRAY_STATE, YELLOW_STATE, GREEN_STATE, WORD_LENGTH, validate_guess
import statistics


# Load the pattern dict from a pickle file
with open('pattern_dict.pkl', 'rb') as f:
    pattern_dict = pickle.load(f)

# Define counters
end_turns = []
failed_words = []

for ans in answers:
    # Define the set of all possible answers
    remaining = set(answers)

    for turn_num in range(GUESSES):
        # Get the best guess
        if len(remaining) == 1:
            guess = next(iter(remaining))
        else:
            if turn_num == 0:
                guess = 'soare'  # Use a fixed first guess to reduce processing time
            else:
                # Rank the guesses based on the remaining candidates
                guess = rank_guesses(remaining, pattern_dict, 1)[0][0]

        # Get the feedback pattern
        feedback_pattern = validate_guess(guess, ans)

        # Check if answer is found
        if feedback_pattern == (GREEN_STATE,)*WORD_LENGTH:
            end_turns.append(turn_num+1)
            print(f"[{ans}] solved in {turn_num+1} turns.")
            break

        # Prune the remaining set
        remaining = remaining.intersection(pattern_dict[guess][feedback_pattern])

        # Unable to find answer due to invalid feedback
        if not remaining:
            failed_words.append(ans)
            print(f"[{ans}] Inconsistent feedback on turn {turn_num} (no candidates left)")
            break

    else:
        failed_words.append(ans)
        print(f"[{ans}] NOT SOLVED in {GUESSES} turns; remaining candidates: {remaining}")


# Print statistics
print("End of benchmark")
print("Average turns: ", statistics.mean(end_turns))
print("Median turns: ", statistics.median(end_turns))
print("Max turns: ", max(end_turns))

most_turns_words = sorted(zip(answers, end_turns), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 hardest:", most_turns_words)
