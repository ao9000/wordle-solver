from wordle_wordlists import allowed, answers, all_possible_words
import pickle
from solver import GUESSES, rank_guesses, GRAY_STATE, YELLOW_STATE, GREEN_STATE, WORD_LENGTH, validate_guess
import statistics
import matplotlib.pyplot as plt
from tqdm import tqdm


# Load the pattern dict from a pickle file
with open('pattern_dict.pkl', 'rb') as f:
    pattern_dict = pickle.load(f)

# Define counters
end_turns = []
failed_words = []

for ans in tqdm(answers, desc="Benchmarking all answers", unit="word"):
    # Define the set of all possible answers
    remaining = set(answers)

    for turn_num in range(GUESSES):
        # Get the best guess
        if len(remaining) == 1:
            guess = next(iter(remaining))
        else:
            if turn_num == 0:
                guess = 'tares'  # Use a fixed first guess to reduce processing time
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
avg_turns = statistics.mean(end_turns)
med_turns = statistics.median(end_turns)
max_turns = max(end_turns) if end_turns else 0
print("End of benchmark")
print("Average turns: ", avg_turns)
print("Median turns: ", med_turns)
print("Max turns: ", max_turns)

# Hardest words
most_turns_words = sorted(zip(answers, end_turns), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 hardest:", most_turns_words)

# Easiest words
least_turns_words = sorted(zip(answers, end_turns), key=lambda x: x[1], reverse=False)[:10]
print("Top 10 easiest:", least_turns_words)

# Save stats in txt
with open('wordle_stats.txt', 'w') as fout:
    fout.write(f"Average turns: {avg_turns}\n")
    fout.write(f"Median turns:  {med_turns}\n")
    fout.write(f"Max turns:     {max_turns}\n\n")
    fout.write("Top 10 hardest words (word, turns):\n")
    for word, turns in most_turns_words:
        fout.write(f"  {word}: {turns}\n")
    fout.write("\nAll end_turns values:\n")
    fout.write(", ".join(map(str, end_turns)))
    # Write failed words
    if failed_words:
        fout.write("\n\nFailed words:\n")
        for word in failed_words:
            fout.write(f"  {word}\n")
print("Saved stats and raw data to wordle_stats.txt")

# Plot distribution of turns
plt.figure(figsize=(6,4))
plt.hist(end_turns, bins=range(1, GUESSES+2), edgecolor='black')
plt.title('Distribution of Turns to Solve')
plt.xlabel('Turns')
plt.ylabel('Number of Words')
plt.xticks(range(1, GUESSES+1))
plt.tight_layout()
plt.savefig('all_words_benchmark_histogram.png')

print("Done")
