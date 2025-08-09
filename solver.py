import math
from wordle_wordlists import all_possible_words
from itertools import product

# Wordle constants
WORD_LENGTH = 5
GUESSES = 6

GREEN_STATE = 0
YELLOW_STATE = 1
GRAY_STATE = 2
WHITE_STATE = 3

def isValid(color_state):
    if color_state == GREEN_STATE or color_state == YELLOW_STATE or color_state == GRAY_STATE:
        return True
    return False


# For ranking guesses, we need to generate all possible patterns
all_patterns = list(product((GREEN_STATE, YELLOW_STATE, GRAY_STATE), repeat=WORD_LENGTH))

# Return a tuple representing the result of a guess against the answer
# GREEN_STATE (0) for correct letter in the correct position,
# YELLOW_STATE (1) for correct letter in the wrong position,
# GRAY_STATE (2) for incorrect letter.
def validate_guess(guess, answer):
    # Using wordle rules, validate a guess against the answer
    guess = guess.lower()
    answer = answer.lower()

    # Initialize all as grey at first
    pattern = [GRAY_STATE] * WORD_LENGTH

    # First pass: check for GREEN matches, exact matches in both position and letter
    # Record wrong letters for pass 2
    leftovers = {}
    for idx in range(WORD_LENGTH):
        if guess[idx] == answer[idx]:
            # Correct letter in the correct position
            # Mark as GREEN_STATE
            pattern[idx] = GREEN_STATE
        else:
            # Wrong position or letter
            # See correct answer letter for this position
            # Store the letter in leftovers for later
            leftovers[answer[idx]] = leftovers.get(answer[idx], 0) + 1

    # Second pass: Check for yellow matches, correct letter but wrong position
    for idx in range(WORD_LENGTH):
        # Only consider wrong, non-green letters
        if pattern[idx] == GRAY_STATE:
            # If the letter exists in the leftovers, aka answer, mark it as YELLOW_STATE
            # Get letter of guess at this index
            letter = guess[idx]

            # If letter exists in leftovers and has a count > 0
            if leftovers.get(letter, 0) > 0:
                # Mark as YELLOW_STATE
                pattern[idx] = YELLOW_STATE
                # Decrease the count of that letter in leftovers
                # This make sures that edge cases with 2 or more of the same letter are handled
                leftovers[letter] -= 1

    return tuple(pattern)

# Calculate Shannon entropy for a list of counts
def calculate_shannon_entropy(patter_matrix_counts):
    # Formula: I = Log2(1/p)
    # I = -Log2(p)
    total = sum(patter_matrix_counts)
    return -sum((c/total) * math.log(c/total) for c in patter_matrix_counts if c > 0)


# From the calculated pattern dict, rank all choices based on entropy
def rank_guesses(remaining, pattern_dict, top_n=5):
    scores = []
    for g in all_possible_words:
        counts = [len(pattern_dict[g][p] & remaining) for p in all_patterns]
        ent = calculate_shannon_entropy(counts)
        scores.append((g, ent))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]
