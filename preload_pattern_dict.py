from wordle_wordlists import allowed, answers, all_possible_words
from solver import validate_guess
from collections import defaultdict
import pickle
from tqdm import tqdm

# Build a 1 level tree of patterns based on each word in allowed validating against all answers
# Complexity: O(allowed * Answers)
# Will return a nested dict containing key = allowed word, value = dict of patterns as keys, and a set of answers as values
def build_pattern_dict(allowed_plus_answers, answers):
    # Create a dict to contain all possible answer patterns for each allowed word
    pattern_dict = { word: defaultdict(set) for word in allowed_plus_answers }
    for word in tqdm(allowed_plus_answers, desc="Allowed words", unit="word"):
        for ans in answers:
            pattern = validate_guess(word, ans)
            pattern_dict[word][pattern].add(ans)
    return pattern_dict

if __name__ == "__main__":
    print(f"Length of allowed words: {len(allowed)}")
    print(f"Length of answers: {len(answers)}")
    print(f"Total unique words: {len(all_possible_words)}")
    # Preprocess the pattern dict
    pattern_dict = build_pattern_dict(all_possible_words, answers)
    # Save pattern dict to a pickle file
    with open('pattern_dict.pkl', 'wb') as f:
        pickle.dump(pattern_dict, f)
    print("Pattern dictionary built and saved.")