# Load both allowed and answers txt
allowed_path = 'wordle_wordlists/allowed.txt'
answers_path = 'wordle_wordlists/answers.txt'

with open(allowed_path, 'r') as f:
    allowed = list(f.read().splitlines())

with open(answers_path, 'r') as f:
    answers = list(f.read().splitlines())

all_possible_words = list(set(allowed) | set(answers))