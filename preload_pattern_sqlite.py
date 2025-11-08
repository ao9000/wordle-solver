# Optimizations:
# 1. Use base 3 to represent pattern int
# There is 3^5 = 243 possible patterns, using base-3 because there is 3 states per letter (gray, yellow, green).
# So each pattern can be represented as an integer in the range [0, 242]

# 2. Instead of using a single word multiple times, represent words & answers as integer IDs instead of strings to save space
# e.g. ID 0  -> "aahed" and so on

# 3. Varint Encoding
# Before: Store as 5 integers × 4 bytes each = 20 bytes
# But notice: small numbers like 5 and 12 don't NEED 4 bytes
# After: Use variable-length integer encoding (varint) to store indices more compactly.
# Each byte has 8 bits. We use:
# - **7 bits** for the actual number
# - **1 bit** (the highest bit) as a "continuation flag"
#   - If bit is 1: "More bytes coming"
#   - If bit is 0: "This is the last byte"

# 4. WITHOUT ROWID
# SQLite normally adds a hidden "rowid" column to every row (8 bytes per row)
# but we can avoid this overhead by using "WITHOUT ROWID" tables when we have a natural primary key.

# 5. Delta encoding for answer indices


import sqlite3
from collections import defaultdict
from tqdm import tqdm
from wordle_wordlists import allowed, answers, all_possible_words
from solver import validate_guess


# Optimization 1: Encode pattern as base-3 integer
def encode_pattern_to_int(pattern):
    # Input: tuple of ints of length 5, of range 0-2
    # Example: (1, 2, 2, 0, 2) representing Y G G B G

    # Convert tuple into list of ints
    digits = [int(x) for x in pattern]

    # Convert int to base-3
    base3_val = 0
    for digit in digits:
        base3_val = base3_val * 3 + int(digit)
    return base3_val

# Optimization 3: Varint Encoding
def encode_varint(pattern_idx):
    # Input: Int to be encoded to blob

    blob_parts = []
    # Loop until the int is fully encoded
    while True:
        # 0x7F = 01111111
        # Use bitwise AND to only extract the lowest 7 bits
        lowest_7_bits = pattern_idx & 0x7F
        # Shift right by 7 bits for next iteration
        pattern_idx >>= 7
        # Checks if there is any more bits left after shifting
        if pattern_idx:
            # 0x80 = 10000000
            # Use bitwise OR to set the continuation flag without changing the other bits
            blob_parts.append(0x80 | lowest_7_bits)
        else:
            # No more bits left, this is the last byte
            blob_parts.append(lowest_7_bits)
            # Break out of inf loop
            break
    # Convert list of ints to bytes
    return bytes(blob_parts)

def pack_indices_varint(indices):
    # Loop all indices and encode each to varint
    return b''.join(encode_varint(i) for i in indices)

def decode_varints(blob):
    # Input: Blob of varint-encoded integers
    # Output: List of decoded integers

    # Store decoded integers here
    decoded_outputs = []

    # Track current position in blob
    i = 0
    # Current decoded integer being built from multiple bytes
    cur = 0
    # Current bit shift amount
    shift = 0

    # Loop through each byte in the blob
    while i < len(blob):
        # Read current byte
        b = blob[i]
        # Increment to next byte for next iteration
        i += 1
        # 0x7F = 01111111
        # Bitwise AND to extract the lowest 7 bits of the byte
        # Shift left by current shift amount and add to current integer
        # Use Bitwise OR to combine the shifted bits into current integer
        cur |= (b & 0x7F) << shift

        # 0x80 = 10000000
        # Check continuation flag
        if (b & 0x80) == 0:
            # No more bytes for this integer
            # Store final integer into output list
            decoded_outputs.append(cur)
            # Reset for next integer
            cur = 0
            shift = 0
        else:
            # More bytes to come for this integer
            # Increase shift amount by 7 for next byte
            shift += 7

    # Return list of decoded integers
    return decoded_outputs

# Optimization 5: Delta encoding for answer indices
def encode_delta(indices):
    # Input: A list of sorted ints representing answer indices
    # Output: A list of deltas (Differences between consecutive indices)

    deltas = [indices[0]]  # First value is stored as it is
    # Loop through indices and compute deltas
    for i in range(1, len(indices)):
        # Current index minus previous index
        delta = indices[i] - indices[i - 1]
        # Difference is stored
        deltas.append(delta)

    # Return list of deltas
    return deltas


def decode_delta(deltas):
    # Input: A list of deltas (Differences between consecutive indices)
    # Output: A list of reconstructed indices
    indices = [deltas[0]]  # First value is the absolute value
    # Loop through deltas and reconstruct indices
    for i in range(1, len(deltas)):
        # Current index is previous + current delta
        indices.append(indices[-1] + deltas[i])

    return indices


def build_pattern_db(batch_limit=2000, output_db="pattern_dict.sqlite"):
    # Get lists of possible words (possible guesses + answers) list
    guesses = list(all_possible_words)
    answers_list = list(answers)

    # Optimization 2: Encode answers to indexes
    answers_map = {a: i for i, a in enumerate(answers_list)}

    # Check number of answers to encode
    print(f"Length of allowed words: {len(allowed)}")
    print(f"Length of answers: {len(answers)}")
    print(f"Total unique words: {len(all_possible_words)}")


    # Create connection and cursor to sqlite3 DB
    conn = sqlite3.connect(output_db)
    c = conn.cursor()

    # Performance pragmas for bulk insert
    c.execute("PRAGMA journal_mode = WAL;")
    c.execute("PRAGMA synchronous = OFF;")
    c.execute("PRAGMA temp_store = MEMORY;")
    conn.commit()

    # Create compact schema. Use WITHOUT ROWID for guess_pattern to save space
    # Words table -> to map word IDs to words (Answer + guesses)
    # Answers table -> to map answer IDs to answers
    # guess_pattern table -> Stores which answers produce which patterns for each guess, the main table
    c.executescript("""
    CREATE TABLE IF NOT EXISTS words (
        id INTEGER PRIMARY KEY,
        word TEXT NOT NULL UNIQUE
    );
    CREATE TABLE IF NOT EXISTS answers (
        id INTEGER PRIMARY KEY,
        word TEXT NOT NULL UNIQUE
    );
    CREATE TABLE IF NOT EXISTS guess_pattern (
        guess_id INTEGER NOT NULL,
        pattern_int INTEGER NOT NULL,
        answer_blob BLOB NOT NULL,
        PRIMARY KEY (guess_id, pattern_int)
    ) WITHOUT ROWID;
    CREATE INDEX IF NOT EXISTS idx_guess ON guess_pattern(guess_id);
    """)
    conn.commit()

    # Insert into answer and words mapping tables
    print("Inserting words and answers...")
    c.executemany("INSERT OR REPLACE INTO words(id, word) VALUES (?, ?);", [(i, w) for i, w in enumerate(guesses)])
    c.executemany("INSERT OR REPLACE INTO answers(id, word) VALUES (?, ?);", [(i, a) for i, a in enumerate(answers_list)])
    conn.commit()

    # Insert into guess_pattern table
    insert_sql = "INSERT OR REPLACE INTO guess_pattern(guess_id, pattern_int, answer_blob) VALUES (?, ?, ?);"
    to_insert = []

    # Create the core pattern DB by building a 1 level tree of patterns based on each word in allowed validating against all answers
    for guess_id, guess in enumerate(tqdm(guesses, desc="Guesses")):
        # Init pattern map
        pattern_dict = defaultdict(list)
        # Loop each answer
        for ans in answers_list:
            # Get pattern
            # Returns a tuple of ints
            pattern = validate_guess(guess, ans)
            # Encode tuple of ints into a Base-3 int to save space
            # For example: (2,2,2,1,2) -> 239
            pattern_base3_int = encode_pattern_to_int(pattern)
            # Store entry into pattern map
            pattern_dict[pattern_base3_int].append(answers_map[ans])

        # Convert each pattern value into packed blob to save space
        for pattern_base3_int, ans_idx_list in pattern_dict.items():
            # Sort indices for delta encoding
            ans_idx_list.sort()
            # Apply delta encoding to answer indices
            ans_idx_list = encode_delta(ans_idx_list)

            # Pack indices into blob
            blob = pack_indices_varint(ans_idx_list)
            to_insert.append((guess_id, pattern_base3_int, sqlite3.Binary(blob)))

        # flush in batches
        if len(to_insert) >= batch_limit:
            c.executemany(insert_sql, to_insert)
            conn.commit()
            to_insert = []

    # final flush batch
    if to_insert:
        c.executemany(insert_sql, to_insert)
        conn.commit()

    print("Checkpointing WAL and compacting DB...")
    # Merges all data from the -wal file back into the main database file
    c.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    # Changes from WAL mode back to traditional DELETE mode
    c.execute("PRAGMA journal_mode = DELETE;")
    conn.commit()
    # Rebuilds the entire database file from scratch
    # Reclaims unused space (from deleted/updated rows)
    print("Running VACUUM")
    c.execute("VACUUM;")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    build_pattern_db()
