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

# 6. Remove redundant entries such as all green pattern since that is trivial

# 7. Zlib compression for large blobs only
# Zlib has a header overhead of about ~8 bytes, so dont compress small blobs where overhead outweighs savings
# Total blobs: 1278668
# Average: 29.8 bytes
# Median: 7 bytes
# Min: 1 bytes
# Max: 1568 bytes
# Percentiles:
#   50th: 7 bytes
#   75th: 23 bytes
#   90th: 67 bytes
#   95th: 133 bytes
#   99th: 383 bytes
# Estimated avg row size: 44.8 bytes

# 8. Increased page size from default 4096 to 8192 bytes
# 4kb is 300mb, 8kb is 244 mb, 16kb is 221mb, 32kb is 219mb
# We choose 16384 due to new appstore regulations of supporting devices with 16kb page size
# Phone has to load entire page into memory when accessing db, so cannot be too large to eat up all memory


import sqlite3
from collections import defaultdict
from tqdm import tqdm
from wordle_wordlists import allowed, answers, all_possible_words
from solver import validate_guess, GREEN_STATE
import zlib


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

def decode_int_to_pattern(base3_val):
    digits = []
    # Extract 5 digits from the base-3 integer, because Wordle pattern len are length 5
    for _ in range(5):
        # Get the rightmost digit (remainder when divided by 3)
        digit = base3_val % 3
        digits.append(digit)
        # Remove the rightmost digit (integer division by 3)
        base3_val //= 3

    # Reverse because we extracted digits from right to left
    digits.reverse()
    return tuple(digits)

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


def zlib_compress(indices):
    # First encode as varint
    uncompressed = pack_indices_varint(indices)

    # Then compress with zlib at maximum compression level
    compressed = zlib.compress(uncompressed, level=9)

    return compressed


def hybrid_zlib_and_pack_indices_varint(indices):
    varint_blob = pack_indices_varint(indices)

    threshold = 64 # Compress only if blob is >= threshold bytes
    min_savings = 4 # Minimum savings to commit to compression, this is to consider zlib tradeoff
    blob_size = len(varint_blob)
    # Only compress if blob is >= 80 bytes
    if blob_size >= threshold:
        compressed = zlib.compress(varint_blob, level=9)

        # Check if compression is effective
        if len(compressed) < blob_size - min_savings:
            return compressed, 1

    return varint_blob, 0


def build_pattern_db(batch_limit=2000, output_db="pattern_dict.sqlite"):
    # Get lists of possible words (possible guesses + answers) list
    guesses = list(all_possible_words)

    answers_list = list(answers)
    # answers_list = list(all_possible_words)

    # Optimization 2: Encode answers to indexes
    answers_map = {a: i for i, a in enumerate(answers_list)}

    # Check number of answers to encode
    print(f"Length of allowed words: {len(allowed)}")
    print(f"Length of answers: {len(answers)}")
    print(f"Total unique words: {len(all_possible_words)}")


    # Create connection and cursor to sqlite3 DB
    conn = sqlite3.connect(output_db)
    c = conn.cursor()

    # # Reduced page size for more savings from default 4096
    c.execute("PRAGMA page_size = 16384;")
    # conn.commit()

    # Performance pragmas for bulk insert
    c.execute("PRAGMA journal_mode = WAL;")
    c.execute("PRAGMA synchronous = OFF;")
    c.execute("PRAGMA temp_store = MEMORY;")
    conn.commit()

    # Create compact schema. Use WITHOUT ROWID for guess_pattern to save space
    # guess table -> to map word IDs to words (Answer + other words = all possible guesses)
    # Answers table -> to map answer IDs to answers
    # guess_pattern table -> Stores which answers produce which patterns for each guess, the main table
    c.executescript("""
    CREATE TABLE IF NOT EXISTS guess (
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
        is_compressed INTEGER NOT NULL,
        answer_blob BLOB NOT NULL,
        PRIMARY KEY (guess_id, pattern_int)
    ) WITHOUT ROWID;
    """)

    # # If only have 1 table, aka answer and guess are the same, we can optimize further by merging both tables
    # c.executescript("""
    # CREATE TABLE IF NOT EXISTS guess (
    #     id INTEGER PRIMARY KEY,
    #     word TEXT NOT NULL UNIQUE
    # );
    # CREATE TABLE IF NOT EXISTS guess_pattern (
    #     guess_id INTEGER NOT NULL,
    #     pattern_int INTEGER NOT NULL,
    #     is_compressed INTEGER NOT NULL,
    #     answer_blob BLOB NOT NULL,
    #     PRIMARY KEY (guess_id, pattern_int)
    # ) WITHOUT ROWID;
    # """)
    conn.commit()

    # Insert into answer and words mapping tables
    print("Inserting words and answers...")
    c.executemany("INSERT OR REPLACE INTO guess(id, word) VALUES (?, ?);", [(i, w) for i, w in enumerate(guesses)])
    c.executemany("INSERT OR REPLACE INTO answers(id, word) VALUES (?, ?);", [(i, a) for i, a in enumerate(answers_list)])
    conn.commit()

    # Insert into guess_pattern table
    insert_sql = "INSERT OR REPLACE INTO guess_pattern(guess_id, pattern_int, is_compressed, answer_blob) VALUES (?, ?, ?, ?);"
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

            # Optimization 6: Skip all-green pattern as it is trivial
            if all(p == GREEN_STATE for p in pattern):
                continue

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
            # No compression
            # blob = pack_indices_varint(ans_idx_list)
            
            # # Use zlib compression
            # blob = zlib_compress(ans_idx_list)

            # Use smart compression, only compress large blobs
            blob, is_compressed = hybrid_zlib_and_pack_indices_varint(ans_idx_list)

            to_insert.append((guess_id, pattern_base3_int, is_compressed, sqlite3.Binary(blob)))

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
