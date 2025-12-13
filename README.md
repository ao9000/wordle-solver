# Wordle Solver - Technical Documentation

## Overview

This readme documents my own Wordle puzzle solver called Wordle Lens.
Specifically explaining the 3 main components:
1. The entropy-based solver algorithm
2. Wordle Grid extractor using computer vision system for extracting game state from images 
3. All optimizations to store pattern dictionary very efficiently

---

## 1. Wordle Puzzle Solving Algorithm

### How It Works

The solver uses an **information theory approach** based on Shannon entropy to determine the optimal guesses that maximize information gain.

### Core Algorithm Components

#### Pattern Validation

The solver uses a two-pass algorithm to validate guesses against answers, generating patterns that indicate which letters are correct (green), present but misplaced (yellow), or absent (gray).

The validation process:
- **First pass**: Identifies exact matches (green tiles) and records unused letters
- **Second pass**: Identifies letters in wrong positions (yellow tiles) while handling duplicates correctly

#### Entropy-Based Ranking

The solver ranks potential guesses by calculating Shannon entropy, which measures the expected information gain from each guess. 

The entropy calculation uses the formula: **H = -Σ(p × log₂(p))** where p is the probability of each pattern occurring.

#### Guess Ranking Process

For each possible guess, the algorithm:
1. Generates all 243 possible patterns (3^5 for 5-letter words with 3 states per letter)
2. Counts how many remaining answers would produce each pattern
3. Calculates entropy based on these counts
4. Returns the top-N guesses with highest entropy

### Wordlist Structure

The solver uses two distinct wordlists:
- **Answers list**: Valid Wordle solutions (~2,300 words)
- **Allowed list**: All acceptable guesses including answers (~13,000 words)
- **All possible words**: Union of both lists for comprehensive coverage 

### Interactive Solving

The solver provides an interactive mode where users input guesses and feedback, and the system narrows down possibilities iteratively.

---

## 2. Camera Feature: Alphabet Extraction & Recognition

### Architecture Overview

The camera feature uses a two-stage pipeline: **image processing for cell extraction** and **CNN-based character recognition**.

### Stage 1: Wordle Board Extraction

#### Grid Detection Process

| Step | Process | Purpose |
|------|---------|---------|
| 1 | Grayscale conversion & Gaussian blur | Reduce noise and simplify image |
| 2 | Adaptive thresholding | Create binary image for contour detection |
| 3 | Contour detection | Find all closed shapes in image |
| 4 | Square filtering | Keep only square-shaped contours (aspect ratio 0.85-1.15) |
| 5 | Row grouping | Group squares by y-coordinate into 6 rows |
| 6 | Validation | Filter to exactly 5 cells per row with consistent spacing | 

The algorithm implements three critical validation checks:
- **Check 1**: Each row must contain exactly 5 boxes
- **Check 2**: All rows must have similar cell dimensions (within 10% tolerance)
- **Check 3**: Rows must have consistent vertical spacing to filter out keyboard elements

#### Cell Preprocessing

Each extracted cell undergoes margin cropping to remove borders that could interfere with letter detection.

#### Color Extraction

The system determines cell state (green/yellow/gray/white) by calculating the mean BGR color and finding the closest match to Wordle's official color scheme using Euclidean distance.

### Stage 2: Alphabet Recognition

#### CNN Model Architecture

The system uses a custom CNN classifier based on the MNIST architecture, adapted for 26 alphabet classes:

| Layer | Type | Details |
|-------|------|---------|
| Conv1 | Convolution | 32 filters, 3×3 kernel |
| Conv2 | Convolution | 64 filters, 3×3 kernel |
| Pool | Max Pooling | 2×2 |
| Dropout1 | Dropout | 25% rate |
| FC1 | Fully Connected | 9216 → 128 units |
| Dropout2 | Dropout | 50% rate |
| FC2 | Output | 128 → 26 classes | [9](#0-8) 

#### Image Preprocessing Pipeline

Before feeding to the CNN, each cell undergoes specialized preprocessing:

1. **Thresholding**: Convert to binary using Otsu's method 

2. **Noise Reduction & Centering**: 
   - Detect largest contour (the letter)
   - Crop to letter boundaries with small buffer
   - Resize maintaining aspect ratio to 17×17 pixels
   - Center on 28×28 black canvas 

3. **Normalization**: Apply MNIST dataset statistics (mean=0.11070, std=0.2661)

#### Inference Process

The complete recognition pipeline in the live solver

The system only attempts letter recognition for cells with valid colors (green/yellow/gray), skipping white/empty cells.

---

## 3. Pattern Dictionary Storage Optimizations

The pattern dictionary is the core data structure that maps every possible guess to all possible feedback patterns and their corresponding answer sets. Without optimization, this would require huge amount of space since there are around 2 million rows.


### Optimization Techniques

The repository implements optimizations to reduce storage from ~2GB to ~44MB (more than 90% reduction). 
This is especially useful for deploying onto Android devices where space is limited

#### Optimization Summary Table

| # | Technique | Description | Impact                                                                                                 |
|---|-----------|-------------|--------------------------------------------------------------------------------------------------------|
| 1 | Base-3 Pattern Encoding | Encode 5-tuple patterns as single integers (0-242) | 20 bytes → 1 byte per pattern                                                                          |
| 2 | Word ID Mapping | Replace string words with integer IDs | ~5 bytes → 2 bytes per word                                                                            |
| 3 | Varint Encoding | Variable-length integer encoding for small numbers | 4 bytes → 1-2 bytes avg                                                                                |
| 4 | WITHOUT ROWID | Remove SQLite's hidden rowid column | -8 bytes per row                                                                                       |
| 5 | Delta Encoding | Store differences between consecutive sorted indices | Further reduction of bytes required, makes lower digits more common thus improves compression later on |
| 6 | Skip Trivial Patterns | Don't store all-green patterns (always = answer) | -13K entries                                                                                           |
| 7 | Hybrid Zlib Compression | Compress large blobs only (≥64 bytes) | ~30-50% on large entries                                                                               |
| 8 | Increased Page Size | Use 16KB pages instead of 4KB | Better disk utilization                                                                                |

#### Detailed Optimization Explanations

**1. Base-3 Pattern Encoding**

Patterns are tuples of 5 integers (0=green, 1=yellow, 2=gray), representing 3^5 = 243 possibilities. These are encoded as base-3 integers

Example: Pattern `(1, 2, 2, 0, 2)` → Base-3 integer `239`

**2. Word ID Mapping**

Instead of storing "crane" repeatedly, map it to integer ID 147 once

**3. Varint Encoding**

Small numbers don't need 4 bytes. Varint uses 1 continuation bit + 7 data bits per byte 

Examples:
- Number 5: `00000101` (1 byte)
- Number 300: `10101100 00000010` (2 bytes)

**4. WITHOUT ROWID**

SQLite tables normally have a hidden 8-byte rowid. Using `WITHOUT ROWID` with natural primary keys eliminates this

**5. Delta Encoding**

For sorted answer indices `[5, 12, 15, 20]`, store deltas `[5, 7, 3, 5]` for better compression

**6. Skip All-Green Patterns**

If all tiles are green, the answer equals the guess—no need to store this

**7. Hybrid Zlib Compression**

Only compress blobs ≥64 bytes where compression saves ≥4 bytes (to offset zlib overhead)

**8. Increased Page Size**

Using 16KB pages instead of default 4KB reduces overhead and improves performance on modern devices 

This is especially important for the new devices which now support 16KB page sizes.

### Database Schema

The optimized SQLite database uses three tables:

- **guess**: Maps word IDs to guess words
- **answers**: Maps answer IDs to answer words  
- **guess_pattern**: Stores compressed pattern-to-answers mappings


---

## References
1. 3Blue1Brown. (2022, February 6). Solving Wordle using information theory [Video recording]. https://www.youtube.com/watch?v=v68zYyaEmEA


