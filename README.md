# Professor-Assignment Problem
Project prepared for the course Discrete Structures for Computer Science Fall 2023

## Requirements
- Python 3.7 or above
- numpy

## Usage

1. **Input File:**
   - Create an input file named `input.txt` with the following format:
     - The first line contains a space-separated list of CDCs (Compulsory Disciplinary Courses).
     - The second line contains a space-separated list of electives.
     - Subsequent lines contain preferences of each professor, where each line corresponds to a professor's preferences.

2. **Run the Script:**
   - Execute the script by running the command:
     ```bash
     python main.py
     ```

3. **Output:**
   - The script will generate an output file named `output.txt` containing the assignments for each solution.

## Parameters

Adjust the parameters within the script:
- `n1`: Number of professors of category x1.
- `n2`: Number of professors of category x2.
- `n3`: Number of professors of category x3.
- `num_solns`: Number of solutions required.

## File Formats

### Input File (`input.txt`)

The input file should have the following format:

```plaintext
CDC1 CDC2 ... CDCn
Elective1 Elective2 ... Electivek
Prof1_Pref1 Prof1_Pref2 ... Prof1_Prefm
Prof2_Pref1 Prof2_Pref2 ... Prof2_Prefp
...
```

Note: There shouldn't be any blank spaces after each line

### Output File ('output.txt')

The output file will contain the assignments in a dictionary format for each solution:

```plaintext
{'x1': {'Prof 1': ['Course1'], 'Prof 2': ['Course2', 'Course3'], ...},
 'x2': {'Prof 3': ['Course4', 'Course5'], 'Prof 4': ['Course6', 'Course7'], ...},
 'x3': {'Prof 5': ['Course8', 'Course9', 'Course10'], ...},
 ...}
```
Note: A professor being assigned two of the same course means that that professor will be taking the entire course
