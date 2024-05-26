import numpy as np
from random import sample, shuffle


def initialise(n1, n2, n3, num_solns, file_name):
    """
    Sets up the cost matrices for the hungarian algorithm.
    :param n1: Number of professors of category x1
    :param n2: Number of professors of category x2
    :param n3: Number of professors of category x3
    :param num_solns: Number of solutions required in the output
    :param file_name: Input file name
    :return: Tuple of cost matrices and course lists for each cost matrix
    """
    result = [] # List of cost matrices for each solution
    course_lists = [] # List of courses selected for each solution
    with open(file_name, 'r') as f:
        cdcs = [course.strip('\n') for course in f.readline().split(' ')] # List of CDCS
        electives = [course.strip('\n') for course in f.readline().split(' ')] # List of electives
        preferences = []
        for line in f.readlines():
            preferences += [[course.strip("\n") for course in line.split(' ')]] # List of preferences for each professor
    electives_weights = [0] * len(electives) # Sum of weights of each elective in each preference list
    courses = cdcs.copy()
    for i in range(len(electives)):
        for preference in preferences:
            if electives[i] in preference:
                electives_weights[i] += preference.index(electives[i])
            else:
                electives_weights[i] += 2*len(preference) # Assign the elective a large weight if it is not in the list
    num_electives = int(0.5 * n1 + n2 + 1.5 * n3 - len(courses))
    while num_electives > 0: # Selecting the electives with the minimum weights for the optimal solution
        mini = min(electives_weights)
        idx = electives_weights.index(mini)
        courses += [electives[idx]]
        electives_weights[idx] = max(electives_weights) + 1
        num_electives -= 1
    course_lists += [flatten(2 * [item] for item in courses)] # Adding 2 columns for each course (splitting into 0.5)
    Q = []
    for preference in preferences:
        l = []
        for course in courses:
            if course in preference:
                l += 2 * [preference.index(course) + 1]
            else:
                l += 2 * [len(preference)] # Assign the course a large weight if it is not in the list
        Q += [l] # Add each cost vector to the empty list
    La = [1] * n1 + [2] * n2 + [3] * n3 # List containing number of 0.5 courses for each professor
    M = [] # Cost matrix for optimal solution
    for i in range(len(La)):
        for j in range(La[i]):
            M += [Q[i]] # 1 row for x1, 2 rows for x2, 3 rows for x3
    result += [M] # Optimal solution
    while num_solns > 1:
        c_list = cdcs + sample(electives, len(courses) - len(electives))  # Randomly select electives
        course_lists += [flatten(2 * [item] for item in c_list)] # Similar code to optimal solution
        Q = []
        for preference in preferences:
            l = []
            for course in c_list:
                if course in preference:
                    l += 2 * [preference.index(course) + 1]
                else:
                    l += 2 * [len(preference)]
            Q += [l]
        La = [1] * n1 + [2] * n2 + [3] * n3
        M = []
        for i in range(len(La)):
            for j in range(La[i]):
                M += [Q[i]]
        if M not in result: # Check uniqueness of solution
            result += [M]
            num_solns -= 1
        if num_solns == 1:
            break
        lis = courses.copy() # Providing non preference optimal solutions by shuffling the same courses
        shuffle(lis)
        course_lists += [flatten(2 * [item] for item in lis)] # Similar code to optimal solution
        Q = []
        for preference in preferences:
            l = []
            for course in lis:
                if course in preference:
                    l += 2 * [preference.index(course) + 1]
                else:
                    l += 2 * [len(preference)]
            Q += [l]
        La = [1] * n1 + [2] * n2 + [3] * n3
        M = []
        for i in range(len(La)):
            for j in range(La[i]):
                M += [Q[i]]
        if M not in result:  # Check uniqueness of solution
            result += [M]
            num_solns -= 1


    return result, course_lists


def flatten(l): # Auxiliary function for flattening a list of lists
    return [item for sublist in l for item in sublist]


def reduce_matrix(M):
    """
    Subtracts the minimum value of each row and column of M from every value in the row/column
    :param M: Any matrix
    :return: The reduced matrix
    """
    for i in range(len(M)): # Iterate over rows
        mini = np.min(M[i]) # Minimum value
        for j in range(len(M[i])): # Iterate over values in each row
            M[i, j] -= mini # Subtract
    M = M.transpose()
    for i in range(len(M)): # Iterate over columns
        mini = np.min(M[i]) # Minimum value
        for j in range(len(M[i])): # Iterate over values in each column
            M[i, j] -= mini # Subtract
    M = M.transpose()
    return M


def covered_rows_cols(M):
    """
    Finds the minimum number of lines required to cover every zero in M.
    The function finds the row/column with the maximum number of uncovered zeros, and covers it, until all zeros
    are covered
    :param M: Any matrix
    :return: Number of lines needed
    """
    lines = 0
    covered_rows = np.zeros(len(M), dtype=bool) # Boolean array to check if a row is covered by a line
    covered_cols = np.zeros(len(M), dtype=bool) # Boolean array to check if a column is covered by a line
    N = M.transpose()
    done = False
    while not done:
        maxi = 0
        column = False # Boolean variable to check if a column has the most zeros
        idx = -1 # Index of row/column with the most zeros
        for i in range(len(M)): # Iterate over rows
            count = 0 # Number of zeros in the current row
            if not covered_rows[i]: # Check if current row is covered
                for j in range(len(M[i])): # Iterate over values in each row
                    if M[i, j] == 0 and not covered_cols[j]: # If value is a zero that is not covered, add to count
                        count += 1
                if count > maxi: # Set variables if maximum number of zeros
                    maxi = count
                    idx = i
        for i in range(len(N)): # Iterate over columns, similarly to rows
            count = 0
            if not covered_cols[i]:
                for j in range(len(N[i])):
                    if N[i, j] == 0 and not covered_rows[j]:
                        count += 1
                if count > maxi:
                    column = True # Maximum number of zeros is in a column
                    maxi = count
                    idx = i
        if column:
            covered_cols[idx] = True # Cover the column
            lines += 1 # Add line
        else:
            covered_rows[idx] = True # Cover the row
            lines += 1 # Add line
        zeroes = np.argwhere(M == 0)
        all = True
        for indices in zeroes: # Check if all zeros are covered
            i = indices[0]
            j = indices[1]
            if not covered_rows[i] and not covered_cols[j]:
                all = False
        if all:
            done = True # Loop is done
    return lines


def make_optimal_assignment(M):
    """
    Given a reduced matrix, chooses one zero per row and column, for the final assignment.
    Similar to the covered_rows_cols function, except it finds the row/column with the minimum number of assignable
    zeros and assigns the first assignable zero.
    :param M: Reduced Matrix
    :return: List of indices of assigned zeros
    """
    N = M.transpose()
    starred_zero_rows = np.zeros(len(M), dtype=bool) # Boolean array to check if a row contains a starred(assigned) zero
    starred_zero_cols = np.zeros(len(M), dtype=bool) # Boolean array to check if a column contains a starred
                                                                                                       # (assigned) zero
    stars = []
    while len(stars) < len(M): # While number of assigned zeros is less than the length of the matrix
        mini = len(M)+1 # Maximum possible number of unassigned zeros plus one
        column = False # Check if a column contains the maximum number of unassigned zeros
        idx = -1 # index of row/column with the maximum number of unassigned zeros
        for i in range(len(M)): # Iterate over rows
            count = 0 # Number of unassigned zeros in the current row
            if not starred_zero_rows[i]: # If the row doesn't contain an assigned zero
                for j in range(len(M[i])): # Iterate over elements in the row
                    if M[i, j] == 0 and not starred_zero_cols[j]: # If the element is a zero, and the column doesn't
                        count += 1                                # contain an assigned zero, add to count
                if count < mini: # Set variables if the row has the minimum assigned zeros
                    mini = count
                    idx = i
        for i in range(len(N)): # Iterate over columns, similar to rows
            count = 0
            if not starred_zero_cols[i]:
                for j in range(len(N[i])):
                    if N[i, j] == 0 and not starred_zero_rows[j]:
                        count += 1
                if count < mini:
                    column = True # A column contains the minimum number of unassigned zeros
                    mini = count
                    idx = i
        if column:
            for j in range(len(N[idx])): # Assign the first unassigned zero
                if N[idx, j] == 0 and not starred_zero_rows[j]:
                    starred_zero_rows[j] = True
                    starred_zero_cols[idx] = True
                    stars += [(j, idx)]
                    break
        else:
            for j in range(len(M[idx])):
                if M[idx, j] == 0 and not starred_zero_cols[j]: # Assign the first unassigned zero
                    starred_zero_rows[idx] = True
                    starred_zero_cols[j] = True
                    stars += [(idx, j)]
                    break
    return stars


def hungarian_algorithm(M):
    """
    Runs the hungarian algorithm on cost matrix M
    :param M: A square cost matrix
    :return: Tuple of indices of assigned tasks to workers (Courses to profs in this case)
    """
    M = reduce_matrix(M) # Reduce the matrix
    minlines = covered_rows_cols(M) # Minimum number of lines required to cover every zero of the reduced matrix
    if minlines == len(M): # If minimum number of lines is equal to the order of the matrix,
        T = make_optimal_assignment(M) # Assign the zeros
    else:
        while minlines < len(M): # Subtract the minimum value in M from every value in M
            val = np.min(M[np.nonzero(M)]) # Minimum nonzero value in M
            for i in range(len(M)):
                for j in range(len(M[i])):
                    if M[i][j] != 0:
                        M[i][j] -= val
            minlines = covered_rows_cols(M) # Recalculate the minimum number of lines needed
        T = make_optimal_assignment(M) # Assign the zeros
    return T


def assign_profs(n1, n2, n3, num_solns):
    """
    Assigns professors to courses by reading courses and preferences from the input file and returns as many solutions
    as needed
    :param n1: Number of professors of category x1
    :param n2: Number of professors of category x2
    :param n3: Number of professors of category x3
    :param num_solns: Number of solutions needed
    :return: Nothing, writes the assignments to the output file
    """
    if not (0.5*n1 + n2 + 1.5*n3).is_integer(): # Check if an integer number of courses is to be assigned
        with open('output.txt', 'w') as f:
            f.write("Assignment not possible")
        return
    cost_matrices, course_lists = initialise(n1, n2, n3, num_solns, 'input.txt')
    assignments = [] # Final lists of assignments of professors to courses
    for M, courses in zip(cost_matrices, course_lists): # Get the cost matrix and course list for each needed solution
        stars = hungarian_algorithm(np.array(M)) # Run the hungarian algorithm on each cost matrix
        assignment = {'x1': {}, 'x2': {}, 'x3': {}} # Initialize assignments dictionary
        for i in range(1, n1+1): # Give each professor of each category an empty list of courses
            assignment['x1'][f"Prof {i}"] = []
        for i in range(n1+1, n1+n2+1):
            assignment['x2'][f'Prof {i}'] = []
        for i in range(n1+n2+1, n1+n2+n3+1):
            assignment['x3'][f'Prof {i}'] = []
        prof_no = 1
        for i in range(n1): # Iterate over x1 category professors
            for index in stars: # Iterate over assignment indices
                if index[0] == i:
                    j = index[1]
            assignment['x1'][f'Prof {prof_no}'] += [courses[j]] # Assign the required course
            prof_no += 1
        for i in range(n1, n1 + 2 * n2, 2): # Iterate over x2 category professors
            for index in stars: # Iterate over assignment indices
                if index[0] == i: # Check the first row for the current professor
                    j1 = index[1]
                if index[0] == i + 1: # Check the second row for the current professor
                    j2 = index[1]
            assignment['x2'][f'Prof {prof_no}'] += [courses[j1]] # Assign the required courses
            assignment['x2'][f'Prof {prof_no}'] += [courses[j2]]
            prof_no += 1
        for i in range(n1 + 2 * n2, n1 + 2 * n2 + 3 * n3, 3): # Iterate over x3 category professors
            for index in stars: # Iterate over assignment indices
                if index[0] == i: # Check the first row for the current professor
                    j1 = index[1]
                if index[0] == i + 1: # Check the second row for the current professor
                    j2 = index[1]
                if index[0] == i + 2: # Check the third row for the current professor
                    j3 = index[1]
            assignment['x3'][f'Prof {prof_no}'] += [courses[j1]] # Assign the required courses
            assignment['x3'][f'Prof {prof_no}'] += [courses[j2]]
            assignment['x3'][f'Prof {prof_no}'] += [courses[j3]]
            prof_no += 1
        assignments += [assignment] # Add solution to list of solutions
    with open('output.txt', 'w') as f: # Write assignments to output file
        for assignment in assignments:
            f.write(str(assignment) + '\n')
    return
assign_profs(10,10,10,30)