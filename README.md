# module 1
Comprehensive Exam Notes with Examples: Module 1 (NumPy & Foundations)1. Understanding Python Data Types vs. NumPy ArraysTheoretical ComparisonPython Integer: Implemented as a C structure containing dynamic metadata (PyObject_HEAD). A standard C integer requires only raw bytes (e.g., 4 or 8 bytes). In contrast, a Python integer wraps ob_refcnt, ob_type, ob_size, and the actual digit value ob_digit.   Python List: A list is a collection of pointers, where each pointer refers to a separate, complete Python object. This enables heterogeneous data types (e.g., mixing integers, strings, booleans) but incurs significant memory overhead and cache misses.   NumPy ndarray: A NumPy array holds a pointer to a single, contiguous block of uniformly typed, fixed-size data in memory. It lacks type flexibility but enables fast operations compiled directly at the C layer.   Code ExamplePythonimport sys
import numpy as np

# Python integer overhead
x = 42
print("Size of Python int object:", sys.getsizeof(x))  # Typically 28 bytes

# Python list vs NumPy array memory layout
py_list = [1, 2, 3, 4, 5]
np_arr = np.array([1, 2, 3, 4, 5], dtype='int32')

print("NumPy element size:", np_arr.itemsize)          # 4 bytes per integer
print("NumPy total data bytes:", np_arr.nbytes)        # 20 bytes total
   2. Array Creation & AttributesKey Attributesndim: Total number of dimensions (axes).   shape: Tuple indicating the length of each dimension.   size: Total number of elements across all axes.   dtype: Specific data type of array elements.   itemsize: Size of one array element in bytes.   nbytes: Total bytes occupied by elements ($nbytes = size \times itemsize$).   Code Example: Built-in Creation Routines & Attribute InspectionPythonimport numpy as np

# 1. Array creation routines
zeros_arr = np.zeros((2, 3), dtype=int)         # 2x3 matrix of zeros
ones_arr = np.ones((3, 3), dtype=float)         # 3x3 matrix of ones
full_arr = np.full((2, 2), 3.14)               # Filled with 3.14
range_arr = np.arange(0, 10, 2)                 # [0, 2, 4, 6, 8]
linear_arr = np.linspace(0, 1, 5)               # 5 values evenly spaced between 0 and 1
ident_mat = np.eye(3)                           # 3x3 identity matrix

# 2. Inspecting Array Attributes
x3 = np.random.randint(10, size=(3, 4, 5))      # 3D array (depth=3, rows=4, cols=5)

print("Number of dimensions (ndim):", x3.ndim)      # Output: 3
print("Array shape (shape):", x3.shape)            # Output: (3, 4, 5)
print("Total elements (size):", x3.size)           # Output: 60
print("Data type (dtype):", x3.dtype)              # Output: int64 (or int32 depending on OS)
print("Size per element (itemsize):", x3.itemsize)  # Output: 8 bytes
print("Total array memory (nbytes):", x3.nbytes)    # Output: 480 bytes
   3. Indexing, Slicing, Reshaping & JoiningSlicing Syntax and the "No-Copy View" ConceptSyntax follows standard Python slicing: arr[start:stop:step].   Crucial Concept: NumPy array slices return views, not copies. Modifying a sliced subarray directly updates the parent array. Call .copy() if an isolated duplicate is required.   Code Example: Slicing, Views, and ReshapingPythonimport numpy as np

grid = np.arange(1, 10).reshape((3, 3))
print("Original 3x3 Grid:\n", grid)
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

# Subarray slicing (rows 0 and 1; columns 0 and 1)
sub_view = grid[:2, :2]
sub_view[0, 0] = 99
print("Original Grid after mutating sub_view:\n", grid)
# Notice grid[0, 0] is now 99!

# Creating an explicit copy
sub_copy = grid[:2, :2].copy()
sub_copy[0, 0] = 1000
print("Original Grid untouched after mutating sub_copy:\n", grid[0, 0])  # Still 99

# Adding dimensions with np.newaxis
a = np.array([1, 2, 3])
row_vector = a[np.newaxis, :]  # Shape: (1, 3)
col_vector = a[:, np.newaxis]  # Shape: (3, 1)
   Code Example: Joining and SplittingPythonx = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# Horizontal and Vertical Stacking
v_stacked = np.vstack([x, y])  # Shape: (2, 3)
print("Vertical Stack:\n", v_stacked)

# Splitting arrays at designated indices
data = np.arange(10)
p1, p2, p3 = np.split(data, [3, 7])
print("Split pieces:", p1, p2, p3)
# p1 = [0, 1, 2], p2 = [3, 4, 5, 6], p3 = [7, 8, 9]
   4. Universal Functions (UFuncs) & ComputationSlowness of Standard LoopsCPython loops incur runtime type-checks and dynamic function dispatches on every single iteration. NumPy universal functions (ufuncs) push array iterations into compiled machine instructions, executing significantly faster.   Code Example: Loop vs. UFunc Performance & Advanced FeaturesPythonimport numpy as np

# 1. Vectorized Arithmetic
arr = np.arange(1, 6)
print("Vectorized (+5):", arr + 5)       # Element-wise addition: [6, 7, 8, 9, 10]
print("Trigonometric (sin):", np.sin(arr))

# 2. Specifying Output Memory (out parameter)
result = np.empty(5)
np.multiply(arr, 10, out=result)
print("Result written to pre-allocated buffer:", result)

# 3. UFunc Aggregations: reduce and accumulate
sum_result = np.add.reduce(arr)             # 1 + 2 + 3 + 4 + 5 = 15
running_sum = np.add.accumulate(arr)        # [1, 3, 6, 10, 15]
print("Reduce sum:", sum_result)
print("Accumulated sum:", running_sum)

# 4. Outer Product
mult_table = np.multiply.outer(arr, arr)
print("Multiplication Table (1-5):\n", mult_table)
   5. Aggregations (Min, Max, and Multi-Dimensional Axes)Axis Mechanismaxis=0: Collapses along the vertical axis (computes column-wise statistics).   axis=1: Collapses along the horizontal axis (computes row-wise statistics).   NaN-safe functions (e.g., np.nansum, np.nanmean) ignore missing values instead of propagating NaN.   Code ExamplePythonimport numpy as np

matrix = np.array([
    [10, 20, 30],
    [5,  15, 25]
])

print("Total sum:", np.sum(matrix))             # Output: 105
print("Column-wise minimum (axis=0):", np.min(matrix, axis=0))  # Output: [5, 15, 25]
print("Row-wise maximum (axis=1):", np.max(matrix, axis=1))     # Output: [30, 25]

# Handling NaNs safely
corrupted_arr = np.array([1.0, 2.0, np.nan, 4.0])
print("Standard mean:", np.mean(corrupted_arr))         # Output: nan
print("NaN-safe mean:", np.nanmean(corrupted_arr))       # Output: 2.3333333333333335
   6. Computation on Arrays: Broadcasting RulesBroadcasting RulesRule 1: If two arrays differ in dimension count, pad the smaller shape with ones on its leading (left) side.   Rule 2: If shape dimensions do not match, any dimension of size 1 is stretched to match the corresponding dimension of the other array.   Rule 3: If along any dimension the sizes disagree and neither equals 1, an error (ValueError) is raised.   Step-by-Step Code ExamplePythonimport numpy as np

# Example: Adding a 2D array and a 1D array
M = np.ones((2, 3))     # Shape: (2, 3)
a = np.arange(3)        # Shape: (3,)

# Rule 1: a.shape becomes (1, 3) by padding left with 1
# Rule 2: a.shape stretches along axis 0 from 1 to 2 -> (2, 3)
# Rule 3: Shapes match!
result = M + a
print("Broadcasting Result (2x3 + 1D array):\n", result)
# [[1. 2. 3.]
#  [1. 2. 3.]]

# Real-world practical example: Centering a matrix
X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0]])

# Calculate mean of each column (shape: (3,))
col_mean = X.mean(axis=0)  # [4., 5., 6.]

# Subtract mean from matrix (broadcasting (3, 3) - (3,))
X_centered = X - col_mean
print("Centered Matrix (Mean=0):\n", X_centered)
   7. Boolean Masking & Fancy IndexingConceptsBoolean Masks: Array comparison operations yield boolean arrays used as filter conditions. Always use bitwise operators (&, |, ~) instead of keyword logical operators (and, or, not).   Fancy Indexing: Passing arrays/lists of index positions to extract or update arbitrary subsets.   Code ExamplePythonimport numpy as np

data = np.array([12, 45, 67, 89, 23, 56, 78, 90])

# 1. Boolean Masking
mask = (data > 30) & (data < 80)
print("Filtered values using mask:", data[mask])  # [45, 67, 56, 78]

# 2. Fancy Indexing (1D)
indices = [0, 3, 5]
print("Selected elements via index list:", data[indices])  # [12, 89, 56]

# 3. In-place modification with ufunc.at to handle duplicate indices
counts = np.zeros(5, dtype=int)
idx_to_increment = [1, 2, 2, 4]

# Incorrect direct assignment:
# counts[idx_to_increment] += 1 (Index 2 only increments once)
# Correct method:
np.add.at(counts, idx_to_increment, 1)
print("Counts with repeated indices handled:", counts)
# Output: [0, 1, 2, 0, 1]
   8. Sorting Arrays & Partial Sorts (Partitioning)Full Sorts vs. Partitionsnp.sort(arr): Generates a new array in ascending order via Quicksort ($O(N \log N)$ average).   np.argsort(arr): Returns the array of indices corresponding to sorted order.   np.partition(arr, k): Reorganizes the array in $O(N)$ expected time so that the smallest $k$ items reside to the left of the $k$-th position in arbitrary order, followed by the remaining elements.   Code ExamplePythonimport numpy as np

arr = np.array([7, 2, 3, 1, 6, 5, 4])

# Standard sorting
sorted_arr = np.sort(arr)
print("Sorted Array:", sorted_arr)          # [1, 2, 3, 4, 5, 6, 7]

# Getting index locations of sorted elements
sort_idx = np.argsort(arr)
print("Indices of sorted values:", sort_idx)  # [3, 1, 2, 6, 5, 4, 0]

# Partitioning: Get the 3 smallest elements without full sort overhead
partitioned = np.partition(arr, 3)
print("Partitioned array at k=3:", partitioned)
# Output: [1, 2, 3, 4, 6, 5, 7]
# First 3 values are the 3 smallest; elements beyond are larger.
   9. Structured Data: NumPy's Structured ArraysConcept & C-Struct AlignmentStructured arrays allow contiguous storage of compound records with heterogeneous fields, directly mirroring C structure layouts in memory.   Code ExamplePythonimport numpy as np

# 1. Defining a compound data type
# 'U10': Unicode string max 10 chars, 'i4': 32-bit int, 'f8': 64-bit float
student_dtype = np.dtype([
    ('name', 'U10'),
    ('sem', 'i4'),
    ('cgpa', 'f8')
])

# 2. Initializing structured array
students = np.zeros(3, dtype=student_dtype)

# 3. Populating fields
students['name'] = ['Alice', 'Bob', 'Charlie']
students['sem'] = [5, 5, 5]
students['cgpa'] = [9.1, 8.4, 7.9]

print("Structured Array:\n", students)
print("All student names:", students['name'])
print("First record:", students[0])

# 4. Record Arrays (np.recarray) for attribute access
students_rec = students.view(np.recarray)
print("Access via attribute notation:", students_rec.cgpa)
