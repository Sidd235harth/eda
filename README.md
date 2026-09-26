# Module 1: Introduction to Python and NumPy

Notes and reference implementations covering foundational concepts for Exploratory Data Analysis (BAI515E)[cite: 1].

---

## Table of Contents
1. [Python Objects vs. NumPy Ndarrays](#1-python-objects-vs-numpy-ndarrays)
2. [Array Creation Routines & Attributes](#2-array-creation-routines--attributes)
3. [Array Slicing, Views, and Reshaping](#3-array-slicing-views-and-reshaping)
4. [Universal Functions (UFuncs) & Vectorization](#4-universal-functions-ufuncs--vectorization)
5. [Multi-Dimensional Aggregations](#5-multi-dimensional-aggregations)
6. [Broadcasting Rules](#6-broadcasting-rules)
7. [Boolean Masking & Fancy Indexing](#7-boolean-masking--fancy-indexing)
8. [Sorting & Partitioning](#8-sorting--partitioning)
9. [Structured & Record Arrays](#9-structured--record-arrays)

---

## 1. Python Objects vs. NumPy Ndarrays

### Definition
* **Python Integer / Object:** A C-level structure (`PyObject_HEAD`) containing metadata such as reference count, type descriptor, and size along with the raw value.
* **Python List:** A sequence of references pointing to distinct Python objects situated arbitrarily in memory, allowing heterogeneous data types at the expense of memory footprint and cache efficiency.
* **NumPy `ndarray`:** A uniform memory segment holding fixed-size, homogeneous items mapped via stride and shape metadata, enabling compiled execution speed.

```python
import sys
import numpy as np

# Python object memory overhead
py_val = 100
print(f"Python int object size: {sys.getsizeof(py_val)} bytes")

# Contiguous layout of NumPy ndarray
arr = np.arange(10, dtype="int32")
print(f"Per-element size: {arr.itemsize} bytes")
print(f"Total buffer size: {arr.nbytes} bytes")
