# WarpDB

WarpDB is a GPU-accelerated SQL query engine that demonstrates how to leverage CUDA for high-performance database operations. It uses JIT (Just-In-Time) compilation to dynamically generate CUDA kernels based on user queries, providing fast data processing capabilities for analytical workloads.

## Features

- **GPU-Accelerated Query Processing**: Execute SQL-like queries directly on GPU memory for maximum performance
- **Dynamic CUDA Kernel Compilation**: JIT-compile custom CUDA kernels at runtime based on user expressions
- **Expression Parsing & Code Generation**: Parse SQL-like expressions and automatically generate optimized CUDA code
- **CSV Data Loading**: Efficiently load data from CSV files directly to GPU memory
- **JSON Data Loading**: Read newline-delimited JSON files
- **Parquet/Arrow/ORC Loading**: Use Apache Arrow to ingest columnar formats
- **CUDA-Based Data Filtering & Projection**: Filter and transform data in parallel on the GPU
- **Arrow Columnar Format**: Optionally load data using Apache Arrow for zero-copy
  interoperability with Pandas, PyTorch, and Spark
- **User-Provided CUDA Functions**: Extend queries with functions defined in `custom.cu`
- **Column Statistics & Optimizer**: Collect min/max/null counts for basic filter pushdown and kernel fusion


## Architecture

WarpDB consists of the following main components:

### CSV Loader
- Loads CSV data directly into GPU memory with minimal CPU intervention
- Handles data type conversion and memory allocation

### JSON Loader
- Parses newline-delimited JSON records containing `price` and `quantity`
- Uploads parsed columns to GPU memory


### Arrow Integration
- When Apache Arrow is available, WarpDB loads data into Arrow tables and
  transfers columns to GPU memory using Arrow's CUDA support. Arrow buffers can
  be shared across processes and enable efficient zero-copy interchange with
  other systems.

### Arrow Loader
- Reads Parquet, Arrow, and ORC files using Apache Arrow
- Transfers columns to GPU memory


### SQL Parser
- Tokenizes and parses SQL-like expressions into an Abstract Syntax Tree (AST)
- Supports basic arithmetic operations, comparisons, and column references

### CUDA JIT Compiler
- Compiles SQL expressions into optimized CUDA kernels at runtime using NVRTC
- Dynamically generates and optimizes code based on the query structure

### Query Execution Engine
- Executes the compiled kernels on the GPU
- Manages memory allocation and data transfer between host and device

## Requirements

- CUDA Toolkit 10.0 or higher
- CMake 3.18 or higher
- C++17 compatible compiler
- NVIDIA GPU with compute capability 7.0 or higher
- [Optional] Apache Arrow with CUDA support for zero-copy columnar data

The build system uses `find_package(CUDAToolkit)` to automatically locate
NVRTC and the CUDA driver. Ensure the CUDA toolkit is installed and available
in your environment.

## Building

```bash
mkdir build
cd build
cmake ..  # CMake will locate the CUDA toolkit automatically
# Arrow is discovered via `find_package(Arrow)` when available
make
```

## Usage

```bash
./warpdb "query_expression [WHERE condition]"
```


### Custom CUDA Functions

WarpDB looks for a file named `custom.cu` in the working directory at runtime.
Any functions defined in this file are appended to the generated kernel and can
be used in expressions. Functions should be marked with `__device__` so they are
callable from GPU kernels.

Example `custom.cu`:

```cpp
__device__ float discount(float price, float rate) {
    return price * rate;
}
```

You can then invoke the function in a query:

```bash
./warpdb "discount(price, 0.9)"
```

### Python API

You can also use WarpDB directly from Python if `pybind11` is available:

```python
import pywarpdb

db = pywarpdb.WarpDB("data/test.csv")  # or data/test.json
result = db.query("price * quantity WHERE price > 10")
print(result)
```

### Example Queries

```bash
# Calculate revenue (price * quantity)
./warpdb "price * quantity"

# Filter rows where price is greater than 15
./warpdb "price WHERE price > 15"

# Calculate discounted price for items above a threshold
./warpdb "price * 0.9 WHERE price > 20"

# Calculate total cost with tax
./warpdb "price * quantity * 1.08"
```

### Multi-GPU Example

WarpDB includes a helper `run_multi_gpu_jit` demonstrating how to split the
input table across available GPUs and execute the same JIT-compiled kernel on
each device. Results are aggregated back on the host.

## Project Structure

```
├── CMakeLists.txt          # CMake build configuration
├── data/                   # Sample data files
│   ├── test.csv            # Test data
│   └── test.json           # JSON test data
├── include/                # Header files
│   ├── csv_loader.hpp      # CSV loading interface
│   ├── arrow_loader.hpp    # Parquet/Arrow/ORC loading interface
│   ├── expression.hpp      # Expression parsing
│   └── jit.hpp             # JIT compilation interface
├── custom.cu               # User-provided CUDA functions (optional)
└── src/                    # Source files
    ├── csv_loader.cpp      # CSV loading implementation
    ├── arrow_loader.cpp    # Columnar format loading implementation
    ├── expression.cpp      # Expression parsing implementation
    ├── jit.cpp             # JIT compilation implementation
    └── main.cu             # Main application and CUDA kernels
```

## How It Works

1. **CSV Loading**: Input data is loaded from CSV files directly into GPU memory.
2. **Columnar Loading**: Parquet, Arrow, and ORC files are read via Apache Arrow and moved to GPU memory.
3. **Query Parsing**: User queries are tokenized and parsed into an AST.
4. **Code Generation**: The AST is converted into CUDA code.
5. **JIT Compilation**: The generated code is compiled into a CUDA kernel using NVRTC.
6. **Execution**: The compiled kernel is executed on the GPU.
7. **Result Retrieval**: Results are copied back to host memory and displayed.

## Technical Details

### Expression Parsing

WarpDB implements a simple recursive descent parser to transform SQL-like expressions into an AST. The parser supports:

- Column references (e.g., `price`, `quantity`)
- Numeric literals
- Binary operations (`+`, `-`, `*`, `/`)
- Comparison operations (`>`, `<`, `>=`, `<=`, `==`, `!=`)
  - The tokenizer checks two-character operators (e.g., `>=`, `<=`) before
    handling single-character ones.
- Parenthesized expressions

### JIT Compilation

The JIT compiler uses NVIDIA's Runtime Compilation library (NVRTC) to:

1. Generate CUDA C++ code from the AST
2. Compile the code into PTX (Parallel Thread Execution) instructions
3. Load the PTX into a CUDA module
4. Execute the compiled kernel on the GPU

### CUDA Kernels

WarpDB implements several CUDA kernels:

- `filter_price_gt`: Filters rows based on a price threshold
- `project_columns`: Projects specific columns
- `project_revenue`: Calculates revenue (price × quantity)
- `project_revenue_and_adjusted`: Calculates multiple expressions in one pass

## Limitations

- Currently supports a limited subset of SQL functionality
- Only supports simple CSV and JSON files with basic data types
- No support for joins, aggregations, or complex SQL features yet
- Limited error handling for malformed queries

## Future Improvements

- Support for more SQL features (JOINs, GROUP BY, ORDER BY)
- Better error handling and query validation
- Additional data source support (e.g. Avro)
- Query optimization based on data statistics
- Multi-GPU support for larger datasets
- Return results as Arrow buffers for easy sharing
