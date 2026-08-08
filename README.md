# WarpDB
<img width="256" alt="Magnifying Glass on Orange Gradient" src="https://github.com/user-attachments/assets/abd703f5-17be-4537-b097-35c081e223e5" />

WarpDB is a GPU-accelerated SQL query engine that demonstrates how to leverage CUDA for high-performance database operations. It uses JIT (Just-In-Time) compilation to dynamically generate CUDA kernels based on user queries, providing fast data processing capabilities for analytical workloads.

## Features

- **GPU-Accelerated Query Processing**: Execute SQL-like queries directly on GPU memory for maximum performance
- **Dynamic CUDA Kernel Compilation**: JIT-compile custom CUDA kernels at runtime based on user expressions
- **Expression Parsing & Code Generation**: Parse SQL-like expressions and automatically generate optimized CUDA code
- **CSV Data Loading**: Efficiently load data from CSV files directly to GPU memory
- **JSON Data Loading**: Read newline-delimited JSON files
- **Parquet/Arrow/ORC Loading (Optional)**: Use Apache Arrow (when available at build time) to ingest columnar formats
- **CUDA-Based Data Filtering & Projection**: Filter and transform data in parallel on the GPU
- **Arrow Columnar Format**: Optionally load data using Apache Arrow for zero-copy
  interoperability with Pandas, PyTorch, and Spark
- **Arrow Results**: Retrieve query results as Arrow buffers for easy sharing
  (optionally using a custom shared memory name)
- **User-Provided CUDA Functions**: Extend queries with functions defined in `custom.cu`
- **Column Statistics & Optimizer**: Collect min/max/null counts for basic filter pushdown and kernel fusion
- **Multi-GPU Execution**: Robust support for running queries across multiple GPUs, including streaming large CSV files

## Architecture

WarpDB consists of the following main components:

### CSV Loader
- Loads CSV data directly into GPU memory with minimal CPU intervention
- Handles data type conversion and memory allocation

### JSON Loader
- Parses newline-delimited JSON records with dynamic schema inference
- Supports optional user-provided JSON schema maps for deterministic typing
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
- [nlohmann/json](https://github.com/nlohmann/json) for JSON parsing
- NVIDIA GPU supported by your installed CUDA toolkit/driver
- [Optional] Apache Arrow with CUDA support for zero-copy columnar data
- [Optional] `pybind11` to build the Python module (set `-DWARPDB_BUILD_PYTHON=ON`)

The build system uses `find_package(CUDAToolkit)` to automatically locate
NVRTC and the CUDA driver. Ensure the CUDA toolkit is installed and available
in your environment.

## Building

```bash
mkdir build
cd build
cmake ..  # CMake will locate the CUDA toolkit automatically
# Arrow support is optional and enabled only when its libraries are found
# Use -DWARPDB_BUILD_PYTHON=OFF to skip building the Python bindings
make
```

If `WARPDB_BUILD_PYTHON=ON` (default) and `pybind11` is detected, a `pywarpdb`
module is produced in the build directory alongside the C++ binaries. When
`pybind11` is not found, C++ targets still build normally and only Python
bindings are skipped.

## Testing

Run `ctest` from the `build` directory to execute the project's tests. Some
tests rely on CUDA and optional libraries like Arrow or pybind11.

The parser and code-generation tests depend only on the host C++ sources and
can be run without a CUDA toolkit or GPU:

```bash
./tests/run_host_tests.sh
```

CI runs this GPU-free lane on every change in addition to the full build.

## Usage

```bash
./warpdb "<sql_query>" [data_file]
```

The CLI entrypoint expects a SQL query string. For expression-style execution
(for example, `"price * quantity WHERE price > 10"`), use the C++/Python API
`WarpDB::query`/`WarpDB.query`.

If `data_file` is omitted, WarpDB loads `data/test.csv` by default.


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

WarpDB ships optional Python bindings powered by `pybind11`.
Ensure `pybind11` is installed and build the module with CMake:

```bash
mkdir build
cd build
cmake .. -DWARPDB_BUILD_PYTHON=ON  # default
make pywarpdb
```

Apache Arrow is optional; when present, results can be exchanged as Arrow
arrays. To skip building the Python bindings entirely, configure CMake with
`-DWARPDB_BUILD_PYTHON=OFF`.

You can also install the module directly from the source tree:

```bash
pip install .
```

During Python builds, `setup.py` automatically downloads the pinned
`nlohmann/json` header (`v3.2.0`, matching the CMake `FetchContent` pin) into a
local build dependency directory and adds that include path before compiling the
`Pybind11Extension`. Python-only builders do not need to pre-install
`nlohmann/json`.

Or build a wheel for redistribution:

```bash
pip wheel . -w dist
```

With the bindings installed you can query data directly from Python:


```python
import pywarpdb

db = pywarpdb.WarpDB("data/test.csv")  # or data/test.json
result = db.query("price * quantity WHERE price > 10")
print(result)

# Export result as an Arrow array stored in shared memory
arr_capsule, schema_capsule = db.query_arrow(
    "price * quantity", shared_memory=True, shm_name="/my_result")
import pyarrow as pa
arrow_arr = pa.Array._import_from_c(arr_capsule, schema_capsule)
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

# Use the SQL helper for GROUP BY
./warpdb "SELECT SUM(price) FROM test GROUP BY quantity"
# Filter groups by an aggregate with HAVING
./warpdb "SELECT SUM(price) FROM test GROUP BY quantity HAVING SUM(price) > 20"
# Limit results after sorting
./warpdb "SELECT price FROM test ORDER BY price DESC LIMIT 5"

# Non-GROUP-BY SQL can project multiple columns; each result row prints the
# selected columns in order:
./warpdb "SELECT price, quantity FROM test"

# Note: GROUP BY queries still support only a single SELECT expression.
```

### Multi-GPU Example


WarpDB exposes `query_multi_gpu` and `query_multi_gpu_csv` to run expressions on
all available GPUs. The CSV variant streams the file in chunks so datasets can
exceed a single GPU's memory.

```python
db = pywarpdb.WarpDB("data/test.csv")
result = db.query_multi_gpu("price * quantity")

# Process a huge CSV without loading the entire file
big_res = pywarpdb.WarpDB.query_multi_gpu_csv(
    "large.csv", "price * quantity", rows_per_chunk=1_000_000)
```

WarpDB includes helpers `run_multi_gpu_jit` and `run_multi_gpu_jit_large`
demonstrating how to split the input table across available GPUs and execute the
same JIT-compiled kernel on each device. Both functions now take the CSV file
path as their first argument. The `run_multi_gpu_jit_large` variant streams the
CSV file in chunks, enabling processing of datasets larger than a single GPU's
memory. Results are aggregated back on the host.


### Benchmark Visualization

To illustrate the benefit of GPU acceleration WarpDB ships with a helper script
that benchmarks CPU and GPU execution paths and plots execution time, memory
throughput, and GPU utilization (with multi-GPU scaling where available).

```bash
python examples/gpu_cpu_benchmark.py --mode sample
```

The sample mode uses curated metrics to showcase how JIT-compiled CUDA kernels
outpace CPU execution even when multiple GPUs are involved. To execute live
benchmarks on your hardware, install the optional Python bindings and run:

```bash
python examples/gpu_cpu_benchmark.py --mode live --dataset data/test.csv \
       --enable-multi-gpu --output-dir visualizations/live
```

The live mode times Pandas-based CPU evaluation versus WarpDB's GPU kernels,
computes approximate memory throughput, and generates comparison plots for each
query in the specified output directory.


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
  - A single `=` is also accepted and treated as equality (equivalent to `==`),
    matching SQL syntax such as `WHERE price = 10` and `JOIN ... ON a.id = b.id`.
  - SQL's `<>` is accepted as not-equal (equivalent to `!=`), e.g.
    `WHERE price <> 10`.
- Parenthesized expressions

### JIT Compilation

The JIT compiler uses NVIDIA's Runtime Compilation library (NVRTC) to:

1. Generate CUDA C++ code from the AST
2. Compile the code into PTX (Parallel Thread Execution) instructions
3. Load the PTX into a CUDA module
4. Execute the compiled kernel on the GPU

### CUDA Kernels

WarpDB primarily relies on JIT-generated CUDA kernels emitted from parsed
expressions/SQL and compiled at runtime with NVRTC. The repository also
contains standalone CUDA examples and tests that exercise lower-level kernels.

### Kernel Launch Error Handling

All CUDA kernel launches should be immediately followed by an error check to
surface misconfigurations early. The expected pattern is:

```cpp
my_kernel<<<grid, block>>>(args...);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize()); // retain for debugging when needed
```

`cudaDeviceSynchronize` calls help catch runtime errors during development and
should remain where explicitly added for debugging.

## Development Progress

The project has recently gained several improvements:

- Optional Apache Arrow integration can be enabled with `USE_ARROW`.
- Basic query optimization uses column statistics for simple filter pushdown.
- RAII wrappers manage CUDA contexts and modules to avoid resource leaks.
- Helper functions demonstrate streaming across multiple GPUs.
- Python bindings are built when `pybind11` is installed and
  `-DWARPDB_BUILD_PYTHON=ON`.

## Limitations

- Currently supports a limited subset of SQL functionality
- CSV and JSON paths are the most mature; JSON loading reads newline-delimited
  objects and infers each column's type, with an optional user-supplied schema
  map for deterministic typing
- SQL support includes filtering, aggregations, ordering, LIMIT/OFFSET, HAVING,
  and a host-side single-inner-equi JOIN path
- GROUP BY execution supports only one SELECT expression and one grouping key
- Limited error handling for malformed queries
- Loading Parquet/Arrow/ORC files requires Apache Arrow
- Building the Python module requires `pybind11` or disable it with
  `-DWARPDB_BUILD_PYTHON=OFF`

## Future Improvements

- Continue extending SQL support beyond JOIN/GROUP BY/ORDER BY, LIMIT, HAVING, and OFFSET
- Better error handling and query validation
- Additional data source support (e.g. Avro)

## License

WarpDB is licensed under the [Apache License 2.0](LICENSE).
