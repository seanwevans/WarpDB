# WarpDB
<img width="256" alt="Magnifying Glass on Orange Gradient" src="https://github.com/user-attachments/assets/abd703f5-17be-4537-b097-35c081e223e5" />

WarpDB is an educational query engine that JIT-compiles SQL-like expressions
into CUDA kernels at runtime with NVRTC and runs them over columnar data held in
GPU memory.

It is a demonstration of the technique, not a database. There is no storage
engine, no catalog, no transactions and no persistence — a WarpDB instance loads
one file and answers queries about it. Two execution paths exist and only one of
them uses the GPU; see [Execution model](#execution-model) for exactly which
query shapes are accelerated. The project publishes no performance claims and
has not been benchmarked against alternatives such as DuckDB, cuDF or
HeavyDB.

## Features

- **GPU Expression Execution**: `query()` evaluates an expression with an optional `WHERE` clause as a JIT-compiled CUDA kernel over columns resident in GPU memory
- **Dynamic CUDA Kernel Compilation**: JIT-compile custom CUDA kernels at runtime based on user expressions
- **Expression Parsing & Code Generation**: Parse SQL-like expressions and automatically generate optimized CUDA code
- **CSV Data Loading**: Efficiently load data from CSV files directly to GPU memory
- **JSON Data Loading**: Read newline-delimited JSON files
- **Parquet/Arrow/ORC Loading (Optional)**: Use Apache Arrow (when available at build time) to ingest columnar formats
- **CUDA-Based Filtering & Projection**: The `query()` path filters and transforms in parallel on the GPU
- **Arrow Columnar Format**: Optionally load data using Apache Arrow for zero-copy
  interoperability with Pandas, PyTorch, and Spark
- **Arrow Results**: Retrieve query results as Arrow buffers for easy sharing
  (optionally using a custom shared memory name)
- **User-Provided CUDA Functions**: Extend queries with functions defined in `custom.cu`
- **Column Statistics (analysis only)**: Collect min/max/null counts and detect
  constant-folding opportunities in filter predicates. These routines are not
  currently invoked by either execution path — see [Execution model](#execution-model)
- **Multi-GPU Execution (sequential)**: Shard an expression's rows across every
  visible device, and stream CSV files larger than one device's memory. Devices
  are driven one at a time rather than concurrently, so this extends capacity,
  not throughput — see [Execution model](#execution-model)

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
- Compiles expressions into CUDA kernels at runtime using NVRTC, caching
  compiled modules so a repeated expression is compiled once
- Optimization is whatever NVRTC applies; WarpDB does not rewrite the query

### Query Execution Engine
- Launches compiled kernels on the GPU for the `query()` path, choosing a block
  size by occupancy
- Manages memory allocation and data transfer between host and device
- The `query_sql()` path does not use this engine; it interprets the query on
  the host (see [Execution model](#execution-model))

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

# SELECT * expands to every column of the table, in schema order
# (not supported together with JOIN):
./warpdb "SELECT * FROM test WHERE price > 10"

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


### Benchmarking

`examples/gpu_cpu_benchmark.py` times WarpDB's JIT-compiled GPU expression path
against a pandas CPU baseline on your own hardware and plots the comparison.

```bash
python examples/gpu_cpu_benchmark.py --dataset /path/to/large.csv \
       --enable-multi-gpu --output-dir visualizations
```

Requires pandas and the `pywarpdb` bindings; pass `--cpu-only` to record just
the pandas baseline. Every figure it prints or plots is measured locally — the
script has no sample or demo mode and will fail rather than report numbers it
did not measure.

**This project publishes no performance claims.** No benchmark results are
checked into the repository, and none of the documentation asserts a speedup.
If you want to know whether WarpDB is faster than the alternatives on your
workload, run the script and find out. Note in particular:

- Timings exclude ingest for both engines, and exclude NVRTC compilation for
  WarpDB (the script warms the JIT cache first). Cold-start cost is real and is
  not what these numbers show.
- Throughput is reported as input CSV bytes ÷ execution time. That is a
  size-relative rate for comparing two engines on one file, not achieved device
  memory bandwidth.
- On small inputs the result is dominated by fixed per-call overhead — kernel
  launch, PCIe transfer, Python dispatch. The 5-row CSVs in `data/` exist to
  test correctness, not performance. The script warns below 100k rows.
- Only `query()` is benchmarked. It is the JIT-compiled GPU path. `query_sql()`
  is a separate, host-side implementation and is not covered here.


## Execution model

WarpDB has two independent execution paths that do not share an implementation.
Which one runs depends on the entry point you call.

| Entry point | Where it runs | What it accepts |
|---|---|---|
| `query(expr)` | **GPU.** Parsed to an AST, emitted as CUDA, compiled with NVRTC, launched as one kernel per query. | A single expression plus an optional `WHERE` clause |
| `query_multi_gpu(expr)` | **GPU**, one device at a time | Same as `query()` |
| `query_sql(sql)` | **Host**, with one narrow GPU exception (below) | `SELECT` / `WHERE` / `GROUP BY` / `HAVING` / `ORDER BY` / `DISTINCT` / `LIMIT` / `OFFSET` / single inner equi-`JOIN` |

`query_sql()` — the path that accepts actual SQL — is implemented as a
row-at-a-time AST interpreter over host memory (`filter_rows`, `eval_node`,
`execute_group_by` and the hash join in `src/warpdb.cpp`). Rows are visited in a
loop and each AST node is dispatched with `dynamic_cast`. Nothing about this
path is GPU-accelerated or vectorized, and it should be assumed slower than an
established CPU engine, not faster.

The one exception is a GROUP BY fast path (`can_use_gpu_group_fast_path`,
`src/warpdb.cpp`) that reduces on the device. It applies only when **all** of
the following hold:

- exactly one `GROUP BY` key, and it is an `Int32` or `Int64` column
- exactly one `SELECT` expression, and it is an aggregate over a plain numeric column
- the aggregate is not `COUNT`
- no `WHERE`, no `HAVING`, no `JOIN`

Any query that misses one of these conditions falls back to the host
interpreter.

### Multi-GPU

`run_multi_gpu_jit_host` splits rows evenly across visible devices, then loops
over them: set device, upload the shard, launch, synchronize, copy back, next
device. Because each iteration synchronizes before the next begins, the devices
never work concurrently. The value of this path is that a dataset larger than
one device's memory can be processed at all; it is not a throughput
optimization, and it has not been measured to be faster than the single-device
path.

### Optimizer

`src/optimizer.cpp` provides column statistics, predicate analysis
(always-true / always-false detection) and greedy join-order planning. None of
it is called by `query()` or `query_sql()` — the routines are exercised only by
`tests/optimizer_test.cpp`. No filter pushdown, join reordering or kernel fusion
is applied to any query you can currently run.

## How It Works

This describes the `query()` path. For `query_sql()`, steps 4-6 are replaced by
host-side interpretation — see [Execution model](#execution-model).

1. **CSV Loading**: Input data is loaded from CSV files directly into GPU memory.
2. **Columnar Loading**: Parquet, Arrow, and ORC files are read via Apache Arrow and moved to GPU memory.
3. **Query Parsing**: User queries are tokenized and parsed into an AST.
4. **Code Generation**: The AST is converted into CUDA code.
5. **JIT Compilation**: The generated code is compiled into a CUDA kernel using NVRTC (cached per expression).
6. **Execution**: The compiled kernel is executed on the GPU.
7. **Result Retrieval**: Results are copied back to host memory as `float32` and displayed.

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
- Column statistics and predicate analysis exist as standalone routines (not
  yet wired into query execution).
- RAII wrappers manage CUDA contexts and modules to avoid resource leaks.
- Helper functions demonstrate streaming across multiple GPUs.
- Python bindings are built when `pybind11` is installed and
  `-DWARPDB_BUILD_PYTHON=ON`.

## Limitations

These are load-bearing. Read them before assuming WarpDB fits a use case.

**Execution**

- `query_sql()` runs on the host as a row-at-a-time interpreter. Only `query()`
  and one narrow GROUP BY shape use the GPU — see
  [Execution model](#execution-model)
- Multi-GPU execution is sequential, so it adds capacity, not throughput
- The optimizer is not connected to query execution

**Types and correctness**

- Results are `float32`. Every literal is emitted as a float and aggregates
  accumulate into float, so integer and fixed-point values lose exactness beyond
  24 bits of mantissa. There is no `DECIMAL` type — WarpDB is not suitable for
  monetary or exact-arithmetic workloads
- There is no `NULL` type and no three-valued logic. Null counts are gathered at
  load time and then discarded; nulls do not propagate through expressions or
  affect aggregates
- No date, time, or timestamp types
- String columns can be loaded and compared for equality, but there is no
  general string function support

**SQL surface**

- No subqueries, CTEs, `UNION`, `CASE`, `IN`, `LIKE`, `BETWEEN`, or `CAST`
- Window functions parse into an AST node but are never executed
- `GROUP BY` supports one grouping key and one `SELECT` expression
- `JOIN` is limited to a single inner equi-join, is not supported together with
  `GROUP BY`, and is not supported with `SELECT *`
- Limited error handling for malformed queries

**Not a database**

- No persistence, storage engine, catalog, indexes, transactions, or
  concurrency control. One instance loads one file and answers queries about it
- No `INSERT` / `UPDATE` / `DELETE` / `CREATE TABLE`

**Build**

- CSV and JSON paths are the most mature; JSON loading reads newline-delimited
  objects and infers each column's type, with an optional user-supplied schema
  map for deterministic typing
- Loading Parquet/Arrow/ORC files requires Apache Arrow
- Building the Python module requires `pybind11`, or disable it with
  `-DWARPDB_BUILD_PYTHON=OFF`

## Future Improvements

Roughly in order of how much they would change what WarpDB is:

- Execute the `query_sql()` path on the GPU instead of interpreting it on the
  host, so the SQL surface and the acceleration stop being disjoint
- Carry column types through execution rather than collapsing results to
  `float32`
- Represent and propagate `NULL`
- Wire the existing statistics and predicate analysis into query planning
- Overlap multi-GPU work across devices instead of driving them serially
- Establish a benchmark against an established engine on a real dataset, so
  performance can be discussed with evidence
- Better error handling and query validation
- Additional data source support (e.g. Avro)

## License

WarpDB is licensed under the [Apache License 2.0](LICENSE).
