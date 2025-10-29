#include "arrow_loader.hpp"
#include "csv_loader.hpp"
#include <cassert>
#include <iostream>

int main() {
#ifndef USE_ARROW
  std::cout << "Arrow not enabled; skipping test.\n";
  return 0;
#else
  std::shared_ptr<arrow::cuda::CudaDeviceManager> manager;
  arrow::Status status = arrow::cuda::CudaDeviceManager::GetInstance(&manager);
  if (!status.ok()) {
    std::cout << "Arrow CUDA unavailable: " << status.ToString() << "\n";
    return 0;
  }

  std::shared_ptr<arrow::cuda::CudaContext> context;
  status = manager->GetContext(0, &context);
  if (!status.ok()) {
    std::cout << "Failed to acquire CUDA context: " << status.ToString() << "\n";
    return 0;
  }

  Table table = load_csv_to_gpu("data/test.csv");
  assert(table.num_rows == 4);
  assert(table.columns.size() == 2);

  const ColumnDesc &price = table.columns[0];
  const ColumnDesc &quantity = table.columns[1];

  assert(price.name == "price");
  assert(quantity.name == "quantity");
  assert(price.length == table.num_rows);
  assert(quantity.length == table.num_rows);

  std::cout << "Arrow loader smoke test passed\n";
  return 0;
#endif
}
