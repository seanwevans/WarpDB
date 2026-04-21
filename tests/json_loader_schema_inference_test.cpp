#include "json_loader.hpp"

#include <cassert>
#include <iostream>
#include <unordered_map>

int main() {
  HostTable inferred = load_json_to_host("data/arbitrary.json", ParsePolicy::Strict);
  assert(inferred.columns.size() == 3);

  const HostColumn *id_col = inferred.get_column("id");
  const HostColumn *name_col = inferred.get_column("name");
  const HostColumn *score_col = inferred.get_column("score");

  assert(id_col != nullptr && id_col->type == DataType::Int32);
  assert(name_col != nullptr && name_col->type == DataType::String);
  assert(score_col != nullptr && score_col->type == DataType::Float32);

  const auto &ids = std::get<std::vector<int32_t>>(id_col->data);
  const auto &names = std::get<std::vector<std::string>>(name_col->data);
  const auto &scores = std::get<std::vector<float>>(score_col->data);

  assert(ids.size() == 3 && names.size() == 3 && scores.size() == 3);
  assert(ids[0] == 1 && names[1] == "beta");

  std::unordered_map<std::string, DataType> schema = {
      {"id", DataType::Int64},
      {"name", DataType::String},
      {"score", DataType::Float64},
  };
  HostTable typed = load_json_to_host("data/arbitrary.json", schema, ParsePolicy::Strict);
  assert(typed.get_column("id")->type == DataType::Int64);
  assert(typed.get_column("score")->type == DataType::Float64);

  std::cout << "json loader schema inference test passed\n";
  return 0;
}
