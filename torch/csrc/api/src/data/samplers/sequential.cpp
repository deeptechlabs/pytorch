#include <torch/data/samplers/sequential.h>
#include <torch/tensor.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace torch {
namespace data {
namespace samplers {
SequentialSampler::SequentialSampler(size_t size) : size_(size) {}

void SequentialSampler::reset() {
  index_ = 0;
}

optional<std::vector<size_t>> SequentialSampler::next(size_t batch_size) {
  const auto remaining_indices = size_ - index_;
  if (remaining_indices == 0) {
    return nullopt;
  }
  std::vector<size_t> index_batch(std::min(batch_size, remaining_indices));
  for (auto& i : index_batch) {
    i = index_++;
  }
  return index_batch;
}

} // namespace samplers
} // namespace data
} // namespace torch
