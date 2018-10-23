#pragma once

#include <torch/data/samplers/base.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace data {
namespace samplers {

/// A `Sampler` that returns random indices.
class RandomSampler : public Sampler<> {
 public:
  /// Constructs a `RandomSampler` with a size and dtype for the stored indices.
  ///
  /// The constructor will eagerly allocate all required indices, which is the
  /// sequence `0 ... size - 1`. `index_dtype` is the data type of the stored
  /// indices. You can change it to influence memory usage.
  explicit RandomSampler(int64_t size, Dtype index_dtype = torch::kInt64);

  /// Resets the `RandomSampler` to a new set of indices.
  void reset() override;

  /// Returns the next batch of indices.
  optional<std::vector<size_t>> next(size_t batch_size) override;

 private:
  Tensor indices_;
  int64_t index_ = 0;
};
} // namespace samplers
} // namespace data
} // namespace torch
