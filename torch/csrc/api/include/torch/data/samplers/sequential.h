#pragma once

#include <torch/data/samplers/base.h>
#include <torch/tensor.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace data {
namespace samplers {

/// A `Sampler` that returns indices sequentially.
class SequentialSampler : public Sampler<> {
 public:
  /// Creates a `SequentialSampler` that will return indices in the range
  /// `0...size - 1`.
  explicit SequentialSampler(size_t size);

  /// Resets the `SequentialSampler` to zero.
  void reset() override;

  /// Returns the next batch of indices.
  optional<std::vector<size_t>> next(size_t batch_size) override;

 private:
  size_t size_;
  size_t index_{0};
};

} // namespace samplers
} // namespace data
} // namespace torch
