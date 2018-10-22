#pragma once

#include <torch/tensor.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace data {
namespace samplers {
/// A `Sampler` is an object that yields an index with which to access a
/// dataset.
template <typename Index = std::vector<size_t>>
class Sampler {
 public:
  using IndexType = Index;

  virtual ~Sampler() = default;

  /// Resets the `Sampler`'s internal state.
  /// Typically called before a new epoch.
  virtual void reset() = 0;

  /// Returns the next index if possible, or an empty optional if the
  /// sampler is exhausted for this epoch.
  virtual optional<Index> next(size_t batch_size) = 0;
};

} // namespace samplers
} // namespace data
} // namespace torch
