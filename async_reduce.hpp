#pragma once

#include <utility>

#include <agency/agency.hpp>
#include <agency/experimental.hpp>
#include <agency/cuda.hpp>

#include "cooperative_reduce.hpp"
#include "collective_reducer.hpp"

template<size_t block_size, size_t grain_size = 1, size_t heap_size = 0>
auto static_grid(int num_blocks) ->
  decltype(agency::cuda::par(num_blocks, agency::cuda::experimental::static_concurrent_execution_policy<block_size, grain_size, heap_size>()))
{
  return agency::cuda::par(num_blocks, agency::cuda::experimental::static_concurrent_execution_policy<block_size, grain_size, heap_size>());
}

template<size_t block_size, size_t grain_size = 1, size_t heap_size = 0>
using static_grid_agent = agency::parallel_group<agency::experimental::static_concurrent_agent<block_size, grain_size, heap_size>>;


template<class Range, class T, class BinaryOperation>
agency::cuda::async_future<T> async_reduce(Range&& input, T init, BinaryOperation binary_op)
{
  if(input.size() == 0) return agency::cuda::make_ready_async_future<T>(std::move(init));

  using namespace agency;
  using namespace agency::cuda;
  using namespace agency::experimental;
  using namespace ::experimental;

  static const constexpr int group_size = 128;
  static const constexpr int grain_size = 8;

  // XXX might be a better idea to tile_evenly to ensure
  //     that the number of partial sums fits into the second kernel
  static const constexpr int tile_size = group_size * grain_size;
  size_t num_groups = (input.size() + tile_size - 1) / tile_size;

  // make a view of the input that we can capture by value
  auto input_view = all(std::forward<Range>(input));

  auto tile_sums_future = bulk_async(static_grid<128,8>(num_groups),
    [=] __host__ __device__ (static_grid_agent<128,8>& self)
    {
      // find this group's tile of the input
      auto this_tile = tile(input_view, tile_size)[self.outer().rank()];

      // sum across the tile
      auto tile_sum = cooperative_uninitialized_reduce(self.inner(), this_tile, binary_op);

      // rank 0 returns the result
      return self.inner().rank() == 0 ? scope_result<1,T>(tile_sum) : no_result<T>();
    }
  );

  using tile_sums_t = typename future_traits<decltype(tile_sums_future)>::value_type;

  return bulk_then(static_grid<512,4>(1),
    [=] __host__ __device__ (static_grid_agent<512,4>& self, tile_sums_t& tile_sums)
    {
      // sum across the tile
      auto tile_sum = cooperative_reduce(self.inner(), tile_sums, init, binary_op);

      // rank 0 returns the result
      return self.inner().rank() == 0 ? single_result<T>(tile_sum) : no_result<T>();
    },
    tile_sums_future
  );
}

