#include <iostream>
#include <agency/agency.hpp>
#include <agency/experimental.hpp>
#include <agency/cuda.hpp>

#include "cooperative_reduce.hpp"
#include "collective_reducer.hpp"

#include "measure_bandwidth_of_invocation.hpp"

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

  using phase_one_reducer_t = collective_reducer<T,group_size>;

  auto tile_sums_future = bulk_async(static_grid<128,8>(num_groups),
    [=] __host__ __device__ (static_grid_agent<128,8>& self, phase_one_reducer_t& reducer)
    {
      // find this group's tile of the input
      auto this_tile = tile(input_view, tile_size)[self.outer().rank()];

      // sum across the tile
      auto tile_sum = cooperative_uninitialized_reduce(self.inner(), reducer, this_tile, binary_op);

      // rank 0 returns the result
      return self.inner().rank() == 0 ? scope_result<1,T>(tile_sum) : no_result<T>();
    },
    share_at_scope<1,phase_one_reducer_t>()
  );

  using tile_sums_t = typename future_traits<decltype(tile_sums_future)>::value_type;

  using phase_two_reducer_t = collective_reducer<T,512>;

  return bulk_then(static_grid<512,4>(1),
    [=] __host__ __device__ (static_grid_agent<512,4>& self, tile_sums_t& tile_sums, phase_two_reducer_t& reducer)
    {
      // sum across the tile
      auto tile_sum = cooperative_reduce(self.inner(), reducer, tile_sums, init, binary_op);

      // rank 0 returns the result
      return self.inner().rank() == 0 ? single_result<T>(tile_sum) : no_result<T>();
    },
    tile_sums_future,
    share_at_scope<1,phase_two_reducer_t>()
  );
}


struct plus
{
  template<class T1, class T2>
  __host__ __device__
  auto operator()(const T1& a, const T2& b) const -> decltype(a + b)
  {
    return a + b;
  }
};


int main(int argc, char** argv)
{
  using namespace agency;

  size_t tile_size = 128 * 8;
  std::vector<size_t> sizes = {0, 1, tile_size - 1, tile_size, tile_size + 1, 100000 * tile_size + 13, (1 << 30) + 13};

  for(auto n : sizes)
  {
    std::cout << "n: " << n << std::endl;

    // Prepare the fibonacci numbers on the host.
    std::vector<int, cuda::allocator<int>> input(n);
    for(int i = 0; i < input.size(); ++i)
      input[i] = (i + 1) * (i + 1);

    int init = 13;

    // compare to reference
    assert(std::accumulate(input.begin(), input.end(), init, plus()) == async_reduce(input, init, plus()).get());
  }

  size_t n = (1 << 30) + 13;

  // Prepare the fibonacci numbers on the host.
  std::vector<int, cuda::allocator<int>> input(n);
  for(int i = 0; i < input.size(); ++i)
    input[i] = (i + 1) * (i + 1);

  // warm up
  async_reduce(input, 0, plus()).get();

  std::cout << "Measuring performance" << std::endl;

  auto bandwidth = measure_bandwidth_of_invocation_in_gigabytes_per_second(100, sizeof(int) * n, [&]
  {
    async_reduce(input, 0, plus());
  });

  std::cout << "Mean bandwidth: " << bandwidth << " GB/s " << std::endl;

  std::cout << "OK" << std::endl;

  return 0;
}

