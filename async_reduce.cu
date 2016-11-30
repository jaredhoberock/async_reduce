#include <iostream>
#include <agency/cuda.hpp>

#include "async_reduce.hpp"
#include "measure_bandwidth_of_invocation.hpp"

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

