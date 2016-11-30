#include <iostream>
#include <numeric>
#include <random>

#include <agency/cuda.hpp>

#include "async_inner_product.hpp"
#include "measure_bandwidth_of_invocation.hpp"


int main(int argc, char** argv)
{
  using namespace agency;

  size_t tile_size = 128 * 8;
  std::vector<size_t> sizes = {0, 1, tile_size - 1, tile_size, tile_size + 1, 100000 * tile_size + 13, (1 << 30) + 13};

  for(auto n : sizes)
  {
    std::cout << "n: " << n << std::endl;

    std::vector<int, cuda::allocator<int>> input1(n);
    std::vector<int, cuda::allocator<int>> input2(n);

    std::mt19937 rng;

    std::generate(input1.begin(), input1.end(), rng);
    std::generate(input2.begin(), input2.end(), rng);

    int init = rng();

    // compare to reference
    int reference = std::inner_product(input1.begin(), input1.end(), input2.begin(), init);
    int result = async_inner_product(input1, input2, init).get();

    std::cout << "reference: " << reference << std::endl;
    std::cout << "result: " << result << std::endl;

    assert(reference == result);
  }

  size_t n = ((1 << 30) + 13) / 2;

  std::vector<int, cuda::allocator<int>> input1(n);
  std::vector<int, cuda::allocator<int>> input2(n);

  std::mt19937 rng;

  std::generate(input1.begin(), input1.end(), rng);
  std::generate(input2.begin(), input2.end(), rng);

  int init = rng();

  // warm up
  async_inner_product(input1, input2, init).get();

  std::cout << "Measuring performance" << std::endl;

  auto bandwidth = measure_bandwidth_of_invocation_in_gigabytes_per_second(100, 2 * sizeof(float) * n, [&]
  {
    async_inner_product(input1, input2, init);
  });

  std::cout << "Mean bandwidth: " << bandwidth << " GB/s " << std::endl;

  std::cout << "OK" << std::endl;

  return 0;
}

