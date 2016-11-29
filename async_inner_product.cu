#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>

#include <agency/agency.hpp>
#include <agency/experimental.hpp>
#include <agency/cuda.hpp>

#include "cooperative_reduce.hpp"
#include "collective_reducer.hpp"

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


struct multiplies
{
  template<class T1, class T2>
  __host__ __device__
  auto operator()(const T1& a, const T2& b) const -> decltype(a * b)
  {
    return a * b;
  }
};


template<class Range1, class Range2, class T, class BinaryOperation1, class BinaryOperation2>
agency::cuda::async_future<T> async_inner_product(Range1&& input1, Range2&& input2, T init, BinaryOperation1 binary_op1, BinaryOperation2 binary_op2)
{
  return agency::cuda::make_ready_async_future<T>(std::move(init));
}


template<class Range1, class Range2, class T>
agency::cuda::async_future<T> async_inner_product(Range1&& input1, Range2&& input2, T init)
{
  return async_inner_product(std::forward<Range1>(input1), std::forward<Range2>(input2), init, plus(), multiplies());
}


template<class Float>
bool almost_equal(Float a, Float b, Float epsilon = 0.0001f)
{
  return std::abs(a - b) <= epsilon;
}


int main(int argc, char** argv)
{
  using namespace agency;

  size_t tile_size = 128 * 8;
  std::vector<size_t> sizes = {0, 1, tile_size - 1, tile_size, tile_size + 1, 100000 * tile_size + 13, (1 << 30) + 13};

  for(auto n : sizes)
  {
    std::cout << "n: " << n << std::endl;

    std::vector<float, cuda::allocator<float>> input1(n);
    std::vector<float, cuda::allocator<float>> input2(n);

    std::mt19937_64 rng;
    std::uniform_real_distribution<float> dist;

    std::generate(input1.begin(), input1.end(), [&]{ return dist(rng); });
    std::generate(input2.begin(), input2.end(), [&]{ return dist(rng); });

    float init = dist(rng);

    // compare to reference
    float reference = std::inner_product(input1.begin(), input1.end(), input2.begin(), init);
    float result = async_inner_product(input1, input2, init).get();

    assert(almost_equal(reference, result));
  }

  size_t n = (1 << 30) + 13;

  std::vector<float, cuda::allocator<float>> input1(n);
  std::vector<float, cuda::allocator<float>> input2(n);

  std::default_random_engine rng;
  std::uniform_real_distribution<float> dist;

  std::generate(input1.begin(), input1.end(), [&]{ return dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]{ return dist(rng); });

  float init = dist(rng);

  // warm up
  async_inner_product(input1, input2, init);

  std::cout << "Measuring performance" << std::endl;

  auto bandwidth = measure_bandwidth_of_invocation_in_gigabytes_per_second(100, 2 * sizeof(float) * n, [&]
  {
    async_inner_product(input1, input2, init);
  });

  std::cout << "Mean bandwidth: " << bandwidth << " GB/s " << std::endl;

  std::cout << "OK" << std::endl;

  return 0;
}

