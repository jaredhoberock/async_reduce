#pragma once

#include <utility>

#include <agency/cuda.hpp>
#include <agency/experimental.hpp>

#include "async_reduce.hpp"

template<class Range1, class Range2, class T, class BinaryOperation1, class BinaryOperation2>
agency::cuda::async_future<T> async_inner_product(Range1&& input1, Range2&& input2, T init, BinaryOperation1 binary_op1, BinaryOperation2 binary_op2)
{
  // get all of the input ranges to workaround problems with passing std::vector::iterator to __device__ functions
  auto all1 = agency::experimental::all(std::forward<Range1>(input1));
  auto all2 = agency::experimental::all(std::forward<Range2>(input2));

  auto zipped = agency::experimental::zip_with(binary_op2, all1, all2);
  return async_reduce(zipped, init, binary_op1);
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


struct multiplies
{
  template<class T1, class T2>
  __host__ __device__
  auto operator()(const T1& a, const T2& b) const -> decltype(a * b)
  {
    return a * b;
  }
};


template<class Range1, class Range2, class T>
agency::cuda::async_future<T> async_inner_product(Range1&& input1, Range2&& input2, T init)
{
  return async_inner_product(std::forward<Range1>(input1), std::forward<Range2>(input2), init, plus(), multiplies());
}

