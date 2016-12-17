#pragma once

#include <utility>
#include <agency/agency.hpp>
#include <agency/experimental.hpp>

#include "reduce.hpp"
#include "collective_reducer.hpp"


namespace experimental
{


template<std::size_t group_size, std::size_t grain_size, std::size_t heap_size, int reducer_size, class Range, class T, class BinaryOperator>
__AGENCY_ANNOTATION
T cooperative_reduce(agency::experimental::static_concurrent_agent<group_size, grain_size, heap_size>& self,
                     collective_reducer<agency::experimental::range_value_t<Range>, reducer_size>& reducer,
                     Range&& rng,
                     T init,
                     BinaryOperator binary_op)
{
  using namespace agency::experimental;

  auto agent_rank = self.rank();

  // each agent strides through its group's chunk of the input...
  auto my_values = stride(drop(rng, agent_rank), size_t(group_size));
  
  // ...and sequentially computes a partial sum
  auto partial_sum = uninitialized_reduce(unroll<grain_size>(), my_values, binary_op);
  
  // the entire group cooperatively reduces the partial sums
  int num_partials = rng.size() < group_size ? rng.size() : group_size;

  return reducer.reduce(self, partial_sum, num_partials, init, binary_op);
}


template<class ConcurrentAgent, int reducer_size, class Range, class T, class BinaryOperator>
__AGENCY_ANNOTATION
T cooperative_reduce(ConcurrentAgent& self,
                     collective_reducer<agency::experimental::range_value_t<Range>, reducer_size>& reducer,
                     Range&& rng,
                     T init,
                     BinaryOperator binary_op)
{
  using namespace agency::experimental;

  auto agent_rank = self.rank();
  auto group_size = self.group_size();

  // each agent strides through its group's chunk of the input...
  auto my_values = stride(drop(rng, agent_rank), group_size);
  
  // ...and sequentially computes a partial sum
  auto partial_sum = uninitialized_reduce(agency::seq, my_values, binary_op);
  
  // the entire group cooperatively reduces the partial sums
  int num_partials = rng.size() < group_size ? rng.size() : group_size;

  return reducer.reduce(self, partial_sum, num_partials, init, binary_op);
}


// special case for when the heap_size is smaller than the size of the collective_reducer type we need to allocate
template<std::size_t group_size, std::size_t grain_size, std::size_t heap_size, class Range, class T, class BinaryOperator,
         __AGENCY_REQUIRES(
           heap_size < sizeof(
             collective_reducer<agency::experimental::range_value_t<Range>, group_size>
           )
         )>
__AGENCY_ANNOTATION
T cooperative_reduce(agency::experimental::static_concurrent_agent<group_size, grain_size, heap_size>& self,
                     Range&& rng, T init, BinaryOperator binary_op)
{
#ifdef __CUDA_ARCH__
  using value_type = agency::experimental::range_value_t<Range>;

  using reducer_type = collective_reducer<value_type, group_size>;

  // put the reducer directly in __shared__ memory
  __shared__ reducer_type reducer;

  return cooperative_reduce(self, reducer, std::forward<Range>(rng), init, binary_op);
#else
  throw std::bad_alloc();
#endif
}


// this is the overload of cooperative_reduce() for static_concurrent_agent with large heap_size
template<std::size_t group_size, std::size_t grain_size, std::size_t heap_size, class Range, class T, class BinaryOperator,
         __AGENCY_REQUIRES(
           heap_size >= sizeof(
             collective_reducer<agency::experimental::range_value_t<Range>, group_size>
           )
         )>
__AGENCY_ANNOTATION
T cooperative_reduce(agency::experimental::static_concurrent_agent<group_size, grain_size, heap_size>& self,
                     Range&& rng, T init, BinaryOperator binary_op)
{
  using agent_type = agency::experimental::static_concurrent_agent<group_size, grain_size, heap_size>;
  using value_type = agency::experimental::range_value_t<Range>;

  // note that the collective_reducer has a static size
  using reducer_type = collective_reducer<value_type, group_size>;

  // create a shared reducer
  agency::shared<reducer_type, agent_type> reducer(self);

  return cooperative_reduce(self, reducer.value(), std::forward<Range>(rng), init, binary_op);
}


// this is the overload of cooperative_reduce() for generic ConcurrentAgent
template<class ConcurrentAgent, class Range, class T, class BinaryOperator>
__AGENCY_ANNOTATION
T cooperative_reduce(ConcurrentAgent& self, Range&& rng, T init, BinaryOperator binary_op)
{
  using value_type = agency::experimental::range_value_t<Range>;

  // note that the collective_reducer has a dynamic size
  using reducer_type = collective_reducer<value_type>;

  // create a shared reducer
  agency::shared<reducer_type, ConcurrentAgent> reducer(self, self.group_size(), self.memory_resource());

  return cooperative_reduce(self, reducer.value(), std::forward<Range>(rng), init, binary_op);
}


// XXX we should attempt to implement this with cooperative_reduce()
template<std::size_t group_size, std::size_t grain_size, std::size_t heap_size, class Range, class BinaryOperator>
__AGENCY_ANNOTATION
agency::experimental::optional<agency::experimental::range_value_t<Range>>
  cooperative_uninitialized_reduce(agency::experimental::static_concurrent_agent<group_size, grain_size, heap_size>& self,
                                   collective_reducer<agency::experimental::range_value_t<Range>, (int)group_size>& reducer,
                                   Range&& rng,
                                   BinaryOperator binary_op)
{
  using namespace agency::experimental;

  auto agent_rank = self.rank();

  // each agent strides through its group's chunk of the input...
  auto my_values = stride(drop(rng, agent_rank), size_t(group_size));
  
  // ...and sequentially computes a partial sum
  auto partial_sum = uninitialized_reduce(unroll<grain_size>(), my_values, binary_op);
  
  // the entire group cooperatively reduces the partial sums
  int num_partials = rng.size() < group_size ? rng.size() : group_size;
  
  return reducer.reduce(self, partial_sum, num_partials, binary_op);
}


// special case for when the heap_size is smaller than the size of the collective_reducer type we need to allocate
// XXX we should attempt to implement this with cooperative_reduce()
template<std::size_t group_size, std::size_t grain_size, std::size_t heap_size, class Range, class BinaryOperator,
         __AGENCY_REQUIRES(
           heap_size < sizeof(
             collective_reducer<agency::experimental::range_value_t<Range>, group_size>
           )
         )>
__AGENCY_ANNOTATION
agency::experimental::optional<agency::experimental::range_value_t<Range>>
  cooperative_uninitialized_reduce(agency::experimental::static_concurrent_agent<group_size, grain_size, heap_size>& self,
                                   Range&& rng,
                                   BinaryOperator binary_op)
{
#ifdef __CUDA_ARCH__
  using value_type = agency::experimental::range_value_t<Range>;

  using reducer_type = collective_reducer<value_type, group_size>;

  // put the reducer directly in __shared__ memory
  __shared__ reducer_type reducer;

  return cooperative_uninitialized_reduce(self, reducer, std::forward<Range>(rng), binary_op);
#else
  throw std::bad_alloc();
#endif
}


} // end experimental

