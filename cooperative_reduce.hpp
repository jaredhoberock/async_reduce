#pragma once

#include <utility>
#include <agency/agency.hpp>
#include <agency/experimental.hpp>
#include <agency/execution/executor/experimental/unrolling_executor.hpp>
#include "collective_reducer.hpp"


namespace experimental
{


// XXX unrolling stuff should go in its own file
template<size_t factor_>
class unrolling_execution_policy : public agency::basic_execution_policy<
  agency::sequenced_agent,
  agency::experimental::unrolling_executor<factor_>,
  unrolling_execution_policy<factor_>
>
{
  private:
    using super_t = agency::basic_execution_policy<
      agency::sequenced_agent,
      agency::experimental::unrolling_executor<factor_>,
      unrolling_execution_policy<factor_>
    >;

  public:
    static constexpr size_t factor = factor_;

    using super_t::basic_execution_policy;
};

template<size_t factor>
__AGENCY_ANNOTATION
unrolling_execution_policy<factor> unroll()
{
  return unrolling_execution_policy<factor>{};
}


template<class ExecutionPolicy, class Range, class Function>
__host__ __device__
void for_each(ExecutionPolicy policy, Range&& rng, Function f)
{
  auto exec = policy.executor();

  exec.bulk_sync_execute([&](size_t i, int, int)
  {
    std::forward<Function>(f)(rng[i]);
  },
  rng.size(),
  []{ return 0; },
  []{ return 0; }
  );
}


template<class Range, class Function>
__host__ __device__
void for_each(agency::sequenced_execution_policy, Range&& rng, Function f)
{
  for(auto i = rng.begin(); i != rng.end(); ++i)
  {
    std::forward<Function>(f)(*i);
  }
}


template<class ExecutionPolicy, class Range, class T, class BinaryOperator>
__host__ __device__
T reduce(ExecutionPolicy policy, Range&& rng, T init, BinaryOperator binary_op)
{
  using reference = typename agency::experimental::range_value_t<Range>;

  for_each(policy, std::forward<Range>(rng), [&](reference x)
  {
    init = binary_op(init, x);
  });

  return init;
}


template<class ExecutionPolicy, class Range, class BinaryOperator>
__host__ __device__
agency::experimental::range_value_t<Range>
  reduce_nonempty(ExecutionPolicy policy, Range&& rng, BinaryOperator binary_op)
{
  return reduce(policy, agency::experimental::drop(std::forward<Range>(rng), 1), std::forward<Range>(rng)[0], binary_op);
}


template<class ExecutionPolicy, class Range, class BinaryOperator>
__host__ __device__
agency::experimental::optional<agency::experimental::range_value_t<Range>>
  uninitialized_reduce(ExecutionPolicy policy, Range&& rng, BinaryOperator binary_op)
{
  if(!std::forward<Range>(rng).empty())
  {
    return reduce_nonempty(policy, std::forward<Range>(rng), binary_op);
  }

  return agency::experimental::nullopt;
}


template<std::size_t group_size, std::size_t grain_size, std::size_t heap_size, class Range, class BinaryOperator>
__host__ __device__
agency::experimental::range_value_t<Range>
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


template<std::size_t group_size, std::size_t grain_size, std::size_t heap_size, class Range, class T, class BinaryOperator>
__host__ __device__
T cooperative_reduce(agency::experimental::static_concurrent_agent<group_size, grain_size, heap_size>& self,
                     collective_reducer<agency::experimental::range_value_t<Range>, (int)group_size>& reducer,
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


} // end experimental

