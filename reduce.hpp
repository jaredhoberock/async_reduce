#pragma once

#include <agency/agency.hpp>
#include <agency/experimental.hpp>
#include <agency/execution/executor/experimental/unrolling_executor.hpp>

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
__AGENCY_ANNOTATION
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
__AGENCY_ANNOTATION
void for_each(agency::sequenced_execution_policy, Range&& rng, Function f)
{
  for(auto i = rng.begin(); i != rng.end(); ++i)
  {
    std::forward<Function>(f)(*i);
  }
}


template<class ExecutionPolicy, class Range, class T, class BinaryOperator>
__AGENCY_ANNOTATION
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
__AGENCY_ANNOTATION
agency::experimental::range_value_t<Range>
  reduce_nonempty(ExecutionPolicy policy, Range&& rng, BinaryOperator binary_op)
{
  return reduce(policy, agency::experimental::drop(std::forward<Range>(rng), 1), std::forward<Range>(rng)[0], binary_op);
}


template<class ExecutionPolicy, class Range, class BinaryOperator>
__AGENCY_ANNOTATION
agency::experimental::optional<agency::experimental::range_value_t<Range>>
  uninitialized_reduce(ExecutionPolicy policy, Range&& rng, BinaryOperator binary_op)
{
  if(!std::forward<Range>(rng).empty())
  {
    return reduce_nonempty(policy, std::forward<Range>(rng), binary_op);
  }

  return agency::experimental::nullopt;
}


} // end experimental

