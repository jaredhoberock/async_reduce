#pragma once

#include <agency/agency.hpp>
#include <agency/experimental.hpp>
#include <cstddef>

#include "reduce.hpp"


namespace experimental
{


// XXX this bounded_ stuff should go in some other file
template<size_t bound>
struct bounded_executor
{
  using execution_category = agency::sequenced_execution_tag;

  __AGENCY_ANNOTATION
  constexpr std::size_t unit_shape() const
  {
    return bound;
  }

  template<class Function, class ResultFactory, class SharedFactory>
  __AGENCY_ANNOTATION
  typename std::result_of<ResultFactory()>::type
    bulk_sync_execute(Function f, std::size_t n, ResultFactory result_factory, SharedFactory shared_factory)
  {
    auto result = result_factory();
    auto shared_arg = shared_factory();

    for(size_t idx = 0; idx < bound; ++idx)
    {
      if(idx < n)
      {
        f(idx, result, shared_arg);
      }
    }

    return std::move(result);
  }
};


template<size_t bound_>
class bounded_execution_policy : public agency::basic_execution_policy<
  agency::sequenced_agent,
  bounded_executor<bound_>,
  bounded_execution_policy<bound_>
>
{
  private:
    using super_t = agency::basic_execution_policy<
      agency::sequenced_agent,
      bounded_executor<bound_>,
      bounded_execution_policy<bound_>
    >;

  public:
    static constexpr size_t bound = bound_;

    using super_t::basic_execution_policy;
};

template<size_t b>
__AGENCY_ANNOTATION
bounded_execution_policy<b> bound()
{
  return bounded_execution_policy<b>{};
}


template<class T>
__AGENCY_ANNOTATION
constexpr T minimum(const T& a, const T& b)
{
  return (b < a) ? b : a;
}

template<class T>
__AGENCY_ANNOTATION
constexpr T maximum(const T& a, const T& b)
{
  return (a < b) ? b : a;
}


template<class Integer>
__AGENCY_ANNOTATION
constexpr bool is_pow2(Integer x)
{
  return 0 == (x & (x - 1));
}


template<class Integer>
__AGENCY_ANNOTATION
constexpr Integer div_up(Integer numerator, Integer denominator)
{
  return (numerator + denominator - Integer(1)) / denominator;
}


template<class Integer>
__AGENCY_ANNOTATION
constexpr Integer log2(Integer x)
{
  return (x > 1) ? (1 + log2(x/2)) : 0;
}


#ifdef __NVCC__

// XXX seems like this is really only valid for POD
// XXX if we destroy x and properly construct the result, how correct does that make it?
template<class T>
__device__
T shuffle_down(const T& x, int offset, int width)
{ 
  constexpr std::size_t num_words = div_up(sizeof(T), sizeof(int));

  union
  {
    int words[num_words];
    T value;
  } u;
  u.value = x;

  for(int i = 0; i < num_words; ++i)
  {
    u.words[i] = __shfl_down(u.words[i], offset, width);
  }

  return u.value;
}


template<class T>
__device__
agency::experimental::optional<T> optionally_shuffle_down(const agency::experimental::optional<T>& x, int offset, int width)
{
  constexpr std::size_t num_words = div_up(sizeof(T), sizeof(int));

  union
  {
    int words[num_words];
    T value;
  } u;

  if(x)
  {
    u.value = *x;
  }

  for(int i = 0; i < num_words; ++i)
  {
    u.words[i] = __shfl_down(u.words[i], offset, width);
  }

  // communicate whether or not the words we shuffled came from a valid object
  bool is_valid = x ? true : false;
  is_valid = __shfl_down(is_valid, offset, width);

  return is_valid ? agency::experimental::make_optional(u.value) : agency::experimental::nullopt;
}


// requires __CUDA_ARCH__ >= 300.
// num_threads can be any power-of-two <= warp_size.
// warp_reduce_t returns the reduction only in lane 0.
template<class T, int num_threads>
struct warp_reducer
{
  static_assert(num_threads <= 32 && is_pow2(num_threads), "warp_reducer must operate on a pow2 number of threads <= CUDA warp size (32)");

  template<class BinaryOperation>
  __device__
  static agency::experimental::optional<T> reduce_and_elect(int lane, agency::experimental::optional<T> x, int count, BinaryOperation binary_op)
  {
    if(count == num_threads)
    { 
      for(int pass = 0; pass < log2(num_threads); ++pass)
      {
        int offset = 1 << pass;
        auto y = shuffle_down(*x, offset, num_threads);
        x = binary_op(*x, y);
      }
    }
    else
    {
      for(int pass = 0; pass < log2(num_threads); ++pass)
      {
        int offset = 1 << pass;
        auto y = optionally_shuffle_down(x, offset, num_threads);
        if((lane + offset < count) && y) *x = binary_op(*x, *y);
      }
    }

    return (lane == 0) ? x : agency::experimental::nullopt;
  }

};


#endif


constexpr int dynamic_group_size = -1;


namespace detail
{


template<class ExecutionAgent>
struct execution_agent_static_group_size_impl {};


template<size_t group_size, size_t grain_size, size_t heap_size>
struct execution_agent_static_group_size_impl<agency::experimental::static_concurrent_agent<group_size,grain_size,heap_size>>
{
  using type = std::integral_constant<size_t, group_size>;
};


template<class ExecutionAgent>
using execution_agent_static_group_size = typename detail::execution_agent_static_group_size_impl<ExecutionAgent>::type;

template<class ExecutionAgent>
using execution_agent_has_static_group_size = agency::detail::is_detected<execution_agent_static_group_size, ExecutionAgent>;


template<class T, int group_size>
class basic_collective_reducer
{
  private:
    using storage_type = agency::experimental::array<T, maximum(group_size, 2 * minimum(group_size, 32))>;
    storage_type storage_;

  protected:
    static_assert(0 == group_size % 32, "group_size must be a multiple of warp_size (32)");


    __AGENCY_ANNOTATION
    basic_collective_reducer()
    {
      // XXX note that we're explicitly opting out of default constructing the storage_ array
      //     we do this because constructing the storage_ array yields a big slowdown
      // XXX we should probably only apply this optimization when T is POD
      // XXX we might consider a constructor which takes a ConcurrentAgent parameter
      //     we could concurrently construct the storage_ array to speed things up
      //     instead of making agent 0 do all the work sequentially
      // XXX alternatively, we could placement new into storage_ upon entry into reduce_and_elect()
      //     when we do the assignment:
      //
      //         storage_[agent_rank] = *partial_sum;
      //
      //     if we went in this direction, we'd need to destroy storage_[agent_rank] right before we exit the function
    }


    __AGENCY_ANNOTATION
    storage_type& storage()
    {
      return storage_;
    }

   
    // do_reduce_and_elect() implements the reduction logic
    // it assumes that initialize_storage() has been called to intialize the storage range with the values to reduce
    template<class ConcurrentAgent, class ContiguousRange, class BinaryOperation>
    __AGENCY_ANNOTATION
    static agency::experimental::optional<T> do_reduce_and_elect(ConcurrentAgent& self, ContiguousRange& storage, int count, BinaryOperation binary_op)
    {
      using namespace agency::experimental;

      static constexpr const int num_participating_agents = minimum(32, group_size);
      static constexpr const int num_sequential_sums_per_agent = group_size / num_participating_agents;

      optional<T> partial_sum;
      auto partial_sums = span<T>(storage.data(), count);
      auto agent_rank = self.rank();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
      using warp_barrier = warp_reducer<T, num_participating_agents>;

      if(agent_rank < num_participating_agents)
      {
        // stride through the input and compute a partial sum per agent
        auto my_partial_sums = stride(drop(partial_sums, agent_rank), num_participating_agents);

        partial_sum = uninitialized_reduce(bound<num_sequential_sums_per_agent>(), my_partial_sums, binary_op);

        // reduce across the warp
        partial_sum = warp_barrier::reduce_and_elect(agent_rank, partial_sum, minimum(count, num_participating_agents), binary_op);
      }
      self.wait();
#else
      if(agent_rank < num_participating_agents)
      {
        // stride through the input and compute a partial sum per agent
        auto my_partial_sums = stride(drop(partial_sums, agent_rank), num_participating_agents);

        partial_sum = uninitialized_reduce(bound<num_sequential_sums_per_agent>(), my_partial_sums, binary_op);

        if(partial_sum)
        {
          storage[agent_rank] = *partial_sum;
        }
      }
      self.wait();

      int num_partial_sums = minimum(count, num_participating_agents);

      static constexpr const int num_passes = log2(num_participating_agents);
      int first = (1 & num_passes) ? num_participating_agents : 0;

      if(agent_rank < num_participating_agents && partial_sum)
      {
        storage[first + agent_rank] = *partial_sum;
      }
      self.wait();


      int offset = 1;
      for(int pass = 0; pass < num_passes; ++pass, offset *= 2)
      {
        if(agent_rank < num_participating_agents)
        {
          if(agent_rank + offset < num_partial_sums) 
          {
            partial_sum = binary_op(*partial_sum, storage[first + offset + agent_rank]);
          }

          first = num_participating_agents - first;
          storage[first + agent_rank] = *partial_sum;
        }
        self.wait();
      }
#endif

      return agent_rank == 0 ? partial_sum : agency::experimental::nullopt;
    }
};


template<class T>
class simple_vector
{
  private:
    class deallocator
    {
      public:
        template<class MemoryResource>
        __AGENCY_ANNOTATION
        deallocator(MemoryResource& resource)
          : resource_ptr_(&resource),
            deallocate_function_(call_member_deallocate<MemoryResource>)
        {}

        __AGENCY_ANNOTATION
        void deallocate(T* ptr, size_t size)
        {
          deallocate_function_(resource_ptr_, ptr, size);
        }

      private:
        template<class MemoryResource>
        __AGENCY_ANNOTATION
        static void call_member_deallocate(void* resource_ptr, T* ptr, size_t size)
        {
          return reinterpret_cast<MemoryResource*>(resource_ptr)->deallocate(ptr, size);
        }

        void* resource_ptr_;
        void (*deallocate_function_)(void*, T*, size_t);
    };
    

  public:
    template<class MemoryResource>
    __AGENCY_ANNOTATION
    simple_vector(size_t size, MemoryResource& resource)
      : data_(static_cast<T*>(resource.allocate(size * sizeof(T)))),
        size_(size),
        deallocator_(resource)
    {}

    __AGENCY_ANNOTATION
    size_t size() const
    {
      return size_;
    }

    __AGENCY_ANNOTATION
    T& operator[](size_t i)
    {
      return data_[i];
    }

    __AGENCY_ANNOTATION
    const T& operator[](size_t i) const
    {
      return data_[i];
    }

  private:
    T* data_;
    size_t size_;
    deallocator deallocator_;
};


// XXX this requires collective destruction because shared_vector does
template<class T>
class basic_collective_reducer<T,dynamic_group_size>
{
  private:
    // XXX should find a way to use experimental::vector<T>
    using storage_type = simple_vector<T>;
    storage_type storage_;


  public:
    template<class MemoryResource>
    __AGENCY_ANNOTATION
    basic_collective_reducer(size_t size, MemoryResource& resource)
      : storage_(size, resource)
    {}


  protected:
    __AGENCY_ANNOTATION
    storage_type& storage()
    {
      return storage_;
    }


    // do_reduce_and_elect() implements the reduction logic
    // it assumes that initialize_storage() has been called to intialize the storage range with the values to reduce
    __agency_exec_check_disable__
    template<class ConcurrentAgent, class ContiguousRange, class BinaryOperation,
             __AGENCY_REQUIRES(
               !execution_agent_has_static_group_size<ConcurrentAgent>::value
             )>
    __AGENCY_ANNOTATION
    static agency::experimental::optional<T> do_reduce_and_elect(ConcurrentAgent& self, ContiguousRange& storage, int count, BinaryOperation binary_op)
    {
      auto agent_rank = self.rank();

      agency::experimental::optional<T> partial_sum;

      // each agent corresponding to a summand initializes its partial_sum
      if(agent_rank < count)
      {
        partial_sum = storage[agent_rank];
      }

      self.wait();

      while(count > 1)
      {
        if(agent_rank < count/2)
        {
          // accumulate from storage into partial_sum
          // XXX we might wish to preseve the order of parameters given to binary_op
          *partial_sum = binary_op(*partial_sum, storage[count - agent_rank - 1]);

          // store partial_sum back to storage
          // XXX the last time we do this, it's redundant
          storage[agent_rank] = *partial_sum;
        }

        // wait for every agent in the group to reach this point
        self.wait();

        // cut the number of active agents in half
        count -= count/2;
      }

      return agent_rank == 0 ? partial_sum : agency::experimental::nullopt;
    }
};


} // end detail


template<class T, int group_size = dynamic_group_size>
class collective_reducer : public detail::basic_collective_reducer<T,group_size>
{
  private:
    using super_t = detail::basic_collective_reducer<T,group_size>;

  public:
    using super_t::super_t;

    // reduce_and_elect() with init
    template<class ConcurrentAgent, class BinaryOperation>
    __AGENCY_ANNOTATION
    agency::experimental::optional<T> reduce_and_elect(ConcurrentAgent& self, const agency::experimental::optional<T>& value, int count, T init, BinaryOperation binary_op)
    {
      initialize_storage(self, super_t::storage(), value, count, init, binary_op);
      return super_t::do_reduce_and_elect(self, super_t::storage(), count, binary_op);
    }


    // reduce_and_elect() without init
    template<class ConcurrentAgent, class BinaryOperation>
    __AGENCY_ANNOTATION
    agency::experimental::optional<T> reduce_and_elect(ConcurrentAgent& self, const agency::experimental::optional<T>& value, int count, BinaryOperation binary_op)
    {
      initialize_storage(self, super_t::storage(), value, count);
      return super_t::do_reduce_and_elect(self, super_t::storage(), count, binary_op);
    }


    // reduce() with init
    template<class ConcurrentAgent, class BinaryOperation>
    __AGENCY_ANNOTATION
    T reduce(ConcurrentAgent& self, const agency::experimental::optional<T>& value, int count, T init, BinaryOperation binary_op)
    {
      auto result = reduce_and_elect(self, value, count, init, binary_op);

      // XXX we're using inside knowledge that reduce_and_elect() always elects agent_rank == 0
      if(self.rank() == 0)
      {
        super_t::storage()[0] = *result;
      }

      self.wait();

      return super_t::storage()[0];
    }


    // reduce() without init
    template<class ConcurrentAgent, class BinaryOperation>
    __AGENCY_ANNOTATION
    agency::experimental::optional<T> reduce(ConcurrentAgent& self, const agency::experimental::optional<T>& value, int count, BinaryOperation binary_op)
    {
      // check for empty input early
      if(count == 0) return agency::experimental::nullopt;

      auto result = reduce_and_elect(self, value, count, binary_op);

      // the agent with the result puts it in shared storage
      // XXX we're using inside knowledge that reduce_and_elect() always elects agent_rank == 0
      if(self.rank() == 0)
      {
        super_t::storage()[0] = *result;
      }

      self.wait();

      return agency::experimental::make_optional(super_t::storage()[0]);
    }


  private:
    // initialize_storage() without init value
    template<class ConcurrentAgent, class ContiguousRange>
    __AGENCY_ANNOTATION
    static void initialize_storage(ConcurrentAgent& self, ContiguousRange& storage, const agency::experimental::optional<T>& value, int count)
    {
      auto agent_rank = self.rank();

      // store partial sum to storage
      if(agent_rank < count)
      {
        storage[agent_rank] = *value;
      }

      self.wait();
    }


    // initialize_storage() with init value
    __agency_exec_check_disable__
    template<class ConcurrentAgent, class ContiguousRange, class BinaryOperation>
    __AGENCY_ANNOTATION
    static void initialize_storage(ConcurrentAgent& self, ContiguousRange& storage, const agency::experimental::optional<T>& value, int count, T init, BinaryOperation binary_op)
    {
      auto agent_rank = self.rank();

      if(agent_rank == 0)
      {
        // agent 0 sums in the init
        if(count == 0)
        {
          storage[agent_rank] = init;
        }
        else
        {
          storage[agent_rank] = binary_op(init, *value);
        }
      }
      else if(agent_rank < count)
      {
        storage[agent_rank] = *value;
      }

      self.wait();
    }
};


} // end experimental

