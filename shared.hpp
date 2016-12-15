#pragma once

#include <agency/agency.hpp>
#include <agency/experimental/array.hpp>
#include <utility>
#include <initializer_list>

namespace experimental
{


namespace detail
{


template<class T, class ConcurrentAgent, class... Args>
__AGENCY_ANNOTATION
T* collective_new(ConcurrentAgent& self, Args&&... args)
{
  T* ptr = nullptr;

  if(self.elect())
  {
    // allocate the storage
    ptr = reinterpret_cast<T*>(self.memory_resource().allocate(sizeof(T)));

    // construct the object
    ::new(ptr) T(std::forward<Args>(args)...);
  }

  // we wait because .broadcast() may do side effects on self.memory_resource()
  // XXX we can eliminate this barrier in cases where we can prove that .broadcast() does not use self.memory_resource()
  self.wait();

  using namespace agency::experimental;
  return self.broadcast(ptr ? make_optional(ptr) : nullopt);
}


template<class ConcurrentAgent, class T>
__AGENCY_ANNOTATION
void collective_delete(ConcurrentAgent& self, T* ptr)
{
  if(self.elect())
  {
    // destroy the object
    ptr->~T();

    // deallocate the storage
    self.memory_resource().deallocate(ptr, sizeof(T));
  }

  self.wait();
}


// general case
template<class T, class ConcurrentAgent = void>
class default_collective_delete
{
  public:
    using execution_agent_type = ConcurrentAgent;

    __AGENCY_ANNOTATION
    default_collective_delete(execution_agent_type& self)
      : self_(self)
    {}

    __AGENCY_ANNOTATION
    void operator()(T* ptr)
    {
      self_.wait();
      collective_delete(self_, ptr);
    }

  private:
    execution_agent_type& self_;
};


// type-erased agent case
template<class T>
class default_collective_delete<T,void>
{
  public:
    using execution_agent_type = void;

    template<class ConcurrentAgent>
    __AGENCY_ANNOTATION
    default_collective_delete(ConcurrentAgent& self)
      : self_ptr_(&self),
        delete_function_(call_collective_delete<ConcurrentAgent>)
    {}

    __AGENCY_ANNOTATION
    void operator()(T* ptr)
    {
      delete_function_(self_ptr_, ptr);
    }

  private:
    template<class ConcurrentAgent>
    __AGENCY_ANNOTATION
    static ConcurrentAgent& self(void* self_ptr)
    {
      return *reinterpret_cast<ConcurrentAgent*>(self_ptr);
    }

    template<class ConcurrentAgent>
    __AGENCY_ANNOTATION
    static void call_collective_delete(void* self_ptr, T* ptr)
    {
      ConcurrentAgent& s = self<ConcurrentAgent>(self_ptr);
      s.wait();
      collective_delete(s, ptr);
    }

    void* self_ptr_;
    void (*delete_function_)(void*, T*);
};


template<class T, class ConcurrentAgent, class... Args>
__AGENCY_ANNOTATION
T* collective_new_array(ConcurrentAgent& self, size_t n, Args&&... args)
{
  T* ptr = nullptr;

  if(self.elect())
  {
    // allocate the storage
    std::size_t num_bytes = n * sizeof(T);
    ptr = reinterpret_cast<T*>(self.memory_resource().allocate(num_bytes));

    // construct each array element
    // XXX we could use all the agents for this
    for(T* element = ptr; element != ptr + n; ++element)
    {
      ::new(element) T(std::forward<Args>(args)...);
    }
  }

  // we wait because .broadcast() may do side effects on self.memory_resource()
  // XXX we can eliminate this barrier in cases where we can prove that .broadcast() does not use self.memory_resource()
  self.wait();

  using namespace agency::experimental;
  return self.broadcast(ptr ? make_optional(ptr) : nullopt);
}


template<class ConcurrentAgent, class T>
__AGENCY_ANNOTATION
void collective_delete_array(ConcurrentAgent& self, T* ptr, size_t n)
{
  if(self.elect())
  {
    // destroy each array element
    // XXX we could use all the agents for this
    for(T* element = ptr; element != ptr + n; ++element)
    {
      element->~T();
    }

    // deallocate the storage
    self.memory_resource().deallocate(ptr, n * sizeof(T));
  }

  self.wait();
}


// general case
template<class T, class ConcurrentAgent = void>
class default_collective_delete_array
{
  public:
    using execution_agent_type = ConcurrentAgent;

    __AGENCY_ANNOTATION
    default_collective_delete_array(execution_agent_type& self)
      : self_(self)
    {}

    __AGENCY_ANNOTATION
    void operator()(T* ptr, size_t n)
    {
      self_.wait();
      collective_delete_array(self_, ptr, n);
    }

  private:
    execution_agent_type& self_;
};


// type-erased agent case
template<class T>
class default_collective_delete_array<T,void>
{
  public:
    using execution_agent_type = void;

    template<class ConcurrentAgent>
    __AGENCY_ANNOTATION
    default_collective_delete_array(ConcurrentAgent& self)
      : self_ptr_(&self),
        delete_function_(call_collective_delete_array<ConcurrentAgent>)
    {}

    __AGENCY_ANNOTATION
    void operator()(T* ptr, size_t n)
    {
      delete_function_(self_ptr_, ptr, n);
    }

  private:
    template<class ConcurrentAgent>
    __AGENCY_ANNOTATION
    static ConcurrentAgent& self(void* self_ptr)
    {
      return *reinterpret_cast<ConcurrentAgent*>(self_ptr);
    }

    template<class ConcurrentAgent>
    __AGENCY_ANNOTATION
    static void call_collective_delete_array(void* self_ptr, T* ptr, size_t n)
    {
      ConcurrentAgent& s = self<ConcurrentAgent>(self_ptr);
      s.wait();
      collective_delete_array(s, ptr, n);
    }

    void* self_ptr_;
    void (*delete_function_)(void*, T*, size_t);
};


} // end detail


template<class T, class ConcurrentAgent = void>
class shared
{
  private:
    using collective_deleter_type = detail::default_collective_delete<T,ConcurrentAgent>;

  public:
    shared() = delete;

    shared(const shared&) = delete;

    template<class ConcurrentAgent1,
             class... Args,
             class = typename std::enable_if<
               std::is_constructible<
                 collective_deleter_type,
                 ConcurrentAgent1&
               >::value
             >::type>
    __AGENCY_ANNOTATION
    shared(ConcurrentAgent1& self, Args&&... args)
      : ptr_(detail::collective_new<T>(self, std::forward<Args>(args)...)),
        collective_deleter_(self)
    {}

    __AGENCY_ANNOTATION
    ~shared()
    {
      // note that the first thing collective_deleter_() does is a barrier
      // so, this destructor implicitly synchronizes before doing any deallocation
      collective_deleter_(ptr_);
    }

    __AGENCY_ANNOTATION
    T& value()
    {
      return *ptr_;
    }

    __AGENCY_ANNOTATION
    constexpr const T& value() const
    {
      return *ptr_;
    }

  private:
    T* ptr_;
    collective_deleter_type collective_deleter_;
};


template<class T, std::size_t N, class ConcurrentAgent = void>
class shared_array : private shared<agency::experimental::array<T,N>,ConcurrentAgent>
{
  private:
    using super_t = shared<agency::experimental::array<T,N>,ConcurrentAgent>;

  public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using iterator = pointer;
    using const_iterator = pointer;

    using super_t::super_t;
    using super_t::value;

    template<class ConcurrentAgent1>
    __AGENCY_ANNOTATION
    shared_array(ConcurrentAgent1& self, const agency::experimental::array<T,N>& other)
      : super_t(self, other)
    {}

    template<class ConcurrentAgent1>
    __AGENCY_ANNOTATION
    shared_array(ConcurrentAgent1& self, agency::experimental::array<T,N>&& other)
      : super_t(self, std::move(other))
    {}

    __AGENCY_ANNOTATION
    reference operator[](size_type pos)
    {
      return value()[pos];
    }

    __AGENCY_ANNOTATION
    constexpr const_reference operator[](size_type pos) const
    {
      return value()[pos];
    }

    __AGENCY_ANNOTATION
    reference front()
    {
      return value().front();
    }

    __AGENCY_ANNOTATION
    constexpr const_reference front() const
    {
      return value().front();
    }

    __AGENCY_ANNOTATION
    reference back()
    {
      return value().back();
    }

    __AGENCY_ANNOTATION
    constexpr const_reference back() const
    {
      return value().back();
    }

    __AGENCY_ANNOTATION
    pointer data()
    {
      return value().data();
    }

    __AGENCY_ANNOTATION
    constexpr const_pointer data() const
    {
      return value().data();
    }

    __AGENCY_ANNOTATION
    iterator begin()
    {
      return data();
    }

    __AGENCY_ANNOTATION
    constexpr const_iterator cbegin() const
    {
      return data();
    }

    __AGENCY_ANNOTATION
    constexpr const_iterator begin() const
    {
      return cbegin();
    }

    __AGENCY_ANNOTATION
    iterator end()
    {
      return begin() + N;
    }

    __AGENCY_ANNOTATION
    constexpr const_iterator cend() const
    {
      return cbegin() + N;
    }

    __AGENCY_ANNOTATION
    constexpr const_iterator end() const
    {
      return cend();
    }

    __AGENCY_ANNOTATION
    constexpr bool empty() const
    {
      return false;
    }

    __AGENCY_ANNOTATION
    constexpr size_type size() const
    {
      return N;
    }

    __AGENCY_ANNOTATION
    constexpr size_type max_size() const
    {
      return size();
    }
};


template<class T, class ConcurrentAgent = void>
class shared_vector
{
  private:
    using collective_array_deleter_type = detail::default_collective_delete_array<T,ConcurrentAgent>;

  public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = pointer;
    using const_iterator = const_pointer;

    shared_vector() = delete;

    shared_vector(const shared_vector&) = delete;

    template<class ConcurrentAgent1,
             class = typename std::enable_if<
               std::is_constructible<
                 collective_array_deleter_type,
                 ConcurrentAgent1&
               >::value
             >::type>
    __AGENCY_ANNOTATION
    shared_vector(ConcurrentAgent1& self, size_t n)
      : data_(detail::collective_new_array<T>(self, n)),
        size_(n),
        collective_array_deleter_(self)
    {}

    template<class ConcurrentAgent1,
             class = typename std::enable_if<
               std::is_constructible<
                 collective_array_deleter_type,
                 ConcurrentAgent1&
               >::value
             >::type>
    __AGENCY_ANNOTATION
    shared_vector(ConcurrentAgent1& self, size_t n, const T& value)
      : data_(detail::collective_new_array<T>(self, n, value)),
        size_(n),
        collective_array_deleter_(self)
    {}

    template<class ConcurrentAgent1,
             class Iterator,
             class = typename std::enable_if<
               std::is_constructible<
                 collective_array_deleter_type,
                 ConcurrentAgent1&
               >::value
             >::type>
    __AGENCY_ANNOTATION
    shared_vector(ConcurrentAgent1& self, Iterator first, Iterator last)
      : data_(detail::collective_new_array<T>(self, last - first)),
        size_(last - first),
        collective_array_deleter_(self)
    {
      if(self.elect())
      {
        // XXX we could use all the agents for this
        for(auto iter = begin(); first != last; ++first, ++iter)
        {
          *iter = *first;
        }
      }

      self.wait();
    }

    template<class ConcurrentAgent1,
             class = typename std::enable_if<
               std::is_constructible<
                 collective_array_deleter_type,
                 ConcurrentAgent1&
               >::value
             >::type>
    __AGENCY_ANNOTATION
    shared_vector(ConcurrentAgent1& self, std::initializer_list<T> list)
      : shared_vector(self, list.begin(), list.end())
    {}

    __AGENCY_ANNOTATION
    ~shared_vector()
    {
      // note that the first thing collective_array_deleter_() does is a barrier
      // so, this destructor implicitly synchronizes before doing any deallocation
      collective_array_deleter_(data_, size_);
    }

    __AGENCY_ANNOTATION
    size_t size() const
    {
      return size_;
    }

    __AGENCY_ANNOTATION
    pointer data()
    {
      return data_;
    }

    __AGENCY_ANNOTATION
    const_pointer data() const
    {
      return data_;
    }

    __AGENCY_ANNOTATION
    iterator begin()
    {
      return data();
    }

    __AGENCY_ANNOTATION
    const_iterator cbegin() const
    {
      return data();
    }

    __AGENCY_ANNOTATION
    const_iterator begin() const
    {
      return cbegin();
    }

    __AGENCY_ANNOTATION
    iterator end()
    {
      return begin() + size();
    }

    __AGENCY_ANNOTATION
    const_iterator cend() const
    {
      return cbegin() + size();
    }

    __AGENCY_ANNOTATION
    const_iterator end() const
    {
      return cend();
    }

    __AGENCY_ANNOTATION
    reference operator[](size_t i)
    {
      return data()[i];
    }

    __AGENCY_ANNOTATION
    const_reference operator[](size_t i) const
    {
      return data()[i];
    }

  private:
    T* data_;
    size_t size_;
    collective_array_deleter_type collective_array_deleter_;
};


template<class T, class ConcurrentAgent1, class ConcurrentAgent2>
class shared<shared<T,ConcurrentAgent1>,ConcurrentAgent2>
{
  static_assert(sizeof(T) && false, "shared<shared<...>,...> is not allowed.");
};


template<class T, size_t N, class ConcurrentAgent1, class ConcurrentAgent2>
class shared<shared_array<T,N,ConcurrentAgent1>,ConcurrentAgent2>
{
  static_assert(sizeof(T) && false, "shared<shared_array<...>,...> is not allowed.");
};


template<class T, class ConcurrentAgent1, class ConcurrentAgent2>
class shared<shared_vector<T,ConcurrentAgent1>,ConcurrentAgent2>
{
  static_assert(sizeof(T) && false, "shared<shared_vector<...>,...> is not allowed.");
};


template<class T, size_t N, class ConcurrentAgent1, class ConcurrentAgent2>
class shared_array<shared<T,ConcurrentAgent1>,N,ConcurrentAgent2>
{
  static_assert(sizeof(T) && false, "shared_array<shared<...>,...> is not allowed.");
};


template<class T, size_t N1, class ConcurrentAgent1, size_t N2, class ConcurrentAgent2>
class shared_array<shared_array<T,N1,ConcurrentAgent1>,N2,ConcurrentAgent2>
{
  static_assert(sizeof(T) && false, "shared_array<shared_array<...>,...> is not allowed.");
};


template<class T, size_t N, class ConcurrentAgent1, class ConcurrentAgent2>
class shared_array<shared_vector<T,ConcurrentAgent1>,N,ConcurrentAgent2>
{
  static_assert(sizeof(T) && false, "shared_array<shared_vector<...>,...> is not allowed.");
};


template<class T, class ConcurrentAgent1, class ConcurrentAgent2>
class shared_vector<shared<T,ConcurrentAgent1>,ConcurrentAgent2>
{
  static_assert(sizeof(T) && false, "shared_vector<shared<...>,...> is not allowed.");
};


template<class T, size_t N, class ConcurrentAgent1, class ConcurrentAgent2>
class shared_vector<shared_array<T,N,ConcurrentAgent1>,ConcurrentAgent2>
{
  static_assert(sizeof(T) && false, "shared_vector<shared_array<...>,...> is not allowed.");
};


template<class T, class ConcurrentAgent1, class ConcurrentAgent2>
class shared_vector<shared_vector<T,ConcurrentAgent1>,ConcurrentAgent2>
{
  static_assert(sizeof(T) && false, "shared_vector<shared_vector<...>,...> is not allowed.");
};


} // end experimental

