/*
 * Copyright 2011-2016 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __UTIL_LRU_H__
#define __UTIL_LRU_H__

#include <OpenImageIO/unordered_map_concurrent.h>

#include "util_math.h"
#include "util_vector.h"
#include "util_map.h"
#include "util_thread.h"
#include "util_hash.h"
#include "util_list.h"
#include "util_atomic.h"

CCL_NAMESPACE_BEGIN

OIIO_NAMESPACE_USING

#define LRU_DEFAULT_CACHE_SIZE (1024*1024*256) /* 256 mb */

template<class key_t, class value_t, class hash_t>
class LRU {
public:
	struct ref_t{
		struct intrusive_ref_t {
			uint ready;
			uint used;
			uint refcount;
			uint size;

			intrusive_ref_t() {
				memset(this, 0, sizeof(intrusive_ref_t));
			}
		};

		value_t* value;

		ref_t() : value(NULL) {}

		ref_t(value_t* value) : value(value) {
			inc();
		}

		/* copy constructor */
		ref_t(const ref_t& other) : value(other.value) {
			inc();
		}

#if 0
		/* move constructor */
		ref_t(ref_t&& other) : value(other.valye) {
			other.value = NULL;
		}
#endif

		~ref_t() {
			dec();
		}

		/* copy assignment */
		ref_t& operator = (const ref_t& other) {
			if(value == other.value)
				return *this;

			dec();

			value = other.value;
			inc();

			return *this;
		}

#if 0
		/* move assignment */
		ref_t& operator = (ref_t&& other) {
			dec();

			value = other.value;
			other.value = NULL;

			return *this;
		}
#endif

		int inc() const {
			if(value) {
				return atomic_add_uint32(&value->intrusive_ref.refcount, 1);
			}
			return 0;
		}

		int dec() {
			if(value) {
				int count = atomic_sub_uint32(&value->intrusive_ref.refcount, 1);
				if(count == 0) {
					delete value;
					value = NULL;
				}
				return count;
			}
			return 0;
		}

		value_t* operator = (value_t* value_) {
			if(value == value_) return value;
			dec();

			value = value_;
			inc();

			return value;
		}

		void set_size(size_t size) const {
			if(!value) return;
			value->intrusive_ref.size = size;
		}

		size_t get_size() const {
			if(!value) return 0;
			return value->intrusive_ref.size;
		}

		void mark_used() const {
			if(!value) return;
			atomic_cas_uint32(&value->intrusive_ref.used, 0, 1);
		}

		bool mark_unused() const {
			if(!value) return false;
			return atomic_cas_uint32(&value->intrusive_ref.used, 1, 0);
		}

		void mark_ready() const {
			if(!value) return;
			atomic_cas_uint32(&value->intrusive_ref.ready, 0, 1);
		}

		void wait_till_ready() const {
			if(!value) return;

			atomic_backoff backoff;
			while(!atomic_cas_uint32(&value->intrusive_ref.ready, 0, 0))
				backoff();
		}

		value_t* operator -> () const { return value; }
		value_t& operator * () const { return *value; }
		operator value_t* () const { return value; }
	};

	struct thread_data_t {
		int locked_bin;

		thread_data_t() : locked_bin(-1) {}
	};

	typedef unordered_map_concurrent<key_t, ref_t, hash_t> map_t;

	thread_specific_ptr<thread_data_t> thread_data;

	uint max_size;
	uint current_size;
	thread_mutex check_size_lock;
	key_t cursor;

	map_t entries;

	thread_data_t* get_thread_data() {
		thread_data_t* tdata = thread_data.get();
		if(!tdata) {
			tdata = new thread_data_t;
			thread_data.reset(tdata);
		}
		return tdata;
	}

	LRU(size_t max_size=LRU_DEFAULT_CACHE_SIZE) : max_size(max_size), current_size(0) {
		set_max_size(max_size);
	}

	void set_max_size(uint size) {
		max_size = std::min(size, (uint)1024*1024*1000*4); /* just under 4gb */
		max_size = std::max(max_size, (uint)1024*1024*128); /* 128 mb */
	}

	void check_size() {
		if(current_size <= max_size)
			return;

		if(!check_size_lock.try_lock())
			return;

		typename map_t::iterator it = entries.find(cursor);
		int num_loops = 0;

		while(atomic_cas_uint32(&current_size, 0, 0) > max_size && num_loops < 100) {
			if(it == entries.end()) {
				it = entries.begin();

				if(it == entries.end())
					break;

				num_loops++;
			}

			ref_t& ref = it->second;

			bool was_used = ref.mark_unused();
			if(!was_used) {
				key_t key = it->first;
				atomic_sub_uint32(&current_size, ref.get_size());

				it++;
				if(it)
					cursor = it->first;
				it.clear();

				entries.erase(key);

				it = entries.find(cursor);
			}
			else {
				it++;
			}
		}

		if(it)
			cursor = it->first;

		check_size_lock.unlock();
	}

	bool find(const key_t& key, ref_t& value) {
		return entries.retrieve(key, value);
	}

	bool insert(const key_t& key, const ref_t& value) {
		check_size();

		value.mark_used();
		atomic_add_uint32(&current_size, value.get_size());

		return entries.insert(key, value);
	}

	bool find_or_lock(const key_t& key, ref_t& value, thread_data_t* tdata=NULL) {
		if(!tdata) tdata = get_thread_data();

		assert(tdata->locked_bin < 0);
		tdata->locked_bin = entries.lock_bin(key);

		if(entries.retrieve(key, value, false)) {
			entries.unlock_bin(tdata->locked_bin);
			tdata->locked_bin = -1;

			return true;
		}

		return false;
	}

	void insert_and_unlock(const key_t& key, const ref_t& value, thread_data_t* tdata=NULL) {
		if(!tdata) tdata = get_thread_data();

		assert(tdata->locked_bin >= 0);

		entries.insert(key, value, false);

		entries.unlock_bin(tdata->locked_bin);
		tdata->locked_bin = -1;

		value.mark_used();
		check_size();
		value.mark_used();
		atomic_add_uint32(&current_size, value.get_size());
	}

	void clear() {
		entries.~map_t();
		new (&entries) map_t;
		current_size = 0;
	}
};

CCL_NAMESPACE_END

#endif /* __UTIL_LRU_H__ */

