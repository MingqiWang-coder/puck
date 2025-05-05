//Copyright (c) 2023 Baidu, Inc.  All Rights Reserved.
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
/**
 * @file max_heap.h
 * @author huangben@baidu.com
 * @author yinjie06@baidu.com
 * @date 2019/8/20 10:43
 * @brief
 *
 **/
#pragma once
#include <cstdio>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <functional>
#include <vector>
namespace puck {
class MaxHeap {
public:
    MaxHeap(const uint32_t size, float* val, uint32_t* tag);

    // 定义回调类型（旧值, 新值）
    using UpdateCallback = std::function<void(float old_val, float new_val)>;

    // 设置回调函数
    void set_update_callback(UpdateCallback cb) {
        _callback = std::move(cb);
    }

    ~MaxHeap() {}
    /*
     * @brief 小于堆顶元素时，需要更新堆
     * @@param [in] new_val : 距离
     * @@param [in] new_tag : 标记
     **/
    void max_heap_update(const float new_val, const uint32_t new_tag);
    /*
     * @brief 排序
     **/
    void reorder();
    /*
     * @brief 获取堆内元素个数
     **/
    uint32_t get_heap_size() {
        return _heap_size - _default_point_cnt;
    }
    /*
     * @brief 获取堆顶val的指针
     **/
    float* get_top_addr() const {
        return _heap_val + 1;
    }
private:
    MaxHeap();
    /*
     * @brief 在某个范围内查找合适的位置，插入新的值
     * @@param [in] heap_size : 堆的大小
     * @@param [in] father_idx : 开始查找的节点位置
     * @@param [in] new_val : 距离
     * @@param [in] new_tag : 标记
     **/
    void insert(uint32_t heap_size, uint32_t father_idx, float  new_val, uint32_t new_tag);
    void pop_top(uint32_t heap_size);
private:
    float* _heap_val;
    uint32_t* _heap_tag;
    uint32_t _default_point_cnt;
    const uint32_t _heap_size;

    UpdateCallback _callback; // 回调函数对象

};

}