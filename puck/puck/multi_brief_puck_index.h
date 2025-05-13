/**
 * @file    multi_brief_puck_index.h
 * @author  yinjie06(yinjie06@baidu.com)
 * @date    2023/09/12 16:49
 * @brief
 *
 **/

#pragma once
#include <vector>
#include <string>
#include <memory>

#include <mutex>
#include <fstream>
#include <stdlib.h>
#include "puck/puck/puck_index.h"
#include "puck/hierarchical_cluster/max_heap.h"

namespace puck {

template <typename T>
T clamp(const T& v, const T& lo, const T& hi) {
      return (v < lo) ? lo : (v > hi) ? hi : v;
    }

//内存索引结构
class MultiBriefPuckIndex : public puck::PuckIndex {
public:
    MultiBriefPuckIndex();
    ~MultiBriefPuckIndex();
    /*
     * @brief 检索最近的topk个样本
     * @@param [in] request : request
     * @@param [out] response : response
     * @@return (int) : 正常返回0，错误返回值<0
     **/
    virtual int search(const Request* request, Response* response) override;
private:
    ////训练建库
    /*
    * @brief 检索过程中会按某种规则调整样本在内存的顺序（memory_idx），计算对应的信息
    * @@param [out] cell_start_memory_idx : 每个cell下样本中最小的memory_idx
    * @@param [out] local_to_memory_idx : 每个样本local_idx 与 memory_idx的映射关系
    * @@return (int) : 正常返回0，错误返回值<0
    **/
    virtual int convert_local_to_memory_idx(uint32_t* cell_start_memory_idx, uint32_t* local_to_memory_idx);
    int check_index_type();
private:
    int get_brief_points_cnt(int brief_id) {
        int start_cell_idx = _briefs_cell_indptr[brief_id];
        int end_cell_idx = _briefs_cell_indptr[brief_id + 1];
        int start_point_idx = _cell_point_indptr[start_cell_idx];
        int end_point_idx = _cell_point_indptr[end_cell_idx];
        return end_point_idx - start_point_idx;
    }
    ///检索
    /*
     * @brief 计算query与一级聚类中心的距离并排序
     * @@param [in\out] context : context由内存池管理
     * @@param [in] feature : query的特征向量
     * @@param [in] top_coarse_cnt : 保留top_coarse_cnt个最近的一级聚类中心
     * @@return (int) : 堆的size
     **/
    int search_nearest_coarse_cluster(SearchContext* context, const float* feature,
                                      const uint32_t top_coarse_cnt, uint32_t& true_top_coarse);
    /*
     * @brief 计算query与top_coarse_cnt个一级聚类中心的下所有二级聚类中心的距离
     * @@param [in\out] context : context由内存池管理
     * @@param [in] feature : query的特征向量
     * @@return (int) : 正常返回保留的cell个数(>0)，错误返回值<0
     **/
    int search_nearest_filter_points(SearchContext* context, const float* feature,
                                     const uint32_t true_coarse_cnt);

    int compute_quantized_distance(SearchContext* context, const int cell_point_idx,
                                   const float cell_dist, MaxHeap& result_heap);
private:
    //memory idx order，point has brief ids
    std::unique_ptr<int32_t[]> _briefs_indptr;
    std::unique_ptr<int32_t[]> _briefs_indices;
    //标记coase下样本与的brief信息
    std::unique_ptr<bool[]> _briefs_coarse;
    //每个brief下，样本在的cell ids
    std::unique_ptr<int32_t[]> _briefs_cell_indptr;
    std::unique_ptr<int32_t[]> _briefs_cell_indices;
    //cell下，样本的memory ids
    std::unique_ptr<int32_t[]> _cell_point_indptr;
    std::unique_ptr<int32_t[]> _cell_point_indices;

}; // class MultiBriefPuckIndex


struct IDSelector {
    size_t nb;
    using TL = int32_t;
    const TL* lims;
    const int32_t* indices;
    int32_t w1 = -1, w2 = -1;

    IDSelector(
        size_t nb, const TL* lims, const int32_t* indices):
        nb(nb), lims(lims), indices(indices) {}

    void set_query_words(int32_t w1, int32_t w2) {
        this->w1 = w1;
        this->w2 = w2;
    }

    // binary search in the indices array
    bool find_sorted(TL l0, TL l1, int32_t w) const {
        while (l1 > l0 + 1) {
            TL lmed = (l0 + l1) / 2;

            if (indices[lmed] > w) {
                l1 = lmed;
            } else {
                l0 = lmed;
            }
        }

        return indices[l0] == w;
    }

    bool is_member(int64_t id) const {
        TL l0 = lims[id], l1 = lims[id + 1];

        if (l1 <= l0) {
            return false;
        }

        //return find_sorted(l0, l1, w1);
        if (!find_sorted(l0, l1, w1)) {
            return false;
        }

        //if (w2 >= 0 && !find_sorted(l0, l1, w2)) {
        //    return false;
        //}

        return true;
    }

    ~IDSelector() {}
};

struct BriefRequest : public  Request {
    int* briefs;
    int brief_size;
    BriefRequest(): Request() {
        briefs = nullptr;
        brief_size = 0;
    }
};

// 仿照你的剪枝逻辑，改写为与原搜索流程一致的风格
struct PivotUpdater {
    // 自适应参数配置
    float alpha_base = 0.2f;          // 基础平滑系数
    float alpha_decay = 0.95f;        // 平滑系数衰减率（每10次迭代）
    float delta_threshold = 0.005f;   // 触发动态步长的变化阈值
    float max_step_size = 0.05f;      // 步长上限
    float min_step_size = 0.001f;     // 步长下限

    // 运行时状态
    float previous_pivot = 0.0f;
    float previous_top = 0.0f;
    int update_count = 0;             // 更新计数器
    bool use_aggressive_phase = true; // 初始激进阶段标识

    float update(MaxHeap& filter_heap, float query_norm, float radius_rate) {
        const float* heap_vals = filter_heap.get_top_addr();
        const float heap_top = heap_vals[0];

        // 第一阶段：激进剪枝（使用原始方法快速收敛）
        if (use_aggressive_phase) {
            if (update_count < 5) { // 前5次更新保持激进
                previous_pivot = (heap_top) / (2 * radius_rate);
                previous_top = heap_top;
                ++update_count;
                return previous_pivot;
            } else {
                use_aggressive_phase = false; // 切换至平滑阶段
            }
        }

        // 第二阶段：自适应平滑策略
        // 1. 计算动态alpha（随更新次数衰减）
        float alpha = alpha_base * std::pow(alpha_decay, update_count/10.0f);
        alpha = clamp(alpha, 0.05f, 0.3f); // 限制alpha范围

        // 2. 计算基础pivot（带噪声抑制）
        float base_pivot = (heap_top) / (2 * radius_rate);
        if (std::abs(heap_top - previous_top) < delta_threshold * previous_top) {
            base_pivot *= 0.98f; // 当堆顶稳定时略微收紧
        }

        // 3. 动态步长计算（基于堆顶变化率）
        float delta = (previous_top - heap_top) / std::abs(previous_top);
        float dynamic_step = clamp(
            min_step_size + delta * max_step_size,
            min_step_size,
            max_step_size
        );

        // 4. 综合更新
        float smoothed_pivot = alpha * base_pivot + (1 - alpha) * previous_pivot;
        smoothed_pivot += dynamic_step; // 添加动态推进

        // 状态更新
        previous_top = heap_top;
        previous_pivot = smoothed_pivot;
        ++update_count;

        return smoothed_pivot;
    }
};



} // namespace puck
