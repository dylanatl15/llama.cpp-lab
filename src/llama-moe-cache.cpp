#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <linux/futex.h>
#include <climits>
#include <atomic>
#include <immintrin.h>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

// LRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct cpy_job { uint8_t * dst; const uint8_t * src; size_t n; };
struct cpy_pool {
    std::vector<std::thread> th;
    std::atomic<int> gen{0}, done{0}, next{0};
    std::atomic<bool> quit{false};
    const cpy_job * jobs = nullptr; int njobs = 0; int nw = 0;
    double spin_us = 3000;
    void start(int n) {
        nw = n;
        if (const char * e = getenv("ENGINE_LAB_CPY_SPIN_US")) { spin_us = atof(e); }
        for (int i = 0; i < n; ++i) th.emplace_back([this] {
            int seen = 0; int idle = 0; double t_idle = 0;
            while (!quit.load(std::memory_order_relaxed)) {
                int g = gen.load(std::memory_order_acquire);
                if (g == seen) {
                    // spin for spin_us after the last job so the next decode layer finds the workers awake, then sleep so an idle server does not burn the cores
                    if ((++idle & 255) == 1) {
                        const double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
                        if (t_idle == 0) t_idle = t;
                        if (t - t_idle > spin_us) {
                            syscall(SYS_futex, (int *) &gen, FUTEX_WAIT_PRIVATE, seen, nullptr, nullptr, 0);
                            idle = 0; t_idle = 0;
                        }
                    }
                    _mm_pause();
                    continue;
                }
                idle = 0; t_idle = 0;
                seen = g; work(); done.fetch_add(1, std::memory_order_release);
            }
        });
    }
    void work() { for (;;) { int j = next.fetch_add(1); if (j >= njobs) break; memcpy(jobs[j].dst, jobs[j].src, jobs[j].n); } }
    void run(const cpy_job * j, int n) {
        jobs = j; njobs = n; next = 0; done = 0;
        gen.fetch_add(1, std::memory_order_release);
        syscall(SYS_futex, (int *) &gen, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
        work();
        while (done.load(std::memory_order_acquire) < nw) _mm_pause();
    }
    ~cpy_pool() { quit = true; gen.fetch_add(1); syscall(SYS_futex, (int *) &gen, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0); for (auto & t : th) t.join(); }
};

struct moe_cache_lru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // doubly linked list of the slots, head is the least recently used
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head = -1;
    int32_t tail = -1;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = 0;
        tail = n_slots - 1;
        seen.assign(n_expert, 0);
    }

    // move slot s to the tail (most recently used)
    void touch(int32_t s) {
        if (s == tail) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        prev[next[s]] = prev[s];

        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // hybrid: mark the cached experts of ids as recently used
    void touch_hits(int32_t il, const int32_t * ids, size_t n_ids) {
        int32_t * slots = slot_map[il];
        for (size_t i = 0; i < n_ids; ++i) {
            if (slots[ids[i]] >= 0) {
                touch(slots[ids[i]]);
            }
        }
    }

    // hybrid: give expert e of layer il the least recently used slot
    int32_t insert(int32_t il, int32_t e) {
        const int32_t s = head;
        if (key_of[s] >= 0) {
            slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
        }
        key_of[s] = il*n_expert + e;
        slot_map[il][e] = s;
        touch(s);
        return s;
    }

    // give a slot to each expert selected by ids in layer il, the misses evict the least recently used experts
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // hits go to the tail first, so the head can be evicted below
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = head;
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            touch(s);
            fills.push_back({ e, s });
        }
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    // hybrid decode
    bool hybrid = false;
    int  insert_max = 1;
    ggml_tensor * miss_t = nullptr; // I32 [n_expert_used, 1] in host memory
    std::vector<int32_t> scratch;
    size_t hy_calls = 0, hy_hits = 0, hy_miss = 0, hy_inserted = 0;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size();
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host + 1);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        hybrid = getenv("ENGINE_LAB_HYBRID") != nullptr && atoi(getenv("ENGINE_LAB_HYBRID")) != 0;
        if (getenv("ENGINE_LAB_INSERT_MAX")) {
            insert_max = atoi(getenv("ENGINE_LAB_INSERT_MAX"));
        }
        if (hybrid) {
            miss_t = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, n_expert_used, 1);
            ggml_set_name(miss_t, "moe_cache.miss_ids");
            buf_host_size += GGML_PAD(ggml_nbytes(miss_t), alignment_host);
            LLAMA_LOG_INFO("%s: hybrid decode enabled, insert_max = %d\n", __func__, insert_max);
        }

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + 1, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        log_stats();
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_backend_buffer_ptr mb_buf;
    volatile int32_t * mb_ptr = nullptr;
    bool mb_tried = false;
    static constexpr int MB_STRIDE = 32; // int32 per layer, 128 bytes

    volatile int32_t * get_mailbox(int32_t il, int64_t n_tokens, int64_t n_used) {
        if (getenv("ENGINE_LAB_NO_MAILBOX") || n_tokens != 1 || n_used > MB_STRIDE || get_slot_map(il, n_tokens, n_used) == nullptr) {
            return nullptr;
        }
        if (!mb_tried) {
            mb_tried = true;
            ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(devices[0].backend));
            if (host != nullptr) {
                mb_buf.reset(ggml_backend_buft_alloc_buffer(host, layers.size()*MB_STRIDE*sizeof(int32_t)));
                if (mb_buf) {
                    mb_ptr = (volatile int32_t *) ggml_backend_buffer_get_base(mb_buf.get());
                    for (size_t i = 0; i < layers.size()*MB_STRIDE; ++i) { mb_ptr[i] = -1; }
                }
            }
        }
        return mb_ptr != nullptr ? mb_ptr + (size_t) il*MB_STRIDE : nullptr;
    }

    ggml_tensor * get_miss_ids(int32_t il, int64_t n_tokens, int64_t n_used) const {
        if (!hybrid || miss_t == nullptr || n_tokens != 1 || n_used != n_expert_used || get_slot_map(il, n_tokens, n_used) == nullptr) {
            return nullptr;
        }
        return miss_t;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    // pinned host staging for the uploads: set_async from pageable memory makes the backend synchronize on every call
    ggml_backend_buffer_ptr stg_buf;
    cpy_pool pool; bool pool_on = false; std::vector<cpy_job> jobs;
    uint8_t * stg_ptr  = nullptr;
    size_t    stg_size = 0;
    bool      stg_tried = false;
    size_t    big_off = 0;
    static constexpr size_t big_half = (size_t) 96 << 20;
    bool      inflight[2] = {false, false};
    int       cur_half = 0;
    size_t    pos[2] = {0, 0};

    void ensure_pool() {
        if (!pool_on) { const char * e = getenv("ENGINE_LAB_CPY_THREADS"); int nt = e ? atoi(e) : 12; if (nt > 0) pool.start(nt); pool_on = true; }
    }

    // large uploads (prefill): parallel memcpy into one half of the pinned buffer while the other half is in flight
    double bu_mc = 0, bu_sub = 0, bu_sync = 0; size_t bu_bytes = 0, bu_calls = 0, bu_syncs = 0;
    bool upload(ggml_backend_t backend, ggml_tensor * dst, const void * host, size_t offset, size_t size) {
        if (stg_ptr == nullptr || getenv("ENGINE_LAB_NO_BIGUP")) {
            return false;
        }
        ensure_pool();
        const uint8_t * src = (const uint8_t *) host;
        for (size_t done = 0; done < size; ) {
            if (pos[cur_half] >= big_half) {
                cur_half ^= 1;
                if (inflight[cur_half]) {
                    const double ts = now_us();
                    ggml_backend_synchronize(backend);
                    bu_sync += now_us() - ts; bu_syncs++;
                    inflight[0] = inflight[1] = false;
                }
                pos[cur_half] = 0;
            }
            const int h = cur_half;
            const size_t n = std::min(big_half - pos[h], size - done);
            uint8_t * buf = stg_ptr + big_off + (size_t) h*big_half + pos[h];
            const size_t chunk = 128 << 10;
            const double tm = now_us();
            if (n <= 2*chunk) {
                memcpy(buf, src + done, n);
            } else {
                std::vector<cpy_job> js;
                for (size_t o = 0; o < n; o += chunk) {
                    js.push_back({buf + o, src + done + o, std::min(chunk, n - o)});
                }
                if (pool.nw > 0) pool.run(js.data(), (int) js.size()); else for (auto & j : js) memcpy(j.dst, j.src, j.n);
            }
            const double tm2 = now_us();
            ggml_backend_tensor_set_async(backend, dst, buf, offset + done, n);
            bu_mc += tm2 - tm; bu_sub += now_us() - tm2; bu_bytes += n; bu_calls++;
            inflight[h] = true;
            pos[h] += n;
            done += n;
        }
        return true;
    }
    size_t cpy_chunk = getenv("ENGINE_LAB_CPY_CHUNK") ? (size_t) atol(getenv("ENGINE_LAB_CPY_CHUNK")) : (size_t) 64 << 10;
    static constexpr size_t stg_map = 65536; // head of the staging buffer: copy of the slot map

    void stg_init(ggml_backend_t backend) {
        stg_tried = true;
        if (getenv("ENGINE_LAB_NO_STAGING")) {
            return;
        }
        size_t per_expert = 0;
        for (const group & g : groups) {
            size_t sz = 0;
            for (const ggml_tensor * t : g.ref) {
                sz += t->nb[2];
            }
            per_expert = std::max(per_expert, sz);
        }
        ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
        if (host == nullptr) {
            return;
        }
        big_off  = stg_map + per_expert*(size_t) 8*n_expert_used;
        stg_size = big_off + 2*big_half;
        stg_buf.reset(ggml_backend_buft_alloc_buffer(host, stg_size));
        if (!stg_buf) {
            stg_size = 0;
            return;
        }
        stg_ptr = (uint8_t *) ggml_backend_buffer_get_base(stg_buf.get());
        if (!getenv("ENGINE_LAB_NO_PREFAULT")) {
            // fault in all the host expert weights once, concurrent minor faults on the model mapping are slow when done lazily by the copies
            const double tp = now_us();
            std::vector<std::pair<uint8_t *, size_t>> ranges;
            for (const layer & l : layers) for (const ggml_tensor * t : l.experts) {
                if (t != nullptr && t->data != nullptr) ranges.push_back({(uint8_t *) t->data, ggml_nbytes(t)});
            }
            std::atomic<size_t> nx{0};
            std::vector<std::thread> ths;
            for (int i = 0; i < 8; ++i) ths.emplace_back([&] {
                for (;;) {
                    const size_t k = nx.fetch_add(1); if (k >= ranges.size()) break;
                    uintptr_t a = (uintptr_t) ranges[k].first & ~(uintptr_t) 4095;
                    size_t len = ranges[k].second + ((uintptr_t) ranges[k].first - a);
                    madvise((void *) a, len, 22 /* MADV_POPULATE_READ */);
                }
            });
            for (auto & t : ths) t.join();
            LLAMA_LOG_INFO("llama_moe_cache: prefaulted %zu host weights in %.1f s\n", ranges.size(), (now_us() - tp)/1e6);
        }
        LLAMA_LOG_INFO("llama_moe_cache: pinned upload staging = %.2f MiB\n", stg_size/1024.0/1024.0);
    }

    double t_read = 0, t_plan = 0, t_up = 0, t_gap = 0, t_memcpy = 0, last_end = 0; size_t n_copy_calls = 0, n_up_calls = 0; size_t d_calls = 0; double d_read = 0, d_up = 0, d_gap = 0, d_memcpy = 0; size_t d_bytes = 0;
    size_t mb_polls = 0, mb_fallbacks = 0;
    static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        const double t0 = now_us();
        const double gap = last_end > 0 ? t0 - last_end : 0;
        ids.resize(ggml_nelements(sel));
        volatile int32_t * mb = nullptr;
        {
            const ggml_tensor * mbn = sel->view_src ? sel->view_src : sel;
            if (mbn->op == GGML_OP_MOE_MAILBOX) {
                int64_t p; memcpy(&p, mbn->op_params, sizeof(p));
                mb = (volatile int32_t *) (intptr_t) p;
            }
        }
        bool polled = false;
        if (mb != nullptr) {
            // the GPU publishes the ids as soon as they are selected, the rest of the split keeps running while the CPU plans
            const size_t n = ids.size();
            for (size_t spin = 0; !polled; ++spin) {
                size_t k = 0;
                while (k < n && mb[k] >= 0) { ++k; }
                if (k == n) {
                    for (size_t j = 0; j < n; ++j) { ids[j] = mb[j]; mb[j] = -1; }
                    polled = true;
                } else if ((spin & 0xfff) == 0xfff && now_us() - t0 > 50000) {
                    break;
                }
            }
            if (polled) { mb_polls++; }
        }
        if (!polled) {
            ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
            ggml_backend_synchronize(backend);
            if (mb != nullptr) { for (size_t j = 0; j < ids.size(); ++j) { mb[j] = -1; } mb_fallbacks++; }
        }

        const double t1 = now_us();
        if (!stg_tried) {
            stg_init(backend);
        }
        const size_t map_bytes = ggml_nbytes(src);

        if (hybrid && stg_ptr != nullptr && ids.size() == (size_t) n_expert_used && map_bytes <= stg_map) {
            // hybrid decode: the cached experts are read from the cache, the others get the zero slot and are computed on the CPU
            int32_t * slots = g.lru.slot_map[il];
            g.lru.touch_hits(il, ids.data(), ids.size());
            int32_t * miss = (int32_t *) miss_t->data;
            const int32_t n_exp = (int32_t) (map_bytes/sizeof(int32_t));
            int32_t * out = (int32_t *) stg_ptr;
            for (int32_t e = 0; e < n_exp; ++e) {
                out[e] = slots[e] >= 0 ? slots[e] : g.n_slots; // the slot after the last one is never written, its weights are zero
            }
            fills.clear();
            size_t n_h = 0;
            for (size_t j = 0; j < ids.size(); ++j) {
                if (slots[ids[j]] >= 0) {
                    miss[j] = -1;
                    n_h++;
                } else {
                    miss[j] = ids[j];
                }
            }
            const size_t n_m = ids.size() - n_h;
            // the slot map is uploaded before the inserts change it
            ggml_backend_tensor_set_async(backend, dst, out, 0, map_bytes);

            // insert a few of the missed experts, they are used by the next tokens
            size_t stg_off2 = stg_map;
            size_t bytes2 = 0;
            int n_ins = 0;
            for (size_t j = 0; j < ids.size() && n_ins < insert_max; ++j) {
                if (miss[j] < 0) {
                    continue;
                }
                const int32_t s = g.lru.insert(il, miss[j]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    const size_t es = w->nb[2];
                    if (stg_off2 + es > stg_size) {
                        break;
                    }
                    const double tm = now_us();
                    memcpy(stg_ptr + stg_off2, (const uint8_t *) w->data + (size_t) miss[j]*es, es);
                    d_memcpy += now_us() - tm;
                    ggml_backend_tensor_set_async(backend, g.banks[ip], stg_ptr + stg_off2, (size_t) s*es, es);
                    n_up_calls++;
                    stg_off2 += es;
                    bytes2 += es;
                }
                n_ins++;
            }
            stats_small.hits   += n_h;
            stats_small.misses += n_m;
            stats_small.bytes  += bytes2;
            hy_calls++; hy_hits += n_h; hy_miss += n_m; hy_inserted += n_ins;
            const double t3 = now_us();
            t_read += t1 - t0; t_up += t3 - t1; n_copy_calls++;
            d_calls++; d_read += t1 - t0; d_up += t3 - t1; d_gap += gap; d_bytes += bytes2;
            last_end = t3;
            return true;
        }

        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        const double t2 = now_us();
        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        size_t stg_off = stg_map;
        double mc = 0;
        if (stg_ptr != nullptr) { ensure_pool(); }
        inflight[0] = inflight[1] = false; pos[0] = pos[1] = 0; // the readback synchronized the backend, or the ids were published after the uploads of the previous layer completed
        jobs.clear();
        struct up_t { ggml_tensor * bank; const uint8_t * host; size_t off, n; };
        std::vector<up_t> ups;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                const uint8_t * host = (const uint8_t *) w->data + fills[i].expert*expert_size;
                if (stg_ptr != nullptr && stg_off + n*expert_size <= stg_size) {
                    // split into chunks so the workers share the copy evenly
                    for (size_t o = 0; o < n*expert_size; o += cpy_chunk) {
                        jobs.push_back({stg_ptr + stg_off + o, host + o, std::min(cpy_chunk, n*expert_size - o)});
                    }
                    host = stg_ptr + stg_off;
                    stg_off += n*expert_size;
                }
                ups.push_back({bank, host, fills[i].slot*expert_size, n*expert_size});
                bytes += n*expert_size;
                i += n;
            }
        }
        if (!jobs.empty()) {
            const double tm = now_us();
            if (pool.nw > 0) pool.run(jobs.data(), (int) jobs.size()); else for (auto & j : jobs) memcpy(j.dst, j.src, j.n);
            mc += now_us() - tm;
        }
        for (auto & u : ups) { ggml_backend_tensor_set_async(backend, u.bank, u.host, u.off, u.n); n_up_calls++; }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        if (stg_ptr != nullptr && map_bytes <= stg_map) {
            memcpy(stg_ptr, src->data, map_bytes);
            ggml_backend_tensor_set_async(backend, dst, stg_ptr, 0, map_bytes);
        } else {
            ggml_backend_tensor_set_async(backend, dst, src->data, 0, map_bytes);
        }
        const double t3 = now_us();
        t_read += t1 - t0; t_plan += t2 - t1; t_up += t3 - t2; n_copy_calls++;
        last_end = t3;
        if (ids.size() <= (size_t) 8*n_expert_used && ids.size() <= (size_t) n_expert_used) { // single-token decode
            d_calls++; d_read += t1 - t0; d_up += t3 - t2; d_gap += gap; d_memcpy += mc; d_bytes += bytes;
        }

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        if (mb_polls || mb_fallbacks) LLAMA_LOG_INFO("llama_moe_cache: mailbox: %zu polled, %zu fell back to readback\n", mb_polls, mb_fallbacks);
        if (n_copy_calls) LLAMA_LOG_INFO("llama_moe_cache: timing: %zu copy calls, readback+sync %.1f us/call, plan %.1f us/call, upload-submit %.1f us/call, %zu upload calls\n", n_copy_calls, t_read/n_copy_calls, t_plan/n_copy_calls, t_up/n_copy_calls, n_up_calls);
        if (bu_calls) LLAMA_LOG_INFO("llama_moe_cache: big uploads: %zu chunks, %.1f MiB, memcpy %.2f s, submit %.2f s, sync %.2f s (%zu syncs)\n", bu_calls, bu_bytes/1048576.0, bu_mc/1e6, bu_sub/1e6, bu_sync/1e6, bu_syncs);
        if (d_calls) LLAMA_LOG_INFO("llama_moe_cache: decode(1 tok): %zu calls, gap(gpu compute+cpu) %.1f us, readback+sync %.1f us, upload %.1f us (memcpy %.1f us), %.2f MiB/call\n", d_calls, d_gap/d_calls, d_read/d_calls, d_up/d_calls, d_memcpy/d_calls, d_bytes/1048576.0/d_calls);
        if (hy_calls) LLAMA_LOG_INFO("llama_moe_cache: hybrid: %zu calls, %.2f hits/call, %.2f cpu-computed/call, %.3f inserted/call\n", hy_calls, (double) hy_hits/hy_calls, (double) hy_miss/hy_calls, (double) hy_inserted/hy_calls);
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
    pimpl(new impl(model, backends, bufts, size)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

volatile int32_t * llama_moe_cache::get_mailbox(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_mailbox(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_miss_ids(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_miss_ids(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

bool llama_moe_cache::upload(ggml_backend_t backend, ggml_tensor * dst, const void * host, size_t offset, size_t size) {
    if (!pimpl->stg_tried) { pimpl->stg_init(backend); }
    return pimpl->upload(backend, dst, host, offset, size);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
