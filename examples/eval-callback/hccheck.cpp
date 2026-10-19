// lab tool: compare the fused hc_mix op against the unfused chain on real activations (ENGINE_LAB_HCMIX_CHECK=1).
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct chk {
    std::map<std::string, std::vector<float>> t;
    double worst_rel = 0.0, worst_inj = 0.0;
    int    worst_il = -1, n = 0;
    double sum_rel = 0.0, sum_inj = 0.0;
};

static void fetch(ggml_tensor * t, std::vector<float> & out) {
    out.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size()*sizeof(float));
}

static double rel(const std::vector<float> & a, const std::vector<float> & b, double * maxabs_ref) {
    double num = 0.0, den = 0.0, mx = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - (double) b[i];
        num += d*d; den += (double) b[i]*b[i];
        mx = std::max(mx, std::fabs((double) b[i]));
    }
    if (maxabs_ref) *maxabs_ref = mx;
    return std::sqrt(num/(den + 1e-30));
}

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    chk * c = (chk *) ud;
    const char * n = t->name;
    static FILE * dumpf = getenv("HCCHECK_DUMP") ? fopen(getenv("HCCHECK_DUMP"), "wb") : nullptr;
    static int dump_n = 0;
    const bool dump_want = dumpf && dump_n < 2*92*3 && (!strncmp(n, "hc_combine-", 11) || !strncmp(n, "hc_mixed-", 9) || !strncmp(n, "hc_inject-", 10));
    if (ask && dump_want) return true;
    if (!ask && dump_want) {
        std::vector<float> v; fetch(t, v);
        char nm[32] = {0}; strncpy(nm, n, 31);
        fwrite(nm, 1, 32, dumpf); int nn = (int) v.size(); fwrite(&nn, 4, 1, dumpf); fwrite(v.data(), 4, nn, dumpf);
        fflush(dumpf); dump_n++;
        return true;
    }
    const bool want = !strncmp(n, "hc_mixed-", 9) || !strncmp(n, "hc_inject-", 10) ||
                      !strncmp(n, "ref_mixed-", 10) || !strncmp(n, "ref_inject-", 11);
    if (ask) return want;
    if (!want) return true;
    fetch(t, c->t[n]);
    const bool is_inj = strstr(n, "inject") != nullptr;
    const char * dash = strchr(n, '-');
    const int il = atoi(dash + 1);
    const std::string fk = std::string(is_inj ? "hc_inject-" : "hc_mixed-") + std::to_string(il);
    const std::string rk = std::string(is_inj ? "ref_inject-" : "ref_mixed-") + std::to_string(il);
    if (!c->t.count(fk) || !c->t.count(rk) || c->t[fk].empty() || c->t[rk].empty()) return true;
    auto & f = c->t[fk];
    auto & r = c->t[rk];
    double mx;
    const double e = rel(f, r, &mx);
    if (!is_inj) {
        c->sum_rel += e; c->n++;
        if (e > c->worst_rel) { c->worst_rel = e; c->worst_il = il; }
        if (il % 8 == 0 || e > 1e-3) printf("  layer %2d mixed rel L2 %.3e  (max|ref| %.3g)\n", il, e, mx);
    } else {
        c->sum_inj += e;
        if (e > c->worst_inj) c->worst_inj = e;
        if (il % 8 == 0 || e > 1e-3) printf("  layer %2d inject rel L2 %.3e  fused[%g %g %g %g] ref[%g %g %g %g]\n", il, e,
            f[0], f[1], f[2], f[3], r[0], r[1], r[2], r[3]);
    }
    c->t.erase(fk);
    c->t.erase(rk);
    return true;
}

static void segv(int) {
    void * bt[40];
    int n = backtrace(bt, 40);
    backtrace_symbols_fd(bt, n, 2);
    _exit(139);
}

int main(int argc, char ** argv) {
    signal(SIGSEGV, segv);
    std::setlocale(LC_NUMERIC, "C");
    setvbuf(stdout, nullptr, _IOLBF, 0);
    chk c;
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    llama_backend_init();
    params.cb_eval = cb;
    params.cb_eval_user_data = &c;
    params.warmup = false;
    fprintf(stderr, "hccheck: params parsed\n");
    auto init = common_init_from_params(params);
    fprintf(stderr, "hccheck: init done\n");
    auto * model = init->model();
    auto * ctx   = init->context();
    if (!model || !ctx) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    fprintf(stderr, "hccheck: %zu tokens\n", toks.size());
    for (size_t i = 0; i < toks.size(); ++i) {
        std::vector<llama_token> one = { toks[i] };
        common_batch batch = common_batch_get_one(ctx, one);
        printf("token %zu\n", i);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) return 1;
        if (const char * lf = getenv("HCCHECK_LOGITS")) {
            const float * lg = llama_get_logits_ith(ctx, -1);
            const int nv = llama_vocab_n_tokens(vocab);
            if (lg) {
                FILE * f = fopen(lf, i == 0 ? "wb" : "ab");
                fwrite(lg, sizeof(float), nv, f);
                fclose(f);
            }
        }
    }
    printf("SUMMARY: pairs %d, mean mixed rel %.3e, worst mixed rel %.3e (layer %d), mean inject rel %.3e, worst inject rel %.3e\n",
           c.n, c.n ? c.sum_rel/c.n : 0.0, c.worst_rel, c.worst_il, c.n ? c.sum_inj/c.n : 0.0, c.worst_inj);
    llama_backend_free();
    return 0;
}
