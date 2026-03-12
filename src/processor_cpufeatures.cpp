// This file is a part of Julia. License is MIT: https://julialang.org/license

// Unified processor detection and dispatch using the cpufeatures library.
// Replaces processor_x86.cpp, processor_arm.cpp, and processor_fallback.cpp.
// No hand-maintained CPU/feature tables - all data comes from LLVM TableGen
// via generated headers committed to the cpufeatures repository.

// Include cpufeatures generated tables (defines FeatureBits, feature_table, etc.)
#if defined(_CPU_X86_64_) || defined(_CPU_X86_)
#include <cpufeatures/target_tables_x86_64.h>
#elif defined(_CPU_AARCH64_)
#include <cpufeatures/target_tables_aarch64.h>
#elif defined(__riscv) && __riscv_xlen == 64
#include <cpufeatures/target_tables_riscv64.h>
#else
// Minimal fallback - no feature tracking
#define TARGET_FEATURE_WORDS 1
typedef struct { uint64_t bits[1]; } FeatureBits;
static inline int feature_test(const FeatureBits *, unsigned) { return 0; }
static inline void feature_set(FeatureBits *, unsigned) {}
static const unsigned num_features = 0;
typedef struct { const char *name; unsigned bit; FeatureBits implies; } FeatureEntry;
static const FeatureEntry *feature_table = nullptr;
typedef struct { const char *name; FeatureBits features; } CPUEntry;
static const FeatureEntry *find_feature(const char *) { return nullptr; }
static const CPUEntry *find_cpu(const char *) { return nullptr; }
static void expand_implied(FeatureBits *) {}
static const FeatureBits hw_feature_mask = {{0}};
#endif

#include <cpufeatures/target_parsing.h>

// Verify the cpufeatures tables were generated from a compatible LLVM version.
// LLVM_VERSION_MAJOR comes from LLVM headers (via LLVM_CXXFLAGS in the build).
#if defined(TARGET_TABLES_LLVM_VERSION_MAJOR) && defined(LLVM_VERSION_MAJOR)
static_assert(TARGET_TABLES_LLVM_VERSION_MAJOR == LLVM_VERSION_MAJOR,
    "cpufeatures tables were generated with a different LLVM major version than Julia uses");
#endif

// ============================================================================
// Debug output (enabled via JULIA_DEBUG=cpufeatures or JULIA_DEBUG=all)
// ============================================================================

static bool cpufeatures_debug_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char *debug_env = getenv("JULIA_DEBUG");
        enabled = debug_env && (strstr(debug_env, "cpufeatures") || strstr(debug_env, "all"));
    }
    return enabled;
}

#define CF_DEBUG(...) do { if (cpufeatures_debug_enabled()) jl_safe_printf(__VA_ARGS__); } while (0)

// Helper: build a feature string for debug output
static std::string debug_feature_str(const FeatureList<TARGET_FEATURE_WORDS * 2> &features) {
    FeatureBits fb;
    memcpy(&fb, &features, sizeof(fb));
    return tp::build_feature_string(fb);
}

// ============================================================================
// Feature size bridging: cpufeatures uses uint64_t, processor.cpp uses uint32_t
// On little-endian (all supported arches), reinterpret_cast is safe.
// ============================================================================

static constexpr size_t feature_sz = TARGET_FEATURE_WORDS * 2;

static inline void featurebits_to_list(const FeatureBits &fb, FeatureList<feature_sz> &fl) {
    static_assert(sizeof(fb) == sizeof(fl), "FeatureBits and FeatureList size mismatch");
    memcpy(&fl.eles[0], &fb.bits[0], sizeof(fb));
}

static inline void list_to_featurebits(const FeatureList<feature_sz> &fl, FeatureBits &fb) {
    static_assert(sizeof(fb) == sizeof(fl), "FeatureBits and FeatureList size mismatch");
    memcpy(&fb.bits[0], &fl.eles[0], sizeof(fb));
}

// ============================================================================
// Feature name table for reflection (used by jl_reflect_feature_names)
// ============================================================================

static FeatureName feature_names_storage[512];
static uint32_t nfeature_names_val = 0;

static void init_feature_names() {
    if (nfeature_names_val > 0) return;
    uint32_t n = num_features < 512 ? num_features : 512;
    for (uint32_t i = 0; i < n; i++) {
        feature_names_storage[i].name = feature_table[i].name;
        feature_names_storage[i].bit = feature_table[i].bit;
        feature_names_storage[i].llvmver = 0; // all features available
    }
    nfeature_names_val = n;
}

// These must be visible to processor.cpp after the #include
static const FeatureName *feature_names = feature_names_storage;
static uint32_t &nfeature_names = nfeature_names_val;

// ============================================================================
// Feature masks (set of all known feature bits)
// ============================================================================

static FeatureList<feature_sz> compute_feature_masks() {
    FeatureList<feature_sz> m = {};
    for (unsigned i = 0; i < num_features; i++) {
        set_bit(m, feature_table[i].bit, true);
    }
    return m;
}

static const FeatureList<feature_sz> feature_masks = compute_feature_masks();

// ============================================================================
// Host CPU detection
// ============================================================================

static inline const std::string &host_cpu_name()
{
    return tp::get_host_cpu_name();
}

static inline const std::pair<uint32_t, FeatureList<feature_sz>> &get_host_cpu()
{
    static auto host = [] {
        auto fb = tp::get_host_features();
        FeatureList<feature_sz> fl = {};
        featurebits_to_list(fb, fl);
        return std::make_pair(uint32_t(0), fl);
    }();
    return host;
}

// Get host CPU feature string for Julia
static std::string get_host_feature_string()
{
    auto fb = tp::get_host_features();
    return tp::build_feature_string(fb);
}

// ============================================================================
// Command line target parsing using cpufeatures
// ============================================================================

static const llvm::SmallVector<TargetData<feature_sz>, 0> &get_cmdline_targets_cpufeatures(const char *cpu_target)
{
    auto feature_cb = [] (const char *str, size_t len, FeatureList<feature_sz> &list) {
        // Look up feature by name
        char buf[64];
        if (len >= sizeof(buf)) return false;
        memcpy(buf, str, len);
        buf[len] = '\0';
        const FeatureEntry *fe = find_feature(buf);
        if (!fe) return false;
        set_bit(list, fe->bit, true);
        return true;
    };
    return get_cmdline_targets<feature_sz>(cpu_target, feature_cb);
}

// ============================================================================
// JIT target management
// ============================================================================

static llvm::SmallVector<TargetData<feature_sz>, 0> jit_targets;

static TargetData<feature_sz> arg_target_data(const TargetData<feature_sz> &arg, bool require_host)
{
    TargetData<feature_sz> res = arg;
    const FeatureList<feature_sz> *cpu_features = nullptr;

    CF_DEBUG("[cpufeatures] arg_target_data: name='%s' require_host=%d\n",
             arg.name.c_str(), require_host);

    if (res.name == "native") {
        res.name = host_cpu_name();
        cpu_features = &get_host_cpu().second;
        CF_DEBUG("[cpufeatures]   resolved 'native' -> '%s'\n", res.name.c_str());
    }
    else {
        // Look up CPU in cpufeatures database
        const CPUEntry *entry = find_cpu(res.name.c_str());
        if (entry) {
            FeatureList<feature_sz> fl = {};
            featurebits_to_list(entry->features, fl);
            // Store as a static to take a pointer
            static thread_local FeatureList<feature_sz> cpu_fl;
            cpu_fl = fl;
            cpu_features = &cpu_fl;
            CF_DEBUG("[cpufeatures]   found CPU '%s' in database\n", res.name.c_str());
        }
        else {
            res.en.flags |= JL_TARGET_UNKNOWN_NAME;
            CF_DEBUG("[cpufeatures]   CPU '%s' NOT in database (unknown)\n", res.name.c_str());
        }
    }

    if (cpu_features) {
        for (size_t i = 0; i < feature_sz; i++) {
            res.en.features[i] |= (*cpu_features)[i];
        }
    }

    // Resolve feature dependencies
    FeatureBits fb;
    list_to_featurebits(res.en.features, fb);
    expand_implied(&fb);
    featurebits_to_list(fb, res.en.features);

    // Apply disabled features
    for (size_t i = 0; i < feature_sz; i++)
        res.en.features[i] &= ~res.dis.features[i];

    // If requiring host, mask to host features
    if (require_host) {
        for (size_t i = 0; i < feature_sz; i++) {
            res.en.features[i] &= get_host_cpu().second[i];
        }
    }

    // Disable features whose dependencies are missing
    // (converting back to FeatureBits for expand_implied won't help here,
    //  we just mask against known features)

    if (cpu_features) {
        // Fill in disable features as complement of enabled within known HW mask.
        // Only hw features go into dis - tuning features must not block sysimg matching.
        FeatureList<feature_sz> hw_mask = {};
        featurebits_to_list(hw_feature_mask, hw_mask);
        for (size_t i = 0; i < feature_sz; i++) {
            res.dis.features[i] = hw_mask[i] & ~res.en.features[i];
        }
    }

    return res;
}

// ============================================================================
// Vector register size (for dispatch tie-breaking)
// ============================================================================

static int max_vector_size(const FeatureList<feature_sz> &features)
{
#if defined(_CPU_X86_64_) || defined(_CPU_X86_)
    const FeatureEntry *avx512f = find_feature("avx512f");
    const FeatureEntry *avx = find_feature("avx");
    if (avx512f && test_nbit(features, avx512f->bit))
        return 64;
    if (avx && test_nbit(features, avx->bit))
        return 32;
    return 16; // SSE
#elif defined(_CPU_AARCH64_)
    const FeatureEntry *sve = find_feature("sve");
    if (sve && test_nbit(features, sve->bit))
        return 256; // SVE is scalable, use large number
    return 16; // NEON
#elif defined(__riscv)
    const FeatureEntry *rv_v = find_feature("v");
    const FeatureEntry *zve64d = find_feature("zve64d");
    if ((rv_v && test_nbit(features, rv_v->bit)) ||
        (zve64d && test_nbit(features, zve64d->bit)))
        return 128; // RVV scalable
    return 0;
#else
    (void)features;
    return 16;
#endif
}

// ============================================================================
// Clone flag computation
// ============================================================================

static void compute_clone_flags(llvm::SmallVector<TargetData<feature_sz>, 0> &targets)
{
    auto ntargets = targets.size();
    for (size_t i = 1; i < ntargets; i++) {
        auto &t = targets[i];
        if (t.en.flags & JL_TARGET_CLONE_ALL)
            continue;
        // Always clone when code checks CPU features or has loops
        t.en.flags |= JL_TARGET_CLONE_CPU;
        t.en.flags |= JL_TARGET_CLONE_LOOP;

        auto &features0 = targets[t.base].en.features;

        auto check_feature = [&](const char *name) -> bool {
            const FeatureEntry *fe = find_feature(name);
            return fe && !test_nbit(features0, fe->bit) && test_nbit(t.en.features, fe->bit);
        };

#if defined(_CPU_X86_64_) || defined(_CPU_X86_)
        // KNL/KNM special case
        if (!(t.dis.flags & JL_TARGET_CLONE_ALL)) {
            if ((t.name == "knl" || t.name == "knm") &&
                targets[t.base].name != "knl" && targets[t.base].name != "knm") {
                t.en.flags |= JL_TARGET_CLONE_ALL;
                continue;
            }
        }
        // Math cloning (FMA)
        if (check_feature("fma") || check_feature("fma4"))
            t.en.flags |= JL_TARGET_CLONE_MATH;

        // SIMD cloning
        static const char *simd_features[] = {
            "sse3", "ssse3", "sse4.1", "sse4.2", "avx", "avx2",
            "avx512f", "avx512dq", "avx512bw", "avx512vl",
            "avx512vnni", "avx512bf16", "avx512fp16", nullptr
        };
        for (const char **f = simd_features; *f; f++) {
            if (check_feature(*f)) { t.en.flags |= JL_TARGET_CLONE_SIMD; break; }
        }
        if (check_feature("avx512fp16"))
            t.en.flags |= JL_TARGET_CLONE_FLOAT16;
        if (check_feature("avx512bf16"))
            t.en.flags |= JL_TARGET_CLONE_BFLOAT16;

#elif defined(_CPU_AARCH64_)
        if (check_feature("sve") || check_feature("sve2"))
            t.en.flags |= JL_TARGET_CLONE_SIMD;
        if (check_feature("fullfp16"))
            t.en.flags |= JL_TARGET_CLONE_FLOAT16;
        if (check_feature("bf16"))
            t.en.flags |= JL_TARGET_CLONE_BFLOAT16;

#elif defined(__riscv)
        if (check_feature("v") || check_feature("zve32x") || check_feature("zve64d"))
            t.en.flags |= JL_TARGET_CLONE_SIMD;
        if (check_feature("zfh"))
            t.en.flags |= JL_TARGET_CLONE_FLOAT16;
        if (check_feature("zvfbfmin"))
            t.en.flags |= JL_TARGET_CLONE_BFLOAT16;
#else
        // Fallback: clone everything for non-default targets
        t.en.flags |= JL_TARGET_CLONE_ALL;
#endif
    }
}

// ============================================================================
// LLVM target string generation
// ============================================================================

static std::pair<std::string, llvm::SmallVector<std::string, 0>>
get_llvm_target_noext(const TargetData<feature_sz> &data)
{
    std::string name = data.name;

    // Normalize generic names
#if defined(_CPU_X86_64_)
    if (name == "generic" || name == "x86-64" || name == "x86_64")
        name = "x86-64";
#elif defined(_CPU_X86_)
    if (name == "generic" || name == "i686" || name == "pentium4")
        name = "pentium4";
#endif

    // Only emit hardware features to LLVM (not tuning features).
    // hw_feature_mask from the generated tables identifies real ISA extensions.
    FeatureBits hw_mask_fb;
    memcpy(&hw_mask_fb, &hw_feature_mask, sizeof(hw_mask_fb));

    llvm::SmallVector<std::string, 0> features;
    for (uint32_t i = 0; i < nfeature_names; i++) {
        auto &fename = feature_names_storage[i];
        if (!feature_test(&hw_mask_fb, fename.bit))
            continue; // Skip tuning features
        if (test_nbit(data.en.features, fename.bit)) {
            features.insert(features.begin(), std::string("+") + fename.name);
        }
        else if (test_nbit(data.dis.features, fename.bit)) {
            features.push_back(std::string("-") + fename.name);
        }
    }

    // Baseline features always required
#if defined(_CPU_X86_64_) || defined(_CPU_X86_)
    features.push_back("+sse2");
    features.push_back("+mmx");
    features.push_back("+fxsr");
#ifdef _CPU_X86_64_
    features.push_back("+64bit");
#endif
    features.push_back("+cx8");
#endif

    return std::make_pair(std::move(name), std::move(features));
}

static std::pair<std::string, llvm::SmallVector<std::string, 0>>
get_llvm_target_vec(const TargetData<feature_sz> &data)
{
    auto res = get_llvm_target_noext(data);
    append_ext_features(res.second, data.ext_features);
    return res;
}

static std::pair<std::string, std::string>
get_llvm_target_str(const TargetData<feature_sz> &data)
{
    auto res = get_llvm_target_noext(data);
    auto features = join_feature_strs(res.second);
    append_ext_features(features, data.ext_features);
    return std::make_pair(std::move(res.first), std::move(features));
}

// ============================================================================
// Sysimage initialization callbacks
// ============================================================================

static uint32_t sysimg_init_cb(void *ctx, const void *id, jl_value_t **rejection_reason)
{
    const char *cpu_target = (const char *)ctx;
    CF_DEBUG("[cpufeatures] sysimg_init_cb: cpu_target='%s'\n",
             cpu_target ? cpu_target : "(null)");
    CF_DEBUG("[cpufeatures]   host CPU: '%s'\n", host_cpu_name().c_str());

    auto &cmdline = get_cmdline_targets_cpufeatures(cpu_target);
    CF_DEBUG("[cpufeatures]   cmdline has %zu target(s)\n", cmdline.size());
    TargetData<feature_sz> target = arg_target_data(cmdline[0], true);

    CF_DEBUG("[cpufeatures]   JIT target: name='%s' features=%s\n",
             target.name.c_str(), debug_feature_str(target.en.features).c_str());

    // Deserialize sysimage targets
    auto sysimg = deserialize_target_data<feature_sz>((const uint8_t *)id);
    CF_DEBUG("[cpufeatures]   sysimg has %zu target(s):\n", sysimg.size());
    for (uint32_t i = 0; i < sysimg.size(); i++) {
        CF_DEBUG("[cpufeatures]     [%u] name='%s' flags=0x%x features=%s\n",
                 i, sysimg[i].name.c_str(), sysimg[i].en.flags,
                 debug_feature_str(sysimg[i].en.features).c_str());
    }

    // Match using the existing template function
    auto match = match_sysimg_targets(sysimg, target, max_vector_size, rejection_reason);
    if (match.best_idx == UINT32_MAX) {
        CF_DEBUG("[cpufeatures]   NO compatible target found!\n");
        return match.best_idx;
    }

    CF_DEBUG("[cpufeatures]   selected target %u '%s' (vreg_size=%d)\n",
             match.best_idx, sysimg[match.best_idx].name.c_str(), match.vreg_size);

    // Adjust JIT target vector register size to match sysimg
    if (match.vreg_size != max_vector_size(target.en.features) &&
        (sysimg[match.best_idx].en.flags & JL_TARGET_VEC_CALL)) {
#if defined(_CPU_X86_64_) || defined(_CPU_X86_)
        // Disable wider vector features if sysimg doesn't have them
        if (match.vreg_size < 64) {
            // Disable AVX512
            const char *avx512_features[] = {
                "avx512f", "avx512dq", "avx512ifma", "avx512cd",
                "avx512bw", "avx512vl", "avx512vbmi", "avx512vpopcntdq",
                "avx512vbmi2", "avx512vnni", "avx512bitalg",
                "avx512vp2intersect", "avx512bf16", "avx512fp16", nullptr
            };
            for (const char **f = avx512_features; *f; f++) {
                const FeatureEntry *fe = find_feature(*f);
                if (fe) unset_bits(target.en.features, fe->bit);
            }
        }
        if (match.vreg_size < 32) {
            // Disable AVX
            const char *avx_features[] = {
                "avx", "avx2", "fma", "f16c", "fma4", "xop",
                "vaes", "vpclmulqdq", nullptr
            };
            for (const char **f = avx_features; *f; f++) {
                const FeatureEntry *fe = find_feature(*f);
                if (fe) unset_bits(target.en.features, fe->bit);
            }
        }
#elif defined(_CPU_AARCH64_)
        if (match.vreg_size < 256) {
            const FeatureEntry *sve = find_feature("sve");
            const FeatureEntry *sve2 = find_feature("sve2");
            if (sve) unset_bits(target.en.features, sve->bit);
            if (sve2) unset_bits(target.en.features, sve2->bit);
        }
#endif
    }

    jit_targets.push_back(std::move(target));
    return match.best_idx;
}

static uint32_t pkgimg_init_cb(void *ctx, const void *id, jl_value_t **rejection_reason)
{
    TargetData<feature_sz> target = jit_targets.front();
    auto pkgimg = deserialize_target_data<feature_sz>((const uint8_t *)id);
    auto match = match_sysimg_targets(pkgimg, target, max_vector_size, rejection_reason);
    return match.best_idx;
}

static void ensure_jit_target(const char *cpu_target, bool imaging)
{
    init_feature_names();
    auto &cmdline = get_cmdline_targets_cpufeatures(cpu_target);
    check_cmdline(cmdline, imaging);
    if (!jit_targets.empty())
        return;
    CF_DEBUG("[cpufeatures] ensure_jit_target: '%s' imaging=%d, %zu cmdline target(s)\n",
             cpu_target ? cpu_target : "(null)", imaging, cmdline.size());
    for (auto &arg : cmdline) {
        auto data = arg_target_data(arg, jit_targets.empty());
        jit_targets.push_back(std::move(data));
    }
    compute_clone_flags(jit_targets);
    CF_DEBUG("[cpufeatures]   resolved %zu JIT target(s):\n", jit_targets.size());
    for (size_t i = 0; i < jit_targets.size(); i++) {
        CF_DEBUG("[cpufeatures]     [%zu] name='%s' flags=0x%x base=%d features=%s\n",
                 i, jit_targets[i].name.c_str(), jit_targets[i].en.flags,
                 jit_targets[i].base,
                 debug_feature_str(jit_targets[i].en.features).c_str());
    }
}

// ============================================================================
// Exported functions
// ============================================================================

// X86-specific: CPUID wrappers
#if defined(_CPU_X86_64_) || defined(_CPU_X86_)

extern "C" JL_DLLEXPORT void jl_cpuid(int32_t CPUInfo[4], int32_t InfoType)
{
    asm volatile (
#if defined(__i386__) && defined(__PIC__)
        "xchg %%ebx, %%esi;"
        "cpuid;"
        "xchg %%esi, %%ebx;" :
        "=S" (CPUInfo[1]),
#else
        "cpuid" :
        "=b" (CPUInfo[1]),
#endif
        "=a" (CPUInfo[0]),
        "=c" (CPUInfo[2]),
        "=d" (CPUInfo[3]) :
        "a" (InfoType)
    );
}

extern "C" JL_DLLEXPORT void jl_cpuidex(int32_t CPUInfo[4], int32_t InfoType, int32_t subInfoType)
{
    asm volatile (
#if defined(__i386__) && defined(__PIC__)
        "xchg %%ebx, %%esi;"
        "cpuid;"
        "xchg %%esi, %%ebx;" :
        "=S" (CPUInfo[1]),
#else
        "cpuid" :
        "=b" (CPUInfo[1]),
#endif
        "=a" (CPUInfo[0]),
        "=c" (CPUInfo[2]),
        "=d" (CPUInfo[3]) :
        "a" (InfoType),
        "c" (subInfoType)
    );
}

#endif // x86

JL_DLLEXPORT void jl_dump_host_cpu(void)
{
    init_feature_names();
    jl_safe_printf("CPU: %s\n", host_cpu_name().c_str());
    jl_safe_printf("Features:");
    auto &host = get_host_cpu();
    bool first = true;
    for (uint32_t i = 0; i < nfeature_names; i++) {
        if (test_nbit(host.second, feature_names_storage[i].bit)) {
            if (first) {
                jl_safe_printf(" %s", feature_names_storage[i].name);
                first = false;
            }
            else {
                jl_safe_printf(", %s", feature_names_storage[i].name);
            }
        }
    }
    jl_safe_printf("\n");
}

JL_DLLEXPORT jl_value_t *jl_check_pkgimage_clones(char *data)
{
    jl_value_t *rejection_reason = NULL;
    JL_GC_PUSH1(&rejection_reason);
    uint32_t match_idx = pkgimg_init_cb(NULL, data, &rejection_reason);
    JL_GC_POP();
    if (match_idx == UINT32_MAX)
        return rejection_reason;
    return jl_nothing;
}

JL_DLLEXPORT jl_value_t *jl_cpu_has_fma(int bits)
{
#if defined(_CPU_X86_64_) || defined(_CPU_X86_)
    if (bits == 32 || bits == 64) {
        const FeatureEntry *fma_fe = find_feature("fma");
        const FeatureEntry *fma4_fe = find_feature("fma4");
        auto &host = get_host_cpu();
        if ((fma_fe && test_nbit(host.second, fma_fe->bit)) ||
            (fma4_fe && test_nbit(host.second, fma4_fe->bit)))
            return jl_true;
    }
#elif defined(_CPU_AARCH64_)
    // AArch64 always has FMA
    if (bits == 32 || bits == 64)
        return jl_true;
#endif
    return jl_false;
}

jl_image_t jl_init_processor_sysimg(jl_image_buf_t image, const char *cpu_target)
{
    init_feature_names();
    if (!jit_targets.empty())
        jl_error("JIT targets already initialized");
    return parse_sysimg(image, sysimg_init_cb, (void *)cpu_target);
}

jl_image_t jl_init_processor_pkgimg(jl_image_buf_t image)
{
    if (jit_targets.empty())
        jl_error("JIT targets not initialized");
    if (jit_targets.size() > 1)
        jl_error("Expected only one JIT target");
    return parse_sysimg(image, pkgimg_init_cb, NULL);
}

std::pair<std::string, llvm::SmallVector<std::string, 0>>
jl_get_llvm_target(const char *cpu_target, bool imaging, uint32_t &flags)
{
    ensure_jit_target(cpu_target, imaging);
    flags = jit_targets[0].en.flags;
    return get_llvm_target_vec(jit_targets[0]);
}

const std::pair<std::string, std::string> &jl_get_llvm_disasm_target(void)
{
    static const auto res = get_llvm_target_str(TargetData<feature_sz>{"generic", "",
            {feature_masks, 0}, {{}, 0}, 0});
    return res;
}

#ifndef __clang_gcanalyzer__
llvm::SmallVector<jl_target_spec_t, 0> jl_get_llvm_clone_targets(const char *cpu_target)
{
    init_feature_names();
    auto &cmdline = get_cmdline_targets_cpufeatures(cpu_target);
    check_cmdline(cmdline, true);
    llvm::SmallVector<TargetData<feature_sz>, 0> image_targets;
    for (auto &arg : cmdline) {
        auto data = arg_target_data(arg, image_targets.empty());
        image_targets.push_back(std::move(data));
    }
    compute_clone_flags(image_targets);

    if (image_targets.empty())
        jl_error("No targets specified");

    // Compute hw-only feature mask as FeatureList for masking
    FeatureList<feature_sz> hw_mask = {};
    featurebits_to_list(hw_feature_mask, hw_mask);

    llvm::SmallVector<jl_target_spec_t, 0> res;
    for (auto &target : image_targets) {
        jl_target_spec_t ele;
        std::tie(ele.cpu_name, ele.cpu_features) = get_llvm_target_str(target);
        // Only serialize hardware features (not tuning features) into sysimage data.
        // Tuning features differ between CPU vendors and would cause false
        // mismatches during sysimage loading (e.g. znver4 loading skylake-avx512 image).
        FeatureList<feature_sz> hw_en = {}, hw_dis = {};
        for (size_t i = 0; i < feature_sz; i++) {
            hw_en[i] = target.en.features[i] & hw_mask[i];
            hw_dis[i] = target.dis.features[i] & hw_mask[i];
        }
        ele.data = serialize_target_data(target.name, hw_en, hw_dis, target.ext_features);
        ele.flags = target.en.flags;
        ele.base = target.base;
        res.push_back(ele);
    }
    return res;
}
#endif

extern "C" int jl_test_cpu_feature(jl_cpu_feature_t feature)
{
    if (feature >= 32 * feature_sz)
        return 0;
    return test_nbit(&get_host_cpu().second[0], feature);
}

// ============================================================================
// CPU feature query API (used by cpuid.jl)
// ============================================================================

// Return the number of feature bytes (for buffer allocation on Julia side)
extern "C" JL_DLLEXPORT size_t jl_cpufeatures_nbytes(void)
{
    return sizeof(FeatureBits);
}

// Look up hardware features for a named CPU. Only returns bits in hw_feature_mask.
// Returns 0 on success, -1 if CPU not found.
extern "C" JL_DLLEXPORT int jl_cpufeatures_lookup(const char *cpu_name,
                                                    uint8_t *features_out,
                                                    size_t bufsize)
{
    if (bufsize < sizeof(FeatureBits))
        return -1;
    const CPUEntry *entry = find_cpu(cpu_name);
    if (!entry)
        return -1;
    // Mask to hardware features only (exclude tuning hints)
    FeatureBits hw;
    for (int i = 0; i < TARGET_FEATURE_WORDS; i++)
        hw.bits[i] = entry->features.bits[i] & hw_feature_mask.bits[i];
    memcpy(features_out, &hw, sizeof(FeatureBits));
    return 0;
}

// Get host CPU hardware features.
extern "C" JL_DLLEXPORT void jl_cpufeatures_host(uint8_t *features_out, size_t bufsize)
{
    if (bufsize < sizeof(FeatureBits))
        return;
    auto fb = tp::get_host_features();
    // Mask to hardware features
    for (int i = 0; i < TARGET_FEATURE_WORDS; i++)
        fb.bits[i] &= hw_feature_mask.bits[i];
    memcpy(features_out, &fb, sizeof(FeatureBits));
}

// ============================================================================
// FPU control (architecture-specific)
// ============================================================================

#if defined(_CPU_X86_64_) || defined(_CPU_X86_)

#include <xmmintrin.h>

static uint32_t subnormal_flags = [] {
    int32_t info[4];
    jl_cpuid(info, 0);
    if (info[0] >= 1) {
        jl_cpuid(info, 1);
        if (info[3] & (1 << 26)) {
            return 0x00008040u; // SSE2: FZ + DAZ
        }
        else if (info[3] & (1 << 25)) {
            return 0x00008000u; // SSE: FZ only
        }
    }
    return 0u;
}();

extern "C" JL_DLLEXPORT int32_t jl_get_zero_subnormals(void)
{
    return _mm_getcsr() & subnormal_flags;
}

extern "C" JL_DLLEXPORT int32_t jl_set_zero_subnormals(int8_t isZero)
{
    uint32_t flags = subnormal_flags;
    if (flags) {
        uint32_t state = _mm_getcsr();
        if (isZero)
            state |= flags;
        else
            state &= ~flags;
        _mm_setcsr(state);
        return 0;
    }
    return isZero;
}

extern "C" JL_DLLEXPORT int32_t jl_get_default_nans(void)
{
    return 0;
}

extern "C" JL_DLLEXPORT int32_t jl_set_default_nans(int8_t isDefault)
{
    return isDefault;
}

#elif defined(_CPU_AARCH64_)

extern "C" JL_DLLEXPORT int32_t jl_get_zero_subnormals(void)
{
    uint64_t fpcr;
    asm volatile ("mrs %0, fpcr" : "=r"(fpcr));
    return (fpcr & (1 << 24)) != 0; // FZ bit
}

extern "C" JL_DLLEXPORT int32_t jl_set_zero_subnormals(int8_t isZero)
{
    uint64_t fpcr;
    asm volatile ("mrs %0, fpcr" : "=r"(fpcr));
    if (isZero)
        fpcr |= (1 << 24);
    else
        fpcr &= ~(uint64_t)(1 << 24);
    asm volatile ("msr fpcr, %0" :: "r"(fpcr));
    return 0;
}

extern "C" JL_DLLEXPORT int32_t jl_get_default_nans(void)
{
    uint64_t fpcr;
    asm volatile ("mrs %0, fpcr" : "=r"(fpcr));
    return (fpcr & (1 << 25)) != 0; // DN bit
}

extern "C" JL_DLLEXPORT int32_t jl_set_default_nans(int8_t isDefault)
{
    uint64_t fpcr;
    asm volatile ("mrs %0, fpcr" : "=r"(fpcr));
    if (isDefault)
        fpcr |= (1 << 25);
    else
        fpcr &= ~(uint64_t)(1 << 25);
    asm volatile ("msr fpcr, %0" :: "r"(fpcr));
    return 0;
}

#else
// Fallback: no FPU control

extern "C" JL_DLLEXPORT int32_t jl_get_zero_subnormals(void)
{
    return 0;
}

extern "C" JL_DLLEXPORT int32_t jl_set_zero_subnormals(int8_t isZero)
{
    return isZero;
}

extern "C" JL_DLLEXPORT int32_t jl_get_default_nans(void)
{
    return 0;
}

extern "C" JL_DLLEXPORT int32_t jl_set_default_nans(int8_t isDefault)
{
    return isDefault;
}

#endif
