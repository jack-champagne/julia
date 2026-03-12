# This file is a part of Julia. License is MIT: https://julialang.org/license

module CPUID

export cpu_isa

"""
    ISA(features::Set{UInt32})

A structure which represents the Instruction Set Architecture (ISA) of a
computer.  It holds the `Set` of features of the CPU.

Feature bit indices come from the cpufeatures library's generated tables
(extracted from LLVM's TableGen data at build time).
"""
struct ISA
    features::Set{UInt32}
end

Base.:<=(a::ISA, b::ISA) = a.features <= b.features
Base.:<(a::ISA,  b::ISA) = a.features <  b.features
Base.isless(a::ISA,  b::ISA) = a < b

include(string(Base.BUILDROOT, "features_h.jl"))  # include($BUILDROOT/base/features_h.jl)

"""
    _featurebytes_to_isa(buf::Vector{UInt8}) -> ISA

Convert a raw feature byte buffer (from cpufeatures) into an ISA.
"""
function _featurebytes_to_isa(buf::Vector{UInt8})
    features = Set{UInt32}()
    for byte_idx in 0:length(buf)-1
        b = buf[byte_idx + 1]
        b == 0 && continue
        for bit in 0:7
            if (b >> bit) & 1 != 0
                push!(features, UInt32(byte_idx * 8 + bit))
            end
        end
    end
    return ISA(features)
end

"""
    _lookup_cpu(name::String) -> ISA

Look up hardware features for the named CPU from the cpufeatures library.
Returns an empty ISA if the CPU name is not found.
"""
function _lookup_cpu(name::String)
    nbytes = ccall(:jl_cpufeatures_nbytes, Csize_t, ())
    buf = Vector{UInt8}(undef, nbytes)
    ret = ccall(:jl_cpufeatures_lookup, Cint, (Cstring, Ptr{UInt8}, Csize_t), name, buf, nbytes)
    ret != 0 && return ISA(Set{UInt32}())
    return _featurebytes_to_isa(buf)
end

"""
    _host_isa() -> ISA

Get the hardware features of the host CPU from the cpufeatures library.
"""
function _host_isa()
    nbytes = ccall(:jl_cpufeatures_nbytes, Csize_t, ())
    buf = Vector{UInt8}(undef, nbytes)
    ccall(:jl_cpufeatures_host, Cvoid, (Ptr{UInt8}, Csize_t), buf, nbytes)
    return _featurebytes_to_isa(buf)
end

# ISA definitions per architecture family.
# CPU names map to LLVM names in the cpufeatures database.
# Names like "skylake_avx512" are BinaryBuilder tier labels;
# they use dashes for the actual LLVM lookup.
# Entries with empty string ("") use an empty feature set (generic baseline).
#
# Keep in sync with `arch_march_isa_mapping` in binaryplatforms.jl.
const ISAs_by_family = Dict(
    "i686" => [
        "pentium4" => ISA(Set{UInt32}()),
        "prescott" => _lookup_cpu("prescott"),
    ],
    "x86_64" => [
        "x86_64" => ISA(Set{UInt32}()),
        "core2" => _lookup_cpu("core2"),
        "nehalem" => _lookup_cpu("nehalem"),
        "sandybridge" => _lookup_cpu("sandybridge"),
        "haswell" => _lookup_cpu("haswell"),
        "skylake" => _lookup_cpu("skylake"),
        "skylake_avx512" => _lookup_cpu("skylake-avx512"),
    ],
    "aarch64" => [
        "armv8.0-a" => ISA(Set{UInt32}()),
        "armv8.1-a" => _lookup_cpu("cortex-a76"),    # representative v8.1
        "armv8.2-a+crypto" => _lookup_cpu("cortex-a78"), # representative v8.2+crypto
        "a64fx" => _lookup_cpu("a64fx"),
        "apple_m1" => _lookup_cpu("apple-a14"),
    ],
    "armv6l" => [
        "arm1176jzfs" => ISA(Set{UInt32}()),
    ],
    "armv7l" => [
        "armv7l" => ISA(Set{UInt32}()),
        "armv7l+neon" => ISA(Set{UInt32}()),
        "armv7l+neon+vfpv4" => ISA(Set{UInt32}()),
    ],
    "riscv64" => [
        "riscv64" => ISA(Set{UInt32}()),
    ],
    "powerpc64le" => [
        "power8" => ISA(Set{UInt32}()),
    ],
)

# Test a CPU feature exists on the currently-running host
test_cpu_feature(feature::UInt32) = ccall(:jl_test_cpu_feature, Bool, (UInt32,), feature)

# Normalize some variation in ARCH values (which typically come from `uname -m`)
function normalize_arch(arch::String)
    arch = lowercase(arch)
    if arch ∈ ("amd64",)
        arch = "x86_64"
    elseif arch ∈ ("i386", "i486", "i586")
        arch = "i686"
    elseif arch ∈ ("armv6",)
        arch = "armv6l"
    elseif arch ∈ ("arm", "armv7", "armv8", "armv8l")
        arch = "armv7l"
    elseif arch ∈ ("arm64",)
        arch = "aarch64"
    elseif arch ∈ ("ppc64le",)
        arch = "powerpc64le"
    end
    return arch
end

"""
    cpu_isa()

Return the [`ISA`](@ref) (instruction set architecture) of the current CPU.
"""
function cpu_isa()
    return _host_isa()
end

end # module CPUID
