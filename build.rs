use std::env;
use std::path::PathBuf;

fn main() {
    // Rerun build script if build.rs or hardware flags change
    println!("cargo:rerun-if-changed=build.rs");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    println!("cargo:warning=Building Kronos Core for target OS: {}, arch: {}", target_os, target_arch);

    // --- Apple Metal Hardware Acceleration Flags (macOS) ---
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-arg=-fapple-link-rtlib");
    }

    // --- NVIDIA CUDA Support Flags (Linux / Windows) ---
    if cfg!(feature = "cuda") {
        let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
        let cuda_lib_path = PathBuf::from(&cuda_path).join("lib64");

        if cuda_lib_path.exists() {
            println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
            println!("cargo:rustc-link-lib=dylib=cuda");
            println!("cargo:rustc-link-lib=dylib=cudart");
            println!("cargo:rustc-link-lib=dylib=cublas");
        } else {
            println!("cargo:warning=CUDA feature enabled, but lib64 not found at {:?}", cuda_lib_path);
        }
    }

    // --- CPU SIMD Optimization Flags ---
    // Enables AVX2/FMA vector instructions on x86_64 for fast tensor math
    if target_arch == "x86_64" {
        println!("cargo:rustc-env=RUSTFLAGS=-C target-cpu=native");
    }
}
