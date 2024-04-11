set(CMAKE_CUDA_COMPILER "/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/cuda-11.7.0-7sdye3id7ahz34mzhyzzqbxowjxgxkhu/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/gcc-10.3.0-o57x6h2gubo7bzh7evmy4mvibdqrlghr/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "11.7.64")
set(CMAKE_CUDA_DEVICE_LINKER "/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/cuda-11.7.0-7sdye3id7ahz34mzhyzzqbxowjxgxkhu/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/cuda-11.7.0-7sdye3id7ahz34mzhyzzqbxowjxgxkhu/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "14")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "10.3")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/cuda-11.7.0-7sdye3id7ahz34mzhyzzqbxowjxgxkhu")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/cuda-11.7.0-7sdye3id7ahz34mzhyzzqbxowjxgxkhu")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "11.7.64")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/cuda-11.7.0-7sdye3id7ahz34mzhyzzqbxowjxgxkhu")

set(CMAKE_CUDA_ARCHITECTURES_ALL "35;37;50;52;53;60;61;62;70;72;75;80;86;87")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "35;50;60;70;80")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/cuda-11.7.0-7sdye3id7ahz34mzhyzzqbxowjxgxkhu/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/cuda-11.7.0-7sdye3id7ahz34mzhyzzqbxowjxgxkhu/targets/x86_64-linux/lib/stubs;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/cuda-11.7.0-7sdye3id7ahz34mzhyzzqbxowjxgxkhu/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/zstd-1.5.2-726gdzljowscmdg6vy2txu3q5j3ncfc6/include;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/mpc-1.2.1-zoh6w2uzicslpz2vw6bp2cd3vawomaag/include;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/mpfr-4.1.0-32gcbvddeljaaw35yzaciq5ostd4o2d4/include;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/gmp-6.2.1-mw6xsf4g4ul4v57rnyyxtufvcvwpao6p/include;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/libxml2-2.9.13-d4fgivwilgwkucth7kcsmpbcefmngewz/include;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/libiconv-1.16-pbdcxjpgrdv3k3eaisme7cw5fnl7zjmh/include;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/ncurses-6.2-qhoz4g3phdfhwu3ian4zdcx7uotuijrt/include;/storage/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/gcc-10.3.0-o57x6h2gubo7bzh7evmy4mvibdqrlghr/include/c++/10.3.0;/storage/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/gcc-10.3.0-o57x6h2gubo7bzh7evmy4mvibdqrlghr/include/c++/10.3.0/x86_64-pc-linux-gnu;/storage/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/gcc-10.3.0-o57x6h2gubo7bzh7evmy4mvibdqrlghr/include/c++/10.3.0/backward;/storage/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/gcc-10.3.0-o57x6h2gubo7bzh7evmy4mvibdqrlghr/lib/gcc/x86_64-pc-linux-gnu/10.3.0/include;/storage/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/gcc-10.3.0-o57x6h2gubo7bzh7evmy4mvibdqrlghr/lib/gcc/x86_64-pc-linux-gnu/10.3.0/include-fixed;/usr/local/include;/storage/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/gcc-10.3.0-o57x6h2gubo7bzh7evmy4mvibdqrlghr/include;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/cuda-11.7.0-7sdye3id7ahz34mzhyzzqbxowjxgxkhu/targets/x86_64-linux/lib/stubs;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/cuda-11.7.0-7sdye3id7ahz34mzhyzzqbxowjxgxkhu/targets/x86_64-linux/lib;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/gcc-10.3.0-o57x6h2gubo7bzh7evmy4mvibdqrlghr/lib/gcc/x86_64-pc-linux-gnu/10.3.0;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/gcc-10.3.0-o57x6h2gubo7bzh7evmy4mvibdqrlghr/lib/gcc;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/gcc-10.3.0-o57x6h2gubo7bzh7evmy4mvibdqrlghr/lib64;/storage/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/gcc-10.3.0-o57x6h2gubo7bzh7evmy4mvibdqrlghr/lib64;/lib64;/usr/lib64;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/gcc-10.3.0-o57x6h2gubo7bzh7evmy4mvibdqrlghr/lib;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/zstd-1.5.2-726gdzljowscmdg6vy2txu3q5j3ncfc6/lib;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/mpc-1.2.1-zoh6w2uzicslpz2vw6bp2cd3vawomaag/lib;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/mpfr-4.1.0-32gcbvddeljaaw35yzaciq5ostd4o2d4/lib;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/gmp-6.2.1-mw6xsf4g4ul4v57rnyyxtufvcvwpao6p/lib;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/cuda-11.7.0-7sdye3id7ahz34mzhyzzqbxowjxgxkhu/lib64;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/libxml2-2.9.13-d4fgivwilgwkucth7kcsmpbcefmngewz/lib;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/libiconv-1.16-pbdcxjpgrdv3k3eaisme7cw5fnl7zjmh/lib;/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/ncurses-6.2-qhoz4g3phdfhwu3ian4zdcx7uotuijrt/lib;/storage/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/gcc-10.3.0-o57x6h2gubo7bzh7evmy4mvibdqrlghr/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")
