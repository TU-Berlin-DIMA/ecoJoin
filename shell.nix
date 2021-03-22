with import <nixpkgs> {};

let
  CUDATOOLKIT=cudatoolkit_10_0;
in
gcc8Stdenv.mkDerivation {
  name = "efficient-gpu-joins-env";
  nativeBuildInputs = [
    CUDATOOLKIT linuxPackages.nvidia_x11 cpufrequtils tbb boost170
  ];

  # makeFlags = [ "NVCC=$(CUDA_PATH)/bin/nvcc" "LINKER=$(CUDA_PATH)/bin/nvcc" ];

  # Set Environment Variables
  CUDA_PATH=CUDATOOLKIT;
  NVCC="${CUDATOOLKIT}/bin/nvcc";
  LINKER="${CUDATOOLKIT}/bin/nvcc";

  # LD_FLAGS = "-L${pkgs.cudatoolkit_10_2}/lib -L${pkgs.cudatoolkit_10_2.lib}/lib";
  LD_LIBRARY_PATH = "${CUDATOOLKIT}/lib:${CUDATOOLKIT}/lib/stubs:${CUDATOOLKIT.lib}/lib:$LD_LIBRARY_PATH -Lnative=${pkgs.linuxPackages.nvidia_x11}/lib";
}
