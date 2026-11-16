"""Print the CUDA libraries shipped as pip `nvidia-*` packages (InsightFace / onnxruntime-gpu need them on LD_LIBRARY_PATH)."""
import glob, os, sysconfig
print(":".join(glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "*", "lib"))))
