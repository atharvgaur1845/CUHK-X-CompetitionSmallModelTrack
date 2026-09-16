# CUHK-X Small Model Track, team AGX -- verification image.
# Base = the toolchain that reproduced our submission on the cluster (EXP-159):
# Python 3.11, torch 2.7.1+cu126, torchvision 0.22.1. CUDA 12.6 runs on NVIDIA driver >= 525
# (CUDA 12 minor-version compatibility). Everything needed at inference is inside the image;
# nothing is downloaded when it runs (the Committee runs it with --network none).
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 YOLO_OFFLINE=1 \
    DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

# ultralytics is installed --no-deps so that the GUI OpenCV build (needs libGL) stays out;
# its runtime dependencies are pinned explicitly. Versions are recorded by `pip freeze`
# into environment/requirements.txt after the build.
RUN pip install --no-cache-dir \
        opencv-python-headless==5.0.0.93 polars==1.44.2 py-cpuinfo==9.0.0 matplotlib==3.11.2 \
        nvidia-ml-py==13.610.43 ultralytics-thop==2.1.6 \
    && pip install --no-cache-dir --no-deps ultralytics==8.4.118
# numpy 2.2.6, pillow 11.0.0, pyyaml 6.0.2, requests 2.32.3, psutil 7.0.0, filelock 3.18.0
# come with the base image and are left untouched.

WORKDIR /package
COPY . /package

# Build-time self-checks: artifact hashes, the package opens, the detectors rebuild from the
# package, and every video architecture builds with NO torch hub cache and NO network.
RUN sha256sum -c artifacts/SHA256SUMS \
 && python -c "import torch, torchvision, ultralytics, cv2, PIL, numpy; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'ultralytics', ultralytics.__version__)" \
 && python src/code/unpack_stage2.py --package artifacts/stage2_ship7.pth --check integrity \
 && python src/code/extract_detectors.py --package artifacts/stage2_ship7.pth --out /tmp/selfcheck \
 && TORCH_HOME=/tmp/empty_torch_home python -c "import sys; sys.path.insert(0, 'src/kaggle'); import cuhkx_224_kaggle as K; [K.build_model('mvit_v2_s', image_size=s, in_channels=c, pretrained=False) for s, c in ((384, 4), (224, 4), (224, 3))]; print('offline model build OK')" \
 && rm -rf /tmp/selfcheck /tmp/empty_torch_home /root/.config/Ultralytics /package/work

ENTRYPOINT []
CMD ["bash", "inference.sh", "/data", "/output/submission.csv"]
