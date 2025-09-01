# Installation

Our implementation mainly relies on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). Our server has CUDA 11.8 installed. The Installation steps are as follows:

a. Env: Create a conda virtual environment and activate it.

    conda create -n cooptrack python=3.8 -y
    conda activate cooptrack

b. Torch: Install PyTorch and torchvision.

    pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

c. CUDA: Specify the CUDA_HOME to compile operators on the gpu, e.g. spconv, mmdet3d. (Optional)

    export CUDA_HOME=YOUR_CUDA_PATH/
    # Eg: export CUDA_HOME=/user/local/cuda-11.8/

d. Install mmcv-full, mmdet and mmseg.

    pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
    pip install mmdet==2.14.0
    pip install mmsegmentation==0.14.1

e. Install mmdet3d from source code.

    git clone https://github.com/open-mmlab/mmdetection3d.git
    cd mmdetection3d
    git checkout v0.17.1
    pip install -v -e .

f. Clone our project
    
    git clone https://github.com/zhongjiaru/CoopTrack.git
    cd cooptrack
    pip install -r requirement.txt

Note that it seems that the NVIDIA 40xx series GPUs are not compatible with this environmental configuration and will result in the following error. 
```
File "./v2xtrack/projects/mmdet3d_plugin/uniad/detectors/uniad_track.py", line 333, in velo_update
    g2l_r = torch.linalg.inv(l2g_r2).type(torch.float)
RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR, when calling `cusolverDnCreate(handle)`
```
Several open-source projects have reported this issue, e.g. [UniAD](https://github.com/OpenDriveLab/UniAD/issues/54), [StreamPETR](https://github.com/exiawsh/StreamPETR/issues/161). Following their kind advice, I modified `line 333 in projects/mmdet3d_plugin/uniad/detectors/uniad_track.py`.


---
-> Next Page: [Prepare The Dataset](./DATA_PREP.md)