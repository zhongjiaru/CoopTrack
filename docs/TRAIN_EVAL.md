# Train/Eval Models
- **General Script Usage**

    For training on multiple gpus:
    ```
    CUDA_VISIBLE_DEVICES=GPU_ID ./tools/dist_train.sh CONFIG_FILE NUM_GPUS
    ```
    For inference (single GPU):
    ```
    CUDA_VISIBLE_DEVICES=0 ./tools/eval_tracking.sh CONFIG_FILE CHECKPOINT 1
    ```

- **Training**

    To train the cooperative model, we first train the single-end models, which are the vehicle-side and infrastructure-side models. Then, we run the inference for infrastructure-side model to save the perception information to the specified path, which will be used for training the cooperative model.
Below, we provide the complete training process.
  - vehicle-side
  ```
  CUDA_VISIBLE_DEVICES=GPU_ID ./tools/dist_train.sh CONFIG_FILE_VEHICLE NUM_GPUS
  ```
  - infrastructure-side
  ```
  CUDA_VISIBLE_DEVICES=GPU_ID ./tools/dist_train.sh CONFIG_FILE_INF NUM_GPUS
  ```
  Modify the parameters in CONFIG_FILE_INF.
  ```
  save_track_query=True
  save_track_query_file_root=data/infos/inf_query
  ```
  And then run inference.
  ```
  CUDA_VISIBLE_DEVICES=0 ./tools/eval_tracking.sh CONFIG_FILE_INF CHECKPOINT 1
  ```
  - cooperative

    Modify the parameters in CONFIG_FILE_COOP.
  ```
  is_track_cooperation=True
  read_track_query_file_root=data/infos/inf_query
  ```
  And then run training.
  ```
  CUDA_VISIBLE_DEVICES=GPU_ID ./tools/dist_train.sh CONFIG_FILE_COOP NUM_GPUS
  ```
  Finally, you will get the cooperative model.

- **Inference**
Inference is straightforward; you just need to load the corresponding configuration and checkpoint.
  - vehicle-side
  ```
  CUDA_VISIBLE_DEVICES=0 ./tools/eval_tracking.sh CONFIG_FILE_VEH CHECKPOINT_VEH 1
  ```
  - infrastructure-side
  ```
  CUDA_VISIBLE_DEVICES=0 ./tools/eval_tracking.sh CONFIG_FILE_INF CHECKPOINT_INF 1
  ```
  - cooperative
  ```
  CUDA_VISIBLE_DEVICES=0 ./tools/eval_tracking.sh CONFIG_FILE_COOP CHECKPOINT_COOP 1
  ```

## Trained Checkpoints
We provide trained checkpoints on [hugging face](https://huggingface.co/zhongjiaru/CoopTrack). The links and md5sum of the model are listed as follows.
| Model | HF Link | MD5 |
|----------|----------|-----------|
| cooptrack_det_r50_veh.pth | [Link](https://huggingface.co/zhongjiaru/CoopTrack/resolve/main/cooptrack_det_r50_veh.pth) | f79a1eea02bdb43f864b8e431b01c8b5 |
| cooptrack_det_r50_inf.pth | [Link](https://huggingface.co/zhongjiaru/CoopTrack/resolve/main/cooptrack_det_r50_inf.pth) | f93c88f4beb59da7267d073bb777beb1 |
| cooptrack_r50_veh.pth | [Link](https://huggingface.co/zhongjiaru/CoopTrack/resolve/main/cooptrack_r50_veh.pth) | 34d9f0553e4eb421bdb4bb3b7dae41aa |
| cooptrack_r50_inf.pth | [Link](https://huggingface.co/zhongjiaru/CoopTrack/resolve/main/cooptrack_r50_inf.pth) | 0657439da18e951724f6b4e5a623ea23 |
| cooptrack_r50_coop.pth | [Link](https://huggingface.co/zhongjiaru/CoopTrack/resolve/main/cooptrack_r50_coop.pth) | 12d4658b70f21931f3b4a793cd1bf033 |

## Visualization
Modify the arguments accordingly in `tools/vis_result.sh` and run:

```
bash tools/vis_result.sh
```

---
<- Last Page: [Prepare The Dataset](./DATA_PREP.md)