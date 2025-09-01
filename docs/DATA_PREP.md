# Data Preparation
Downlowd [V2X-Seq-SPD](https://github.com/AIR-THU/DAIR-V2X?tab=readme-ov-file#dataset) dataset, organize the files as official instructions and then follow the steps below to prepare the dataset.

a. Create V2X-Seq-SPD-Example (as an example, you can give it a name that you like) according to given sequence ids.

```
python ./tools/spd_data_converter/gen_example_data.py --input YOUR_V2X-Seq-SPD_ROOT --output ./datasets/V2X-Seq-SPD-Example --sequences 0000 0001 0002 --update-label
```
Note that if you want to use all sequences, you can directly use `--sequences 'all'`. We report results on all sequences.

b. Generate data info for training and nuscenes style dataset for evaluation. 
You should modify `./tools/spd_example_converter.sh` according to the path used in the previous step, and then execute the following commands in the terminal.
```
bash ./tools/spd_example_converter.sh vehicle-side
bash ./tools/spd_example_converter.sh infrastructure-side
bash ./tools/spd_example_converter.sh cooperative
```

### Prepare pretrained models
Download checkpoints of R101DCN and BEVFormer from [this project](https://github.com/fundamentalvision/BEVFormer), and create a new folder named `ckpt` and place the downloaded models in it.

### Overall Structure
As a result, you will see the overall structure as follows:
```
V2XTrack
├── projects/
├── tools/
├── ckpts/
│   ├── r101_dcn_fcos3d_pretrain.pth
│   ├── bevformer_r101_dcn_24ep.pth
├── data/
│   ├── infos/
|   |   ├── V2X-Seq-SPD-Example
|   |   |   ├── cooperative
|   |   │   │   ├── spd_infos_temporal_train.pkl
|   |   │   │   ├── spd_infos_temporal_val.pkl
|   |   |   ├── vehicle-side
|   |   |   ├── infrastructure-side
│   ├── split_datas/
│   │   ├── cooperative-split-data-spd.json
├── datasets/
│   ├── V2X-Seq-SPD-Example/
│   │   ├── cooperative/
│   │   │   ├── calib /
│   │   │   ├── image /
│   │   │   ├── label /
│   │   │   ├── velodyne /
│   │   │   ├── v1.0-trainval /
│   │   │   ├── data_info.json
│   │   ├── vehicle-side/
│   │   ├── infrastructure-side/
```

---
<- Last Page:  [Installation](./INSTALL.md)

-> Next Page: [Train/Eval UniAD](./TRAIN_EVAL.md)