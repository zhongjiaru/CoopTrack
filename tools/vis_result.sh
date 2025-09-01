#!/bin/bash
export PYTHONPATH=$PYTHONPATH:./
python ./tools/analysis_tools/visual_spd \
    --predroot test/tiny_track_r50_stream_bs8_48epoch_3cls/Sun_Dec_29_18_24_23_2024/results_nusc.json \
    --out_folder result_vis/pf-track \
    --is_side vehicle-side \
    --is_gt \
    --dataroot datasets/V2X-Seq-SPD-Batch-65-10-10761/ \