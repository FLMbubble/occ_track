python occ_sort_app.py \
    --detection_file=./detections/MY_PANOOCC/det-scene-0004.npy \
    --timestamp_file=./detections/MY_PANOOCC/time-scene-0004.npy \
    --aux_file=./detections/MY_PANOOCC/aux-scene-0004.npz \
    --ori_dir=./detections/track_data \
    --min_confidence=0.65 \
    --nn_budget=100 \
    --display=False