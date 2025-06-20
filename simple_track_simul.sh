# python occ_sort_app.py \
#     --detection_outdir=./detections/Simul \
#     --detection_file=./detections/Simul/det-1.npy \
#     --timestamp_file=./detections/Simul/time-1.npy \
#     --aux_file=./detections/Simul/aux-1.npz \
#     --ori_dir=/home/wuxiyang/Research/occ_vis/track_data \
#     --min_confidence=0.65 \
#     --nn_budget=100 \
#     --display=False \
#     --occ_size=0.1 \
#     --occ_range=[-10,-10,10,10]
    # --occ_size=0.4 \
    # --occ_range=[-40,-40,40,40]


python sort_v1.py \
    --detection_outdir=./detections/Simul \
    --detection_file=./detections/Simul/det-1.npy \
    --timestamp_file=./detections/Simul/time-1.npy \
    --aux_file=./detections/Simul/aux-1.npz \
    --ori_dir=/home/wuxiyang/Research/occ_vis/track_data \
    --min_confidence=0.65 \
    --nn_budget=100 \
    --display=False \
    --occ_size=0.1 \
    --occ_range="-10,-10,10,10"