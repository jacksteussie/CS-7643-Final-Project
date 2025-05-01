#!/bin/bash

# Define paths
CONFIG=/teamspace/studios/this_studio/mmrotate/demo/rotated_frcnn_4_lr/rotated_faster_rcnn_r50_fpn_1x_dota_le90_val_1p5.py""
CHECKPOINT_DIR="/teamspace/studios/this_studio/mmrotate/demo/rotated_frcnn_4_lr"
OUTPUT_FILE="/teamspace/studios/this_studio/mmrotate/demo/rotated_frcnn_4_lr/results_dist_test_0.log"

# Clear output file before writing new results
> $OUTPUT_FILE

# Loop through epochs from 2 to 20
for i in $(seq 30 -1 25)
do
    CHECKPOINT="${CHECKPOINT_DIR}/epoch_${i}.pth"
    echo "Testing: $CHECKPOINT" | tee -a $OUTPUT_FILE
    bash /teamspace/studios/this_studio/mmrotate/tools/dist_test.sh $CONFIG $CHECKPOINT 4 --eval mAP | tee -a $OUTPUT_FILE
done

echo "All evaluations completed! Results saved in $OUTPUT_FILE"