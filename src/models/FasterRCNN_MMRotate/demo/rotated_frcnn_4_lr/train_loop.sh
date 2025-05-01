#!/bin/bash

# Define paths
CONFIG="/teamspace/studios/this_studio/mmrotate/demo/rotated_frcnn_4_lr/testing_on_training_set.py"
CHECKPOINT_DIR="/teamspace/studios/this_studio/mmrotate/demo/rotated_frcnn_4_lr"
OUTPUT_FILE="/teamspace/studios/this_studio/mmrotate/demo/rotated_frcnn_4_lr/results_dist_traincontinue.log"

# Clear output file before writing new results
> $OUTPUT_FILE

# Loop through epochs from 2 to 20
for i in $(seq 30 -3 3)
do
    CHECKPOINT="${CHECKPOINT_DIR}/epoch_${i}.pth"
    echo "Testing: $CHECKPOINT" | tee -a $OUTPUT_FILE
    bash /teamspace/studios/this_studio/mmrotate/tools/dist_test.sh $CONFIG $CHECKPOINT 4 --eval mAP | tee -a $OUTPUT_FILE
done

echo "All evaluations completed! Results saved in $OUTPUT_FILE"