#!/bin/bash
cd ~ec2-user
export DATA_DIR="/home/ec2-user/purify/data"
export CODE_DIR="/home/ec2-user/purify/code"

# Setup filesystem and folder layout
mkfs.xfs /dev/xvdh
mkdir -p $DATA_DIR
mount /dev/xvdh $DATA_DIR
mkdir -p $DATA_DIR/dataset $DATA_DIR/models $DATA_DIR/tb_logs

# Download the dataset if not exists
cd $DATA_DIR/dataset
if [ ! -d "train" ]; then
    aws s3 cp s3://${s3_bucket}/${s3_object} ./
    dataset_filename=$(basename "${s3_object}")
    tar zxf $dataset_filename --exclude='._*'
    rm $dataset_filename
    # Override any non-image suffixes, assuming we only have images
    find . -type f ! -name '*.jpg' -and ! -name '*.png' -and ! -name '*.jpeg' -and ! -name '*.gif' | xargs -I{} mv {} "{}.jpg"
fi

# Download the code
git clone -b ${git_branch} --single-branch ${git_repo} $CODE_DIR

chown -R ec2-user:ec2-user $DATA_DIR
chown -R ec2-user:ec2-user $CODE_DIR

# Start training
sudo -E -u ec2-user -i bash << EOF
source activate tensorflow_p36
pip install efficientnet
cd $CODE_DIR/training
nohup tensorboard --logdir="$DATA_DIR/tb_logs" &> nohup_tb.out &
#nohup python3 train_inceptionv3.py -d $DATA_DIR/dataset -m $DATA_DIR/models -t $DATA_DIR/tb_logs &
EOF