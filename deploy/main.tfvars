# US East (Ohio)
region = "us-east-2"
avail_zone = "us-east-2c"

# EU (Ireland)
#region = "eu-west-1"
#avail_zone = "eu-west-1a"

instance_type   = "p2.xlarge"
spot_price      = "0.35"  # USD/hour

ebs_volume_size = "30"    # GB
ebs_snapshot_id = "snap-0192b2031e6dbb2cd"

dataset_bucket = "private-data.purify.ai"
dataset_object = "training_data/v1.1/dataset-8k.tar.gz"

ssh_public_key = "~/.ssh/id_rsa_rustam.pub"
training_repo  = "https://github.com/purify-ai/geacc-models.git"
training_branch= "v2"