# US East (Ohio)
#region = "us-east-2"
#avail_zone = "us-east-2c"

# EU (Ireland)
region = "eu-west-1"
avail_zone = "eu-west-1a"

instance_type   = "p2.xlarge"
spot_price      = "0.35"  # USD/hour

ebs_volume_size = "10"    # GB
ebs_snapshot_id = ""

dataset_bucket = "my-s3-bucket-name"
dataset_object = "path/to/dataset.tar.gz"

ssh_public_key = "~/.ssh/id_rsa.pub"
training_repo  = "https://github.com/purify-ai/geacc-models.git"