variable "region" {
  type    = "string"
  description = "The AWS region to deploy into (i.e. us-east-1)"
}

variable "avail_zone" {
  type    = "string"
  description = "The AWS availability zone location within the selected region (i.e. us-east-2a)."
}

variable "my_cidr_block" {
  type    = "string"
  default = "10.0.20.0/24"
}

variable "ssh_public_key" {
  description = "Public SSH key to install onto the instances."
}

variable "instance_type" {
  type    = "string"
  description = "The instance type to provision the instances from (i.e. p2.xlarge)."
}

variable "spot_price" {
  type    = "string"
  description = "The maximum hourly price (bid) you are willing to pay for the specified instance, i.e. 0.10. This price should not be below AWS' minimum spot price for the instance based on the region."
}

variable "ebs_volume_size" {
  type    = "string"
  description = "The Amazon EBS volume size (1 GB - 16 TB)."
}

variable "ebs_snapshot_id" {
  type    = "string"
  default = ""
  description = "The Amazon EBS snapshot ID to mount. Useful for saving training datasets and models across trainings."
}

variable "num_instances" {
  type    = "string"
  default = "1"
  description = "The number of AWS EC2 instances to provision."
}

# Use latest Deep Learning AMI (Amazon Linux)
data "aws_ami" "aws_deep_learning_ami" {
  most_recent = true
  owners = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI (Amazon Linux) Version*"]
  }
}

variable "dataset_bucket" {
  type    = "string"
  description = "S3 bucket from where to download training dataset."
}

variable "dataset_object" {
  type    = "string"
  description = "Training dataset S3 object path (excluding bucket name)."
}

variable "training_repo" {
  type    = "string"
  default = "https://github.com/purify-ai/geacc-models.git"
  description = "Public Git repository where training code is stored."
}
