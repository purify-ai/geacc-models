variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  description = "The GCP region to deploy into (i.e. us-east1)"
}

variable "zone" {
  type        = string
  description = "The GCP zone to deploy into (i.e. us-east1-c)"
}

variable "machine_type" {
  type        = string
  description = "The machine type to provision the instances from (i.e. f1-micro)."
}

variable "num_instances" {
  type        = string
  default     = "1"
  description = "The number of AWS EC2 instances to provision."
}

variable "gpu_type" {
  type        = string
  default     = "nvidia-tesla-k80"
  description = "GPU accelerator type: nvidia-tesla-v100 (1, 8); nvidia-tesla-p100 (1, 2, or 4); nvidia-tesla-p4 (1, 2, or 4); nvidia-tesla-k80 (1, 2, 4, or 8)"
}

variable "gpu_number" {
  type        = number
  default     = 1
  description = "Number of GPU accelerators to attach to a machine."
}

# variable "dataset_bucket" {
#   type        = string
#   description = "S3 bucket from where to download training dataset."
# }

# variable "dataset_object" {
#   type        = string
#   description = "Training dataset S3 object path (excluding bucket name)."
# }

variable "training_repo" {
  type        = string
  default     = "https://github.com/purify-ai/geacc-models.git"
  description = "Public Git repository where training code is stored."
}

variable "training_branch" {
  type        = string
  default     = "master"
  description = "Git branch to checkout."
}
