provider "google" {
  project = var.project_id
  region  = var.region
  zone = var.zone
}

resource "google_compute_address" "vm_static_ip" {
  name = "terraform-static-ip"
}

resource "google_compute_instance" "vm_instance" {
  name         = "purify-instance"
  machine_type = var.machine_type

  boot_disk {
    initialize_params {
      size = 50
      # type = "pd-ssd"
      image = "deeplearning-platform-release/tf2-latest-gpu"
    }
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.vm_static_ip.address
    }
  }

  scheduling {
    automatic_restart = false
    preemptible = true
  }

  allow_stopping_for_update = true

  # guest_accelerator {
  #   type = var.gpu_type
  #   count = var.gpu_number
  # }

  # metadata = {
  #   install-nvidia-driver = "True"
  # }

  service_account {
    #email = "9700401775-compute@developer.gserviceaccount.com"
    scopes = ["cloud-platform"]
  }
}