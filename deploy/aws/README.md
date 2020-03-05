1. Configure vars in `main.tfvars`
1. Request AWS Spot instance and start training:
    ```
    terraform apply -var-file="main.tfvars"
    ```