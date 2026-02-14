# SageMaker Secure MLOps Pipeline

CloudFormation-based MLOps reference architecture for AWS SageMaker with security-first design. Covers VPC-isolated processing, encrypted model training, automated model registry with approval gates, secure inference endpoints, and event-driven retraining.

Built from production experience deploying ML pipelines at enterprise scale with strict compliance requirements.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        VPC (Private Subnets)                   │
│                                                                │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────────┐ │
│  │ S3 Data  │──▶│  Processing  │──▶│  Training Job          │ │
│  │ (KMS)    │   │  Job (VPC)   │   │  (Encrypted EBS)       │ │
│  └──────────┘   └──────────────┘   └────────┬───────────────┘ │
│                                              │                 │
│                                              ▼                 │
│  ┌──────────────────────────┐   ┌────────────────────────────┐│
│  │  Model Registry          │◀──│  Hyperparameter Tuning     ││
│  │  (Approval Required)     │   │  (Bayesian/Random)         ││
│  └────────────┬─────────────┘   └────────────────────────────┘│
│               │                                                │
│               ▼                                                │
│  ┌────────────────────────────────────────────────────────────┐│
│  │  Real-time Endpoint (API GW + WAF + Auto-scaling)         ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                │
│  ┌──────────────────────────┐   ┌────────────────────────────┐│
│  │  EventBridge Rules       │──▶│  Retraining Trigger        ││
│  │  (Model Drift / Schedule)│   │  (Step Functions)          ││
│  └──────────────────────────┘   └────────────────────────────┘│
└────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
.
├── README.md
├── infrastructure/
│   ├── network.yaml
│   ├── security.yaml
│   ├── sagemaker-roles.yaml
│   ├── s3-buckets.yaml
│   └── kms-keys.yaml
├── pipeline/
│   ├── processing.yaml
│   ├── training.yaml
│   ├── model-registry.yaml
│   ├── endpoint.yaml
│   └── retraining.yaml
├── scripts/
│   ├── preprocessing.py
│   ├── train.py
│   └── inference.py
├── monitoring/
│   ├── model-monitor.yaml
│   └── alarms.yaml
└── .github/
    └── workflows/
        └── deploy-pipeline.yml
```

## Quick Start

```bash
git clone https://github.com/vatsayanvivek/sagemaker-secure-mlops.git
cd sagemaker-secure-mlops

# Deploy infrastructure first
aws cloudformation deploy \
  --template-file infrastructure/network.yaml \
  --stack-name mlops-network \
  --parameter-overrides Environment=dev

aws cloudformation deploy \
  --template-file infrastructure/kms-keys.yaml \
  --stack-name mlops-kms

aws cloudformation deploy \
  --template-file infrastructure/s3-buckets.yaml \
  --stack-name mlops-storage \
  --parameter-overrides KMSKeyArn=$(aws cloudformation describe-stacks --stack-name mlops-kms --query 'Stacks[0].Outputs[?OutputKey==`KMSKeyArn`].OutputValue' --output text)

# Deploy pipeline components
aws cloudformation deploy \
  --template-file infrastructure/sagemaker-roles.yaml \
  --stack-name mlops-roles \
  --capabilities CAPABILITY_NAMED_IAM

aws cloudformation deploy \
  --template-file pipeline/endpoint.yaml \
  --stack-name mlops-endpoint
```

## Stack Components

| Stack | Template | Description |
|-------|----------|-------------|
| Network | `infrastructure/network.yaml` | VPC with private subnets, VPC endpoints for SageMaker |
| KMS | `infrastructure/kms-keys.yaml` | Encryption keys for S3, EBS, and model artifacts |
| Storage | `infrastructure/s3-buckets.yaml` | Encrypted S3 buckets with lifecycle policies  |
| IAM | `infrastructure/sagemaker-roles.yaml` | Execution roles with least-privilege, VPC-only |
| Security | `infrastructure/security.yaml` | Security groups, VPC endpoint policies |
| Processing | `pipeline/processing.yaml` | Data preprocessing in VPC-isolated jobs |
| Training | `pipeline/training.yaml` | Model training with encrypted volumes |
| Registry | `pipeline/model-registry.yaml` | Model package group with approval workflow |
| Endpoint | `pipeline/endpoint.yaml` | Real-time inference behind API Gateway + WAF |
| Retraining | `pipeline/retraining.yaml` | EventBridge + Step Functions for auto retrain |
| Monitoring | `monitoring/model-monitor.yaml` | Data quality and model quality monitors |

## Security Features

- All data encrypted at rest with customer-managed KMS keys
- SageMaker runs in VPC-isolated mode (no internet access)
- VPC endpoints for S3, SageMaker API, SageMaker Runtime, CloudWatch, KMS
- IAM roles scoped to specific S3 prefixes and KMS keys  
- Model artifacts signed and tracked with SBOM metadata
- API Gateway with WAF rules for inference endpoint
- CloudWatch alarms for model drift and data quality
- Audit trail via CloudTrail for all SageMaker API calls

## Requirements

- AWS CLI >= 2.0
- CloudFormation (no additional tooling needed)
- Python >= 3.9 (for SageMaker scripts)

## License

MIT
