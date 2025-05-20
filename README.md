# Tethys-Speech

A TensorFlow-based repository for speech recognition model implementation and distributed training.

## Overview

This project includes precise TensorFlow implementations of two major speech recognition models, Whisper and Wav2Vec2, with full support for distributed training in Kubernetes environments. These implementations faithfully reproduce the original model architectures with high fidelity to their published specifications.

The jobs in this repository are specifically designed to serve as workloads for scheduler performance evaluation in distributed training environments.

## Main Models

- **Whisper**: Speech-to-text model developed by OpenAI, implemented with precise architecture matching the original design
- **Wav2Vec2**: Self-supervised learning-based speech recognition model developed by Meta, implemented with detailed attention to the original architecture specifications

Both models are fully implemented in TensorFlow, providing an alternative to the original PyTorch implementations.

## Directory Structure

```
tethys-speech/
├── speech_jobs/         # Speech recognition model implementation files
│   ├── whisper_dist.py  # Whisper model and distributed training code
│   └── wav2vec2_dist.py # Wav2Vec2 model and distributed training code
├── stable_jobs/         # Stabilized implementation files
├── sample_tfjobs/       # Kubeflow TFJob configuration files
│   ├── whisper-dist.yaml
│   └── wav2vec2-dist.yaml
```

## Features

- Whisper and Wav2Vec2 models precisely implemented in TensorFlow
- Full distributed training support using TensorFlow's MultiWorkerMirroredStrategy
- TFJob configurations optimized for performance evaluation of Kubernetes schedulers
- Training monitoring and automatic checkpoint saving
- Compatible with Kubeflow and Training Operator 1.7.0

## Usage

### Local Training

```bash
python speech_jobs/whisper_dist.py --batch_size 4 --num_batches 30
```

### Distributed Training (Kubeflow)

```bash
kubectl apply -f sample_tfjobs/whisper-dist.yaml
```

## Performance Metrics

The following metrics are automatically recorded during model training:
- Training loss and accuracy
- GPU and network usage
- Job Completion Time (JCT)

## Docker Image

A pre-built Docker image with all dependencies is available on DockerHub:

```
potato4332/speech-image:0.0.1-beta
```

## Dependencies

- TensorFlow 2.x
- CUDA 11.x and cuDNN 8.x
- NumPy
- TensorFlow Datasets
- Kubernetes (for distributed training)
- Kubeflow Training Operator 1.7.0

## Distributed Training

This implementation leverages TensorFlow's MultiWorkerMirroredStrategy for efficient distributed training across multiple nodes. It has been tested and optimized to work seamlessly with Kubeflow's TFJob operator, specifically version 1.7.0 of the Training Operator.
