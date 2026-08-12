# Multimodal Fake News Detection Using Text-Image-Comment Fusion

A deep learning-based multimodal system for detecting fake news by combining information from text, images, and user comments.

## Project Overview

Fake news can contain misleading information across multiple modalities, including written content, images, and user discussions. This project proposes a multimodal deep learning approach that combines these three sources of information to classify news as real or fake.

The system uses transformer-based text representations, convolutional neural network-based image features, and comment representations, followed by multimodal feature fusion and classification.

## Architecture

The system consists of three main components:

- **Text Encoder:** RoBERTa-base for extracting contextual features from news text.
- **Image Encoder:** ResNet-50 for extracting visual features from news images.
- **Comment Encoder:** RoBERTa-base for representing user comments.

The extracted features are concatenated and passed through fully connected layers for final binary classification.

```text
News Text ───────► RoBERTa ───────┐
                                  │
Images ──────────► ResNet-50 ─────┼──► Feature Fusion ─► Classifier ─► Real/Fake
                                  │
Comments ────────► RoBERTa ───────┘
