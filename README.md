# Vietnamese Hand Sign Language Teaching System

A real-time hand sign language teaching system using computer vision and deep learning to help children learn Vietnamese sign language.

## Overview

This project combines MediaPipe for hand tracking with a GRU neural network to recognize and teach Vietnamese hand sign language. The system includes an interactive interface with a 3D printed prosthetic arm for demonstration.

## Features

- **Real-time Gesture Recognition**: Recognizes hand gestures with 97.3% accuracy
- **Three Learning Modes**:
  - **Learn**: Watch and learn sign language symbols
  - **Test**: Practice with multiple-choice exercises
  - **Interpret**: Convert speech to sign language
- **3D Prosthetic Arm**: Physical demonstration tool
- **User-friendly Interface**: Built with PyQt5

## System Components

### 1. Hand Tracking
- Uses MediaPipe to extract 63 hand keypoints (x, y, z)
- Captures 30 frames (~1.5 seconds) per gesture

### 2. GRU Model
![](https://github.com/user-attachments/assets/bb6cb8a3-c868-4eee-b8e4-267ecdec92a3)


### 3. User Interface
Built with PyQt5, includes three modes for learning and practicing sign language.

## Dataset

- 12 classes (Vietnamese letters with accents and tones)
- 170 videos per class
- 30 frames per video
- Total: 2,040 videos

## Model Performance

| Model | Accuracy | Parameters |
|-------|----------|------------|
| **GRU** | **97.3%** | 440,884 |
| LSTM | 96.1% | 520,000+ |
| RNN | 93.8% | 380,000 |

GRU was chosen for its balance of accuracy and efficiency.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vsl-teaching-system.git
```

## Screenshots

### Main Interface Pipline
![Main Interface](https://github.com/user-attachments/assets/6722a477-a8c5-44dc-99a7-897a2c52defb)

## Future Improvements

- Add more gesture vocabulary
- Mobile app version
- Multi-user support
- Progress tracking

## Acknowledgments

- MediaPipe for hand tracking framework
- Vietnamese Sign Language community

## Contact

For questions or suggestions, please open an issue.

---

**Note**: This is a research project for educational purposes.
