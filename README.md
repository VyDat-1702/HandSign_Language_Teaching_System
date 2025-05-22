🔍 Overview of the hand sign language teaching system model
We propose a real-time Vietnamese hand sign language teaching system, using computer vision combined with a deep recurrent neural network. The model has three main parts:

🧠 1. Gesture Recognition Module
MediaPipe is used to extract 63 keypoints (x, y, z coordinates) from the user's hand video.

The keypoints in a 30-frame sequence are fed into a Gated Recurrent Unit (GRU) model - a variant of RNN - to learn and classify dynamic hand gestures.

The GRU model has:

Two GRU layers (256 and 128 units)

Dropout layer to avoid overfitting

Flatten and Dense layers with softmax function to output classification probability

A total of 440,884 parameters

The accuracy reached 97.3% with the test set.

🧑‍🏫 3. Interactive learning interface (User Interface)
The interface is designed with PyQt5, supporting 3 modes:

Learn - learn letter and number symbols by observing the prosthetic arm.

Test - test skills through multiple choice questions to recognize gestures.

Interpret - enter speech → the system translates into sign language.

Intuitive interface, friendly to children.

📊 Results & evaluation
Using 12 symbol classes (Vietnamese letters with accents, tone symbols).

Each class has 170 videos, each video has 30 frames (≈ 1.5 seconds).

Comparison between GRU, LSTM and RNN shows that GRU is superior, compared to the number of parameters and efficiency:

Accuracy: 97.3%

Precision, Recall, F1 score and ROC AUC are quite high

✅ Summary
Combined model:

MediaPipe → hand skeleton extraction

GRU → dynamic hand sign recognition

3D printed prosthetic arm → sign representation

PyQt5 interface → support interactive learning and testing

=> This is a complete teaching system, suitable for children with hearing or speech impairments, contributing to solving the shortage of sign language teachers.
![hinh10](https://github.com/user-attachments/assets/b9552b8a-a909-4e80-8cce-cdb92b68594a)
