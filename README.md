# ISL to Text Translation Using Deep Learning and NLP
This project focuses on translating Indian Sign Language (ISL) gestures into meaningful text using vision-based gesture recognition and Natural Language Processing (NLP).

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [Contributing](#contributing)

## Overview
This project utilizes a dynamic vision-based system to convert ISL gestures into text in real-time. It leverages MediaPipe Holistic for gesture recognition and LLaMA 3 for generating contextually accurate sentences from recognized gestures.

## Features
- Real-time ISL gesture detection and translation.
- Dynamic gesture recognition.
- Converts gestures into text with coherent sentence formation.
- Custom-made dataset for ISL with 30 words.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/ISL-to-Text.git
    cd ISL-to-Text
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the gesture recognition system:
    ```bash
    python main.py
    ```

2. Interact with the interface to detect ISL gestures and convert them into text.

## Model
- **Gesture Recognition**: MediaPipe Holistic is used to identify hand keypoints and track gestures.
- **Text Generation**: LLaMA 3 is employed to form sentences from recognized words.

## Dataset
A custom ISL dataset with 30 common words is used for training and testing the model.

## Technologies
- Python
- MediaPipe Holistic
- LLaMA 3
- NumPy

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to suggest improvements or report bugs.


