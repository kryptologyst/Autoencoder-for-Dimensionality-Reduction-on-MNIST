# Autoencoder for Dimensionality Reduction on MNIST

This project demonstrates how to use a simple autoencoder to perform dimensionality reduction on the MNIST dataset. The autoencoder is built with TensorFlow/Keras and learns to compress the 784-dimensional images into a lower-dimensional representation.

## Project Overview

Autoencoders are a type of neural network used for unsupervised learning. They consist of two main parts:

1.  **Encoder**: Compresses the input data into a lower-dimensional latent space.
2.  **Decoder**: Reconstructs the original data from the compressed representation.

By training the autoencoder to minimize the reconstruction error, the encoder learns a meaningful, compressed representation of the data. This makes autoencoders a powerful tool for dimensionality reduction, especially for complex, nonlinear data.

## Features

*   **Modular Code**: The script is organized into functions for data loading, model building, training, and visualization.
*   **Hyperparameter Tuning**: Easily experiment with hyperparameters like `encoding_dim`, `epochs`, and `batch_size` using command-line arguments.
*   **Visualization**: The script generates two plots:
    *   A 2D scatter plot of the encoded data, which helps visualize the learned latent space.
    *   A comparison of original and reconstructed images to evaluate the model's performance.
*   **Model Saving**: The trained autoencoder and encoder models are saved to the `models/` directory for future use.

## Getting Started

### Prerequisites

*   Python 3.x
*   TensorFlow
*   NumPy
*   Matplotlib

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/autoencoder-mnist.git
    cd autoencoder-mnist
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Usage

You can run the script with default parameters:

```bash
python autoencoder_mnist.py
```

This will train the autoencoder for 20 epochs with an encoding dimension of 32. The script will then display the visualization plots and save the trained models to the `models/` directory.

To experiment with different hyperparameters, you can use the following command-line arguments:

```bash
python autoencoder_mnist.py --encoding_dim 16 --epochs 30 --batch_size 128
```

*   `--encoding_dim`: The dimension of the encoded representation (default: 32).
*   `--epochs`: The number of epochs to train for (default: 20).
*   `--batch_size`: The batch size for training (default: 256).

## Output

The script will generate the following files:

*   `models/autoencoder.h5`: The trained autoencoder model.
*   `models/encoder.h5`: The trained encoder model.
*   `encoded_data_projection.png`: A 2D scatter plot of the encoded test data.
*   `reconstructed_images.png`: A comparison of original and reconstructed images.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
# Autoencoder-for-Dimensionality-Reduction-on-MNIST
