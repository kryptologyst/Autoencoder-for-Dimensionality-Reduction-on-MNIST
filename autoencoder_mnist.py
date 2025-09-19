# Autoencoders for dimensionality reduction
# Description:
# This script trains a simple autoencoder on the MNIST dataset to reduce the dimensionality of the images.
# The autoencoder learns to compress the input data into a lower-dimensional representation and then reconstruct it.

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import os

def load_data():
    """Loads and preprocesses the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.
    x_test = x_test.astype("float32") / 255.
    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))
    return (x_train, y_train), (x_test, y_test)

def build_autoencoder(input_dim, encoding_dim):
    """Builds the autoencoder and encoder models."""
    # Input layer
    input_img = Input(shape=(input_dim,))

    # Encoder layers
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded_output = Dense(encoding_dim, activation='relu')(encoded)

    # Decoder layers
    decoded = Dense(64, activation='relu')(encoded_output)
    decoded = Dense(128, activation='relu')(decoded)
    decoded_output = Dense(input_dim, activation='sigmoid')(decoded)

    # Autoencoder model
    autoencoder = Model(input_img, decoded_output)

    # Encoder model
    encoder = Model(input_img, encoded_output)

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def train_model(autoencoder, x_train, x_test, epochs, batch_size):
    """Trains the autoencoder model."""
    autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                    validation_data=(x_test, x_test), verbose=1)

def visualize_embeddings(encoder, x_test, y_test):
    """Visualizes the 2D projection of the encoded data."""
    encoded_imgs = encoder.predict(x_test)
    plt.figure(figsize=(10, 8))
    plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, cmap='viridis', alpha=0.7, s=10)
    plt.colorbar()
    plt.title("2D Projection of MNIST using Autoencoder")
    plt.xlabel("Encoded Feature 1")
    plt.ylabel("Encoded Feature 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("encoded_data_projection.png")
    plt.show()

def visualize_reconstructions(autoencoder, x_test):
    """Visualizes the original and reconstructed images."""
    decoded_imgs = autoencoder.predict(x_test)

    n = 10  # Number of digits to display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle("Original vs Reconstructed Images")
    plt.savefig("reconstructed_images.png")
    plt.show()

def main():
    """Main function to run the autoencoder."""
    parser = argparse.ArgumentParser(description='Autoencoder for MNIST dimensionality reduction.')
    parser.add_argument('--encoding_dim', type=int, default=32, help='Dimension of the encoded representation.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.')
    args = parser.parse_args()

    (x_train, _), (x_test, y_test) = load_data()
    input_dim = x_train.shape[1]

    autoencoder, encoder = build_autoencoder(input_dim, args.encoding_dim)

    print("\nTraining autoencoder...")
    train_model(autoencoder, x_train, x_test, args.epochs, args.batch_size)

    # Create a directory to save models if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the models
    autoencoder.save('models/autoencoder.keras')
    encoder.save('models/encoder.keras')
    print("\nModels saved to 'models/' directory.")

    print("\nVisualizing encoded data projection...")
    visualize_embeddings(encoder, x_test, y_test)

    print("\nVisualizing original vs reconstructed images...")
    visualize_reconstructions(autoencoder, x_test)

if __name__ == '__main__':
    main()
