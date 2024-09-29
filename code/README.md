# Source Code
# Classes
1. **BaseEmbedding**
An abstract base class for embedding generation. It defines the interface for preprocessing data, generating embeddings, and accessing the embedding size. Subclasses must implement the preprocess, get_embedding, and embedding_size methods.
2. **OneHotEmbedding**
Generates one-hot encodings for specified columns in a DataFrame. It implements methods to preprocess the DataFrame, generate embeddings, and return the size of the one-hot encoding.
3. **SMILESEmbedding**
Generates embeddings for SMILES strings in a DataFrame. It preprocesses the DataFrame and extracts molecular properties using RDKit, returning scaled embeddings.
4. **Mol2VecEmbedding**
Generates molecular embeddings using the Mol2Vec approach. It converts SMILES strings into molecular sentences and retrieves embeddings from a pre-trained Mol2Vec model.
5. **MorganFingerPrintEmbedding**
Generates embeddings based on Morgan fingerprints. It preprocesses the DataFrame to extract fingerprints and uses an autoencoder for dimensionality reduction.
6. **Autoencoder**
Defines an autoencoder model for dimensionality reduction. It consists of an encoder and a decoder, mapping input data to a lower-dimensional latent space and reconstructing it back.
7. **Preprocessor**
Applies multiple embeddings to a DataFrame. It manages the indices of each embedding in the final preprocessed DataFrame and generates configurations for expert models.
8. **LSTMExpert**
A neural network module designed for processing sequential data using bidirectional LSTM layers combined with an attention mechanism and fully connected layers.
9. **NNExpert**
A neural network expert module that processes input data through a sequence of linear layers with ReLU activations and dropout layers for regularization.
10. **GatingNetwork**
A neural network module that determines the weights for each expert based on input data. It uses a feature extractor and an attention mechanism to compute gating weights.
11. **MixtureOfExperts**
Combines multiple expert models to generate a final output. It uses a gating network to weigh the outputs of each expert based on the input data.
12. **EarlyStopping**
Implements early stopping in training processes to prevent overfitting by monitoring validation loss and stopping training when it stops improving.
13. **Custom Loss Functions**
Includes functions for calculating various loss metrics, such as RMSE and custom losses that incorporate diversity among expert outputs.
14. **Feature Importance and Visualization Functions**
Functions for computing feature importance using interpretability methods and visualizing expert weights and clusters in the Mixture of Experts model.
These descriptions can be used to document the functionality and purpose of each class in your codebase.
