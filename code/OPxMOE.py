# packages
import numpy as np
import pandas as pd
import random
import os
from abc import ABC, abstractmethod
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, QED
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds, CalcNumHBA, CalcNumHBD, CalcFractionCSP3
from rdkit import DataStructs
from rdkit.Chem import BRICS, Recap
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from captum.attr import IntegratedGradients, Saliency, DeepLift, ShapleyValueSampling
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed=42):
    """
    Set the seed for random number generators to ensure reproducibility.

    This function sets the seed for random number generators in Python, NumPy, and PyTorch.
    It also sets the seed for the Python hash function and the tokenizers parallelism.

    Parameters:
    - seed (int, optional): The seed value to set. Defaults to 42.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print('-----Seed Set!-----')

# Embedding classes
class BaseEmbedding(ABC):
    """
    Abstract base class for embedding generation.

    This class defines the interface for embedding generation. It includes methods for preprocessing data, generating embeddings, and accessing the embedding size.

    Attributes:
        None

    Methods:
        preprocess(df): Abstract method for preprocessing data.
        get_embedding(df): Abstract method for generating embeddings.
        embedding_size: Abstract property for accessing the embedding size.
    """
    @abstractmethod
    def preprocess(self, df):
        pass

    @abstractmethod
    def get_embedding(self, df):
        pass

    @property
    @abstractmethod
    def embedding_size(self):
        pass

class OneHotEmbedding(BaseEmbedding):
    """
    This class is used to generate one-hot encodings for a given set of columns in a dataframe.
    It inherits from the BaseEmbedding class and implements the preprocess, get_embedding, and embedding_size methods.

    Attributes:
        columns (list): A list of column names to be one-hot encoded.
        encoder (OneHotEncoder): An instance of OneHotEncoder for encoding the data.
        _embedding_size (int): The size of the one-hot encoding, set dynamically during encoding.

    Methods:
        preprocess(df): Preprocesses the dataframe by selecting the specified columns.
        get_embedding(df, fit=True): Generates the one-hot encodings for the dataframe.
        embedding_size: Returns the size of the one-hot encoding.
    """
    def __init__(self, columns):
        """
        Initializes the OneHotEmbedding object with the given columns and an instance of OneHotEncoder.

        Parameters:
        - columns (list): A list of column names to be one-hot encoded.
        """
        self.columns = columns
        self.encoder = OneHotEncoder(sparse_output=False)
        self._embedding_size = None

    def preprocess(self, df):
        """
        Preprocesses the dataframe by selecting the specified columns.

        Parameters:
        - df (pd.DataFrame): The input dataframe.

        Returns:
        - pd.DataFrame: The preprocessed dataframe with only the specified columns.
        """
        return df[self.columns]

    def get_embedding(self, df, fit=True):
        """
        Generates the one-hot encodings for the dataframe.
        If fit is True, it fits the encoder to the data. Otherwise, it only transforms the data.

        Parameters:
        - df (pd.DataFrame): The input dataframe.
        - fit (bool, optional): Whether to fit the encoder to the data. Defaults to True.

        Returns:
        - pd.DataFrame: A dataframe containing the one-hot encoded features.
        """
        if fit:
            encoded_features = self.encoder.fit_transform(df[self.columns])
        else:
            encoded_features = self.encoder.transform(df[self.columns])
        self._embedding_size = encoded_features.shape[1]
        return pd.DataFrame(encoded_features, columns=self.encoder.get_feature_names_out(self.columns))

    @property
    def embedding_size(self):
        """
        Returns the size of the one-hot encoding.

        Returns:
        - int: The size of the one-hot encoding.
        """
        return self._embedding_size
    
class SMILESEmbedding(BaseEmbedding):
    """
    This class is used to generate embeddings for SMILES strings in a dataframe.
    It inherits from the BaseEmbedding class and implements the preprocess, get_embedding, and embedding_size methods.

    Attributes:
        scaler (StandardScaler): An instance of StandardScaler for scaling the features.

    Methods:
        preprocess(df): Preprocesses the dataframe by returning it as is, assuming the dataframe is already in the correct format.
        get_embedding(df, fit=True): Generates the embeddings for the SMILES strings in the dataframe.
        embedding_size: Returns the size of the embedding, which is 15 in this case.
    """
    def __init__(self):
        """
        Initializes the SMILESEmbedding object with a StandardScaler for scaling the features.
        """
        self.scaler = StandardScaler()

    def preprocess(self, df):
        """
        Preprocesses the dataframe by returning it as is, assuming the dataframe is already in the correct format.

        Parameters:
        - df (pd.DataFrame): The input dataframe.

        Returns:
        - pd.DataFrame: The preprocessed dataframe.
        """
        return df

    def get_embedding(self, df, fit=True):
        """
        Generates the embeddings for the SMILES strings in the dataframe.
        If fit is True, it fits the scaler to the data. Otherwise, it only transforms the data.

        Parameters:
        - df (pd.DataFrame): The input dataframe.
        - fit (bool, optional): Whether to fit the scaler to the data. Defaults to True.

        Returns:
        - pd.DataFrame: A dataframe containing the scaled embeddings for the SMILES strings.
        """
        smiles_info_list = df['SMILES'].apply(self.extract_smiles_info)
        smiles_info_df = pd.DataFrame(smiles_info_list.tolist())
        if fit:
            smiles_info_df = self.scaler.fit_transform(smiles_info_df)
        else:
            smiles_info_df = self.scaler.transform(smiles_info_df)
        return pd.DataFrame(smiles_info_df, columns=[
            'Molecular Weight', 'LogP', 'TPSA', 'Number of Atoms', 'Number of Bonds',
            'Number of Rotatable Bonds', 'Number of Hydrogen Bond Acceptors', 'Number of Hydrogen Bond Donors',
            'Number of Rings', 'Number of Aromatic Rings', 'Number of Stereocenters',
            'Fraction of sp3 Carbons', 'Balaban J Index', 'Bertz CT', 'QED Score'
        ])

    @staticmethod
    def extract_smiles_info(smiles):
        """
        Extracts various molecular properties from a given SMILES string.
        Returns None if the SMILES string is invalid or empty.

        Parameters:
        - smiles (str): The SMILES string to extract properties from.

        Returns:
        - dict or None: A dictionary containing molecular properties or None if the SMILES string is invalid or empty.
        """
        if smiles is None or smiles == '':
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        info = {
            'Molecular Weight': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': CalcTPSA(mol),
            'Number of Atoms': mol.GetNumAtoms(),
            'Number of Bonds': mol.GetNumBonds(),
            'Number of Rotatable Bonds': CalcNumRotatableBonds(mol),
            'Number of Hydrogen Bond Acceptors': CalcNumHBA(mol),
            'Number of Hydrogen Bond Donors': CalcNumHBD(mol),
            'Number of Rings': Descriptors.RingCount(mol),
            'Number of Aromatic Rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'Number of Stereocenters': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            'Fraction of sp3 Carbons': CalcFractionCSP3(mol),
            'Balaban J Index': Descriptors.BalabanJ(mol),
            'Bertz CT': Descriptors.BertzCT(mol),
            'QED Score': QED.qed(mol)
        }
        return info
    
    @property
    def embedding_size(self):
        """
        Returns the size of the embedding, which is 15 in this case.

        Returns:
        - int: The size of the embedding.
        """
        return 15  
    
class Mol2VecEmbedding(BaseEmbedding):
    """
    A class for generating molecular embeddings using Mol2Vec.
    """
    def __init__(self, model_path):
        """
        Initializes the Mol2VecEmbedding class with a pre-trained model.

        Parameters:
        - model_path (str): The path to the pre-trained Mol2Vec model.
        """
        self.model = word2vec.Word2Vec.load(model_path)
        self.keys = set(self.model.wv.key_to_index.keys())

    def preprocess(self, df):
        """
        Preprocesses a DataFrame containing SMILES strings by converting them into molecular sentences.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing SMILES strings.

        Returns:
        - pd.DataFrame: The preprocessed DataFrame with molecular sentences.
        """
        df['mol'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
        df['sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
        return df

    def get_embedding(self, df, fit=True):
        """
        Generates embeddings for the preprocessed molecular sentences.

        Parameters:
        - df (pd.DataFrame): The preprocessed DataFrame containing molecular sentences.
        - fit (bool, optional): Unused parameter for compatibility with other embedding methods. Defaults to True.

        Returns:
        - pd.DataFrame: A DataFrame containing the embeddings for each molecular sentence.
        """
        df['vector'] = df['sentence'].apply(lambda sentence: self.sentence_to_vector(sentence))
        vector_dim = len(self.model.wv.get_vector(next(iter(self.keys))))
        vector_columns = [f'vector_{i}' for i in range(vector_dim)]
        return pd.DataFrame(df['vector'].tolist(), columns=vector_columns)

    def sentence_to_vector(self, sentence, unseen=False, unseen_vec=np.zeros(300)):
        """
        Converts a molecular sentence into a vector representation.

        Parameters:
        - sentence (list): A list of words representing the molecular sentence.
        - unseen (bool, optional): If True, includes unseen words in the vector calculation. Defaults to False.
        - unseen_vec (np.array, optional): The vector to use for unseen words. Defaults to a zero vector of size 300.

        Returns:
        - np.array: The vector representation of the molecular sentence.
        """
        if unseen:
            vec = sum([self.model.wv.get_vector(word) if word in self.keys else unseen_vec for word in sentence])
        else:
            vec = sum([self.model.wv.get_vector(word) for word in sentence if word in self.keys])
        return vec

    @property
    def embedding_size(self):
        """
        Returns the size of the embedding vector.

        Returns:
        - int: The size of the embedding vector.
        """
        return len(self.model.wv.get_vector(next(iter(self.keys))))
    
class Autoencoder(nn.Module):
    """
    An autoencoder model for dimensionality reduction.

    This class defines an autoencoder model with an encoder and a decoder. The encoder maps the input to a lower-dimensional latent space, and the decoder maps the latent space back to the original space.

    Attributes:
        encoder (nn.Sequential): The encoder network, a sequence of linear and ReLU layers.
        decoder (nn.Sequential): The decoder network, a sequence of linear and Sigmoid layers.
    """

    def __init__(self, input_size, hidden_size):
        """
        Initializes the Autoencoder with the specified input and hidden sizes.

        Parameters:
        - input_size (int): The size of the input data.
        - hidden_size (int): The size of the hidden layer.
        """
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - tuple: A tuple containing the encoded and decoded tensors.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded      


class MorganFingerPrintEmbedding(BaseEmbedding):
    """
    A class for generating embeddings based on Morgan fingerprints.

    This class is designed to generate embeddings for molecular structures using Morgan fingerprints. It includes methods for preprocessing data, generating embeddings, and accessing the embedding size.

    Attributes:
        hidden_size (int): The size of the hidden layer in the autoencoder.
        autoencoder (Autoencoder, optional): The trained autoencoder model. Defaults to None.
    """

    def __init__(self, hidden_size=128):
        """
        Initializes the MorganFingerPrintEmbedding with the specified hidden size.

        Parameters:
        - hidden_size (int, optional): The size of the hidden layer in the autoencoder. Defaults to 128.
        """
        self.hidden_size = hidden_size
        self.autoencoder = None  # This will store the trained autoencoder

    def preprocess(self, df):
        """
        Preprocesses the dataframe by extracting Morgan fingerprints.

        Parameters:
        - df (pd.DataFrame): The input dataframe containing SMILES strings.

        Returns:
        - pd.DataFrame: The preprocessed dataframe with Morgan fingerprints.
        """
        return df

    def get_embedding(self, df, fit=True):
        """
        Generates embeddings for the molecular structures based on Morgan fingerprints.

        Parameters:
        - df (pd.DataFrame): The input dataframe containing SMILES strings.
        - fit (bool, optional): Whether to train the autoencoder. Defaults to True.

        Returns:
        - pd.DataFrame: A dataframe containing the compressed Morgan fingerprints as embeddings.
        """
        # Extract Morgan fingerprints
        morgan_fp_list = df['SMILES'].apply(self.extract_morgan_fingerprint)
        morgan_fp_array = np.stack(morgan_fp_list)

        input_size = morgan_fp_array.shape[1]

        if fit:
            # Train autoencoder
            self.autoencoder = self.train_autoencoder(morgan_fp_array, input_size, self.hidden_size)
        
        # Get compressed fingerprints
        with torch.no_grad():
            compressed_fp = self.autoencoder.encoder(torch.tensor(morgan_fp_array, dtype=torch.float32)).numpy()

        compressed_fp_df = pd.DataFrame(compressed_fp, columns=[f'CompressedFP_{i}' for i in range(self.hidden_size)])
        return compressed_fp_df

    def extract_morgan_fingerprint(self, smiles, radius=2, nBits=2048):
        """
        Extracts the Morgan fingerprint for a given SMILES string.

        Parameters:
        - smiles (str): The SMILES string of the molecule.
        - radius (int, optional): The radius of the Morgan fingerprint. Defaults to 2.
        - nBits (int, optional): The number of bits in the fingerprint. Defaults to 2048.

        Returns:
        - np.array: The Morgan fingerprint as a numpy array.
        """
        if smiles is None or smiles == '':
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        morgan_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nBits)
        fp = morgan_gen.GetFingerprint(mol)
        
        fp_array = np.zeros((nBits,))
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        
        return fp_array

    def train_autoencoder(self, data, input_size, hidden_size, num_epochs=70, batch_size=32, learning_rate=0.001):
        """
        Trains an autoencoder on the given data.

        Parameters:
        - data (np.array): The input data for training.
        - input_size (int): The size of the input data.
        - hidden_size (int): The size of the hidden layer.
        - num_epochs (int, optional): The number of epochs for training. Defaults to 70.
        - batch_size (int, optional): The batch size for training. Defaults to 32.
        - learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.

        Returns:
        - Autoencoder: The trained autoencoder model.
        """
        autoencoder = Autoencoder(input_size=input_size, hidden_size=hidden_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

        dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            for batch in dataloader:
                inputs = batch[0]
                _, reconstructed = autoencoder(inputs)

                loss = criterion(reconstructed, inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        return autoencoder

    @property
    def embedding_size(self):
        """
        Returns the size of the embedding vector.

        Returns:
        - int: The size of the embedding vector.
        """
        return self.hidden_size

class Preprocessor:
    """
    A class for preprocessing data by applying multiple embeddings.

    This class is designed to preprocess data by applying multiple embeddings to a given dataframe. It maintains a record of the indices for each embedding in the final preprocessed dataframe.

    Attributes:
        embeddings (list): A list of embedding objects to apply to the data.
        embedding_indices (dict): A dictionary mapping embedding names to their start and end indices in the preprocessed dataframe.
    """

    def __init__(self, embeddings):
        """
        Initializes the Preprocessor with a list of embeddings.

        Parameters:
        - embeddings (list): A list of embedding objects to apply to the data.
        """
        self.embeddings = embeddings
        self.embedding_indices = {}

    def preprocess(self, df, fit=True):
        """
        Preprocesses the dataframe by applying each embedding in the list.

        This method preprocesses the input dataframe by applying each embedding in the list. It concatenates the preprocessed dataframes along the columns and updates the embedding indices.

        Parameters:
        - df (pd.DataFrame): The input dataframe to preprocess.
        - fit (bool, optional): Whether to fit the embeddings to the data. Defaults to True.

        Returns:
        - pd.DataFrame: The preprocessed dataframe with all embeddings applied.
        """
        processed_dfs = []
        current_index = 0

        for embedding in self.embeddings:
            preprocessed_df = embedding.preprocess(df)
            embedded_df = embedding.get_embedding(preprocessed_df, fit=fit)
            processed_dfs.append(embedded_df)

            embedding_name = embedding.__class__.__name__
            self.embedding_indices[embedding_name] = (current_index, current_index + embedding.embedding_size)
            current_index += embedding.embedding_size

        return pd.concat(processed_dfs, axis=1)

    def generate_expert_config(self, expert_specs):
        """
        Generates an expert configuration based on the given expert specifications.

        This method generates an expert configuration by mapping each expert to a list of indices corresponding to the embeddings specified for that expert.

        Parameters:
        - expert_specs (dict): A dictionary where keys are expert names and values are lists of embedding names to combine.

        Returns:
        - dict: A dictionary mapping each expert to a list of indices for their specified embeddings.
        """
        expert_config = {}
        
        for expert, embedding_names in expert_specs.items():
            indices = []
            for embedding_name in embedding_names:
                start, end = self.embedding_indices[embedding_name]
                indices.extend(range(start, end))
            expert_config[expert] = indices
        
        return expert_config
    
# Expert Models

class LSTMExpert(nn.Module):
    """
    LSTMExpert is a neural network module designed for processing sequential data. It combines bidirectional LSTM layers with an attention mechanism and fully connected layers to generate a final output.

    Attributes:
        input_size (int): The size of the input data.
        hidden_size (int): The size of the hidden state in the LSTM layers.
        num_layers (int): The number of LSTM layers.
        dropout (float): The dropout rate for regularization.

    Methods:
        forward(x): Processes the input data through the LSTMExpert network.
    """
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=2, dropout=0.3):
        super(LSTMExpert, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=4)
        
        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, output_size)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm3 = nn.LayerNorm(hidden_size * 2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Processes the input data through the LSTMExpert network.

        This method processes the input data by first ensuring it is 3D. Then, it applies bidirectional LSTM layers, followed by an attention mechanism. The output is then passed through fully connected layers with residual connections to generate the final output.

        Parameters:
        - x (torch.Tensor): The input data to the LSTMExpert network.

        Returns:
        - torch.Tensor: The output of the LSTMExpert network.
        """
        # Ensure input is 3D
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        lstm_out = lstm_out.permute(1, 0, 2)  # Change to (seq_len, batch, feature)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.permute(1, 0, 2)  # Change back to (batch, seq_len, feature)
        
        # Take the last output or apply global average pooling
        if attn_out.size(1) == 1:
            out = attn_out.squeeze(1)
        else:
            out = F.adaptive_avg_pool1d(attn_out.transpose(1, 2), 1).squeeze(-1)
        
        # Fully connected layers with residual connections
        out = self.layer_norm1(out)
        residual = out
        
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.layer_norm2(out)
        out = out + residual
        
        residual = out
        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.layer_norm3(out)
        out = out + residual
        
        out = self.fc3(out)
        
        return out


class NNExpert(nn.Module):
    """
    A neural network expert module for processing input data.

    This module consists of a sequence of linear layers with ReLU activations and dropout layers for regularization. The output is passed through a Tanh activation function to ensure it is within a specific range.

    Attributes:
        network (nn.Sequential): A sequence of neural network layers.

    Methods:
        forward(x): Processes the input data through the network.
    """

    def __init__(self, input_size, output_size, dropout_rate=0.5):
        """
        Initializes the NNExpert module.

        Parameters:
            input_size (int): The size of the input data.
            output_size (int): The size of the output data.
            dropout_rate (float, optional): The dropout rate for regularization. Defaults to 0.5.
        """
        super(NNExpert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Processes the input data through the network.

        If the input data is 3D, it is flattened to 2D before processing. The output of the network is then returned.

        Parameters:
            x (torch.Tensor): The input data to the NNExpert module.

        Returns:
            torch.Tensor: The output of the NNExpert module.
        """
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        output = self.network(x)
        return output

class GatingNetwork(nn.Module):
    """
    A neural network module for gating experts based on input data.

    This module processes input data through a feature extractor, applies attention, and outputs weights for gating experts.

    Attributes:
        input_size (int): The size of the input data.
        num_experts (int): The number of experts to gate.
        hidden_size (int, optional): The size of the hidden layer in the feature extractor. Defaults to 512.
        feature_extractor (nn.Sequential): A sequence of layers for extracting features from input data.
        attention (nn.MultiheadAttention): A multi-head attention layer for computing attention weights.
        output_layer (nn.Linear): A linear layer for outputting gating weights.

    Methods:
        forward(x): Processes the input data through the gating network and returns gating weights.
    """

    def __init__(self, input_size, num_experts, hidden_size=512):
        """
        Initializes the GatingNetwork module.

        Parameters:
            input_size (int): The size of the input data.
            num_experts (int): The number of experts to gate.
            hidden_size (int, optional): The size of the hidden layer in the feature extractor. Defaults to 512.
        """
        super(GatingNetwork, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        
        self.output_layer = nn.Linear(hidden_size, num_experts)
        
    def forward(self, x):
        """
        Processes the input data through the gating network and returns gating weights.

        Parameters:
            x (torch.Tensor): The input data to the GatingNetwork module.

        Returns:
            torch.Tensor: The gating weights for the experts.
        """
        features = self.feature_extractor(x)
        features = features.unsqueeze(0)  # Add sequence dimension for attention
        attn_output, _ = self.attention(features, features, features)
        attn_output = attn_output.squeeze(0)
        
        logits = self.output_layer(attn_output)
        return F.softmax(logits, dim=-1)

# class GatingNetwork(nn.Module):
#     def __init__(self, input_size, num_experts, initial_temperature=1.0):
#         super(GatingNetwork, self).__init__()

#         # First and second blocks using nn.Sequential
#         self.fc1 = nn.Sequential(
#             nn.Linear(input_size, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3)
#         )

#         self.fc2 = nn.Sequential(
#             nn.Linear(256, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3)
#         )

#         # Attention mechanism
#         self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4)

#         # Final layer for outputting expert weights (logits)
#         self.fc3 = nn.Linear(256, num_experts)

#         # Temperature parameter
#         self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

#     def forward(self, x):
#         identity = self.fc1[0](x)  # Pass through the first Linear layer

#         # Pass through the first and second blocks
#         out = self.fc1(x)
#         out = self.fc2(out)

#         # Residual connection
#         out = out + identity

#         # Attention mechanism: Add sequence dimension
#         out = out.unsqueeze(0)
#         attn_output, _ = self.attention(out, out, out)
#         attn_output = attn_output.squeeze(0)

#         # Pass through the final linear layer to get logits
#         logits = self.fc3(attn_output)

#         # Apply temperature scaling before softmax
#         return F.softmax(logits / self.temperature, dim=-1)
    
# class GatingNetwork(nn.Module):
#     def __init__(self, input_size, num_experts):
#         super(GatingNetwork, self).__init__()
#         self.lstm = nn.LSTM(input_size, 128, 1, batch_first=True)
#         self.linear = nn.Sequential(
#             nn.Linear(128, 256),  # Hidden state output from LSTM to fully connected
#             nn.ReLU(),            # ReLU activation
#             nn.Dropout(0.3),      # Dropout for regularization
#             nn.Linear(256, num_experts),  # Second hidden layer
#             nn.ReLU(),
#             nn.Dropout(0.3)
#         )
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         # Check if input is a sequence or not
#         if x.dim() == 2:
#             # If input is (batch_size, input_size), add a sequence dimension
#             x = x.unsqueeze(1)
        
#         # Now x shape should be (batch_size, sequence_length, input_size)
#         lstm_out, _ = self.lstm(x)
        
#         # Use the last output of the LSTM
#         last_output = lstm_out[:, -1, :]
#         x = self.linear(last_output)
#         x = self.softmax(x)
#         return x

class MixtureOfExperts(nn.Module):
    """
    A neural network module that combines multiple experts to generate a final output.

    This module initializes a mixture of experts based on the provided configurations. It sets up a gating network to determine the weights for each expert's output. The final output is a weighted sum of the outputs from each expert.

    Attributes:
        experts (nn.ModuleDict): A dictionary of expert models.
        feature_indices (dict): A dictionary mapping each expert to its corresponding feature indices.
        total_features (list): A sorted list of all unique feature indices across all experts.
        gating_network (GatingNetwork): The gating network that determines the weights for each expert.
        last_gating_weights (torch.Tensor): The last gating weights computed during the forward pass.

    Methods:
        forward(data, return_gating_logits=False): Processes the input data through the mixture of experts.

    Parameters:
        expert_configs (dict): A dictionary where keys are expert names and values are lists of feature indices.
        output_size (int, optional): The size of the output from each expert. Defaults to 18211.
    """
    def __init__(self, expert_configs, output_size=18211):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleDict()
        self.feature_indices = {}
        total_features = set()

        for expert_name, feature_indices in expert_configs.items():
            input_size = len(feature_indices)
            if expert_name.startswith('lstm'):
                self.experts[expert_name] = LSTMExpert(input_size, output_size)
            elif expert_name.startswith('nn'):
                self.experts[expert_name] = NNExpert(input_size, output_size)
            else:
                raise ValueError(f"Unknown expert type for {expert_name}")
            
            self.feature_indices[expert_name] = feature_indices
            total_features.update(feature_indices)

        # Set up the gating network with the total number of features across all experts
        self.gating_network = GatingNetwork(len(total_features), len(self.experts))
        self.total_features = sorted(list(total_features))  # Keep indices sorted
        self.last_gating_weights = None

    def forward(self, data, return_gating_logits=False):
        """
        Processes the input data through the mixture of experts.

        This method prepares the input for the gating network, computes the gating weights, and then combines the outputs from each expert using these weights. If requested, it also returns the gating logits.

        Parameters:
            data (torch.Tensor): The input data to the mixture of experts.
            return_gating_logits (bool, optional): Whether to return the gating logits. Defaults to False.

        Returns:
            torch.Tensor: The final output from the mixture of experts. If return_gating_logits is True, also returns the gating logits.
        """
        # Prepare input for gating network
        gating_input = data[:, self.total_features]
        gating_logits = self.gating_network(gating_input)
        gating_weights = F.softmax(gating_logits, dim=-1)
        self.last_gating_weights = gating_weights.detach()

        expert_outputs = []
        for expert_name, expert in self.experts.items():
            expert_input = data[:, self.feature_indices[expert_name]]
            expert_output = expert(expert_input)
            expert_outputs.append(expert_output)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        combined_output = torch.sum(expert_outputs * gating_weights.unsqueeze(-1), dim=1)

        if return_gating_logits:
            return combined_output, gating_logits
        else:
            return combined_output
    
# Loss Function
def RMSE_rowwise_loss(y_pred, y_true):
    """
    Computes the row-wise Root Mean Squared Error (RMSE) between predicted and true values.

    The RMSE is calculated using the following formula:

    RMSE = sqrt(1/n * Σ(y_pred_i - y_true_i)^2)

    where n is the number of samples, y_pred_i is the predicted value for the i-th sample, and y_true_i is the true value for the i-th sample.

    This function calculates the RMSE for each row in the predicted and true values, and then returns the mean of these row-wise RMSE values.

    Parameters:
        y_pred (torch.Tensor): The predicted values.
        y_true (torch.Tensor): The true values.

    Returns:
        torch.Tensor: The mean of the row-wise RMSE values.
    """
    return torch.sqrt(torch.mean((y_pred - y_true)**2, dim=1)).mean()

class EarlyStopping:
    """
    A class to implement early stopping in training processes.

    This class is designed to monitor the validation loss during training and stop the training process early if the validation loss stops improving. It is particularly useful for preventing overfitting.

    Attributes:
        patience (int, optional): The number of epochs to wait before stopping the training process if the validation loss does not improve. Defaults to 20.
        min_delta (float, optional): The minimum change in the validation loss to qualify as an improvement. Defaults to 0.
        verbose (bool, optional): Whether to print the early stopping counter. Defaults to False.
        counter (int): The current early stopping counter.
        best_score (float): The best validation score seen so far.
        early_stop (bool): A flag indicating whether early stopping should be triggered.
        val_loss_min (float): The minimum validation loss seen so far.
    """

    def __init__(self, patience=20, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss):
        """
        Checks if the validation loss has improved and updates the early stopping state accordingly.

        This method updates the early stopping state based on the current validation loss. If the validation loss has improved, it resets the counter. If the validation loss has not improved for a specified number of epochs (patience), it sets the early_stop flag to True.

        Parameters:
            val_loss (float): The current validation loss.

        Returns:
            bool: True if early stopping should be triggered, False otherwise.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            return False
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
        return False
    
def train_model(model, train_loader, val_loader, epochs=200, lr=1e-4):
    """
    Trains a model on the given training and validation datasets.

    This function trains a given model on the provided training dataset and validates its performance on the validation dataset. It uses the Adam optimizer and a learning rate scheduler to adjust the learning rate based on the validation loss. The training process is stopped early if the validation loss does not improve for a specified number of epochs.

    Parameters:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        epochs (int, optional): The number of epochs to train the model. Defaults to 200.
        lr (float, optional): The initial learning rate. Defaults to 1e-4.

    Returns:
        list: A list of training losses per epoch.
        list: A list of validation losses per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    early_stopping = EarlyStopping(patience=10)
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        train_loss = 0
        for batch, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(X)
            loss = RMSE_rowwise_loss(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                output = model(X)
                val_loss += RMSE_rowwise_loss(output, y).item()

        val_losses.append(val_loss / len(val_loader))
        
        if val_loss < early_stopping.val_loss_min:
            best_model_state = model.state_dict()
        
        if early_stopping(val_loss):
            print("Early stopping triggered")
            model.load_state_dict(best_model_state)  # Load the best model
            break

        if epoch % 10 == 0 or epoch == epochs - 1:  # Also print on the last epoch
            tqdm.write(f"Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

        scheduler.step(val_loss)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig("training_validation_losses.png")

    return train_losses, val_losses

def noisy_top_k_gating(logits, noise_epsilon=0.1, training=True):
    """
    Applies noisy top-k gating to the given logits.

    This function applies noisy top-k gating to the given logits. In training mode, it adds noise to the logits before applying softmax. In evaluation mode, it directly applies softmax to the logits.

    Parameters:
        logits (torch.Tensor): The input logits.
        noise_epsilon (float, optional): The standard deviation of the noise to be added. Defaults to 0.1.
        training (bool, optional): Whether the model is in training mode. Defaults to True.

    Returns:
        torch.Tensor: The noisy gating weights.
    """
    if training:
        noise = torch.randn_like(logits) * noise_epsilon
        noisy_logits = logits + noise
    else:
        noisy_logits = logits
    return F.softmax(noisy_logits, dim=-1)

def custom_loss(y_pred, y_true, gating_logits, diversity_weight=0.1, noise_epsilon=0.1, training=True):
    """
    Calculates the custom loss for the model.

    This function calculates the custom loss for the model, which includes the main loss (RMSE) and a diversity loss to encourage the use of different experts.

    Parameters:
        y_pred (torch.Tensor): The predicted output.
        y_true (torch.Tensor): The true output.
        gating_logits (torch.Tensor): The logits for gating.
        diversity_weight (float, optional): The weight for the diversity loss. Defaults to 0.1.
        noise_epsilon (float, optional): The standard deviation of the noise for noisy top-k gating. Defaults to 0.1.
        training (bool, optional): Whether the model is in training mode. Defaults to True.

    Returns:
        torch.Tensor: The total loss.
        torch.Tensor: The noisy gating weights.
    """
    # Main loss (RMSE)
    main_loss = RMSE_rowwise_loss(y_pred, y_true)
    
    # Apply noisy top-k gating
    noisy_gating_weights = noisy_top_k_gating(gating_logits, noise_epsilon, training)
    
    # Diversity loss (encourage using different experts)
    avg_expert_usage = noisy_gating_weights.mean(dim=0)
    diversity_loss = -torch.sum(avg_expert_usage * torch.log(avg_expert_usage + 1e-10))
    
    # Combine losses
    total_loss = main_loss - diversity_weight * diversity_loss
    
    return total_loss, noisy_gating_weights

def pretrain_expert(expert, train_loader, val_loader, feature_indices, device, epochs=100, lr=1e-4):
    """
    Pretrains a single expert model on the given training and validation datasets.

    This function pretrains a single expert model on the provided training dataset and validates its performance on the validation dataset. It uses the Adam optimizer and a learning rate scheduler to adjust the learning rate based on the validation loss. The training process is stopped early if the validation loss does not improve for a specified number of epochs.

    Parameters:
        expert (torch.nn.Module): The expert model to be pretrained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        feature_indices (list): The indices of the features to be used.
        device (torch.device): The device to use for training.
        epochs (int, optional): The number of epochs to train the model. Defaults to 100.
        lr (float, optional): The initial learning rate. Defaults to 1e-4.

    Returns:
        list: A list of training losses per epoch.
        list: A list of validation losses per epoch.
    """
    expert.to(device)
    optimizer = torch.optim.Adam(expert.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    early_stopping = EarlyStopping(patience=10)
    
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(epochs), desc=f"Pretraining {type(expert).__name__}"):
        expert.train()
        train_loss = 0
        for batch, (X, y) in enumerate(train_loader):
            X, y = X[:, feature_indices].to(device), y.to(device)
            optimizer.zero_grad()
            output = expert(X)
            loss = RMSE_rowwise_loss(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        
        expert.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X[:, feature_indices].to(device), y.to(device)
                output = expert(X)
                val_loss += RMSE_rowwise_loss(output, y).item()
        
        val_losses.append(val_loss / len(val_loader))
        
        if val_loss < early_stopping.val_loss_min:
            best_model_state = expert.state_dict()
        
        if early_stopping(val_loss):
            print("Early stopping triggered")
            expert.load_state_dict(best_model_state)  # Load the best model
            break
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            tqdm.write(f"Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
        
        scheduler.step(val_losses[-1])
    
    return train_losses, val_losses

def train_gating_network(moe, train_loader, val_loader, device, epochs=50, lr=1e-3, noise_epsilon=1e-2):
    """
    Trains the gating network of a Mixture of Experts (MoE) model.

    This function trains the gating network of a MoE model on the given training and validation datasets. It freezes the expert parameters and trains only the gating network parameters. It uses the Adam optimizer and a learning rate scheduler to adjust the learning rate based on the validation loss. The training process is stopped early if the validation loss does not improve for a specified number of epochs.

    Parameters:
        moe (torch.nn.Module): The MoE model.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device to use for training.
        epochs (int, optional): The number of epochs to train the model. Defaults to 50.
        lr (float, optional): The initial learning rate. Defaults to 1e-3.
        noise_epsilon (float, optional): The standard deviation of the noise for noisy top-k gating. Defaults to 1e-2.

    Returns:
        list: A list of training losses per epoch.
        list: A list of validation losses per epoch.
    """
    # Freeze expert parameters
    for expert in moe.experts.values():
        for param in expert.parameters():
            param.requires_grad = False
    
    # Ensure gating network parameters are unfrozen
    for param in moe.gating_network.parameters():
        param.requires_grad = True
    
    # Move model to device
    moe.to(device)
    
    optimizer = torch.optim.Adam(moe.gating_network.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=10)
    
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(epochs), desc="Training Gating Network"):
        moe.train()
        train_loss = 0
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output, gating_logits = moe(X, return_gating_logits=True)
            
            # Custom loss with noisy top-k gating
            loss, noisy_gating_weights = custom_loss(output, y, gating_logits, 
                                                     noise_epsilon=noise_epsilon, 
                                                     training=True)
            
            loss.backward()
            clip_grad_norm_(moe.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        
        moe.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output, gating_logits = moe(X, return_gating_logits=True)
                loss, _ = custom_loss(output, y, gating_logits, 
                                      noise_epsilon=noise_epsilon, 
                                      training=False)
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(val_loader))
        
        if val_loss < early_stopping.val_loss_min:
            best_model_state = moe.gating_network.state_dict()
        
        if early_stopping(val_loss):
            print("Early stopping triggered")
            moe.gating_network.load_state_dict(best_model_state)  # Load the best model
            break
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            tqdm.write(f"Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
        
        # if epoch % 10 == 0:
            # plot_expert_utilization(moe.last_gating_weights)
        
        scheduler.step(val_losses[-1])
    
    # Unfreeze expert parameters after training
    for expert in moe.experts.values():
        for param in expert.parameters():
            param.requires_grad = True
    
    return train_losses, val_losses

def plot_training_curves(train_losses, val_losses, title):
    """
    Plots the training and validation loss curves.

    This function plots the training and validation loss curves over epochs.

    Parameters:
        train_losses (list): A list of training losses per epoch.
        val_losses (list): A list of validation losses per epoch.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{title}.png")  
    
def plot_expert_utilization(gating_weights):
    """
    Plots the average utilization of each expert.

    This function plots the average utilization of each expert over epochs.

    Parameters:
        gating_weights (torch.Tensor): The gating weights.
    """
    avg_expert_usage = gating_weights.mean(dim=0).cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(avg_expert_usage)), avg_expert_usage)
    plt.title("Average Expert Utilization")
    plt.xlabel("Expert")
    plt.ylabel("Average Usage")
    plt.savefig("expert_utilization.png")

def train_moe(moe, train_loader, val_loader, expert_config, device, epochs=300):
    """
    Trains a Mixture of Experts (MoE) model.

    This function pretrains each expert in the MoE model (if needed) and then trains the gating network.
    It also visualizes the final gating weights.

    Args:
        moe (MixtureOfExperts): The MoE model to train.
        train_loader (torch.utils.data.DataLoader): The data loader for the training data.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation data.
        expert_config (dict): A dictionary specifying the features used by each expert.
        device (torch.device): The device to use for training (e.g., 'cuda' or 'cpu').
        epochs (int, optional): The number of epochs to train for. Defaults to 300.

    Returns:
        None
    """
    # Pretrain experts (if needed)
    for expert_name, expert in moe.experts.items():
        print(f"Pretraining {expert_name}")
        feature_indices = expert_config[expert_name]
        train_losses, val_losses = pretrain_expert(expert, train_loader, val_loader, feature_indices, device, epochs=epochs)
        # plot_training_curves(train_losses, val_losses, f"{expert_name} Pretraining")

    # Train gating network with improvements
    print("Training gating network")
    train_losses, val_losses = train_gating_network(moe, train_loader, val_loader, device, epochs=300, lr=1e-4)
    # plot_training_curves(train_losses, val_losses, "Gating Network Training")

    # Visualize final gating weights
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(moe.experts)), moe.last_gating_weights.cpu().mean(dim=0))
    plt.title("Average Gating Weights")
    plt.xlabel("Expert")
    plt.ylabel("Weight")
    plt.xticks(range(len(moe.experts)), list(moe.experts.keys()), rotation=45)
    plt.savefig("average_gating_weights.png")

def cross_validate_model(model_class, config, X, y, device='cpu', n_splits=5, epochs=200, lr=0.001):
    """
    Cross-validates a PyTorch model.

    This function performs k-fold cross-validation on the given model. It splits the data into k folds, trains the model on k-1 folds, and validates it on the remaining fold. It stores the training and validation losses for each fold and plots the average cross-validation losses.

    Parameters:
        model_class (torch.nn.Module): The model class to be trained.
        config (dict): Configuration parameters for the model.
        X (torch.Tensor): Input data.
        y (torch.Tensor): Target data.
        n_splits (int, optional): The number of folds for cross-validation. Defaults to 5.
        epochs (int, optional): The number of epochs to train the model. Defaults to 200.
        lr (float, optional): The initial learning rate. Defaults to 0.001.

    Returns:
        list: A list of models trained on each fold.
        list: A list of training losses for each fold.
        list: A list of validation losses for each fold.
    """

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_losses_cv = []
    val_losses_cv = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f'Fold {fold+1}/{n_splits}')

        # Create model instance for each fold
        model = model_class(config).to(device)
        model.float()

        # Create Datasets for the fold
        train_dataset = Subset(TensorDataset(X, y), train_idx)
        val_dataset = Subset(TensorDataset(X, y), val_idx)

        # Create DataLoaders for the fold
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        # Train the model for the current fold
        train_losses, val_losses = train_model(model, train_loader, val_loader, epochs, lr)
        
        # Store model and losses for later analysis
        models.append(model)
        train_losses_cv.append(train_losses)
        val_losses_cv.append(val_losses)

    # Ensure all loss lists have the same length by padding with np.nan
    max_epochs = min(len(losses) for losses in train_losses_cv)  # Use min if you want to truncate to shortest
    train_losses_cv = [losses[:max_epochs] for losses in train_losses_cv]  # Truncate to minimum length
    val_losses_cv = [losses[:max_epochs] for losses in val_losses_cv]

    # Convert to numpy arrays for averaging
    avg_train_loss = np.mean(np.array(train_losses_cv), axis=0)
    avg_val_loss = np.mean(np.array(val_losses_cv), axis=0)
    
    # Plotting average cross-validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(avg_train_loss, label='Avg Train Loss')
    plt.plot(avg_val_loss, label='Avg Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{n_splits}-Fold Cross-Validation Losses')
    plt.savefig(f"{n_splits}_fold_cross_validation_losses.png")

    return models, train_losses_cv, val_losses_cv

def compute_feature_importance(model, X, feature_names, k=10, baseline=None, target=0, method = 'integrated_gradients'):
    """
    Compute feature importance using various interpretability methods.

    Parameters:
    model (nn.Module): Trained PyTorch model.
    X (torch.Tensor): Input data.
    feature_names (list): List of feature names.
    k (int, optional): Number of top features to display. Default is 10.
    baseline (torch.Tensor, optional): Baseline input for interpretability methods. If None, a zero baseline is used.
    target (int, optional): Target output index for interpretability methods. Default is 0.
    methods (tuple, optional): Interpretability methods to use. Can include 'integrated_gradients', 'saliency', 'deeplift', and 'shapley'. Default is all methods.
    return_all (bool, optional): If True, returns the top k feature names, their importances, and the full feature importance array for each method. Default is False.

    Returns:
    If return_all is False (default):
        top_k_features (dict): Dictionary mapping method name to list of top k feature names.
    If return_all is True:
        top_k_features (dict), top_k_importances (dict), all_importances (dict)
    """
    # Ensure the model is on CPU
    # model.cpu()
    model.eval()  # Set model to evaluation mode
    
    # Move input to CPU if necessary
    # X = X.cpu()
    # Choose baseline: it should have the same shape as X (defaults to zero baseline)
    if baseline is None:
        baseline = torch.zeros_like(X)
    else:
        baseline = baseline.cpu()  # Ensure baseline is on CPU
    
    top_k_features = {}
    top_k_importances = {}
    all_importances = {}
    
    explainer = IntegratedGradients(model)
    attributions = explainer.attribute(X, baselines=baseline, target=target)
    
    if method == 'integrated_gradients':
        explainer = IntegratedGradients(model)
        attributions = explainer.attribute(X, baselines=baseline, target=target)
    elif method == 'saliency':
        explainer = Saliency(model)
        attributions = explainer.attribute(X, target=target)
    elif method == 'deeplift':
        explainer = DeepLift(model)
        attributions = explainer.attribute(X, baselines=baseline, target=target)
    elif method == 'shapley':
        explainer = ShapleyValueSampling(model)
        attributions = explainer.attribute(X, baselines=baseline, target=target, show_progress=True)
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'integrated_gradients', 'saliency', 'deeplift', or 'shapley'.")
        
        # Convert attributions to numpy for further processing
    attributions_np = attributions.cpu().detach().numpy()
    # Average feature importance across all samples
    avg_importance = attributions_np.mean(axis=0)

    # Get the indices of the top k features
    top_k_indices = np.argsort(np.abs(avg_importance))[-k:][::-1]

    # Get the names and importances of the top k features
    top_k_features[method] = [feature_names[i] for i in top_k_indices]
    top_k_importances[method] = avg_importance[top_k_indices]
    all_importances[method] = avg_importance

    # Plot the attributions for top k features
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[top_k_indices], top_k_importances[method])
    plt.xlabel("Average Importance across samples")
    plt.ylabel("Top Features")
    plt.title(f"Top {k} Feature Importance ({method})")
    plt.savefig(f"top_{k}_feature_importance_{method}.png")

def visualize_expert_weights(model, data_loader, expert_names, num_samples=100):
    """
    Visualizes the expert weights for different input instances.

    This function takes a trained Mixture-of-Experts (MoE) model, a data loader, and a list of expert names.
    It then runs the model on a subset of the data and visualizes the expert weights using a heatmap and a bar plot.

    Args:
        model (torch.nn.Module): The trained MoE model.
        data_loader (torch.utils.data.DataLoader): The data loader for the input data.
        expert_names (list): A list of names for each expert in the MoE model.
        num_samples (int, optional): The number of samples to use for visualization. Defaults to 100.

    Returns:
        None
    """
    model.eval()
    all_weights = []
    all_labels = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            if i >= num_samples:
                break
            _ = model(inputs)  # Forward pass
            weights = model.last_gating_weights
            all_weights.append(weights.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_weights = np.concatenate(all_weights, axis=0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(all_weights, cmap="YlOrRd", annot=False, fmt=".2f", cbar_kws={'label': 'Expert Weight'},
                xticklabels=expert_names)
    plt.title("Expert Weights for Different Input Instances")
    plt.xlabel("Expert")
    plt.ylabel("Input Instance")
    plt.tight_layout()
    plt.savefig("expert_weights_heatmap.png")
    
    # Bar plot for average expert weights
    avg_weights = all_weights.mean(axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(expert_names, avg_weights)
    plt.title("Average Expert Weights")
    plt.xlabel("Expert")
    plt.ylabel("Average Weight")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("average_expert_weights.png")

#     return all_weights, all_labels

def analyze_expert_clusters(model, data_loader, expert_names, num_samples=600, n_clusters=5):
    """
    Analyzes the expert weights and clusters the input data based on their dominant experts.

    This function takes a trained Mixture-of-Experts (MoE) model, a data loader, a list of expert names,
    and performs clustering on the input data based on the dominant expert for each sample.
    It then visualizes the clusters using t-SNE and a heatmap showing the expert dominance per cluster.

    Args:
        model (torch.nn.Module): The trained MoE model.
        data_loader (torch.utils.data.DataLoader): The data loader for the input data.
        expert_names (list): A list of names for each expert in the MoE model.
        num_samples (int, optional): The number of samples to use for analysis. Defaults to 600.
        n_clusters (int, optional): The number of clusters to create. Defaults to 5.

    Returns:
        None
    """
    model.eval()
    all_inputs = []
    all_weights = []
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            if len(all_inputs) * inputs.shape[0] >= num_samples:
                break
            _ = model(inputs)
            weights = model.last_gating_weights
            all_inputs.append(inputs.cpu().numpy())
            all_weights.append(weights.cpu().numpy())
    
    # Concatenate all batches of inputs and weights
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_weights = np.concatenate(all_weights, axis=0)
    
    # Ensure we have exactly num_samples
    num_samples = min(num_samples, all_inputs.shape[0], all_weights.shape[0])
    all_inputs = all_inputs[:num_samples]
    all_weights = all_weights[:num_samples]
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(all_inputs)
    centroids = kmeans.cluster_centers_
    
    # Combine input data and centroids for t-SNE
    combined_data = np.vstack([all_inputs, centroids])
    
    # Dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(combined_data)
    
    # Split reduced data back into inputs and centroids
    reduced_inputs = reduced_data[:num_samples]
    reduced_centroids = reduced_data[num_samples:]
    
    # Ensure dominant_experts is also truncated to num_samples
    dominant_experts = np.argmax(all_weights, axis=1)
    dominant_experts = dominant_experts[:num_samples]
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Scatter plot
    scatter = plt.scatter(reduced_inputs[:, 0], reduced_inputs[:, 1], 
                          c=dominant_experts, cmap='viridis', 
                          alpha=0.6, s=50)
    
    # Add cluster centroids
    plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], 
                marker='x', s=200, linewidths=3, color='r', label='Cluster Centroids')
    
    plt.colorbar(scatter, label='Dominant Expert')
    plt.title('t-SNE Visualization of Input Data\nColored by Dominant Expert with Cluster Centroids')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig("tsne_visualization.png")   
    
    # Heatmap of expert dominance per cluster
    expert_cluster_counts = np.zeros((n_clusters, len(expert_names)))
    for cluster in range(n_clusters):
        cluster_mask = cluster_labels == cluster
        expert_cluster_counts[cluster] = np.sum(all_weights[cluster_mask], axis=0)
    
    expert_cluster_percentages = expert_cluster_counts / expert_cluster_counts.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(expert_cluster_percentages, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=expert_names)
    plt.title('Expert Dominance per Cluster')
    plt.xlabel('Expert')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.savefig("expert_dominance_per_cluster.png")
#     return reduced_inputs, dominant_experts, cluster_labels