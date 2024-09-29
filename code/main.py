from OPxMOE.code.OPxMOE import *
import kagglehub
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

kagglehub.login()

set_seed(seed=42)

data_path = 'opxmoe_data'
model_path = kagglehub.model_download("ankushhv/mol2vec/pyTorch/default")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Using {device}')

df = pd.read_parquet(f"{data_path}/de_train.parquet")

logger.info(df.head())

logger.info("Generating embeddings")

onehot_embedding = OneHotEmbedding(columns=['cell_type', 'sm_name'])
smiles_embedding = SMILESEmbedding()
mol2vec_embedding = Mol2VecEmbedding(f"{model_path}/model_300dim.pkl")
morgan_embedding = MorganFingerPrintEmbedding(hidden_size=128)

preprocessor_train = Preprocessor([onehot_embedding, smiles_embedding, mol2vec_embedding,morgan_embedding])

target_cols = ['cell_type','sm_name','sm_lincs_id','SMILES','control']
targets = df.drop(columns=target_cols)
processed_df_train = preprocessor_train.preprocess(df, fit=True)

logger.info(f"Generated embeddings: {preprocessor_train.embedding_indices}")

# processed_df_train.shape
logger.info("Processed Training DataFrame shape:", processed_df_train.shape)
logger.info("Training Embedding indices:", preprocessor_train.embedding_indices)

feature_names = list(processed_df_train.columns)
logger.info(len(feature_names))
train_array = processed_df_train.to_numpy()

X = torch.tensor(train_array,dtype=torch.float32)
y = torch.tensor(targets.values, dtype=torch.float32)
logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

X = X.to(device)
y = y.to(device)

dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

logger.info("Data loaded")

expert_specs = {
    "lstm_expert_1": ["OneHotEmbedding", "MorganFingerPrintEmbedding"],
    "lstm_expert_2": ["OneHotEmbedding", "Mol2VecEmbedding"],
    "lstm_expert_3": ["OneHotEmbedding", "SMILESEmbedding"],
}

logger.info(f"expert specs: {expert_specs}")

expert_config = preprocessor_train.generate_expert_config(expert_specs)

output_size = 18211
epochs = 5
folds = 3

# model = MixtureOfExperts(expert_config, output_size)
# model.float()
# model.to(device)

# train_losses, val_losses = train_model(model, train_loader, val_loader, epochs)
# models, train_losses_cv, val_losses_cv = cross_validate_model(MixtureOfExperts, expert_config, X, y, folds, epochs)
moe = MixtureOfExperts(expert_config, output_size)
moe.to(device)

logger.info("Training model") 
train_moe(moe, train_loader, val_loader, expert_config=expert_config, device=device, epochs=epochs)