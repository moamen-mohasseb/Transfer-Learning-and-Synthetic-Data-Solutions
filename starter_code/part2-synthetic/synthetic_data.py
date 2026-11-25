from TestModel import test_model
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def load_and_standardize_data(path):
    df = pd.read_csv(path, sep=',')
    df = df.fillna(-99)
    df = df.values.reshape(-1, df.shape[1]).astype('float32')
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)   
    return X_train, X_test, scaler

class DataBuilder(Dataset):
    def __init__(self, path, train=True):
        self.X_train, self.X_test, self.standardizer = load_and_standardize_data(path)
        if train:
            self.x = torch.from_numpy(self.X_train)
            self.len=self.x.shape[0]
        else:
            self.x = torch.from_numpy(self.X_test)
            self.len=self.x.shape[0]
        del self.X_train
        del self.X_test 
    def __getitem__(self,index):      
        return self.x[index]
    def __len__(self):
        return self.len

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
    
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD

class Autoencoder(nn.Module):
    def __init__(self,D_in,H=50,H2=12,latent_dim=3):
        #Encoder
        super(Autoencoder,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2=nn.Linear(H,H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3=nn.Linear(H2,H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)
        
        # Latent vectors
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)
        
        # Decoder
        self.linear4=nn.Linear(H2,H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5=nn.Linear(H2,H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6=nn.Linear(H,D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        
        return r1, r2
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def generate_fake(mu, logvar, no_samples, scaler, model):
    #With trained model, generate some data
    sigma = torch.exp(logvar/2)
    q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
    z = q.rsample(sample_shape=torch.Size([no_samples]))
    with torch.no_grad():
        pred = model.decode(z).cpu().numpy()
    fake_data = scaler.inverse_transform(pred)
    return fake_data

def train(epoch, model, optimizer, loss_mse, trainloader, device, log=False):
    """One training pass over the data. Returns the mean loss per sample."""
    model.train()
    train_loss = 0
    for data in trainloader:
        data = data.to(device)
        optimizer.zero_grad()                       # reset gradients of the previous batch
        recon_batch, mu, logvar = model(data)       # encode -> sample z -> decode
        loss = loss_mse(recon_batch, data, mu, logvar)   # reconstruction (MSE) + KL divergence
        loss.backward()                             # back-propagation
        train_loss += loss.item()
        optimizer.step()                            # update weights
    mean_loss = train_loss / len(trainloader.dataset)
    if log:
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, mean_loss))
    return mean_loss


def validate(epoch, model, loss_mse, validloader, device, log=False):
    """One validation pass (no gradients, eval mode). Returns the mean loss per sample."""
    model.eval()                                    # BatchNorm uses running stats, no sampling noise
    val_loss = 0
    with torch.no_grad():
        for data in validloader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_mse(recon_batch, data, mu, logvar)
            val_loss += loss.item()
    mean_loss = val_loss / len(validloader.dataset)
    if log:
        print('Epoch: {} Validation loss: {:.4f}'.format(epoch, mean_loss))
    return mean_loss


def main():
    # Get a device and set up data paths. You need paths for the original data, the data with just loan status = 1 and the new augmented dataset.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ORIGINAL_PATH = 'data/loan_continuous.csv'
    TARGET1_PATH = 'data/loan_continuous_target1.csv'
    DATA_PATH = 'data/loan_continuous_expanded.csv'

    # Baseline: precision / recall / F1 on the original, imbalanced dataset
    print('=== Baseline (original imbalanced data) ===')
    test_model(ORIGINAL_PATH)

    # Split the data out with loan status = 1
    df = pd.read_csv(ORIGINAL_PATH, sep=',')
    df_target1 = df[df['Loan Status'] == 1]
    df_target1.to_csv(TARGET1_PATH, index=False)
    print('Records with Loan Status = 1: {}'.format(len(df_target1)))

    # Create DataLoaders for training and validation
    traindata_set = DataBuilder(TARGET1_PATH, train=True)
    validdata_set = DataBuilder(TARGET1_PATH, train=False)
    batch_size = 1024
    # drop_last avoids a final batch of size 1, which BatchNorm cannot handle in train mode
    trainloader = DataLoader(traindata_set, batch_size=batch_size, shuffle=True, drop_last=True)
    validloader = DataLoader(validdata_set, batch_size=batch_size, shuffle=False)

    # Train and validate the model (both phases run in EVERY epoch)
    D_in = traindata_set.x.shape[1]
    model = Autoencoder(D_in).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_mse = CustomLoss()
    epochs = 1500
    log_interval = 50
    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        train_losses.append(train(epoch, model, optimizer, loss_mse, trainloader, device, log=(epoch % log_interval == 0)))
        val_losses.append(validate(epoch, model, loss_mse, validloader, device, log=(epoch % log_interval == 0)))

    # Latent distribution (mu, logvar) of the training records, in eval mode
    model.eval()
    with torch.no_grad():
        _, mu, logvar = model(traindata_set.x.to(device))
    mu, logvar = mu.cpu(), logvar.cpu()

    scaler = trainloader.dataset.standardizer
    fake_data = generate_fake(mu, logvar, 50000, scaler, model.cpu())

    # Combine the new data with original dataset
    df_fake = pd.DataFrame(fake_data, columns=df.columns)
    df_fake['Loan Status'] = 1      # every generated record belongs to the minority class
    df_expanded = pd.concat([df, df_fake], ignore_index=True)
    df_expanded.to_csv(DATA_PATH, index=False)
    print('Augmented dataset: {} records\n{}'.format(len(df_expanded), df_expanded['Loan Status'].value_counts()))

    # Test on the FULL augmented dataset
    print('=== Augmented data (original + 50,000 synthetic Loan Status = 1) ===')
    test_model(DATA_PATH)

if __name__ == '__main__':
    main()
    print("done")
