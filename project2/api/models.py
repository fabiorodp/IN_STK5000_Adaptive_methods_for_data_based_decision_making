from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import opendp
import torch


class OurDataset(Dataset):
    def __init__(
            self,
            dataframe: pd.DataFrame,
            drop_features=[],
            response_label='Death',
            device='cpu'
    ):
        self.device = device
        self.response_label = response_label
        self.response_variable = dataframe[self.response_label]
        self.explanatory_variables = dataframe.drop(
            [self.response_label] + drop_features,
            axis=1
        )

    def __getitem__(self, index):
        current_response_variable = [self.response_variable.iloc[index]]

        current_explanatory_variables = \
            list(self.explanatory_variables.iloc[index])

        X_tensor = torch.FloatTensor(current_explanatory_variables).to(
            torch.device(self.device))

        Y_tensor = torch.FloatTensor(current_response_variable).to(
            torch.device(self.device))

        return X_tensor, Y_tensor

    def __len__(self):
        return len(self.explanatory_variables)


class DNN_CV(torch.nn.Module):
    """
    Simple DNN model, but it is only working as Logistic Regression
    by now.
    """

    @staticmethod
    def reset_weights(model):
        """ Resetting model's weights."""
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                # print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()

    @staticmethod
    def init_model(
            in_features,
            out_features,
            num_hl=0,
            num_units=10,
            act_funct=torch.nn.ReLU(),
            out_act_funct=torch.nn.Sigmoid()
    ):

        if num_hl == 0:
            return torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features),
                out_act_funct
            )
        else:
            ValueError("ERROR: This class is only working with num_hl=0. "
                       "The rest will be implemented soon.")

    def __init__(
            self,
            in_features,
            out_features,
            num_hl=0,
            num_units=10,
            out_act_funct=torch.nn.Sigmoid(),
            loss_funct=torch.nn.BCELoss(),              # binary cross entropy
            # loss_funct=torch.nn.CrossEntropyLoss()    # cross entropy
            # loss_funct=torch.nn.MSELoss()             # mean squared error
            optimizer='SGD',
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0,
            seed=1,
    ):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.in_features = in_features
        self.out_features = out_features
        self.num_hl = num_hl
        self.num_units = num_units
        self.loss_funct = loss_funct
        self.out_act_funct = out_act_funct
        self.opt = optimizer
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.models = []
        self.train_losses = []
        self.train_scores = []
        self.valid_scores = []

    def predict(self, X):
        """Forward pass"""
        return self.model(X)

    def fit(self, dataset, num_epochs=10, CV=10, batch_size=32, verbose=True):

        # Initialize Kfold
        k = KFold(
            n_splits=CV,
            shuffle=True
        )

        for fold_idx, (train_ids, test_ids) in enumerate(k.split(dataset)):

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trainloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=train_subsampler
            )

            testloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=test_subsampler
            )

            # Initialize model
            model = self.init_model(
                in_features=self.in_features,
                out_features=self.out_features,
                num_hl=self.num_hl,
                num_units=self.num_units,
                out_act_funct=self.out_act_funct
            )

            model.apply(self.reset_weights)

            # Initialize optimizer
            optimizer = None
            if self.opt == 'SGD':
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=self.lr,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )

            # Set training mode
            model.train()

            fold_train_losses = []
            fold_train_scores = []
            for e in tqdm(range(num_epochs)):
                losses, acc, count = 0, 0, 0
                for i, batch in enumerate(trainloader):
                    optimizer.zero_grad()
                    Y_hat = model(batch[0])
                    loss = self.loss_funct(Y_hat, batch[1])
                    losses += loss.item()  # recording losses
                    loss.backward()
                    optimizer.step()

                    value, index = torch.max(Y_hat, 1)
                    index = torch.unsqueeze(index, dim=-1)
                    correct = (index == batch[1]).sum().item()
                    acc += correct/batch[1].size(0)
                    count += 1

                fold_train_losses.append(losses/count)
                fold_train_scores.append(acc/count)

                if verbose is True:
                    print(f"Kfold: {fold_idx}, "
                          f"Epoch: {e}/{num_epochs}, "
                          f"Loss mean: {losses/count}, "
                          f"Accuracy mean: {acc/count}")

            # Set evaluating mode
            model.eval()
            val_acc, val_count = 0, 0
            with torch.no_grad():
                for i, batch in enumerate(testloader):
                    y_tilde = model(batch[0])

                    value, index = torch.max(y_tilde, 1)
                    index = torch.unsqueeze(index, dim=-1)
                    correct = (index == batch[1]).sum().item()
                    val_acc += correct / batch[1].size(0)
                    val_count += 1

            if verbose is True:
                print(f"Kfold: {fold_idx}, "
                      f"Validation accuracy mean: {val_acc / val_count}")

            self.train_losses.append(fold_train_losses)
            self.train_scores.append(fold_train_scores)
            self.valid_scores.append(val_acc / val_count)
            self.models.append(model)


if __name__ == '__main__':

    labels = ['Covid-Recovered', 'Covid-Positive', 'No_Taste/Smell',
              'Fever', 'Headache', 'Pneumonia', 'Stomach', 'Myocarditis',
              'Blood-Clots', 'Death', 'Age', 'Gender', 'Income'] + \
             [f'g{i}' for i in range(1, 129)] + \
             ['Asthma', 'Obesity', 'Smoking', 'Diabetes', 'Heart-disease',
              'Hypertension', 'Vaccine1', 'Vaccine2', 'Vaccine3']

    observation_features = pd.read_csv(
        "https://raw.githubusercontent.com/fabiorodp/IN_STK5000"
        "_Adaptive_methods_for_data_based_decision_making/main/"
        "project1/data/observation_features.csv.gz",
        header=None,
    )

    observation_features.columns = labels

    dataset = OurDataset(
        dataframe=observation_features,
        drop_features=[],
        response_label='Death',
        device='cpu'
    )

    model_cv = DNN_CV(
        in_features=dataset.explanatory_variables.shape[1],
        out_features=1,
        num_hl=0,
        num_units=10,
        out_act_funct=torch.nn.Sigmoid(),
        loss_funct=torch.nn.BCELoss(),              # binary cross entropy
        # loss_funct=torch.nn.BCEWithLogitsLoss(),      # binary cross entropy
        # loss_funct=torch.nn.CrossEntropyLoss(),     # cross entropy
        # loss_funct=torch.nn.MSELoss(),              # mean squared error
        optimizer='SGD',
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0,
        seed=1
    )

    model_cv.fit(
        dataset=dataset,
        num_epochs=5,
        CV=5,
        batch_size=1000,
        verbose=True
    )
