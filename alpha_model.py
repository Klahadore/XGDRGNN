import torch
import pickle
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import norm
from visualize import visualize_emb
# from data import new_train_dataset, train_dataset
from HGATConv import SimpleHGATConv
from torch_geometric.nn import Linear

metadata = None
with open("data/train_dataset_metadata.pickle", "rb") as file:
    metadata = pickle.load(file)
    print("loaded metadata")

new_train_dataset = None
with open("data/new_train_dataset.pickle", "rb") as file:
    new_train_dataset = pickle.load(file)
    print("loaded new_train_dataset")

torch.manual_seed(42)

class EdgeEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.enc1 = SimpleHGATConv(hidden_channels, hidden_channels, 4, metadata, 64, concat=True, residual=True)
        self.enc1_bn = norm.BatchNorm(hidden_channels * 4)
        self.enc2 = SimpleHGATConv(hidden_channels * 4, hidden_channels * 4, 4, metadata, 64, concat=False, residual=True)

    def forward(self, x, edge_index, node_type, edge_attr, edge_type):
        x = self.enc1(x, edge_index, node_type, edge_attr, edge_type)
        x = self.enc1_bn(x)
        x = self.enc2(x, edge_index, node_type, edge_attr, edge_type)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin1_bn = norm.BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z, edge_label_index):
        row, col = edge_label_index

        z = torch.cat([z[row], z[col]], dim=-1)
        z = self.lin1(z)
        z = self.lin1_bn(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, training=False):
        super().__init__()
        self.encoder = EdgeEncoder(hidden_channels)
        self.decoder = EdgeDecoder(hidden_channels*4)

    def forward(self, batch_data):
        x = batch_data.x
        edge_index = batch_data.edge_index
        node_type = batch_data.node_type
        edge_attr = batch_data.edge_attr
        edge_type = batch_data.edge_type
        edge_label_index = batch_data.edge_label_index

        z = self.encoder(x, edge_index, node_type, edge_attr, edge_type)

        return self.decoder(z, edge_label_index), z


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# currently training only works on cuda

if __name__ == "__main__":

    model = Model(384, training=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    data_loader = LinkNeighborLoader(
        new_train_dataset,
        num_neighbors=[12, 8],
        batch_size=256,
        shuffle=True,
        edge_label=new_train_dataset.edge_label,
        edge_label_index=new_train_dataset.edge_label_index,
        num_workers=4
        )

    scaler = torch.cuda.amp.GradScaler()

    def train(loader):
        model.train()
        total_loss = 0
        total_examples = 0
        for batch_data in loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            pred, _ = model(batch_data)
       # Assuming binary classification for edge prediction

            target = batch_data.edge_label[:len(pred)]
            loss = torch.nn.BCEWithLogitsLoss()(pred, target.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            print(loss.item())

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        return total_loss / total_examples


    with torch.cuda.amp.autocast():
        for epoch in range(8):
            epoch_loss = train(data_loader)
            print(f"Epoch {epoch}: Loss {epoch_loss}")
            if epoch == 1:
                torch.save(model.state_dict(), "cheese_epoch2_2.pt")
                print("epoch 2 saved")
            if epoch == 3:
                torch.save(model.state_dict(), "cheese_epoch4_2.pt")
                print("epoch 4 saved")
            if epoch == 5:
                torch.save(model.state_dict(), "cheese_epoch6_2.pt")
                print("epoch 6 saved")
            if epoch == 7:
                torch.save(model.state_dict(), "cheese_epoch8_2.pt")
                print("epoch 8 saved")

    print("done")


