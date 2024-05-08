##
## Adjust to relative path
##
if __name__ == "__main__":
    import sys

    sys.path.append("./")


##
## Imports
##
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


##
## Bidirectional Graph Neural Network
##
class BiGNN(torch.nn.Module):
    ##
    ## Constructor
    ##
    def __init__(
        self, num_features: int = 2, hidden_dim: int = 32, num_nodes: int = 4
    ) -> None:
        super(BiGNN, self).__init__()

        ##
        ## Forward direction
        ##
        ## This is specified by the flow parameter in GCNConv (source_to_target)
        ##
        self.conv1_fwd = GCNConv(num_features, hidden_dim, flow="source_to_target")
        self.conv2_fwd = GCNConv(hidden_dim, hidden_dim, flow="source_to_target")

        ##
        ## Backward direction
        ##
        ## This is specified by the flow parameter in GCNConv (target_to_source)
        ##
        self.conv1_bwd = GCNConv(num_features, hidden_dim, flow="target_to_source")
        self.conv2_bwd = GCNConv(hidden_dim, hidden_dim, flow="target_to_source")

        ##
        ## We'll have one linear (fully connected) layer at the end.
        ## This will output our shortest tour prediction.
        ##
        ## Consider outputs from both directions (2 * hidden_dim)
        ##
        self.fc = torch.nn.Linear(2 * hidden_dim, num_nodes)

        ##
        ## End of function
        ##

    ##
    ## Forward pass
    ##
    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        ## Forward pass
        x_fwd = F.relu(self.conv1_fwd(x, edge_index))
        x_fwd = F.relu(self.conv2_fwd(x_fwd, edge_index))

        ## Backward pass
        x_bwd = F.relu(self.conv1_bwd(x, edge_index))
        x_bwd = F.relu(self.conv2_bwd(x_bwd, edge_index))

        ## Concatenate the forward and backward representations
        x = torch.cat([x_fwd, x_bwd], dim=1)

        ## Apply the fully connected layer
        out = self.fc(x)

        ##
        ## Return the log softmax of the output
        ##
        return F.log_softmax(out, dim=1)

        ##
        ## End of function
        ##

    ##
    ## End of class
    ##


##
## Training the model
##
if __name__ == "__main__":
    ##
    ## These are the parameters/initializations for the model and training
    ##
    ## These are pretty much the same as for other models.
    ##
    ## 2 features for (x, y) coordinates
    ##
    model = BiGNN(num_features=2, hidden_dim=32, num_nodes=4)

    ## Adam optimizer with learning rate 0.01 (0.001 is also a good choice)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    ##
    ## We'll use the negative log likelihood loss function
    ##
    ## Since we're using log_softmax in the forward pass.
    ##
    loss_fn = torch.nn.NLLLoss()

##
## End of file
##
