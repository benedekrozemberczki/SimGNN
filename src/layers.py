import torch

class AttentionModule(torch.nn.Module):

    def __init__(self, args):
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_layers()
        self.init_parameters()

    def setup_layers(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3, self.args.filters_3)) 
        
    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)


    def forward(self, embedding):
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding,transformed_global.view(-1,1)))
        representation = torch.mm(torch.t(embedding),sigmoid_scores)
        return representation


class TenorNetworkModule(torch.nn.Module):
    
    def __init__(self,args):
        super(TenorNetworkModule, self).__init__()
        self.args = args
        self.setup_layers()
        self.init_parameters()


    def setup_layers(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3, self.args.filters_3, self.args.tensor_neurons))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 2*self.args.filters_3))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

        


    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)


    def forward(self, embedding_1, embedding_2):
        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(self.args.filters_3,-1)).view(self.args.filters_3, self.args.tensor_neurons)
        scoring = torch.mm(torch.t(scoring),embedding_2)
        combined_representation = torch.cat((embedding_1,embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring  + self.bias)
        return scores


