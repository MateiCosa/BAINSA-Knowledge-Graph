import numpy as np
from PathFinding import dijkstra
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def compute_encodings(g, N_NODES, CATEGORIES):
    '''
    Parameters
    ----------
    g : WikiGraph
    N_NODES : int
    CATEGORIES : list
    Returns
    -------
    dist_enc, line_num 
        Compute distances from all nodes to CATEGORIES; 
        Store indices corresponding to each article_id
    '''
    dist_enc = np.zeros((N_NODES, len(CATEGORIES)))
    line_num = dict()
    for i, article_id in enumerate(g.outward_line_id):
        line_num[article_id] = i # store index corresponding to article_id
        current_enc = np.zeros(len(CATEGORIES))
        for j, category in enumerate(CATEGORIES):
            current_enc[j] = dijkstra(g, article_id, category, return_path = False)
        dist_enc[i] = current_enc
    return dist_enc

# Convenience functions

# Get encoding for a given title
def encoding(g, page, p_enc, line_num):
    '''
    Parameters
    ----------
    g : WikiGraph
    page : str
    p_enc : array
    line_num : dict
    Returns
    -------
    array
        Returns encoding of a given page (either article_id or page title)
    '''
    if page not in g.article_id:
        if page not in g.title:
            raise Exception(f"Page {page} not in subgraph!")
    else:
        page = g.article_id[page]
    
    return p_enc[line_num[page]]

# Compute cosine similarity between two pages
def cos_sim(page1, page2, p_enc):
    '''
    Parameters
    ----------
    page1 : str
    page2 : str
    p_enc : array
    Returns
    -------
    float
        Returns cosine similarity value between the two pages

    '''
    return cosine_similarity(np.stack((encoding(page1, p_enc), encoding(page2, p_enc))))[0, 1]

## Heuristic approach

def update(g, dist_enc,line_num,  N_NODES, CATEGORIES, n_times = 5, alpha = 0.8):
    '''
    Parameters
    ----------
    g : WikiGraph
    dist_enc : array
    line_num : dict
    N_NODES : int
    CATEGORIES : list
    n_times : int
    alpha : float
    Returns
    -------
    curr_dist_enc : array
        Update node embeddings by taking a convex linear combination with parameter alpha;
        propagate information for n_time iterations
    '''
    
    # Prepare for pooling update
    curr_dist_enc = dist_enc.copy()
    new_dist_enc = np.zeros((N_NODES, len(CATEGORIES)))
    for _ in range(n_times):
        for i, article_id in enumerate(g.outward_line_id):
            if article_id in CATEGORIES:
                # do not update category pages
                new_dist_enc[line_num[article_id]] = curr_dist_enc[line_num[article_id]]
                continue
            in_neighs = g.get_inward_neighbours(article_id)
            if len(in_neighs) > 0:
                avg_enc_in = np.zeros(len(CATEGORIES))
                for neigh in in_neighs:
                    avg_enc_in += encoding(neigh, p_enc = curr_dist_enc)
                out_neighs = g.get_outward_neighbours(article_id)
                
                if len(out_neighs) > 0:
                    avg_enc_out = np.zeros(len(CATEGORIES))
                    for neigh in out_neighs:
                        avg_enc_out += encoding(neigh, p_enc = curr_dist_enc)
                    avg_enc = avg_enc_in + avg_enc_out
                    avg_enc /= len(in_neighs + out_neighs)
                else:
                    avg_enc = avg_enc_in / len(in_neighs)
                
                new_dist_enc[line_num[article_id]] = alpha * encoding(article_id, 
                                                                      p_enc = curr_dist_enc) + (1 - alpha) * avg_enc
        curr_dist_enc = new_dist_enc.copy()
        new_dist_enc = np.zeros((N_NODES, len(CATEGORIES)))
    return curr_dist_enc

def get_labels(g, scaled_enc, CATEGORIES):
    '''
    Parameters
    ----------
    g : WikiGraph
    scaled_enc : array
    CATEGORIES : list
    Returns
    -------
    cat : dict
        Assign labels to all nodes based on highest similarity score w.r.t. the categories
    '''
    cat = dict()
    for node in g.article_id.keys():
        best_simil_score = -1.
        for category in CATEGORIES:
            simil_score = cos_sim(node, category, p_enc = scaled_enc)
            if simil_score > best_simil_score:
                cat[node] = category
                best_simil_score = simil_score
    return cat

def accuracy(correct_cat, cat):
    '''
    Parameters
    ----------
    correct_cat : dict
    cat : dict
    Returns
    -------
    float
        Compute accuracy score for labels in cat w.r.t. labels in correct_cat

    '''
    correct = 0
    for node in correct_cat.keys():
        if correct_cat[node] == cat[node]:
            correct += 1
    return correct / len(correct_cat)

def tune_hyperparams(g, initial_embedd, alpha_vals, n_iters_vals, correct_labels, line_num, N_NODES, CATEGORIES):
    '''
    Parameters
    ----------
    g : WikiGraph
    initial_embedd : array
    alpha_vals : list
    n_iters_vals : list
    correct_labels : dict
    line_num : dict
    N_NODES : int
    CATEGORIES : list
    Returns
    -------
    best_params : dict
        Run a grid search to find the best values for alpha and n_iters
    '''
    best_params = dict()
    best_params['alpha'] = None
    best_params['n_iters'] = None
    best_params['acc'] = 0
    
    for alpha in alpha_vals:
        for n_iters in n_iters_vals:
            updated_embedd = update(g, initial_embedd, line_num, N_NODES, CATEGORIES, n_iters, alpha)
            scaled_embedd = StandardScaler(with_std = False).fit_transform(updated_embedd) # rescale embedding
            labels = get_labels(g, scaled_embedd, CATEGORIES)
            acc = accuracy(correct_labels, labels)
            if acc > best_params['acc']:
                best_params['alpha'] = alpha
                best_params['n_iters'] = n_iters
                best_params['acc'] = acc
    
    return best_params

## Deep learning approach

def prepare_data(g, scaled_enc, correct_cat, CATEGORIES):
    '''
    Parameters
    ----------
    g : WikiGraph
    scaled_enc : array
    correct_cat : dict
    CATEGORIES : list
    Returns
    -------
    node_embeddings : array
    adjacency_list : dict
    labeled_nodes : array
    labels : array
        Prepare data to fit to deep learning models
    '''
    
    # Build a networkx graph
    networkx_graph = nx.Graph()

    all_nodes = []
    for node in g.title.keys():
        all_nodes.append(int(node))
    networkx_graph.add_nodes_from(all_nodes)

    all_edges = []
    for node in g.title.keys():
        neighbours = g[node]
        for neigh in neighbours:
            all_edges.append([int(node), int(neigh)])
    networkx_graph.add_edges_from(all_edges)

    # Add node embeddings as node attributes
    for node in networkx_graph.nodes:
        embedding = encoding(str(node), p_enc = scaled_enc)  
        networkx_graph.nodes[node]['embedding'] = embedding
    # Convert NetworkX graph to adjacency list
    adjacency_list = nx.to_dict_of_lists(networkx_graph)
    
    labeled_nodes = []
    labels = [] # 1 for Philosophy, 2 for Mathematics, 3 for Sport, 4 for Music
    mapping = {category: i for i, category in enumerate(CATEGORIES)}

    # Collect node embeddings
    node_embeddings = []
    for i, node in enumerate(networkx_graph.nodes):
        embedding = networkx_graph.nodes[node]['embedding']
        node_embeddings.append(embedding)
        if g.get_title(str(node)) in correct_cat.keys():
            labeled_nodes.append(i)
            for category in CATEGORIES:
                if correct_cat[g.get_title(str(node))] == category:
                    labels.append(mapping[category])

    # Convert node_embeddings, labeled_nodes and labels to NumPy arrays
    node_embeddings = np.array(node_embeddings)
    labeled_nodes = np.array(labeled_nodes)
    labels = np.array(labels)

    return node_embeddings, adjacency_list, labeled_nodes, labels

# GraphSAGE model

class GraphSAGEMean(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGEMean, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # GraphSAGE layers
        self.sage_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, node_embeddings, adjacency_list):
        valid_indices = [idx for idx in adjacency_list.keys() if idx < len(node_embeddings)]
        mapping = {node_id: idx for idx, node_id in enumerate(valid_indices)}
        indices = [mapping[node_id] for node_id in adjacency_list.keys() if node_id in mapping]
        neighbors_embeddings = node_embeddings[indices]
        
        # Aggregation step
        aggregated_embeddings = [
            torch.mean(neighbors_embedding, dim=0)
            for neighbors_embedding in neighbors_embeddings
        ]
        
        x = node_embeddings
        
        # GraphSAGE layer computations
        for layer in self.sage_layers:
            x = layer(x)
            x = F.relu(x)
        
        # Final output layer
        x = self.output_layer(x)
        
        return x, aggregated_embeddings

class GraphSAGELSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super(GraphSAGELSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, node_embeddings, adjacency_list):
        valid_indices = [idx for idx in adjacency_list.keys() if idx < len(node_embeddings)]
        mapping = {node_id: idx for idx, node_id in enumerate(valid_indices)}
        indices = [mapping[node_id] for node_id in adjacency_list.keys() if node_id in mapping]
        neighbors_embeddings = node_embeddings[indices]
        
        # Pad the neighbor embeddings for batch processing
        padded_neighbors = nn.utils.rnn.pad_sequence(neighbors_embeddings, batch_first=True)
        
        # Pass the padded neighbors through the LSTM layer
        lstm_out, _ = self.lstm1(padded_neighbors)
        
        # Apply dropout regularization
        aggregated_neighbors = self.dropout(lstm_out)
        
        output = self.linear1(aggregated_neighbors)
        output = F.relu(output)
        output = self.linear2(aggregated_neighbors)
        output = F.relu(output)
        output = self.linear3(aggregated_neighbors)
        output = F.relu(output)
        output = self.linear4(aggregated_neighbors)
        output = F.relu(output)
        
        # Perform linear transformation and activation
        output = self.output_layer(lstm_out)
        
        return output

def train_eval(model_type, node_embeddings, labels, labeled_nodes, adjacency_list, CATEGORIES, hidden_dim = 64, learning_rate = 0.001, num_epochs = 20, batch_size = 32): 
    '''
    Parameters
    ----------
    node_embeddings : array
    labels : array
    labeled_nodes : array
    adjacency_list : dict
    CATEGORIES : list
    hidden_dim : int, optional
    learning_rate : int, optional
    num_epochs : int, optional
    batch_size : int, optional
    Returns
    -------
    model : object
        Train and validate GraphSAGE model
    '''
    
    input_dim = node_embeddings.shape[1]  # Dimensionality of node embeddings
    output_dim = len(CATEGORIES)  # Number of output categories

    # Train/test split
    split = int(0.8 * len(labeled_nodes))
    X_train = (node_embeddings[labeled_nodes])[:split]
    y_train = labels[:split]
    X_test = (node_embeddings[labeled_nodes])[split:]
    y_test = labels[split:]

    # Convert the training data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)

    model = model_type(input_dim, hidden_dim, output_dim)

    # Convert node_embeddings to a PyTorch tensor
    node_embeddings_tensor = torch.FloatTensor(node_embeddings)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
    
        # Mini-batch training
        for batch_start in range(0, len(X_train), batch_size):
            batch_end = min(batch_start + batch_size, len(X_train))
    
            # Forward pass
            outputs, _ = model(X_train_tensor[batch_start:batch_end], adjacency_list)
            loss = criterion(outputs, y_train_tensor[batch_start:batch_end])
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()

    # Print the average loss for the epoch
    average_loss = running_loss / (len(X_train) / batch_size)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")


    # Evaluation
    model.eval()
    with torch.no_grad():
        # Convert the testing data to PyTorch tensors
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
    
        # Forward pass on the testing data
        outputs, _ = model(X_test_tensor, adjacency_list)
        _, predicted_labels = torch.max(outputs, 1)
    
        # Calculate accuracy
        accuracy = (predicted_labels == y_test_tensor).sum().item() / len(y_test)
        print(f"Accuracy on the test set: {accuracy:.4f}")
        
    return model


