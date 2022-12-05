import numpy as np
import time

class WikiGraph(object):
    
    def __init__(self, article_file_path, outward_edges_file_path, debug = False):
        """
        Creates an instance of the WikiGraph object starting from two file paths.
        """
        
        if not isinstance(article_file_path, str) or not isinstance(outward_edges_file_path, str):
            raise Exception("Paths must be strings!")
        
        if debug:
            t0 = time.time()
            print("Initializing object...")
        
        G = dict()
        title = dict()
        article_id = dict()
        
        #read the article ids and use them as dict keys
        
        with open(article_file_path) as f:
            for line in f.readlines():
                line = line[:-1].split(" ")
                node = line[0]  #id   
                article_title = line[1] # title
                G[node] = None
                title[node] = article_title
                article_id[article_title] = node
        
        if debug:
            t1 = time.time()
            print(f"Successfully loaded the dict keys in {t1-t0} seconds...")
        
        #read the source node id and assign its adjacent nodes as a tuple coresponding to the key in the dict
        
        num_edges = 0
        deg = dict()
        with open(outward_edges_file_path) as g:
            line = g.readline()
            while line:
                line = line[:-1].split(" ") #remove the \n at the end of the line
                src_node = line[0]
                if len(line) >= 2:
                    G[src_node] = tuple(line[1:])
                    node_deg = len(line[1:])
                    num_edges += node_deg
                    deg[src_node] = node_deg
                else:
                    deg[src_node] = 0
                line = g.readline()
        
        if debug:
            t2 = time.time()
            print(f"Successfully loaded the dict values in {t2-t1} seconds...")
        
        #store G and number of nodes in the class object
        
        self.G = G
        self.title = title
        self.article_id = article_id
        self.num_nodes = len(G)
        self.num_edges = num_edges
        self.deg = deg
        
        if debug:
            t3 = time.time()
            print(f"Object initialization completed in {t3-t0} seconds.")
            
    def __repr__(self):
        """
        Returns the string represenation of the WikiGraph object.
        """
        graph_description = f"WikiGraph with {self.num_nodes} nodes and {self.num_edges} edges"
        return graph_description
    
    def __len__(self):
        """
        Returns the total number of nodes in the graph.
        """
        return self.num_nodes
    
    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : string
            Either the article id in string format or the article title
        Returns
        -------
        A tuple containing the adjacent nodes of index.

        """
        val = None
        try:
            val = self.G[index]
        except:
            try:
                val = self.G[self.article_id[index]]
            except:
                raise Exception(f"Input index {index} does not match any id or title")
        return val
    
    def __contains__(self, index):
        """
        Parameters
        ----------
        index : string
            Either the article id in string format or the article title
        Returns
        -------
        Bool value: True if index is in WikiGraph, False otherwise.

        """
        return (index in self.title) or (index in self.article_id)
                
    def get_num_nodes(self):
        """
        Returns the number of nodes in the WikiGraph.
        """
        return self.num_nodes
            
    def get_num_edges(self):
        """
        Returns the number of edges in the WikiGraph.
        """
        return self.num_edges
    
    def get_title(self, article_id):
        """
        Parameters
        ----------
        article_id : string
            Page_id in string format
            
        Returns
        -------
        A string representing the title of page if it exsists. Otherwise raises an exception.
        """
        article_title = None
        try:
            article_title = self.title[article_id]
        except:
            raise Exception("Article id not found")
        return article_title
    
    def get_id(self, article_title):
        """
        Parameters
        ----------
        article_tile : string
            Page title in string format
            
        Returns
        -------
        A string representing the id of page if it exsists. Otherwise raises an exception.
        """
        page_id = None
        try:
            page_id = self.article_id[article_title]
        except:
            raise Exception("Article title not found")
        return page_id
    
    def get_stats(self):
        """
        Returns 
        -------
        stats : string
            A string consisting of general statistics about the graph.
        """
        stats = ""
        stats += f"Nodes: {self.num_nodes}\n"
        stats += f"Edges: {self.num_edges}\n"
        
        avg_deg = sum(self.deg.values()) / self.num_nodes
        stats += f"Average out degree: {avg_deg}\n"
        
        min_deg_id = None
        min_val = np.inf
        max_deg_id = None
        max_val = -np.inf
        for key in self.deg:
            if self.deg[key] < min_val:
                min_val = self.deg[key]
                min_deg_id = key
            if self.deg[key] > max_val:
                max_val = self.deg[key]
                max_deg_id = key
        stats += f"Minimum out degree: {min_val} - {self.title[min_deg_id]}\n"
        stats += f"Maximum out degree: {max_val} - {self.title[max_deg_id]}"
        
        print(stats)
        
        return 
    
    def get_weight(self, src_node, dest_node):
        """
        Parameters
        ----------
        src_node : string
            Source node of the edge: either id or title;
        dest_node : string
            Destination node of the edge.
        Takes two string: id or title of source and destination nodes: either id or title;
        Returns
        -------
        weight : float
            A float representing the weight of the edge from src_node to dest_node (for now either 1 if the edge exists or np.inf otherwise).
        """
        if not isinstance(src_node, str) or not isinstance(dest_node, str):
            raise Exception("The two nodes must be both strings!")
        if not self.__contains__(src_node) or not self.__contains__(dest_node):
            raise Exception("The two nodes must belong to the graph!")
        
        weight = None
        if src_node in self.article_id:
            src_node = self.article_id[src_node]
        if dest_node in self.article_id:
            dest_node = self.article_id[dest_node]
        if src_node == dest_node:
            weight = 0.0
        elif dest_node in self.G[src_node]:
            weight = 1.0
        else:
            weight = np.inf
        
        return weight
        
        
        
        
        
