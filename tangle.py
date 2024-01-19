import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import threading
import random



class node_graph():

    def __init__(self, tangle):
        self.time = 1
        self.nodes = []
        self.count = 0
        self.edgelist = []
        self.nodeIDlist = []
        self.nodeWeights = []
        self.tangle = tangle
        self.g = nx.Graph()

    def plot_graph(self):
            self.g.add_nodes_from(self.nodeIDlist)
            self.g.add_edges_from(self.edgelist)
            nx.draw_networkx(self.g)
    
    def printStats(self):
        print("-------------------")
        print("Nodes RAW", self.nodes)
        print("Edges", self.edgelist)
        print("Node IDS", self.nodeIDlist)
        print("Weights", self.nodeWeights)
    
    def assignWW(self):
        #ww = 1 / len(self.nodes)
        ww = random.randint(1, 4)
        self.nodeWeights.clear()
        return ww

    def new_node(self):
        nodeID = self.count
        self.count += 1
        if not self.nodes:
            ww = 1
            n = node([], nodeID,  self.tangle, ww)
            self.nodeIDlist.append(nodeID)
            self.nodeWeights.append([nodeID, n.signature, ww])
            for m in self.nodes:
                m.ww = ww
                self.nodeWeights.append([m.id, m.signature, m.ww])

        elif len(self.nodes) == 1:
            ww = self.assignWW()
            n = node(self.nodes, nodeID, self.tangle, ww)
            for m in self.nodes:
                self.nodeWeights.append([m.id, m.signature, m.ww])
            self.edgelist.append((nodeID, self.nodes[0].id))
            self.nodeIDlist.append(nodeID)
            self.nodeWeights.append([nodeID, n.signature, ww])

        else:
            edges = []
            ww = self.assignWW()
            j = 0
            while j < 2:
                item = random.choice(self.nodes)
                if item not in edges:
                    edges.append(item)
                    j += 1
            n = node(edges, nodeID, self.tangle, ww)
            for m in self.nodes:
                self.nodeWeights.append([m.id, m.signature, m.ww])
            self.nodeWeights.append([nodeID, n.signature, ww])
            self.edgelist.append((nodeID, edges[0].id))
            self.edgelist.append((nodeID, edges[1].id))
            self.nodeIDlist.append(nodeID)

        self.update_neighbours(n)
        self.nodes.append(n)
        print("Finished updating")
        self.printStats()
    
    def delete_node(self):
        #update the graph so that when the node is deleted, its neighbours are connected
        #if they are already connected to eachother then connect to a random other node
        #if there are only 2 nodes then end function
        return 0
    
    def update_neighbours(self, newNode):
        #Update the neighouhoods of all nodes in graph
        if len(self.nodes) == 1:
            single = self.nodes[0]
            single.update_neighbourhood(newNode)

        elif len(self.nodes) > 1:
            for i in self.nodes:
                if i.id == newNode.neighbourhood[0]:
                    i.update_neighbourhood(newNode)
                elif i.id == newNode.neighbourhood[1]:
                    i.update_neighbourhood(newNode)
        else:
            print("First node")
    
class node():

    def __init__(self, edges, nodeID, tangle, ww):
        self.id = nodeID
        self.neighbourhood = edges
        self.signature = np.random.randint(2048)
        self.ww = ww
        self.tangle = tangle
       
    def orphaned_block():
        #Ask neighbours for the block
        return 0
    
    def issue_transaction(self):
        #Take a bunch of parameters for the block and transactions within
        content = random.randint(0, 100)
        nodeSig = self.signature
        self.tangle.next_transaction(self.ww, content)
    
    def get_block_list():
        #Return all the blocks issued by this node
        return 0
    
    def update_neighbourhood(self, newNeighbour):
        self.neighbourhood.append(newNeighbour.id)


class Tangle(object):

    def __init__(self, rate=50, alpha=0.001, tip_selection='mcmc', plot=False):
        self.time = 1.0
        self.rate = rate
        self.alpha = alpha
        self.theta = 5 # confirmation threshold
        
        if plot:
            #self.G = nx.OrderedDiGraph()
            self.G = nx.DiGraph()

        self.genesis = Genesis(self)
        self.transactions = [self.genesis]
        self.count = 1
        self.tip_selection = tip_selection

        self.cw_cache = dict()
        self.t_cache = set()
        self.tip_walk_cache = list()

    def printTransactionStats(self, t):
        print("------------")
        print("Transaction number", t.num)
        print("Transaction Content", t.content)
        print("Node Weight", t.weight)
        print("Confirmed Status", t.confirmed)

    def next_transaction(self, NodeWeight, content):
        dt_time = np.random.exponential(1.0/self.rate)
        self.time += dt_time
        self.count += 1

        if self.tip_selection == 'mcmc':
            approved_tips = set(self.mcmc())
        elif self.tip_selection == 'urts':
            approved_tips = set(self.urts())
        else:
            raise Exception()

        transaction = Transaction(self, self.time, approved_tips, self.count - 1, NodeWeight, content)
        #self.printTransactionStats(transaction)
        for t in approved_tips:
            t.approved_time = np.minimum(self.time, t.approved_time)
            t._approved_directly_by.add(transaction)

            if hasattr(self, 'G'):
                self.G.add_edges_from([(transaction.num, t.num)])

        self.transactions.append(transaction)

        self.cw_cache = {}

    def tips(self):
        return [t for t in self.transactions if t.is_visible() and t.is_tip_delayed()]

    def urts(self):
        tips = self.tips()
        if len(tips) == 0:
            return np.random.choice([t for t in self.transactions if t.is_visible()]),
        if len(tips) == 1:
            return tips[0],
        return np.random.choice(tips, 2)

    def mcmc(self):
        num_particles = 10
        lower_bound = int(np.maximum(0, self.count - 20.0*self.rate))
        upper_bound = int(np.maximum(1, self.count - 10.0*self.rate))

        candidates = self.transactions[lower_bound:upper_bound]
        #at_least_5_cw = [t for t in self.transactions[lower_bound:upper_bound] if t.cumulative_weight_delayed() >= 5]

        particles = np.random.choice(candidates, num_particles)
        distances = {}
        for p in particles:
            t = threading.Thread(target=self._walk2(p))
            t.start()
#            tip, distance = self._walk(p)
#            distances[tip] = distance
            
        #return [key for key in sorted(distances, key=distances.get, reverse=False)[:2]]
        tips = self.tip_walk_cache[:2]
        self.tip_walk_cache = list()

        return tips

    def _walk2(self, starting_transaction):
        p = starting_transaction
        while not p.is_tip_delayed() and p.is_visible():
            if len(self.tip_walk_cache) >= 2:
                return

            next_transactions = p.approved_directly_by()
            if self.alpha > 0:
                p_cw = p.cumulative_weight_delayed()
                c_weights = np.array([])
                for transaction in next_transactions:
                    c_weights = np.append(c_weights, transaction.cumulative_weight_delayed())
                    #print(starting_transaction, c_weights)

                deno = np.sum(np.exp(-self.alpha * (p_cw - c_weights)))
                probs = np.divide(np.exp(-self.alpha * (p_cw - c_weights)), deno)
            else:
                probs = None

            p = np.random.choice(next_transactions, p=probs)

        self.tip_walk_cache.append(p)
    
    def _walk(self, starting_transaction):
        p = starting_transaction
        count = 0
        while not p.is_tip_delayed() and p.is_visible():
            next_transactions = p.approved_directly_by()
            if self.alpha > 0:
                p_cw = p.cumulative_weight_delayed()
                c_weights = np.array([])
                for transaction in next_transactions:
                    c_weights = np.append(c_weights, transaction.cumulative_weight_delayed())

                deno = np.sum(np.exp(-self.alpha * (p_cw - c_weights)))
                probs = np.divide(np.exp(-self.alpha * (p_cw - c_weights)), deno)
            else:
                probs = None

            p = np.random.choice(next_transactions, p=probs)
            count += 1

        return p, count

    def plot(self):
        if hasattr(self, 'G'):
            pos = nx.get_node_attributes(self.G, 'pos')
            nx.draw_networkx_nodes(self.G, pos)
            nx.draw_networkx_labels(self.G, pos)
            nx.draw_networkx_edges(self.G, pos, edgelist=self.G.edges(), arrows=True)
            plt.xlabel('Time')
            plt.yticks([])
            plt.show()


class Transaction(object):

    def __init__(self, tangle, time, approved_transactions, num, NodeWeight, content):
        self.tangle = tangle
        self.time = time
        self.approved_transactions = approved_transactions
        self._approved_directly_by = set()
        self.approved_time = float('inf')
        self.num = num
        self._approved_by = set()
        self.content = content   
        self.weight = NodeWeight     
        self.confirmed = False 

        if hasattr(self.tangle, 'G'):
            self.tangle.G.add_node(self.num, pos=(self.time, np.random.uniform(-1, 1)))

    def is_visible(self):
        return self.tangle.time >= self.time + 1.0

    def is_tip(self):
        return self.tangle.time < self.approved_time

    def is_tip_delayed(self):
        return self.tangle.time - 1.0 < self.approved_time

    def cumulative_weight(self):
        #cw = 1 + len(self.approved_by())
        temp = 0
        for w in self.approved_by():
            temp += w.weight
        cw = self.weight + temp
        self.tangle.t_cache = set()
        if cw >= self.tangle.theta:
            self.confirmed = True
        return cw

    def cumulative_weight_delayed(self):
        cached = self.tangle.cw_cache.get(self.num)
        if cached:
            #print(cached)
            if cached >= self.tangle.theta:
                self.confirmed = True
            return cached

        else:
            temp = 0
            for w in self.approved_by_delayed():
                temp += w.weight
            cached = self.weight + temp
            #cached = 1 + len(self.approved_by_delayed())
            self.tangle.t_cache = set()
            self.tangle.cw_cache[self.num] = cached
        #print(self.weight, cached)

        return cached

    def approved_by(self):
        for t in self._approved_directly_by:
            if t not in self.tangle.t_cache:
                self.tangle.t_cache.add(t)
                self.tangle.t_cache.update(t.approved_by())

        return self.tangle.t_cache

    def approved_by_delayed(self):
        for t in self.approved_directly_by():
            if t not in self.tangle.t_cache:
                self.tangle.t_cache.add(t)
                self.tangle.t_cache.update(t.approved_by_delayed())

        return self.tangle.t_cache

    def approved_directly_by(self):
        return [p for p in self._approved_directly_by if p.is_visible()]

    def __repr__(self):
        return '<Transaction {}>'.format(self.num)


class Genesis(Transaction):

    def __init__(self, tangle):
        self.tangle = tangle
        self.time = 0
        self.approved_transactions = []
        self.approved_time = float('inf')
        self._approved_directly_by = set()
        self.num = 0
        self.content = 0    # add a section in init
        self.weight = 1     # add a section in init
        self.confirmed = True
        if hasattr(self.tangle, 'G'):
            self.tangle.G.add_node(self.num, pos=(self.time, 0))

    def __repr__(self):
        return '<Genesis>'

class watcher():

    def __init__(self, tangle, node_g):

        self.confirmed_transactions = 1
        self.number_of_transactions = 1
        self.number_of_tips = 0
        self.tangle_time = tangle.time
        self.number_of_nodes = len(node_g.nodes)
        self.tangle = tangle
    
    def update(self):
        self.number_of_transactions = len(self.tangle.transactions)
        self.confirmed_transactions = 0
        for c in self.tangle.transactions:
            if c.confirmed == True:
                self.confirmed_transactions += 1
    
    def printStats(self):
        print("--------------")
        print(self.confirmed_transactions)
        print(self.number_of_transactions)

