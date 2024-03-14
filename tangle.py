import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import threading
import random
import pandas as pd



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
        #ww = random.randint(1, 4)
        ww = 1
        self.nodeWeights.clear()
        return ww

    def new_node(self, mal=False, watch=None):
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
            if mal == True:
                n = mal_node(edges, nodeID, self.tangle, ww, watch) # If it's marked as a malicious node make one of those
            else:
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
         
    def issue_transaction(self):
        #Take a bunch of parameters for the block and transactions within
        content = random.randint(0, 100)
        nodeSig = self.signature
        self.tangle.next_transaction(self.ww, content, False)
    
    def update_neighbourhood(self, newNeighbour):
        self.neighbourhood.append(newNeighbour.id)


class mal_node(node):

    def __init__(self, edges, nodeID, tangle, ww, watcher):
        self.id = nodeID
        self.neighbourhood = edges
        self.signature = np.random.randint(2048)
        self.ww = 1
        self.tangle = tangle
        self.ds_start = None
        self.chain = []
        self.watcher = watcher
    
    def issue_bad_transaction(self):
        #Take a bunch of parameters for the block and transactions within
        content = random.randint(0, 100)
        nodeSig = self.signature
        self.tangle.next_transaction(self.ww, content, True)
        self.catch_PC_tr()

    def spam_transactions(self, num):
        for i in range(num):
            content = random.randint(0, 100)
            nodeSig = self.signature
            self.mal_next_transaction(self.ww, content, False)
            self.watcher.update()

    def catch_PC_tr(self):
        for t in self.tangle.transactions:
            if t.DS_transaction == True:
                self.ds_start = t
                self.chain.append(t)
    
    def mal_mcmc(self):
        found = False
        len_low = np.floor(len(self.tangle.transactions))
        while found == False:
            num_particles = 10
            lower_bound = int(np.maximum(0, self.tangle.count - 20.0*self.tangle.rate))
            upper_bound = int(np.maximum(len_low, 
                                         self.tangle.count - 10.0*self.tangle.rate))

            candidates = self.tangle.transactions[lower_bound:upper_bound]

            particles = np.random.choice(candidates, num_particles)
            #print("Particles = ", particles)
            threads = []
            for p in particles:
                t = threading.Thread(target=self.tangle._walk2(p))
                threads.append(t) # added to check if waiting for the threads will solve it
                t.start()
                
            for th in threads:
                th.join()

            #print("Walk Chache", self.tip_walk_cache)

            tips = self.tangle.tip_walk_cache[:1]
            #print("Tips", tips[0].num)
            found = True

        self.tangle.tip_walk_cache = list()
        return tips
        
    def mal_next_transaction(self, NodeWeight, content, DS):
        dt_time = np.random.exponential(1.0/self.tangle.rate)
        self.tangle.time += dt_time
        self.tangle.count += 1
        tip2 = self.mal_mcmc()[0]
        tip1 = self.chain[-1]
        approved_tips = [tip1, tip2]
        transaction = Transaction(self.tangle, self.tangle.time, 
                                    approved_tips, self.tangle.count - 1, NodeWeight, 
                                    content, DS)
            #print(approved_tips, "AP")
        for t in approved_tips:
            #print("approved", t)
            t.approved_time = np.minimum(self.tangle.time, t.approved_time)
            t._approved_directly_by.add(transaction)

            if hasattr(self.tangle, 'G'):
                self.tangle.G.add_edges_from([(transaction.num, t.num)])

        self.tangle.transactions.append(transaction)
        self.chain.append(transaction)
        self.tangle.cw_cache = {}


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
        self.tip_walk_checker = list()

    def printTransactionStats(self, t):
        print("------------")
        print("Transaction number", t.num)
        print("Transaction Content", t.content)
        print("Node Weight", t.weight)
        print("Confirmed Status", t.confirmed)

    def next_transaction(self, NodeWeight, content, DS):
        dt_time = np.random.exponential(1.0/self.rate)
        self.time += dt_time
        self.count += 1
        done = False
        while done == False:
            if self.tip_selection == 'mcmc':
                approved_tips = set(self.mcmc())
                done = True
            elif self.tip_selection == 'urts':
                #approved_tips = set(self.urts())
                approved_tips = self.urts()
                done = True
            else:
                raise Exception()
                done = True
        if DS == False:
            transaction = Transaction(self, self.time, approved_tips, self.count - 1, NodeWeight, content, DS)
            self.printTransactionStats(transaction)
            #print(approved_tips, "AP")
            for t in approved_tips:
                t.approved_time = np.minimum(self.time, t.approved_time)
                t._approved_directly_by.add(transaction)

                if hasattr(self, 'G'):
                    self.G.add_edges_from([(transaction.num, t.num)])

            self.transactions.append(transaction)

            self.cw_cache = {}

        elif DS == True:
            transaction = Transaction(self, self.time, approved_tips, self.count - 1, NodeWeight, content, DS)
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
            return np.random.choice([t for t in self.transactions if t.is_visible()])
        if len(tips) == 1:
            return [tips[0]]
        val = np.random.choice(tips, 2)
        #print(val)
        return val

    def orphan_protocol(self, index):
        for i in self.transactions:
            if i.num == index:
                i.orphaned = True
                #print(i, i.orphaned)

    def tip_no_match(self):
        ret = [self.tip_walk_cache[0]]
        i = 1
        while len(ret) == 1:
            if self.tip_walk_cache[i] != ret[0]:
                ret.append(self.tip_walk_cache[i])
            else:
                if i < len(self.tip_walk_cache)-1:
                    i += 1
                else:
                    #print("Fuck")
                    # add a section about picking another of the latest 3 transactions
                    new_tran = np.random.choice(self.transactions[:2], 1)
                    #print(new_tran)
                    new_tp = new_tran[0]
                    while ret[0] == new_tp:
                        new_tran = np.random.choice(self.transactions[:2], 1) # chooce a random option of the last two
                        new_tp = new_tran[0]
                    ret.append(new_tp)
                    #print(ret)
        return ret

    def mcmc(self):
        found = False
        len_low = np.floor(len(self.transactions))
        while found == False:
            num_particles = 20
            lower_bound = int(np.maximum(0, self.count - 20.0*self.rate))
            upper_bound = int(np.maximum(len_low, self.count - 10.0*self.rate))

            candidates = self.transactions[lower_bound:upper_bound]
            #print("Bounds", lower_bound, upper_bound)
            #print("Candidates", candidates)

            particles = np.random.choice(candidates, num_particles)
            #print("Particles = ", particles)
            threads = []
            for p in particles:
                t = threading.Thread(target=self._walk2(p))
                threads.append(t) # added to check if waiting for the threads will solve it
                t.start()
                
            for th in threads:
                th.join()

            #print("Walk Chache", self.tip_walk_cache)

            tips = self.tip_walk_cache[:2]
            #print("Tips", tips[0].num, tips[1].num)

            if (tips[0].isGenesis == True and tips[1].isGenesis == True) and len(self.transactions) < 4:
                #print("2 Genesis Found")
                found = True
            elif tips[0].num != tips[1].num:
                #print("Selected Different Transactions")
                found = True
            else:
                #break
                tips = self.tip_no_match()
                #print("Same non-genesis selected")
                #print("New Tips", tips[0].num, tips[1].num)
                found = True
                #tips = self.tip_walk_cache[:2] # <- THIS IS A TEMPORARY MEASURE WHILE I SORT OUT PARASITE CHAIN 
                # Try again sucker


        """conflict_flag, index_orphan = self.conflict_check(tips)   #checking for conflicts 
        if conflict_flag == True:
            t_name = tips[index_orphan].num
            self.orphan_protocol(t_name)
            return []"""
        #print(tips)
        self.tip_walk_cache = list()
        return tips
    
    # Without this seperate checking function it goes into forever recursion
    def mcmc_no_check(self):
        num_particles = 10
        lower_bound = int(np.maximum(0, self.count - 20.0*self.rate))
        upper_bound = int(np.maximum(1, self.count - 10.0*self.rate))

        candidates = self.transactions[lower_bound:upper_bound]

        particles = np.random.choice(candidates, num_particles)
        threads = []
        for p in particles:
            t = threading.Thread(target=self._walk2(p))
            threads.append(t)
            t.start()

        for th in threads:
           th.join()

        
        tips = self.tip_walk_cache[:2]
        #print(tips)
        self.tip_walk_cache = list() # This may need changing to something else but if I don't have to I won't
        return tips

    def _walk2(self, starting_transaction):
        #print("In walk")
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

    def test_mcmc(self, tips_to_check):
        tip_0 = 0
        tip_1 = 0
        for i in range(10):
            tips = self.mcmc_no_check()
            if tips[0] == tips_to_check[0]:
                tip_0 += 1
            elif tips[1] == tips_to_check[1]:
                tip_1 += 1
        if tip_0 >= tip_1:
            return 1    # Return the one to be orphaned
        else:
            return 0


    def conflict_check(self, selected_tips):
        tip_1 = selected_tips[0]
        tip_2 = selected_tips[1]
        print("Conflict Check", tip_1.content, tip_1.num, tip_2.content, tip_2.num)
        #print(tip_1.content, tip_2.content)
        if tip_1.isGenesis == True or tip_2.isGenesis == True:
            print("First Approval")
        elif tip_1.content == tip_2.content:
            #print("How the fuck")
            index = self.test_mcmc(selected_tips)
            return True, index
        return False, 2
        

        #return 0

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

    def __init__(self, tangle, time, approved_transactions, num, NodeWeight, content, double_spend=False):
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
        self.orphaned = False
        self.isGenesis = False
        self.DS_transaction = double_spend
        self.cum_weight = NodeWeight

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
        self.cum_weight = cw
        return cw

    def cumulative_weight_delayed(self):
        cached = self.tangle.cw_cache.get(self.num)
        if cached:
            #print("CWD", cached)
            if cached >= self.tangle.theta:
                #print("CWD", cached, self.tangle.theta)
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
        self.cum_weight = cached
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
        self.isGenesis = True
        self.orphaned = False
        self.DS_transaction = False
        if hasattr(self.tangle, 'G'):
            self.tangle.G.add_node(self.num, pos=(self.time, 0))

    def __repr__(self):
        return '<Genesis>'

class watcher():

    def __init__(self, tangle, node_g, ID):

        self.ID = ID    # Single integer number to identify the experiment 
        self.confirmed_transactions = 1
        self.number_of_transactions = 1
        self.number_of_tips = 0
        self.tangle_time = tangle.time
        self.number_of_nodes = len(node_g.nodes)
        self.tangle = tangle
        self.record_of_tips = [self.number_of_tips]
        self.record_of_confirmations = [self.confirmed_transactions]
        self.record_of_transactions = [self.number_of_transactions]
        self.times = [self.tangle_time]
        self.orphans = []
        self.bad_transaction = None
        self.bad_nodes = []
        self.cw_over_time_PC = [] # Cumulative weight of the PC transaction over time
        self.PC_times = []

    def mal_tran_get(self):
        for t in self.tangle.transactions:
            if t.DS_transaction == True:
                self.bad_transaction = t


    def update(self):
        self.mal_tran_get()
        self.number_of_transactions = len(self.tangle.transactions)
        self.record_of_transactions.append(self.number_of_transactions)
        if self.bad_transaction != None:
            self.cw_over_time_PC.append(self.bad_transaction.cum_weight)
            self.PC_times.append(self.tangle.time)
        self.confirmed_transactions = 0
        #print(self.tangle.transactions)
        for c in self.tangle.transactions:
            # If the transaction is confirmed add it here
            if c.confirmed == True:
                self.confirmed_transactions += 1
            # If the transaction is orphaned add it here
            if c.orphaned == True:
                self.orphans.append(c)
        self.number_of_tips = len(self.tangle.tips())

        self.times.append(self.tangle.time)
        self.record_of_tips.append(self.number_of_tips)
        self.record_of_confirmations.append(self.confirmed_transactions)


    def printStats(self):
        print("--------------")
        print("Number of Confirmed transactions", self.confirmed_transactions)
        print("Number of Transactions", self.number_of_transactions)
        print("Orphans", self.orphans)

    def plotTransactions(self):
        plt.plot(self.record_of_transactions, self.times)
        plt.title('Number of Transactions over Time')
        plt.xlabel('Time')
        plt.ylabel('Transactions')

    def plot_tips_over_time(self):
        plt.bar(self.times, self.record_of_tips)
        plt.title('Number of Tips over Time')
        plt.xlabel('Time')
        plt.ylabel('Tips')

    def plot_confirm_over_time(self):
        print(self.record_of_confirmations, self.times)
        plt.plot(self.record_of_confirmations, self.times)
        plt.title('Number of Confirmations over Time')
        plt.xlabel('Time')
        plt.ylabel('Confirmations')

    def plot_cum_weight(self):
        #print(self.cw_over_time_PC, self.PC_times)
        plt.plot(self.cw_over_time_PC, self.PC_times)
        #plt.plot(self.PC_times, self.cw_over_time_PC)
        plt.title('Total Cumulative Weight of Transactions')
        plt.xlabel('Time')
        plt.ylabel('Weight')

    def plot_PC_cum_weight(self):
        plt.plot(self.PC_times, self.cw_over_time_PC)
        plt.title('Parasite Chain Cumulative Weight')
        plt.xlabel('Time')
        plt.ylabel('Weight')

    def output_to_sheet(self):
        res_times = self.times
        new_times =  [round(x) for x in res_times] 
        dict = {'confirmations': self.record_of_confirmations, 'time': new_times}
        df = pd.DataFrame(dict)
        #print(new_times)
        #print(df)
        df.to_csv("confirmations"+str(self.ID)) #save up to i different sets of results
    

class analyser():

    def __init__(self):
        self.help = None 
        self.me = None 
        self.please = None 

    def get_res(self, i):
        num_of_runs = i
        j = 1
        results = []
        
        while j <= num_of_runs:
            df = pd.read_csv("confirmations"+str(j))
            length = df.shape[0]
            curRow = 0
            while curRow < length:
                next_time = df.iloc[curRow, 2]
                next_val = df.iloc[curRow, 1]
                if len(results) == 0:
                    results.append([next_time, next_val, 1])
                else:
                    res_counter = 0
                    len_res = len(results)
                    found = False   # Flag to dictate 
                    while (res_counter <= len_res) and found == False:
                        
                        if res_counter == len_res:
                            results.append([next_time, next_val, 1])
                            #print([next_time, next_val, 1, res_counter])
                            found = True

                        elif results[res_counter][0] == next_time:
                            results[res_counter][2] += 1
                            results[res_counter][1] = (results[res_counter][1]+next_val)/results[res_counter][2] 
                            #print([next_time, next_val, "Fuck", res_counter])
                            found = True
                            # Average the value so I don't have to do it later  

                        elif (results[res_counter][0] > next_time) and (results[res_counter][0] != next_time):
                            i = res_counter
                            results.insert(i, [next_time, next_val, 1])
                            found = True
                        
                        res_counter += 1

                curRow += 1
            j += 1
        print(results)
        return results


            

