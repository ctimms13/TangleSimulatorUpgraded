
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import threading
import random
import tangle as t

class mal_node(t.node):

    def __init__(self, edges, nodeID, tangle, ww):
        self.id = nodeID
        self.neighbourhood = edges
        self.signature = np.random.randint(2048)
        self.ww = 4
        self.tangle = tangle
        ds_start = None
        chain = []

    def build_chain(self):
        #initialise conflicting transaction
        #copy an existing transaction from record and rename/renumber it
        #generate/issue new nodes with a predetermined approval of the double spend
        #add another approval from the mcmc algorithm
        #keep track of tangle time at moment of injection
        #immediately begin spamming these approval transactions
        return 0
    
    def issue_transaction(self):
        #Take a bunch of parameters for the block and transactions within
        content = random.randint(0, 100)
        nodeSig = self.signature
        self.tangle.next_transaction(self.ww, content, True)

    def catch_PC_tr(self):
        for t in self.tangle.transactions:
            if t.double_spend == True:
                self.ds_start = t
                self.chain.append(t)

    
    def mal_mcmc(self):
        found = False
        len_low = np.floor(len(self.transactions))
        while found == False:
            num_particles = 10
            lower_bound = int(np.maximum(0, self.count - 20.0*self.rate))
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
            print("Tips", tips[0].num)

        self.tangle.tip_walk_cache = list()
        return tips
        
    def mal_next_transaction(self, NodeWeight, content, DS):
        dt_time = np.random.exponential(1.0/self.rate)
        self.time += dt_time
        self.count += 1
        tip2 = self.mal_mcmc()
        tip1 = self.chain[-1]
        approved_tips = [tip1, tip2]
        transaction = t.Transaction(self.tangle, self.tangle.time, 
                                    approved_tips, self.tangle.count - 1, NodeWeight, 
                                    content, DS)
        self.printTransactionStats(transaction)
            #print(approved_tips, "AP")
        for t in approved_tips:
            t.approved_time = np.minimum(self.tangle.time, t.approved_time)
            t._approved_directly_by.add(transaction)

            if hasattr(self.tangle, 'G'):
                self.tangle.G.add_edges_from([(transaction.num, t.num)])

        self.tangle.transactions.append(transaction)
        self.tangle.cw_cache = {}

