import networkx as nx 
import requests
from itertools import combinations

# =========================
# Reading in CoinGecko API
# =========================

crypto_api_url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum,bitcoin,litecoin,ripple,cardano,bitcoin-cash,eos&vs_currencies=eth,btc,ltc,xrp,ada,bch,eos"

response = requests.get(crypto_api_url)
if response.status_code == 200:
    data = response.json()
    

coin_mapping = {
    'ripple': 'xrp',
    'cardano': 'ada',
    'bitcoin-cash': 'bch',
    'eos': 'eos',
    'litecoin': 'ltc',
    'ethereum': 'eth',
    'bitcoin': 'btc'
}

# =========================
# Building graph
# =========================

g =  nx.DiGraph()

edges = []

# loop thru each currency
for coin_name, rates in data.items():
    # What ticker symbol represents this coin?
    from_ticker = coin_mapping[coin_name]
    
    # loop thru each exchange rate for this coin
    for to_currency, rate in rates.items():
        edges.append((from_ticker, to_currency, rate))

g.add_weighted_edges_from(edges)

print(g.nodes) # list of all nodes
print("Number of nodes: ", len(g.nodes))
print("Number of edges: ", len(g.edges))

# =========================
# Traversing the Graph
# =========================

min_factor =            float('inf')
max_factor =            0
min_path_to =           None 
min_reverse_path =      None
max_path_to =           None
max_reverse_path =      None

# Going through each of the paths from one currency to the others
# combination function returns all possible currency pairs
# calculating path weight from one currency to another
# then back again
for n1, n2 in combinations(g.nodes,2):
    print("All paths from ", n1, "to", n2, "---------------")
    
    for path in nx.all_simple_paths(g, source=n1, target=n2):
        print("Path To", path)
    
        path_weight_to = 1.0
        # calculating the path weight from the first currency to the second
        # Iterating through all the edges in the path and multiplying them together
        # to get the weight of the entire path
        for i in range(len(path) - 1):
            path_weight_to *= g[path[i]][path[i+1]]['weight'] # multiplying each edge weight
    
        # Reversing the path, to calculate the path weight returning 
        # exchange rates, the paths to and from, in equilibrium multiplied together should be 1.0
        reverse_path = path[::-1]
        print("Path From", reverse_path)
        
        path_weight_from = 1.0
        # calculating the path weight from the second currency back to the first
        for i in range(len(path) - 1):
            path_weight_from *= g[reverse_path[i]][reverse_path[i+1]]['weight']

        factor = path_weight_from * path_weight_to
        print("factor: ", factor)

        if factor < min_factor:
            min_factor = factor
            min_path_to = path
            min_reverse_path = reverse_path
    
        if factor > max_factor:
            max_factor = factor
            max_path_to = path
            max_reverse_path = reverse_path
    
    print("---------------\n")
    
input("press enter")


# =========================
# Results
# =========================
print("Smallest Paths weight factor: ", min_factor)
print("Paths: ", min_path_to, min_reverse_path)
print("Greatest Paths weight factor: ", max_factor)
print("Paths: ", max_path_to, max_reverse_path)