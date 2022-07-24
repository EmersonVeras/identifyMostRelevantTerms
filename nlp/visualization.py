import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def plot_graph(new_df, sum_near_words_list):
    """
    Plots a graph showing connections between the terms and their relative weigths.

    :param new_df: DataFrame with all terms TF
    :param sum_near_words_list: Dict mapping the most important terms to their nearest neighbors.

    """

    edges = []
    for _, row in new_df.iterrows():
        if row['Words'] in sum_near_words_list:
            for near_word in sum_near_words_list[row['Words']]:
                edges.append([row['Words'], row['tf_idf'],near_word])

    df_edges = pd.DataFrame.from_records(edges, columns=['word', 'tf_idf', 'term'])

    g = nx.from_pandas_edgelist(df_edges,source='word',target='term')
    plt.figure(figsize=(30, 30))
    cmap = plt.cm.coolwarm
    colors = [n for n in range(len(g.nodes()))]
    k = 0.14
    pos=nx.spring_layout(g, k=k)
    nodes_size = []
    for node in g.nodes:
        nsize = 100
        if node in new_df['Words'].values:
            nsize = (new_df[new_df['Words']==node].iloc[0]['tf'] ) * 150000
        nodes_size.append(nsize)
    
    nx.draw_networkx(g,pos, node_size=nodes_size, cmap = cmap, 
                    node_color=colors, edge_color='grey', font_size=15, alpha=0.8)
    plt.title("Network diagram of Top Terms", fontsize=16)
    plt.show()
