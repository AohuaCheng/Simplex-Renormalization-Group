import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import copy
import scipy
from scipy.linalg import expm
import networkx as nx
from itertools import permutations

from configs.config_global import FIG_DIR
from analysis.plots import get_plot_path, get_plot_colors

class multi_LRG(object):
    def __init__(self, config, info, Adj):
        super(multi_LRG, self).__init__()
        # graph parameters
        self.dataset = config.dataset # the type of data, e.g. BA, neuro
        self.Adj = Adj
        self.N = Adj.shape[0]
        self.G = nx.Graph(Adj)
        # save path
        self.save_path = osp.join(config.save_path, '{}_{}'.format(info[0],info[1]))
        if not osp.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # LRG parameters
        self.tau_star = config.tau_star
        self.rg_steps = 5
        # saved data of order d
        self.RG_graphs = []
        self.RG_Ls = []
        # parameters for cliques
        self.d = [deg for (_, deg) in self.G.degree()]
        self.store = None
        self.cliques = []
        
    def reset(self):
        self.RG_graphs = []
        self.RG_Ls = []

    # Function to check if the given set of vertices in store array is a clique or not
    def is_clique(self, b):
        # Run a loop for all the set of edges for the select vertex
        for i in range(b):
            for j in range(i + 1, b):
    
                # If any edge is missing
                if (self.Adj[self.store[i]][self.store[j]] == 0):
                    return False
        return True
    
    # Function to find all the cliques of size s by Deep First Searching (DFS) method
    def findCliques(self, i, l, s):
        # Check if any vertices from i+1 can be inserted
        for j in range(i+1, self.N -(s - l)+1):
            # If the degree of the graph is sufficient
            if (self.d[j] >= s - 1):
                # Add the vertex to store
                self.store[l] = j 
                # If the graph is not a clique of size k, then it cannot be a clique by adding another edge
                if (self.is_clique(l + 1)):
                    # If the length of the clique is still less than the desired size
                    if (l < s-1):
                        # Recursion to add vertices
                        self.findCliques(j, l + 1, s)
                    # Size is met
                    else:
                        clique = self.store[:s]
                        self.cliques.append(clique)

    # Multiorder Laplacian functions
    def adj_matrix_of_order(self, d) : 
        """Returns the adjacency matrix of order d of t"""
        if d==1:
            adj_d = self.Adj
        else:
            adj_d = np.zeros_like(self.Adj)
            self.store = [0]*(d+1)
            self.findCliques(-1, 0, d+1)
            for clique in self.cliques :
                for [i_,j_] in permutations(clique, 2):
                    adj_d[i_,j_] += 1
        return adj_d

    def laplacian_of_order(self, d):
        
        Adj_d = self.adj_matrix_of_order(d)
        K_d = sum(Adj_d)
        L_d = np.diag(K_d) - Adj_d
        
        return L_d
    
    def ind2sub(self, array_shape, ind):
        rows = (ind.astype('int') // array_shape[1])
        cols = (ind.astype('int') % array_shape[1])
        return rows, cols

    def LRG(self, d):
        rg_G = copy.deepcopy(self.G)
        self.RG_graphs.append(copy.deepcopy(rg_G))
        rg_L = self.laplacian_of_order(d)
        self.RG_Ls.append(rg_L)
        
        for _ in range(self.rg_steps):
            # step1: 
            Evals, Evecs = np.linalg.eig(rg_L)
            Lsorted_idx = np.argsort(-Evals)
            SEvals = np.real(Evals[Lsorted_idx])
            SEvecs = np.real(Evecs[:, Lsorted_idx])
            factor = (d+1) / 2
            idx, = np.where(SEvals<1/(self.tau_star/factor))
            L_hat = np.zeros_like(rg_L)
            Nn = len(idx)
            for id in idx:
                L_hat = L_hat + SEvals[id]*np.outer(SEvecs[:, id],SEvecs[:, id].T)
            # step2: coarse grain
            K = expm(-self.tau_star*rg_L)
            Rho = K/np.trace(K)
            Abs_Rho = np.triu(Rho, k=1)
            Rhosort_idx = np.argsort(-Abs_Rho.reshape(-1))
            rows, cols = self.ind2sub([Rho.shape[0], Rho.shape[1]], Rhosort_idx)
            
            RefG = nx.Graph() # reference graph
            RefG.add_nodes_from(range(rg_L.shape[0]))
            RefG_com = list(nx.connected_components(RefG))
            for k in range(len(rows)):
                if len(RefG_com) == Nn:
                    break
                if rows[k]==cols[k]:
                    continue
                Node1, Node2 = rows[k], cols[k]
                RefG.add_edges_from([(Node1,Node2)])
                RefG_com = list(nx.connected_components(RefG))
            ToDelete = set([])
            for Nodes in RefG_com:
                Node_list = list(Nodes)
                if len(Node_list) == 1:
                    continue
                Node1 = Node_list[0]
                for node in Node_list[1:]:
                    Neighbors = list(rg_G.neighbors(node))
                    new_edges = [(Node1, Nei) for Nei in Neighbors if Node1!=Nei]
                    rg_G.add_edges_from(new_edges)
                    ToDelete.add(node)
            rg_G.remove_nodes_from(ToDelete)
            rg_G = nx.relabel.convert_node_labels_to_integers(rg_G) # a copy of the graph G with the nodes relabeled using consecutive integers.
            self.RG_graphs.append(copy.deepcopy(rg_G))
            # step3:
            new_Adj = np.zeros([Nn, Nn])
            new_L = np.zeros([Nn, Nn])
            for r, com_r in enumerate(RefG_com):
                for s, com_s in enumerate(RefG_com):
                    if s <= r:
                        continue
                    for m in com_r:
                        for n in com_s:
                            new_Adj[r,s] += -L_hat[m,n]
            new_Adj += new_Adj.T
            new_Deg = sum(new_Adj)
            new_L = np.diag(new_Deg) - new_Adj
            new_Evals, _ = np.linalg.eig(new_L)
            _, new_bins = np.histogram(np.real(new_Evals), bins=100, density=True)
            _, bins = np.histogram(np.real(Evals), bins=100, density=True)
            chi = bins[1] / new_bins[1]
            rg_L =  chi *new_L
            self.RG_Ls.append(rg_L)
        
        # save RG data
        nx.gpickle.write_gpickle(self.RG_graphs, osp.join(self.save_path, 'RG_graphs_d{}.gpickle'.format(d)))
        np.save(osp.join(self.save_path, 'RG_Ls_d{}.npy'.format(d)), self.RG_Ls, allow_pickle=True)
    
    def LRG_plot(self, config, info, d):
        fig_path = get_plot_path(config, info)
        sns.set(context='notebook', style='ticks', font_scale=1.5)

        RG_graphs = nx.gpickle.read_gpickle(osp.join(self.save_path, 'RG_graphs_d{}.gpickle'.format(d)))
        RG_Ls = np.load(osp.join(self.save_path, 'RG_Ls_d{}.npy'.format(d)), allow_pickle=True)
        
        # LRG graph plot
        for i in range(self.rg_steps+1):
            plt.figure(dpi=400)
            G_i = RG_graphs[i]
            L_i = RG_Ls[i]
            EdgeCV = range(len(list(G_i.edges())))
            nx.draw_networkx(G_i,pos=nx.random_layout(G_i),with_labels=False,node_size=100,node_color=np.diag(np.abs(L_i))/np.max(np.abs(L_i)+1)*255, cmap=mpl.colormaps['BuGn'], vmin=-80, edge_color=EdgeCV, edge_cmap=mpl.colormaps['Blues'], width=2,alpha=0.8,linewidths=1,edgecolors=[0,0.5,0.6])
            plt.savefig(osp.join(fig_path, 'LRG_graphs_{}_d_{}_s{}'.format(self.dataset, d, i)+'.png'), bbox_inches='tight', pad_inches=0)
            plt.close()

        # degree distribution
        colors = get_plot_colors(self.rg_steps+1)
        plt.figure(dpi=400)
        for i in range(self.rg_steps+1):
            G_i = RG_graphs[i]
            degree = nx.degree_histogram(G_i)
            x = range(len(degree))
            y = [z/float(sum(degree)) for z in degree]
            plt.scatter(x, y, s=100, color=colors[i], alpha=0.7, linewidths=2, edgecolors=[0,0.5,0.6],label='k={}'.format(i))
        plt.legend(loc='upper right')
        plt.xscale('log')
        plt.yscale('log')
        # plt.xlim([0,900])
        plt.xlabel('Degree')
        plt.ylabel('Probability')
        # plt.tight_layout()
        plt.savefig(osp.join(fig_path, 'LRG_degree_distribution_{}_d_{}'.format(self.dataset, d)+'.png'), bbox_inches='tight')
        plt.close()

        # mean connectivity flow
        colors = get_plot_colors(self.rg_steps+1, 'GnBu')
        plt.figure(dpi=400)
        mean_k = []
        for i in range(self.rg_steps+1):
            G_i = RG_graphs[i]
            degree = nx.degree_histogram(G_i)
            mean = sum([i*degree[i] for i in range(len(degree))])/sum(degree)
            mean_k.append(mean)
        x = range(self.rg_steps+1)
        plt.scatter(x, mean_k, color=colors[-1], s=100, alpha=0.7, linewidths=2, edgecolors=[0,0.5,0.6], label=r'$\tau^*$='+'{}'.format(self.tau_star))
        plt.plot(x, mean_k, color=colors[-1], linewidth=2)
        plt.legend(loc='upper right')
        plt.title('mean connectivity flow')
        plt.yticks(np.linspace(0, 3*round(max(mean_k)), 4))
        # plt.tight_layout()
        plt.savefig(osp.join(fig_path, 'LRG_mean_connectivity_flow_{}_d_{}'.format(self.dataset, d)+'.png'), bbox_inches='tight')
        plt.close()

        # spectral probability distribution
        colors = get_plot_colors(self.rg_steps+1, 'YlGnBu')
        plt.figure(dpi=400)
        for i in range(self.rg_steps+1):
            rg_L = RG_Ls[i]
            Evals, _ = np.linalg.eig(rg_L)
            Evals = np.real(Evals)
            Evals[np.where(Evals<1e-10)] = 0
            hists, bins = np.histogram(Evals, bins=50)
            x = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
            y = [z/float(sum(hists)) for z in hists]
            plt.scatter(x, y, s=100, color=colors[i], alpha=0.7,linewidths=2, edgecolors=[0,0.5,0.6])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Eigenvalue')
        plt.ylabel('Probability')
        plt.tight_layout()
        plt.savefig(osp.join(fig_path, 'LRG_spectral_probability_distribution_{}_d_{}'.format(self.dataset, d)+'.png'), bbox_inches='tight')
        plt.close()
    
    def LRG_condition_plot(self, config, trial_info, d, c):
        fig_path =  osp.join(FIG_DIR, config.experiment_name, config.dataset_name)

        colors = get_plot_colors(self.rg_steps+1)
        fig = plt.figure(dpi=400)
        fig.subplots_adjust(hspace=0.6, wspace=0.3)
        sns.set(context='notebook', style='ticks', font_scale=1.5)
        # degree distribution
        plt.subplot(1,2,1)
        for s in range(self.rg_steps+1):
            deg_list = []
            for info in trial_info:
                save_path = osp.join(config.save_path, '{}_{}'.format(info[0],info[1]))
                RG_graphs = nx.gpickle.read_gpickle(osp.join(save_path, 'RG_graphs_d{}.gpickle'.format(d)))
                deg = nx.degree_histogram(RG_graphs[s])
                deg_list.append(deg)
            max_deg = max([len(deg) for deg in deg_list])
            deg_array = np.zeros([len(deg_list), max_deg])
            for i, deg in enumerate(deg_list):
                for j in range(len(deg)):
                    deg_array[i, j] = deg[j]

            mean_deg = np.mean(deg_array, axis=0)
            x = range(mean_deg.shape[0])
            y = [z/float(sum(mean_deg)) for z in mean_deg]
            y = scipy.signal.savgol_filter(y,2,1)
            plt.scatter(x, y, s=100, color=colors[s], alpha=0.7, linewidths=2, edgecolors=[0,0.5,0.6])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Degree', fontsize=25)
        plt.ylabel('Probability', fontsize=25)

        # spectral probability distribution
        plt.subplot(1,2,2)
        for s in range(self.rg_steps+1):
            Evals_list = []
            for info in trial_info:
                save_path = osp.join(config.save_path, '{}_{}'.format(info[0],info[1]))
                RG_Ls = np.load(osp.join(save_path, 'RG_Ls_d{}.npy'.format(d)), allow_pickle=True)
                Evals, _ = np.linalg.eig(RG_Ls[s])
                Evals = np.real(Evals)
                Evals[np.where(Evals<1e-10)] = 0
                Evals_list.append(Evals)
            All_Evals = np.concatenate(Evals_list)
            hists, bins = np.histogram(All_Evals, bins=100)
            x = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
            y = [z/float(sum(hists)) for z in hists]
            plt.scatter(x, y, s=100, color=colors[s], alpha=0.7, linewidths=2, edgecolors=[0,0.5,0.6])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Eigenvalue', fontsize=25)
        # plt.ylabel('Probability', fontsize=25)
        plt.xticks([1e0,1.5e0,2e0])
        plt.tight_layout()
        plt.savefig(osp.join(fig_path, '{}_mean_LRG_degree_spectral_distribution_{}_d_{}'.format(info[0], config.dataset, d)+'.png'), bbox_inches='tight')
        plt.close()

        # legend plot
        plt.figure(dpi=400, figsize=(18,14))
        ax = plt.subplot(111)
        for s in range(self.rg_steps+1):
            x = [0.5]
            y = [0.5]
            plt.scatter(x,y, s=100, color=colors[s], alpha=0.7, linewidths=2, edgecolors=[0,0.5,0.6], label='k={}'.format(s))
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1),
              fancybox=True, shadow=True, ncol=3, fontsize=25)
        plt.xlim([0,10])
        plt.ylim([0,1])
        plt.savefig(osp.join(fig_path,'legend.png'))
        plt.close()
    
    def KS_statistics(self, cdf_list):
        KS = []
        for i in range(len(cdf_list)-1):
            for j in range(i+1, len(cdf_list)):
                cdf_i, cdf_j = cdf_list[i], cdf_list[j]
                min_len = min(cdf_i.shape[0], cdf_j.shape[0])
                if min_len ==0:
                    KS.append(1)
                else:
                    KS_ = [abs(cdf_i[l]-cdf_j[l]) for l in range(min_len)]
                    KS.append(max(KS_))
        return KS

    def LRG_allplot(self, config, data_info, d):
        fig_path =  osp.join(FIG_DIR, config.experiment_name, config.dataset_name)
        sns.set(context='notebook', style='ticks', font_scale=2)

        # mean degree violin plot
        colors = get_plot_colors(len(data_info)+20, 'GnBu')
        C_KS = []
        NC_KS = []
        C_labels = []
        NC_labels = []
        for trial_info in data_info:
            cdf_steps = []
            for s in range(self.rg_steps+1):
                deg_list= []
                for i, info in enumerate(trial_info):
                    save_path = osp.join(config.save_path, '{}_{}'.format(info[0],info[1]))
                    RG_graphs = nx.gpickle.read_gpickle(osp.join(save_path, 'RG_graphs_d{}.gpickle'.format(d)))
                    degree = nx.degree_histogram(RG_graphs[s])
                    deg_list.append(degree)
                max_deg = max([len(deg) for deg in deg_list])
                deg_array = np.zeros([len(deg_list), max_deg])
                for i, deg in enumerate(deg_list):
                    for j in range(len(deg)):
                        deg_array[i, j] = deg[j]
                mean_deg = np.mean(deg_array, axis=0)
                cdf = np.cumsum(mean_deg)/np.sum(mean_deg)
                cdf_steps.append(cdf)
            KS = self.KS_statistics(cdf_steps)
            idx, = np.where(np.array(KS)>1e-2)
            if len(idx) < 0.5*len(KS):
                C_KS.append(KS)
                C_labels.append(trial_info[0][0])
            else:
                NC_KS.append(KS)
                NC_labels.append(trial_info[0][0])
        
        all_KS = np.array(C_KS + NC_KS).T
        labels = C_labels + NC_labels
        
        plt.figure(dpi=400, figsize=(5,15))
        violin = plt.violinplot(all_KS, vert=False, widths=1)
        for c, patch in enumerate(violin['bodies']):
            patch.set_facecolor(colors[c+20])
            patch.set_alpha(1)
        plt.xlabel('KS statistic', fontsize=30)
        plt.ylabel('Brain regions', fontsize=30)
        plt.yticks(np.arange(1, 1*(len(labels) + 1), 1), labels=labels)
        plt.ylim(0.25, len(labels) + 0.75)
        plt.savefig(osp.join(fig_path, 'LRG_KS_violin_d_{}'.format(d)+'.png'), bbox_inches='tight')
        plt.close()

        # unaverage
        Labels = ['ORB','RSP'] # scale-free regions of order 1
        # Labels = ['APN','MRN','VPM', 'VISrl']  # scale-dependent regions of order 1
        
        colors = get_plot_colors(len(data_info)+20, 'GnBu')
        fig = plt.figure(dpi=400)
        fig.subplots_adjust(hspace=0.5, wspace=0.05)
        # degree distribution
        plt.subplot(1,2,1)
        c = 0
        for trial_info in data_info:
            if trial_info[0][0] not in C_labels:
                continue
            else:
                if trial_info[0][0] not in Labels:
                    c += 1
                    continue
            for s in range(self.rg_steps+1):
                deg_list = []
                for info in trial_info:
                    save_path = osp.join(config.save_path, '{}_{}'.format(info[0],info[1]))
                    RG_graphs = nx.gpickle.read_gpickle(osp.join(save_path, 'RG_graphs_d{}.gpickle'.format(d)))
                    deg = nx.degree_histogram(RG_graphs[s])
                    deg_list.append(deg)
                max_deg = max([len(deg) for deg in deg_list])
                deg_array = np.zeros([len(deg_list), max_deg])
                for i, deg in enumerate(deg_list):
                    for j in range(len(deg)):
                        deg_array[i, j] = deg[j]

                mean_deg = np.mean(deg_array, axis=0)
                x = range(mean_deg.shape[0])
                y = [z/float(sum(mean_deg)) for z in mean_deg]
                y = scipy.signal.savgol_filter(y,2,1)
                plt.scatter(x, y, s=50, color=colors[c+20], alpha=1, linewidths=0, edgecolors=[0,0.5,0.6])
            c += 1
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Degree', fontsize=35)
        plt.ylabel('Probability', fontsize=35)

        # spectral probability distribution
        plt.subplot(1,2,2)
        c = 0
        for trial_info in data_info:
            if trial_info[0][0] not in C_labels:
                continue
            else:
                if trial_info[0][0] not in Labels:
                    c += 1
                    continue
            for s in range(self.rg_steps+1):
                Evals_list = []
                for info in trial_info:
                    save_path = osp.join(config.save_path, '{}_{}'.format(info[0],info[1]))
                    RG_Ls = np.load(osp.join(save_path, 'RG_Ls_d{}.npy'.format(d)), allow_pickle=True)
                    Evals, _ = np.linalg.eig(RG_Ls[s])
                    Evals = np.real(Evals)
                    Evals[np.where(Evals<1e-10)] = 0
                    Evals_list.append(Evals)
                All_Evals = np.concatenate(Evals_list)
                hists, bins = np.histogram(All_Evals, bins=20)
                x = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
                y = [z/float(sum(hists)) for z in hists]
                plt.scatter(x, y, s=50, color=colors[c+20], alpha=1, linewidths=0, edgecolors=[0,0.5,0.6])
            c += 1
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Eigenvalue', fontsize=35)
        plt.ylabel('Probability', fontsize=35)
        plt.tight_layout()
        plt.savefig(osp.join(fig_path, 'CriticalSF_LRG_degree_spectral_distribution_d_{}'.format(d)+'.png'), bbox_inches='tight', pad_inches=0.4)
        plt.close()

        # average
        colors = get_plot_colors(self.rg_steps+1)
        fig = plt.figure(dpi=400)
        fig.subplots_adjust(hspace=0.5, wspace=0.05)
        # degree distribution
        plt.subplot(1,2,1)
        for s in range(self.rg_steps+1):
            deg_list = []
            for trial_info in data_info:
                if trial_info[0][0] not in Labels:
                    continue
                for info in trial_info:
                    save_path = osp.join(config.save_path, '{}_{}'.format(info[0],info[1]))
                    RG_graphs = nx.gpickle.read_gpickle(osp.join(save_path, 'RG_graphs_d{}.gpickle'.format(d)))
                    deg = nx.degree_histogram(RG_graphs[s])
                    deg_list.append(deg)
            max_deg = max([len(deg) for deg in deg_list])
            deg_array = np.zeros([len(deg_list), max_deg])
            for i, deg in enumerate(deg_list):
                for j in range(len(deg)):
                    deg_array[i, j] = deg[j]

            mean_deg = np.mean(deg_array, axis=0)
            x = range(mean_deg.shape[0])
            y = [z/float(sum(mean_deg)) for z in mean_deg]
            y = scipy.signal.savgol_filter(y,2,1)
            plt.scatter(x, y, s=50, color=colors[s], alpha=0.7, linewidths=2, edgecolors=[0,0.5,0.6])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Degree', fontsize=35)
        plt.ylabel('Probability', fontsize=35)

        # spectral probability distribution
        plt.subplot(1,2,2)
        for s in range(self.rg_steps+1):
            Evals_list = []
            for trial_info in data_info:
                if trial_info[0][0] not in Labels:
                    continue
                for info in trial_info:
                    save_path = osp.join(config.save_path, '{}_{}'.format(info[0],info[1]))
                    RG_Ls = np.load(osp.join(save_path, 'RG_Ls_d{}.npy'.format(d)), allow_pickle=True)
                    Evals, _ = np.linalg.eig(RG_Ls[s])
                    Evals = np.real(Evals)
                    Evals_list.append(Evals)
            All_Evals = np.concatenate(Evals_list)
            hists, bins = np.histogram(All_Evals, bins=20)
            x = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
            y = [z/float(sum(hists)) for z in hists]
            plt.scatter(x, y, s=50, color=colors[s], alpha=0.7, linewidths=2, edgecolors=[0,0.5,0.6])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Eigenvalue', fontsize=35)
        plt.ylabel('Probability', fontsize=35)
        plt.tight_layout()
        plt.savefig(osp.join(fig_path, 'CriticalSF_mean_LRG_degree_spectral_distribution_d_{}'.format(d)+'.png'), bbox_inches='tight', pad_inches=0.4)
        plt.close()