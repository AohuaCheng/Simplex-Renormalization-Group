# Simplex Renormalization Group

The toolbox for simplex path integral and simplex renormalization group for high-order interactions.

## Main function and usage of the SRG framework

In application, we have a system, **G**, to process. For the SRG, we need to ensure that **G** is a graph object in the $\mathbf{networkx}$ library. The SRG renormalizes the system on the $p$-order based on $q$-order interactions ($p\leq q$), where $p$ and $q$ can be defined by specifying **p** and **q**. Meanwhile, to select the type of the Laplacian representation (i.e., the Multi-order Laplacian operator or the high-order path Laplacian), we need to specify **L\_Type** in the program. Given these settings, the SRG runs on the system for **IterNum** iterations to generate a renormalization flow. 

```
def SRG_Flow(G,q,p,L_Type,IterNum):
    A=nx.adjacency_matrix(G).toarray()       
    Lq=HighOrderLaplician(A, L_Type, Order=q)
    Lp=HighOrderLaplician(A, L_Type, Order=p)
    Lq_List,Lp_List,Gq_List,Gp_List,C_List,Tracked_Alignment=SRG_Function(Lq,Lp,q,IterNum)
    return  Lq_List,Lp_List,Gq_List,Gp_List,C_List,Tracked_Alignment
```

The main function of the SRG generates four outputs after computation. The first two outputs, \textbf{Lq\_List} and \textbf{Lp\_List}, are the lists of operator $\mathbf{L}^{\left(q\right)}_{k}$ and operator $\mathbf{L}^{\left(p\right)}_{k}$, respectively. For instance, the first element of \textbf{L\_List} is $\mathbf{L}^{\left(q\right)}_{1}$, the second element is $\mathbf{L}^{\left(q\right)}_{2}$, and so on. The number of elements in \textbf{Lq\_List} and \textbf{Lp\_List} is determined by \textbf{IterNum}.

The third and forth outputs, \textbf{Gq\_List} and \textbf{Gp\_List}, are the lists of operator $\mathbf{G}^{\left(q\right)}_{k}$ and operator $\mathbf{G}^{\left(p\right)}_{k}$, respectively. For instance, the first element of \textbf{Gq\_List} is $\mathbf{G}^{\left(q\right)}_{1}$, the second element is $\mathbf{G}^{\left(q\right)}_{2}$, and so on. The number of elements in \textbf{Gq\_List} and \textbf{Gp\_List} is determined by \textbf{IterNum}.

The fifth output, \textbf{C\_List}, is the list of specific heat $X_{1}^{\left(q\right)}$ vector calculated by the initial $q$-order Laplacian, \textbf{Lq\_List[0]}. The number of specific heat vector in \textbf{C\_List} is determined by the number of connected clusters in \textbf{Lq\_List[0]}. For instance, in the ergodicity case, \textbf{C\_List} contains only one element, which is a vector of specific heat derived on the only one connected cluster. 

The last output of the main function is **Tracked\_Alignment**, which indicates the indexes of the initial units aggregated into each macro-unit in every connected cluster after the $k$-th iteration of the SRG. Below, we present a simple instance where system **G** contains only six units and satisfies the ergodicity. 

```
Tracked_Alignment[0]=[[[0, [0, 1, 3]], [2, [2]], [4, [4]], [5, [5]]]]
Tracked_Alignment[1]=[[[0, [0, 5]], [2, [2]], [4, [4]]]]
```

OBefore renormalization, each macro-unit only contains itself (i.e., the initial unit), which can be represented by a structured list [[[0, [0]], [1, [1]], [2, [2]], [3, [3]], [4, [4]], [5, [5]]]] in the form of [macro-unit, list of the units aggregated into this macro-unit]. Note that this trivial list is not included in **Tracked\_Alignment** for convenience. After two iterations of renormalization, **Tracked\_Alignment** is a list of two elements. As shown in the instance presented above, the first element of **Tracked\_Alignment** is [[0, [0, 1, 3]], [2, [2]], [4, [4]], [5, [5]]]. This list means that there remain four macro-units after the first iteration. The first macro-unit, 0, is formed by three initial units, 0, 1, and 3. Initial units 2, 4, and 5 are not coarse grained with other units during the first iteration so they only contain themselves. The second element of is [[[0, [0, 5]], [2, [2]], [4, [4]]]], suggesting that there exist three macro-units after the second iteration. The first macro-unit is derived by grouping 0 and 5, two macro-units generated in the first iteration, together. Consequently, this macro-unit contains four initial units, whose indexes are 0, 1, 3, and 5. Other elements of **Tracked\_Alignment** can be understood in a similar way. In the non-ergodic case, **Tracked\_Alignment**$[k][i][j]$ contains the indexes of the units aggregated into the $j$-th macro-unit in the $i$-th connected cluster after $\left(k+1\right)$-th iteration.

To run the SRG, one can consider the following instances:
```
G=nx.random_graphs.barabasi_albert_graph(1000,3) # Generate a random BA network with 1000 units

# Multiorder Laplacian operator
Lq_List,Lp_List,Gq_List,Gp_List,C_List,Tracked_Alignment=SRG_Flow(G,q=1,p=1,L_Type='MOL',IterNum=5) # Run a SRG for 5 iterations, which renormalize the system on the 1-order based on the 1-order interactions

Lq_List,Lp_List,Gq_List,Gp_List,C_List,Tracked_Alignment=SRG_Flow(G,q=2,p=1,L_Type='MOL',IterNum=5) # Run a SRG for 5 iterations, which renormalize the system on the 1-order based on the 2-order interactions

Lq_List,Lp_List,Gq_List,Gp_List,C_List,Tracked_Alignment=SRG_Flow(G,q=3,p=1,L_Type='MOL',IterNum=5) # Run a SRG for 5 iterations, which renormalize the system on the 1-order based on the 3-order interactions

# High-order path Laplacian
Lq_List,Lp_List,Gq_List,Gp_List,C_List,Tracked_Alignment=SRG_Flow(G,q=1,p=1,L_Type='HOPL',IterNum=5) # Run a SRG for 5 iterations, which renormalize the system on the 1-order based on the 1-order interactions

Lq_List,Lp_List,Gq_List,Gp_List,C_List,Tracked_Alignment=SRG_Flow(G,q=2,p=1,L_Type='HOPL',IterNum=5) # Run a SRG for 5 iterations, which renormalize the system on the 1-order based on the 2-order interactions

Lq_List,Lp_List,Gq_List,Gp_List,C_List,Tracked_Alignment=SRG_Flow(G,q=3,p=1,L_Type='HOPL',IterNum=5) # Run a SRG for 5 iterations, which renormalize the system on the 1-order based on the 3-order interactions
```

## Environment

The SRG depends on several external libraries listed below. Users should prepare these libraries before using the SRG.

python = 3.8.3

```
conda create --name SRG python=3.8.3
conda activate SRG

# main packages
conda install copy
conda install numpy
conda install scipy
conda install networkx
conda install itertools
```

## Citation
If you find our repo useful in your research, please kindly consider cite:
```
@misc{cheng2024simplex,
      title={Simplex path integral and simplex renormalization group for high-order interactions}, 
      author={Aohua Cheng and Yunhui Xu and Pei Sun and Yang Tian},
      year={2024},
      eprint={2305.01895},
      archivePrefix={arXiv},
      primaryClass={cond-mat.stat-mech}
}
```
