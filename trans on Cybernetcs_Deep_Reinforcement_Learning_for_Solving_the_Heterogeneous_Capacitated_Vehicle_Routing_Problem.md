# IEEE Transactions on Cybernetics

## Deep Reinforcement Learning for Solving the Heterogeneous Capacitated Vehicle Routing Problem

**Jingwen Li**, **Yining Ma**, **Ruize Gao**, **Zhiguang Cao**, **Andrew Lim**, **Wen Song**, and **Jie Zhang**

### Abstract
Existing deep reinforcement learning (DRL)-based methods for solving the capacitated vehicle routing problem (CVRP) intrinsically cope with a homogeneous vehicle fleet, in which the fleet is assumed as repetitions of a single vehicle. Hence, their key to construct a solution solely lies in the selection of the next node (customer) to visit excluding the selection of vehicle.

However, vehicles in real-world scenarios are likely to be heterogeneous with different characteristics that affect their capacity (or travel speed), rendering existing DRL methods less effective. In this article, we tackle heterogeneous CVRP (HCVRP), where vehicles are mainly characterized by different capacities. We consider both min–max and min–sum objectives for HCVRP, which aim to minimize the longest or total travel time of the vehicle(s) in the fleet.

To solve those problems, we propose a DRL method based on the attention mechanism with a vehicle selection decoder accounting for the heterogeneous fleet constraint and a node selection decoder accounting for the route construction, which learns to construct a solution by automatically selecting both a vehicle and a node for this vehicle at each step.

Experimental results based on randomly generated instances show that, with desirable generalization to various problem sizes, our method outperforms the state-of-the-art DRL method and most of the conventional heuristics, and also delivers competitive performance against the state-of-the-art heuristic method, that is, slack induction by string removal. In addition, the results of extended experiments demonstrate that our method is also able to solve CVRPLib instances with satisfactory performance.

---

### Manuscript Information
- **Manuscript received**: January 7, 2021  
- **Revised**: June 28, 2021  
- **Accepted**: September 2, 2021  

This work was supported in part by:
- The National Natural Science Foundation of China under Grant 61803104 and Grant 62102228
- The Young Scholar Future Plan of Shandong University under Grant 62420089964188

This article was recommended by Associate Editor D. Zhao. (Corresponding author: Zhiguang Cao.)

---

### Authors' Affiliations

- **Jingwen Li** and **Yining Ma** are with the Department of Industrial Systems Engineering and Management, National University of Singapore, Singapore  
  - Email: `lijingwen@u.nus.edu`, `yiningma@u.nus.edu`

- **Ruize Gao** is with the Department of Computer Science and Engineering, Chinese University of Hong Kong, Hong Kong  
  - Email: `ruizegao@cuhk.edu.hk`

- **Zhiguang Cao** is with the Manufacturing System Division, Singapore Institute of Manufacturing Technology, Singapore  
  - Email: `zhiguangcao@outlook.com`

- **Andrew Lim** is with the School of Computing and Artificial Intelligence, Southwest Jiaotong University, Chengdu 611756, China  
  - Email: `i@limandrew.org`

- **Wen Song** is with the Institute of Marine Science and Technology, Shandong University, Jinan 266237, China  
  - Email: `wensong@email.sdu.edu.cn`

- **Jie Zhang** is with the School of Computer Science and Engineering, Nanyang Technological University, Singapore  
  - Email: `zhangj@ntu.edu.sg`

---

### Additional Information

This article has supplementary material provided by the authors and color versions of one or more figures available at:
[https://doi.org/10.1109/TCYB.2021.3111082](https://doi.org/10.1109/TCYB.2021.3111082)

**Digital Object Identifier**: 10.1109/TCYB.2021.3111082

---

### Index Terms
- Deep reinforcement learning (DRL)
- Heterogeneous CVRP (HCVRP)
- Min–max objective
- Min–sum objective

---

## I. INTRODUCTION

The capacitated vehicle routing problem (CVRP) is a classical combinatorial optimization problem, which aims to optimize the routes for a fleet of vehicles with capacity constraints to serve a set of customers with demands.

Compared with the assumption of multiple identical vehicles in homogeneous CVRP, the settings of vehicles with different capacities (or speeds) are more in line with the real-world practice, which leads to the heterogeneous CVRP (HCVRP) [1], [2]. According to the objectives, CVRP can also be classified as min–max and min–sum ones, respectively.

The former objective requires that the longest (worst-case) travel time (or distance) for a vehicle in the fleet should be as satisfying as possible since fairness is crucial in many real-world applications [3]–[9], and the latter objective aims to minimize the total travel time (or distance) incurred by the whole fleet [10]–[13]. In this article, we study the problem of HCVRP with both min–max and min–sum objectives, that is, MM-HCVRP and MS-HCVRP.

Conventional methods for solving HCVRP include exact and heuristic ones. Exact methods usually adopt branch-and-bound or its variants as the framework and perform well on small-scale problems [10], [11], [14], [15], but may consume a prohibitively long time on large-scale ones given the exponential computation complexity. Heuristic methods usually exploit certain hand-engineered searching rules to guide the solving processes, which often consume much shorter time and are more desirable for large-scale problems in reality [12], [16]–[18]. However, such hand-engineered rules largely rely on human experience and domain knowledge, thus might be incapable of engendering solutions with high quality. Moreover, both conventional exact and heuristic methods always solve the problem instances independently, and fail to exploit the patterns that are potentially shared among the instances.

Recently, researchers tend to apply deep reinforcement learning (DRL) to automatically learn the searching rules in heuristic methods for solving routing problems, including CVRP and traveling salesman problem (TSP) [19]–[24], by discovering the underlying patterns from a large number of instances. Generally, those DRL models are categorized into...

---

# IEEE Transactions on Cybernetics

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.

## Two Classes of Methods

There are two classes, that is:
1) **Construction methods**: Starting with an empty solution, the former constructs a solution by sequentially assigning each customer to a vehicle until all customers are served.
2) **Improvement methods**: Starting with a complete initial solution, the latter selects either candidate nodes (customers or depot) or heuristic operators, or both to improve and update the solution at each step, which are repeated until termination.

By further leveraging the advanced deep-learning architectures like attention mechanism to guide the selection, those DRL models are able to efficiently generate solutions with much higher quality compared to conventional heuristics.

However, existing works only focus on solving homogeneous CVRP which intrinsically cope with vehicles of the same characteristics, in the sense that the complete route of the fleet could be derived by repeatedly dispatching a single vehicle.

Consequently, the key in those works is to solely select the next node to visit excluding the selection of vehicles, since there is only one vehicle essentially. Evidently, those works would be far less effective when applied to solve the more practical HCVRP, given the following issues:

1. The assumption of homogeneous vehicles is unable to capture the discrepancy in vehicles.
2. The vehicle selection is not explicitly considered, which should be of equal importance to the node selection in HCVRP.
3. The contextual information in the attention scheme is insufficient as it lacks states of other vehicles and (partially) constructed routes, which may render it incapable of engendering high-quality solutions in view of the complexity of HCVRP.

In this article, we aim to solve the HCVRP with both min–sum and min–max objectives while emphasizing on addressing the aforementioned issues. We propose a novel neural architecture integrated with the attention mechanism to improve the DRL-based construction method, which combines the decision making for vehicle selection and node selection together to engender solutions of higher quality.

Different from the existing works that construct the routes for each vehicle of the homogeneous fleet in sequence, our policy network is able to automatically and flexibly select a vehicle from a heterogeneous fleet at each step. In specific, our policy network adopts a Transformer-style [25] encoder–decoder structure, where the decoder consists of two parts, that is:
1. **Vehicle selection decoder**
2. **Node selection decoder**

With the problem features (i.e., customer location, customer demand, and vehicle capacity) processed by the encoder for better representation, the policy network first selects a vehicle from the fleet using the vehicle selection decoder based on the states of all vehicles and partial routes and then selects a node for this vehicle using the node selection decoder at each decoding step. This process is repeated until all customers are served.

Accordingly, the major contribution of this article is that we present a DRL method to solve CVRP with multiple heterogeneous vehicles, which is intrinsically different from the homogeneous ones in existing works, as the latter is lacking in selecting vehicles from a fleet. Specifically, we propose an effective neural architecture that integrates the vehicle selection and node selection together, with rich contextual information for selection among the heterogeneous vehicles, where every vehicle in the fleet has the chance to be selected at each step.

We test both min–max and min–sum objectives with various sizes of vehicles and customers. Results show that our method is superior to most of the conventional heuristics and competitive to the state-of-the-art heuristic [i.e., slack induction by string removal (SISR)] with a much shorter computation time. With comparable computation time, our method achieves much better solution quality than that of other DRL methods. In addition, our method generalizes well to problems with larger customer sizes.

The remainder of this article is organized as follows.  
**Section II** briefly reviews conventional methods and deep models for routing problems.  
**Section III** introduces the mathematical formulation of MM-HCVRP and MS-HCVRP and the reformulation in the reinforcement learning (RL) manner.  
**Section IV** elaborates our DRL framework.  
**Section V** provides the computational experiments and analysis. Finally,  
**Section VI** concludes this article and presents future works.

## II. RELATED WORKS

In this section, we briefly review the conventional methods for solving HCVRP with different objective functions, and deep models for solving the general VRPs.

The HCVRP was first studied in [1], where the Clarke and Wright procedure and partition algorithms were applied to generate the lower bound and estimate optimal solution. An efficient constructive heuristic was adopted to solve HCVRP in [26] by merging small start trips for each customer into a complete one, which was also capable for multitrip cases.

Baldacci and Mingozzi [14] presented a unified exact method to solve HCVRP, reducing the number of variables by using three bounding procedures. Feng et al. [27] proposed a novel evolutionary multitasking algorithm to tackle the HCVRP with a time window, and occasional driver, which can also solve multiple optimization tasks simultaneously.

The CVRP with min–sum objective was first proposed by Dantzig and Ramser [28], which was assumed as the generalization of TSP with capacity constraints. To address the large-scale multiobjective optimization problem (MOP), a competitive swarm optimizer (CSO)-based search method was proposed in [29]–[31], which conceived a new particle updating strategy to improve the search accuracy and efficiency.

By transforming the large-scale CVRP (LSCVRP) into a large-scale MOP, an evolutionary multiobjective route grouping method was introduced in [32], which employed a multiobjective evolutionary algorithm to decompose the LSCVRP into small tractable subcomponents. The min–max objective was considered in a multiple TSP [15], which was solved by a tabu search heuristic and two exact search schemes.

An ant colony optimization (ACO) method was proposed to address the min–max single depot CVRP (SDCVRP) [33]. The problem was further extended to the min–max multidepot CVRP [34], which could be reduced to SDCVRP using an equitable region partitioning approach. A swarm intelligence-based heuristic algorithm was presented to...

---

# This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.

## Liet al. : DRL FOR SOLVING THE HCVRP 3

- Address the rich min–max CVRP [18].
- The min–max cumulative CVRP, aiming to minimize the last arrival time at customers, was first studied in [35] and [36], where a two-stage adaptive variable neighborhood search (AVNS) algorithm was introduced and also tested in min–sum objective to verify generalization.

The first deep model for routing problems is Pointer Network, which used supervised learning to solve TSP [37] and was later extended to RL [19]. Afterward, the Pointer Network was adopted to solve CVRP in [21], where the Recurrence Neural-Network architecture in the encoder was removed to reduce computation complexity without degrading solution quality. To further improve the performance, a Transformer-based architecture was incorporated by integrating self-attention in both the encoder and decoder [22].

Different from the above methods which learn constructive heuristics, NeuRewriter was proposed to learn how to pick the next solution in a local search framework [23]. Despite their promising results, these methods are less effective for tackling the heterogeneous fleet in HCVRP. Recently, some learning-based methods have been proposed to solve HCVRP.

Inspired by multiagent RL, Vera and Abad [38] made the first attempt to solve the min–sum HCVRP through cooperative actions of multiple agents for route construction. Qin et al. [39] proposed an RL-based controller to select among several meta-heuristics with different characteristics to solve min–sum HCVRP. Although yielding better performance than conventional heuristics, they are unable to well handle either the min–max objective or heterogeneous speed of vehicles.

## III. PROBLEM FORMULATION

In this section, we first introduce the mathematical formulation of HCVRP with both min–max and min–sum objectives, and then reformulate it as the form of RL.

### A. Mathematical Formulation of HCVRP

Particularly, with n+1 nodes (customers and depot) represented as X={x_i}^n_{i=0} and node x_0 denoting the depot, the customer set is assumed to be X/prime=X\{x_0}. Each node x_i∈R^3 is defined as {(s_i,d_i)}, where s_i contains the 2-dim location coordinates of node x_i, and d_i refers to its demand (the demand for depot is 0). Here, we take heterogeneous vehicles with different capacities into account, which respects the real-world situations. Accordingly, let V={v_i}^m_{i=1} represent the heterogeneous fleet of vehicles, where each element v_i is defined as {(Q_i)}, that is, the capacity of vehicle v_i. The HCVRP problem describes a process that all fully loaded vehicles start from the depot, and sequentially visit the locations of customers to satisfy their demands, with the constraints that each customer can be visited exactly once, and the loading amount for a vehicle during a single trip can never exceed its capacity.

Let D(x_i,x_j) be the Euclidean distance between x_i and x_j. Let y^v_{ij} be a binary variable, which equals to 1 if vehicle v travels directly from customer x_i to x_j, and 0 otherwise. Let l^v_{ij} be the remaining capacity of the vehicle v before traveling from customer x_i to customer x_j. For simplification, we assume that all vehicles have the same speed f, which could be easily extended to take different values. Then, the MM-HCVRP could be naturally defined as follows:

$$
\min \max_{v \in V} \left( \sum_{i \in X} \sum_{j \in X} \frac{D(x_i,x_j)}{f} y^v_{ij} \right) \quad (1)
$$

subject to the following six constraints:

$$
\sum_{v \in V} \sum_{j \in X} y^v_{ij} = 1, \quad i \in X/prime \quad (2)
$$

$$
\sum_{i \in X} y^v_{ij} - \sum_{k \in X} y^v_{jk} = 0, \quad v \in V, j \in X/prime \quad (3)
$$

$$
\sum_{v \in V} \sum_{i \in X} l^v_{ij} - \sum_{v \in V} \sum_{k \in X} l^v_{jk} = d_j, \quad j \in X/prime \quad (4)
$$

$$
d_j y^v_{ij} \leq l^v_{ij} \leq \left(Q^v - d_i \right) \cdot y^v_{ij}, \quad v \in V, i \in X, j \in X \quad (5)
$$

$$
y^v_{ij} = \{0,1\}, \quad v \in V, i \in X, j \in X \quad (6)
$$

$$
l^v_{ij} \geq 0, \quad d_i \geq 0, \quad v \in V, i \in X, j \in X \quad (7)
$$

The objective of the formulation is to minimize the maximum travel time among all vehicles. Constraints (2) and (3) ensure that each customer is visited exactly once and each route is completed by the same vehicle. Constraint (4) guarantees that the difference between the amount of goods loaded by a vehicle before and after serving a customer equals the demand of that customer. Constraint (5) enforces that the amount of goods for any vehicle is able to meet the demands of the corresponding customers and never exceed its capacity. Constraint (6) defines the binary variable and constraint (7) imposes the non-negativity of the variables.

The MS-HCVRP shares the same constraints with MM-HCVRP, while the objective is formulated as follows:

$$
\min \sum_{v \in V} \sum_{i \in X} \sum_{j \in X} \frac{D(x_i,x_j)}{f^v} y^v_{ij} \quad (8)
$$

where f^v represents the speed of vehicle v, and it may vary with different vehicles. Thereby, it is actually minimizing the total travel time incurred by the whole fleet.

### B. Reformulation as RL Form

RL was originally proposed for sequential decision-making problems, such as self-driving cars, robotics, games, etc. [40]–[45]. The construction of routes for HCVRP step by step can be also deemed as a sequential decision-making problem. In our work, we model such process as a Markov decision process (MDP) [46] defined by 4-tuple M={S,A,τ,r}. (An example of the MDP is illustrated in the supplementary material.) Meanwhile, the detailed definition of the state space S, the action space A, the state transition rule τ, and the reward function r is introduced as follows.

**State**: In our MDP, each state s_t=(V_t,X_t)∈S consists of two parts. The first part is the vehicle state V_t, which is expressed as V_t={ v^1_t,v^2_t,..., v^m_t}={(o^1_t,T^1_t,G^1_t),(o^2_t,T^2_t,G^2_t),...,(o^m_t,T^m_t,G^m_t)}, where o^i_t and T^i_t represent the remaining capacity and the accumulated travel time of the vehicle v_i at step t, respectively. G^i_t={g^i_0,g^i_1,...,g^i_t} represents the partial route of the vehicle v_i

---

# IEEE Transactions on Cybernetics

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.

## 4 IEEE TRANSACTIONS ON CYBERNETICS

At step t, where $ g_i^j $ refers to the node visited by the vehicle $ v_i $ at step j. Note that the dimension of partial routes (the number of nodes in a route) for all vehicles keeps the same, that is, if the vehicle $ v_i $ is selected to serve the node $ x_j $ at step t, other vehicles still select their last served nodes. Upon departure from the depot (i.e., t=0), the initial vehicle state is set to $ V_0 = \{(Q_1, 0, \{0\}), (Q_2, 0, \{0\}), ..., (Q_m, 0, \{0\})\} $, where $ Q_i $ is the maximum capacity of vehicle $ v_i $. The second part is the node state $ S_t $, which is expressed as $ X_t = \{x_0^t, x_1^t, ..., x_n^t\} = \{(s_0, d_0^t), (s_1, d_1^t), ..., (s_n, d_n^t)\} $, where $ s_i $ is a 2-dim vector representing the locations of the node, and $ d_i^t $ is a scalar representing the demand of node i ($ d_i^t $ will become 0 once that node has been served). Here, we do not consider demand splitting, and only nodes with $ d_i > 0 $ need to be served.

### Action

The action in our method is defined as selecting a vehicle and a node (a customer or the depot) to visit. In specific, the action $ a_t \in A $ is represented as $ (v_i^t, x_j^t) $, that is, the selected node $ x_j $ will be served (or visited) by the vehicle $ v_i $ at step t. Note that only one vehicle is selected at each step.

### Transition

The transition rule $ \tau $ will transit the previous state $ s_t $ to the next state $ s_{t+1} $ based on the performed action $ a_t = (v_i^t, x_j^t) $, that is, $ s_{t+1} = (V_{t+1}, X_{t+1}) = \tau(V_t, X_t) $. The elements in vehicle state $ V_{t+1} $ are updated as follows:

$$
ok_{t+1} = 
\begin{cases}
ok_t - d_j^t, & \text{if } k = i \\
ok_t, & \text{otherwise}
\end{cases}
$$

$$
Tk_{t+1} = 
\begin{cases}
Tk_t + D(g_k^t, x_j), & \text{if } k = i \\
Tk_t, & \text{otherwise}
\end{cases}
$$

$$
Gk_{t+1} = 
\begin{cases}
[G_k^t, x_j], & \text{if } k = i \\
[G_k^t, g_k^t], & \text{otherwise}
\end{cases}
$$

where $ g_k^t $ is the last element in $ G_k^t $, that is, last visited customer by vehicle $ v_k $ at step t, and [·, ·, ·] is the concatenation operator.

The element in node state $ X_{t+1} $ is updated as follows:

$$
dl_{t+1} = 
\begin{cases}
0, & \text{if } l = j \\
dl_t, & \text{otherwise}
\end{cases}
$$

where each demand will retain 0 after being visited.

### Reward

For the MM-HCVRP, to minimize the maximum travel time of all vehicles, the reward is defined as the negative value of this maximum, where the reward is calculated by accumulating the travel time of multiple trips for each vehicle, respectively. Accordingly, the reward is represented as $ R = - \max_{v \in V} \sum_{t=0}^{T} r_t $, where $ r_t $ is the incremental travel time for all vehicles at step t. Similarly, for the MS-HCVRP, the reward is defined as the negative value of the total travel time of all vehicles, that is, $ R = - \sum_{i=1}^{m} \sum_{t=1}^{T} r_t $. Particularly, assume that nodes $ x_j $ and $ x_k $ are selected at steps t and t+1, respectively, which are both served by the vehicle $ v_i $, then the reward $ r_{t+1} $ is expressed as an m-dim vector as follows:

$$
r_{t+1} = r(s_{t+1}, a_{t+1}) = r\left((V_{t+1}, X_{t+1}), \left(v_i^{t+1}, x_k^{t+1}\right)\right) = 
\begin{cases}
0, ..., 0, D(x_j, x_k)/f, 0, ..., 0
\end{cases}
$$

where $ D(x_j, x_k)/f $ is the time consumed by the vehicle $ v_i $ for traveling from node $ x_j $ to $ x_k $, with other elements in $ r(s_{t+1}, a_{t+1}) $ equal to 0.

![Fig. 1. Framework of our policy network. With raw features of the instance processed by the encoder, our policy network first selects a vehicle ( $ v_i^t $ ) using the vehicle selection decoder and then a node ( $ x_j^t $ ) using the node selection decoder for this vehicle to visit at each route construction step t. Both the selected vehicle and node constitute the action at that step, that is, $ a_t = (v_i^t, x_j^t) $, where the partial solution and state are updated accordingly. To a single instance, the encoder is executed once, while the vehicle and node selection decoders are executed multiple times to construct the solution.]

## IV. METHODOLOGY

In this section, we introduce our DRL-based approach for solving HCVRP with both min–max and min–sum objectives. We first propose a novel attention-based deep neural network to represent the policy, which enables both vehicle selection and node selection at each decision step. Then, we describe the procedure for training our policy network.

### A. Framework of Our Policy Network

In our approach, we focus on learning a stochastic policy $ \pi_\theta(a_t | s_t) $ represented by a deep neural network with trainable parameter $ \theta $. Starting from the initial state $ s_0 $, that is, an empty solution, we follow the policy $ \pi_\theta $ to construct the solution by complying with the MDP in Section III-B until the terminate state $ s_T $ is reached, that is, all customers are served by the whole fleet of vehicles. The T is possibly longer than n+1 due to the fact that sometimes vehicles need to return to the depot for replenishment. Accordingly, the joint probability of this process is factorized based on the chain rule as follows:

$$
p(s_T | s_0) = \prod_{t=0}^{T-1} \pi_\theta(a_t | s_t) p(s_{t+1} | s_t, a_t)
$$

where $ p(s_{t+1} | s_t, a_t) = 1 $ always holds since we adopt the deterministic state transition rule.

As illustrated in Fig. 1, our policy network $ \pi_\theta $ consists of an encoder, a vehicle selection decoder, and a node selection decoder. Since a given problem instance itself remains unchanged throughout the decision process, the encoder is executed only once at the first step (t=0) to simplify the computation, while its outputs could be reused in subsequent steps (t>0) for route construction. To solve the instance, with raw features processed by the encoder for better representation, our policy network first selects a vehicle ( $ v_i $ ) from the whole

Authorized licensed use limited to: Tsinghua University. Downloaded on September 02, 2022 at 01:31:15 UTC from IEEE Xplore. Restrictions apply.

---

# This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.

## LIet al. : DRL FOR SOLVING THE HCVRP 5

### Fig. 2. Architecture of our policy network with heterogeneous vehicles and n customers

It is worth noting that our vehicle selection decoder leverages the vehicle features (last node location and accumulated travel time), the route features (max pooling of the routes for m vehicles), and their combinations to compute the probability of selecting each vehicle.

- Fleet via the vehicle selection decoder and identify its index,
- Then selects a node (xj) for this vehicle to visit via the node selection decoder at each route construction step.
- The selected vehicle and node constitute the action for that step, which is further used to update the states.
- This process is repeated until all customers are served.

## B. Architecture of Our Policy Network

Originating from the field of natural language processing [25], the Transformer model has been successfully extended to many other domains, such as:

- Image processing [47], [48]
- Recommendation systems [49], [50]
- Vehicle routing problems [22], [51]

due to its desirable capability to handle sequential data. Rather than the sequential recurrent or convolutional structures, the Transformer mainly hinges on the self-attention mechanism to learn the relations between arbitrary two elements in a sequence, which allows more efficient parallelization and better feature extraction without the limitation of sequence-aligned recurrence.

Regarding the general vehicle routing problems, the input is a sequence of customers characterized by locations and demands, and the construction of routes could be deemed as a sequential decision making, where the Transformer has desirable potential to engender high-quality solutions with short computation time.

Especially, the Transformer-style models [22], [51] adopt an encoder–decoder structure, where the encoder aims to compute a representation of the input sequence based on the multi-head attention (MHA) mechanism for better feature extraction and the decoder sequentially outputs a customer at each step based on the problem-related contextual information until all customers are visited.

To solve the HCVRP with both min–max and min–sum objectives, we also propose a Transformer-style models as our policy network, which is designed as follows.

As depicted in Fig. 2, our policy network adopts an encoder–decoder structure and the decoder consists of two parts, that is:

1. Vehicle selection decoder
2. Node selection decoder

Based on the stipulation that any vehicle has the opportunity to be selected at each step, our policy network is able to search in a more rational and broader action space given the characteristics of HCVRP. Moreover, we enrich the contextual information for the vehicle selection decoder by adding the features extracted from all vehicles and existing (partial) routes. In doing so, it allows the policy network to capture the heterogeneous roles of vehicles, so that decisions would be made more effectively from a global perspective.

To better illustrate our method, an example of two instances with seven nodes and three vehicles is presented in Fig. 3. Next, we introduce the details of our encoder, vehicle selection decoder, and node selection decoder, respectively.

## Encoder

The encoder embeds the raw features of a problem instance (i.e., customer location, customer demand, and vehicle capacity) into a higher-dimensional space and then processes them through attention layers for better feature extraction.

- We normalize the demand $d_i^0$ of customer $x_i$ by dividing the capacity of each vehicle to reflect the differences of vehicles in the heterogeneous fleet, that is, $\tilde{x}_i = (s_i, d_i^0/Q_1, d_i^0/Q_2, \ldots, d_i^0/Q_m)$.
- Similar to the encoder of Transformer in [22] and [25], the enhanced node feature $\tilde{x}_i$ is then linearly projected to $h_i^0$ in a high-dimensional space with dimension dim = 128.
- Afterward, $h_i^0$ is further transformed to $h_i^N$ through N attention layers for better feature representation, each of which consists of an MHA sublayer and a feedforward (FF) sublayer.

The lth MHA sublayer uses a multihead self-attention network to process the node embeddings $h_l = (h_0^l, h_1^l, \ldots, h_n^l)$.

- We stipulate that $dim_k = (dim/Y)$ is the query/key dimension, $dim_v = (dim/Y)$ is the value dimension, and $Y = 8$ is the number of heads in the attention.
- The lth MHA sublayer first calculates the attention value $Z_{l,y}$ for each head $y \in \{1, 2, \ldots, Y\}$ and then concatenates all these heads and projects them into a new feature space which has the same.

---

# IEEE Transactions on Cybernetics

## Abstract

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.

---

## Figure 3

Illustration of our policy network for two instances with seven nodes and three vehicles, where the red frame indicates the two stacked instances with the same data structure.

Given the current state $ s_t $, the features of nodes and vehicles are processed through the encoder to compute the node embeddings and the graph embedding. In the vehicle selection decoder, the node embeddings of the three tours for three vehicles in current state $ s_t $, that is, $ \{{h_0,h_1},{h_0,h_3},{h_0,h_5}\} $, are processed for route feature extraction, and the current location and the accumulated travel time of three vehicles are processed for vehicle feature extraction, which are then concatenated to compute the probability of selecting a vehicle. With the selected vehicle $ v_1 $ in this example, the current node embedding $ h_1 $ and the current loading ability of this vehicle are first concatenated and linearly propagated, then added with the graph embedding, which is further used to compute the probability of selecting a node with masked softmax, that is, $ \bar{p}_1 = \bar{p}_3 = \bar{p}_5 = 0 $. With the selected node $ x_2 $ in this example, the action is represented as $ a_t = \{v_1,x_2\} $ and the state is updated and transited to $ s_{t+1} $.

---

## Attention Mechanism

The dimension as the input $ h_l $. Concretely, we show these steps as follows:

$$
Q_{l,y} = h_l W^Q_{l,y}, \quad K_{l,y} = h_l W^K_{l,y}, \quad V_{l,y} = h_l W^V_{l,y} \quad (15)
$$

$$
Z_{l,y} = \text{softmax}\left( \frac{Q_{l,y}K_{l,y}^T}{\sqrt{d_k}} \right) V_{l,y} \quad (16)
$$

$$
\text{MHA}(h_l) = \text{MHA}\left( h_l W^Q_{l}, h_l W^K_{l}, h_l W^V_{l} \right) = \text{Concat}(Z_{l,1}, Z_{l,2}, ..., Z_{l,Y}) W^O_l \quad (17)
$$

where $ W^Q_{l}, W^K_{l} \in \mathbb{R}^{Y \times d \times d_k}, W^V_{l} \in \mathbb{R}^{Y \times d \times d_v} $, and $ W^O_{l} \in \mathbb{R}^{d \times d} $ are trainable parameters in layer $ l $ and are independent across different attention layers.

Afterward, the output of the $ l $-th MHA sublayer is fed to the $ l $-th FF sublayer with the ReLU activation function to obtain the next embeddings $ h_{l+1} $. Here, a skip-connection [52] and a batch normalization (BN) layer [53] are used for both MHA and FF sublayers, which are summarized as follows:

$$
r^i_l = \text{BN}\left( h^i_l + \text{MHA}^i(h_l) \right) \quad (18)
$$

$$
h^{i}_{l+1} = \text{BN}\left( r^i_l + \text{FF}(r^i_l) \right) \quad (19)
$$

Finally, we define the final output of the encoder, that is, $ h^i_N $, as the node embeddings of the problem instance, and the mean of the node embeddings, that is, $ \hat{h}_N = \frac{1}{n} \sum_{i \in X} h^i_N $, as the graph embedding of the problem instance, which will be reused for multiple times in the decoders.

---

## Vehicle Selection Decoder

The vehicle selection decoder outputs a probability distribution for selecting a particular vehicle, which mainly leverages two embeddings, that is:  
1) **Vehicle feature embedding**  
2) **Route feature embedding**

### 1) Vehicle Feature Embedding

To capture the states of each vehicle at the current step, we define the vehicle feature context $ C^V_t \in \mathbb{R}^{1 \times 3m} $ at step $ t $ as follows:

$$
C^V_t = \left[ \tilde{g}^1_{t-1}, T^1_{t-1}, \tilde{g}^2_{t-1}, T^2_{t-1}, ..., \tilde{g}^m_{t-1}, T^m_{t-1} \right] \quad (20)
$$

where $ \tilde{g}^i_{t-1} $ denotes the 2-dim location of the last node $ g^i_{t-1} $ in the partial route of vehicle $ v_i $ at step $ t-1 $, and $ T^i_{t-1} $ is the accumulated travel time of vehicle $ v_i $ till step $ t-1 $. Afterward, the vehicle feature context is linearly projected with trainable parameters $ W_1 $ and $ b_1 $ and further processed by a 512-dim FF layer with the ReLU activation function to engender the vehicle feature embedding $ H^V_t $ at step $ t $ as follows:

$$
H^V_t = \text{FF}(W_1 C^V_t + b_1) \quad (21)
$$

### 2) Route Feature Embedding

Route feature embedding extracts information from existing partial routes of all vehicles, which helps the policy network intrinsically learn from the visited nodes in the previous steps, instead of simply masking them as did in previous works [19], [21], [22], [37]. For each vehicle $ v_i $ at step $ t $, we define its route feature context $ \tilde{C}^i_t $ as an arrangement of the node embeddings (i.e., $ h^k_N $ is the node embeddings for node $ x_k $), corresponding to the node in its partial route $ G^i_{t-1} $. Specifically, the route feature context $ \tilde{C}^i_t $ for each vehicle $ v_i $, $ i=1,2,..., m $, is defined as follows:

$$
\tilde{C}^i_t = \left[ \tilde{h}^i_0, \tilde{h}^i_1, ..., \tilde{h}^i_{t-1} \right] \quad (22)
$$

where $ \tilde{C}^i_t \in \mathbb{R}^{t \times d} $ (the first dimension is of size $ t $ since $ G^i_{t-1} $ should have $ t $ elements at step $ t $) and $ \tilde{h}^i_j $ represents the corresponding node embeddings in $ h_N $ of the $ j $-th node in partial route $ G^i_{t-1} $ of vehicle $ v_i $. For example, assume $ t=4 $ and the partial route of vehicle $ v_i $ is $ G^i_3 = \{x_0, x_3, x_3, x_1\} $, then the route feature context of this vehicle at step $ t=4 $ would be $ \tilde{C}^i_4 = [\tilde{h}^i_0, \tilde{h}^i_1, \tilde{h}^i_2, \tilde{h}^i_3] = [h^0_N, h^3_N, h^3_N, h^1_N] $. Afterward, the route feature context of all vehicles is aggregated by a max pooling and then concatenated to yield the route context $ \hat{C}^R_t $ for the whole fleet, which is further processed by a linear projection with trainable parameters $ W_2 $ and $ b_2 $ and a 512-dim FF layer to...

> *Authorized licensed use limited to: Tsinghua University. Downloaded on September 02, 2022 at 01:31:15 UTC from IEEE Xplore. Restrictions apply.*

---

# DRL for Solving the HCVRP

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.

## Route Feature Embedding

The route feature embedding $ HR_t $ is calculated as follows:

$$
\bar{C}_t = \max\left( \tilde{C}_t \right), \quad i=1,2,..., m \quad (23)
$$

$$
\hat{CR}_t = \left[ \bar{C}_1_t, \bar{C}_2_t, ..., \bar{C}_m_t \right] \quad (24)
$$

$$
HR_t = FF\left( W_2 \hat{CR}_t + b_2 \right) \quad (25)
$$

Finally, the vehicle feature embedding $ HV_t $ and route feature embedding $ HR_t $ are concatenated and linearly projected with parameters $ W_3 $ and $ b_3 $, which is further processed by a softmax function to compute the probability vector as follows:

$$
H_t = W_3 \left[ HV_t, HR_t \right] + b_3 \quad (26)
$$

$$
p_t = \text{softmax} (H_t) \quad (27)
$$

Where $ p_t \in \mathbb{R}^m $ and its element $ p_i^t $ represents the probability of selecting vehicle $ v_i $ at time step $ t $. Depending on different strategies, the vehicle can be selected either by retrieving the maximum probability greedily or sampling according to the vector $ p_t $. The selected vehicle $ v_i $ is then used as input to the node selection decoder.

## Node Selection Decoder

Given node embeddings from the encoder and the selected vehicle $ v_i $ from the vehicle selection decoder, the node selection decoder outputs a probability distribution $ \bar{p}_t $ over all unserved nodes (the nodes served in previous steps are masked), which is used to identify a node for the selected vehicle to visit. Similar to [22], we first define a context vector $ H_c^t $ as follows, and it consists of the graph embedding $ \hat{h}_N $, node embedding of the last (previous) node visited by the selected vehicle, and the remaining capacity of this vehicle:

$$
H_c^t = \left[ \hat{h}_N, \tilde{h}_i^{t-1}, o_i^t \right] \quad (28)
$$

Where the second element $ \tilde{h}_i^{t-1} $ has the same meaning as the one defined in (22) and is replaced with trainable parameters for $ t=0 $. The designed context vector highlights the features of the selected vehicle at the current decision step, with the consideration of graph embedding of the instance from the global perspective. The context vector $ H_c^t $ and the node embeddings $ h_N $ are then fed into an MHA layer to synthesis a new context vector $ \hat{H}_c^t $ as a glimpse of the node embeddings [54]. Different from the self-attention in the encoder, the query of this attention comes from the context vector, while the key/value of the attention comes from the node embeddings as shown in:

$$
\hat{H}_c^t = MHA \left( H_c^t W_Q^c, h_N W_K^c, h_N W_V^c \right) \quad (29)
$$

Where $ W_Q^c $, $ W_K^c $, and $ W_V^c $ are trainable parameters similar to (17). We then generate the probability distribution $ \bar{p}_t $ by comparing the relationship between the enhanced context $ \hat{H}_c^t $ and the node embeddings $ h_N $ through a Compatibility Layer. The compatibility of all nodes with context at step $ t $ is computed as follows:

$$
u_t = C \cdot \tanh \left( \frac{q_t^T k_t}{\sqrt{dim_k}} \right) \quad (30)
$$

Where $ q_t = \hat{H}_c^t W_Q^{comp} $ and $ k_t = h_N W_K^{comp} $ are trainable parameters, and $ C $ is set to 10 to control the entropy of $ u_t $. Finally, the probability vector is computed in (31) where all nodes visited in the previous steps are masked for feasibility and element $ \bar{p}_j^t $ represents the probability of selecting node $ x_j $ served by the selected vehicle $ v_i $ at step $ t $ as follows:

$$
\bar{p}_t = \text{softmax} (u_t) \quad (31)
$$

Similar to the decoding strategy of vehicle selection, the nodes could be selected by always retrieving the maximum $ \bar{p}_j^t $, or sampling according to the vector $ \bar{p} $ in a less greedy manner.

## Training Algorithm

The proposed DRL method is summarized in **Algorithm 1**, where we adopt the policy gradient with a baseline to train the policy of vehicle selection and node selection for route construction. The policy gradient is characterized by two networks: 1) the policy network, that is, the policy network $ \pi_\theta $ aforementioned, picks an action and generates probability vectors for both vehicles and nodes with respect to this action at each decoding step and 2) the baseline network $ v_\phi $, a greedy roll-out baseline with a similar structure as the policy network, but computes the reward by always selecting vehicles and nodes with maximum probability. A Monte Carlo method is applied to update the parameters to improve the policy iteratively. At each iteration, we construct routes for each problem instance and calculate the reward with respect to this solution in line 9, and the parameters of the policy network are updated in line 13. In addition, the expected reward of the baseline network $ R_{BL}^b $ comes from a greedy roll-out of the policy in line 10. The parameters of the baseline network will be replaced by the parameters of the latest policy network if the latter significantly outperforms the former according to a paired t-test on several instances in line 15.

### Algorithm 1: DRL Algorithm

**Input:**  
- Initial parameters $ \theta $ for policy network $ \pi_\theta $;  
- Initial parameters $ \phi $ for baseline network $ v_\phi $;  
- Number of iterations $ I $;  
- Iteration size $ N $; number of batches $ M $;  
- Maximum training steps $ T $; significance $ \alpha $.

**1.** For each `iter` = 1, 2, ..., I do  
**2.** Sample N problem instances randomly;  
**3.** For each `i` = 1, 2, ..., M do  
**4.** Retrieve batch $ b = Ni $;  
**5.** For each `t` = 0, 1, ..., T do  
**6.** Pick an action $ a_{t,b} \sim \pi_\theta(a_{t,b}|s_{t,b}) $;  
**7.** Observe reward $ r_{t,b} $, and next state $ s_{t+1,b} $;  
**8.** End  
**9.** $ R_b = -\sum_{t=0}^T r_{t,b} $;  
**10.** Greedy Rollout with baseline $ v_\phi $ and compute its reward $ R_{BL}^b $;  
**11.** $ d\theta \leftarrow \frac{1}{B} \sum_{b=1}^B (R_b - R_{BL}^b) \nabla_\theta \log \pi_\theta(s_T,b|s_0,b) $;  
**12.** $ \theta \leftarrow \text{Adam}(\theta, d\theta) $;  
**13.** End  
**14.** End  
**15.** If `ONESIDED PAIRED TTEST($ \pi_\theta $, $ v_\phi $) < α` then  
**16.** $ \phi \leftarrow \theta $;  
**17.** End  

*Authorized licensed use limited to: Tsinghua University. Downloaded on September 02, 2022 at 01:31:15 UTC from IEEE Xplore. Restrictions apply.*

---

# This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.

## 8 IEEE TRANSACTIONS ON CYBERNETICS

two networks, the policy πθ is improved iteratively toward finding higher-quality solutions.

## V. COMPUTATIONAL EXPERIMENTS

In this section, we conduct experiments to evaluate our DRL method. Particularly, a heterogeneous fleet of fully loaded vehicles with different capacities starts at a depot node and departs to satisfy the demands of all customers by following certain routes, the objective of which is to minimize the longest or total travel time incurred by the vehicles. Moreover, we further verify our method by extending the experiments to benchmark instances from the CVRPLib [55]. Note that the HCVRP with min–max and min–sum objectives are both NP-hard problems, and the theoretical computation complexity grows exponentially as problem size scales up.

### A. Experiment Settings for HCVRP

We describe the settings and the data generation method (which we mainly follow the classic ways in [19], [21], [22], and [37]) for our experiments. Pertaining to MM-HCVRP, the coordinates of depot and customers are randomly sampled within the unit square [0 ,1]×[0,1] using the uniform distribution. The demands of customers are discrete numbers randomly chosen from set {1,2,..., 9} (demand of depot is 0).

To comprehensively verify the performance, we consider two settings of heterogeneous fleets. The first fleet considers three heterogeneous vehicles (called V3), the capacity of which is set to 20, 25, and 30, respectively. The second fleet considers five heterogeneous vehicles (called V5), the capacity of which is set to 20, 25, 30, 35, and 40, respectively. Our method is evaluated with different customer sizes for the two fleets, where we consider 40, 60, 80, 100, and 120 for V3; and 80, 100, 120, 140, and 160 for V5.

In MM-HCVRP, we set the vehicle speed f for all vehicles to be 1.0 for simplification. However, our method is capable of coping with different speeds which is verified in MS-HCVRP. Pertaining to MS-HCVRP, most of the settings are the same as the MM-HCVRP except for the vehicle speeds, which are inversely proportional to their capacities. In doing so, it avoids that only vehicle with the largest capacity is selected to serve all customers to minimize total travel time. Particularly, the speeds are set to (1/4), (1/5), and (1/6) for V3, and (1/4), (1/5), (1/6), (1/7), and (1/8) for V5, respectively.

The hyperparameters are shared to train the policy for all problem sizes. Similar to [22], the training instances are randomly generated on the fly with iteration size of 1 280 000 and are split into 2500 batches for each iteration. Pertaining to the number of iterations, normally more iterations lead to better performance. However, after training with an amount of iterations, if the improvement in the performance is not much significant, we could stop the training before full convergence, which still could deliver competitive performance although not the best. For example, regarding the model of V5-C160 (5 vehicles and 160 customers) with min–max objective trained for 50 iterations, five more iterations can only reduce the gap by less than 0.03%, then we will stop the training.

In our experiments, we use 50 iterations for all problem sizes to demonstrate the effectiveness of our method, while more iterations could be adopted for better performance in practice. The features of nodes and vehicles are embedded into a 128-D space before fed into the vehicle selection and node selection decoder, and we set the dimension of hidden layers in the decoder to be 128 [19], [21], [22]. In addition, the Adam optimizer is employed to train the policy parameters, with an initial learning rate 10^−4 and decaying 0.995 per iteration for convergence. The norm of all gradient vectors is clipped to be within 3.0 and α in Section IV-C is set to 0.05. Each iteration consumes an average training time of 31.61 min, 70.52 min (with single 2080Ti GPU), 93.02 min, 143.62 min (with two GPUs), and 170.14 min (with three GPUs) for problem size of 40, 60, 80, 100, and 120 regarding V3, and 105.25 min, 135.49 min, (with two GPUs) 189.15 min, 264.45 min, and 346.52 min (with three GPUs) for problem size of 80, 100, 120, 140, and 160 regarding V5. Pertaining to testing, 1280 instances are randomly generated for each problem size from the uniform distribution and are fixed for our method and the baselines. Our DRL code in PyTorch is available.

1  
B. Comparison Analysis of HCVRP

For the MM-HCVRP, it is prohibitively time consuming to find optimal solutions, especially for large problem size. Therefore, we adopt a variety of improved classical heuristic methods as baselines, which include: 1) SISRs [56], a state-of-the-art heuristic method for CVRP and its variants; 2) variable neighborhood search (VNS), an efficient heuristic method for solving the consistent VRP [57]; 3) ACO, an improved version of ant colony system for solving HCVRP with time windows [58], where we run the solution construction for all ants in parallel to reduce computation time; 4) the firefly algorithm (FA), an improved version of the standard FA method for solving the heterogeneous fixed fleet vehicle routing problem [59]; and 5) the state-of-the-art DRL-based attention model (AM) [22], learning a policy of node selection to construct a solution for TSP and CVRP. We adapt the objectives and relevant settings of all baselines so that they share the same one with MM-HCVRP. We have fine-tuned the parameters of the conventional heuristic methods using the grid search [60] for adjustable parameters in their original works, such as the number of shifted points in the shaking process, the discounting rate of the pheromones and the scale of the population, and report the best ones in Table I.

Regarding the iterations, we linearly increase the original ones for VNS, ACO, and FA as the problem size scales up for better performance, while the original settings adopt identical iterations across all problem sizes. For SISR, we follow its original setting, where the iterations are increased as the problem size grows up. To fairly compare with AM, we tentatively leverage two external strategies to select a vehicle at each decoding step for AM, that is, by turns and randomly, since it did not cope with vehicle selection originally. The results indicate that vehicle selection by turns is better for AM, which is thereby adopted for both min–max and min–sum objectives in our experiments. Note that we do not compare with OR-Tools as there is no built-in library or function that can directly solve MM-HCVRP. Moreover, we do not compare with Gurobi or

1https://github.com/Demon0312/HCVRP_DRL  
Authorized licensed use limited to: Tsinghua University. Downloaded on September 02,2022 at 01:31:15 UTC from IEEE Xplore.  Restrictions apply.

---

# This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.

## LIet al. : DRL FOR SOLVING THE HCVRP 9

### TABLE I: PARAMETER SETTINGS OF HEURISTIC METHODS

CPLEX either, as our experience shows that they consume days to optimally solve an MM-HCVRP instance even with 3 vehicles and 15 customers. For the MS-HCVRP, with the same three heuristic baselines and AM as used for the MM-HCVRP, we additionally adopted a generic exact solver for solving vehicle routing problems with min–sum objective [61]. The baselines, including VNS, ACO, and FA, are implemented in Python. For SISR, we adopt a publicly available version 2 implemented in Java. Note that the running efficiency of the same algorithm implemented in C++, Java, and Python could be considerably different, which will also be analyzed later for running time comparison.

All these baselines are executed on CPU servers equipped with the Intel i9-10940X CPU at 3.30 GHz. For those which consume much longer running time, we deploy them on multiple identical servers.

Regarding our DRL method and AM, we apply two types of decoders during testing:
1. **Greedy**: which always selects the vehicle and node with the maximum probability at each decoding step.
2. **Sampling**: which engenders N solutions by sampling according to the probability computed in (27) and (31), and then retrieves the best one.

We set N to 1280 and 12800, and term them as Sample1280 and Sample12800, respectively. Then, we record the performance of our methods and baselines on all sizes of MM-HCVRP and MS-HCVRP instances for both three vehicles and five vehicles in Tables II and III, respectively, which include average objective value, gap, and computation time of an instance.

Given the fact that it is prohibitively time consuming to optimally solve MM-HCVRP, the gap here is calculated by comparing the objective value of a method with the best one found among all methods.

> 2https://github.com/chenmingxiang110/tsp_solver  
> 3A program implemented in C/C++ might be 20–50 times faster than that of Python, especially for large-scale problem instances. The running efficiency of Java could be comparable to C/C++ with highly optimized coding but could be slightly slower in general.

---

## From Table II:

We can observe that for the MS-HCVRP with three vehicles, the Exact-solver achieves the smallest objective value and gap and consumes shorter computation time than heuristic methods on V3-C40 and V3-C60. However, its computation time grows exponentially as the problem size scales up. We did not show the results of the Exact-solver for solving instances with more than 100 customers regarding both three vehicles and five vehicles, which consumes a prohibitively long time.

Among the three variants of the DRL method, our DRL(Greedy) outperforms FA and AM(Greedy) in terms of objective values and gap. Although with slightly longer computation time, both Sample1280 and Sample12800 achieve smaller objective values and gaps than Greedy, which demonstrates the effectiveness of sampling strategy in improving the solution quality.

Specifically, our DRL(Sample1280) can outstrip all AM variants and ACO. It also outperforms VNS in most cases except for V3-C40, where our DRL(Sample1280) achieves the same gap with VNS. With Sample12800, our DRL further outperforms VNS in terms of objective value and gap and is slightly inferior to the state-of-the-art heuristic, that is, SISR and the Exact-solver.

Pertaining to the running efficiency, although the computation time of our DRL method is slightly longer than that of AM, it is significantly shorter (at least an order of magnitude faster) than that of conventional methods, even if we eliminate the impact of different programming language via roughly dividing the reported running time by a constant (e.g., 30), especially for large problem sizes.

Regarding the MM-HCVRP, similarly, our DRL method outperforms VNS, ACO, FA, and all AM variants, which performs slightly inferior to SISR but consumes much shorter running time.

Among all heuristic and learning-based methods, the state-of-the-art method SISR achieves lowest objective value and gap, however, the computation time of SISR grows almost exponentially as the problem scale increases and our DRL(Sample12800) grows almost linearly, which is more obvious in large-scale problem sizes.

---

## In Table III:

Similar patterns could be observed in comparison with that of three vehicles, where the superiority of DRL(Sample12800) to VNS, ACO, FA, and AM becomes more obvious. Meanwhile, our DRL method is still competitive to the state-of-the-art method, that is, SISR, on larger problem sizes in comparison with Table II.

Combining both Tables, our DRL method with Sample12800 achieves better overall performance than conventional heuristics and AM on both the MM-HCVRP and MS-HCVRP and also performs competitively well against SISR, with satisfactory computation time.

---

## To further investigate the efficiency of our DRL method against SISR:

We evaluate their performance with a bounded time budget, that is, 500 s. It is much longer than the computation time of our method, given that our DRL computes a solution in a construction fashion rather than the improvement fashion as in SISR.

In Fig. 4, we record the performance of our DRL method and SISR for MS-HCVRP with three vehicles on the same instances in Tables II and III, where the horizontal coordinate refers to the computation time and the vertical one refers to the objective value. We depict the computation time of our DRL method using the hollow circle, and...

*Authorized licensed use limited to: Tsinghua University. Downloaded on September 02, 2022 at 01:31:15 UTC from IEEE Xplore. Restrictions apply.*

---

# This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.

## IEEE TRANSACTIONS ON CYBERNETICS

### Table II
**DRL Method versus Baselines for Three Vehicles (V3)**

### Table III
**DRL Method versus Baselines for Five Vehicles (V5)**

### Figure 4
**Converge curve of the DRL method and SISR (V3).**

We then horizontally extend it for better comparison. We plot the curve of SISR over time since it improves an initial yet complete solution iteratively. We also record the time when SISR achieves the same objective value as our method using a filled circle. We can observe that SISR needs longer computation time to catch up the DRL method as the problem size scales up. When the computation time reaches 500 s for SISR, our DRL method achieves only slightly inferior objective values with much shorter computation time.

### Figure 5
**Converge curve of the DRL method and SISR (V5).**

For example, the DRL method only needs 9.1 s for solving a V3-C120 instance.

> **Authorized licensed use limited to: Tsinghua University. Downloaded on September 02, 2022 at 01:31:15 UTC from IEEE Xplore. Restrictions apply.**

---

# Article Summary

## Abstract

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.

---

## DRL for Solving the HCVRP

### Generalization Performance for MM-HCVRP (V3)

- SISR needs about 453 seconds to achieve the same objective value.
- In Fig. 5, we record the results of our DRL method and SISR for five vehicles also with 500 seconds.
- Similar patterns could be observed to that of three vehicles, where the superiority of our DRL method is more obvious, especially for large-scale problem sizes.
- For example, on V5-C140 and V5-C160, our DRL method with 11.3 and 13.84 seconds even outperforms the SISR with 500 seconds, respectively.
- Combining all the above results, we can conclude that with a relatively short time limit, our DRL method tends to achieve better performance than the state-of-the-art method, that is, SISR, and the superiority of our DRL method is more obvious for larger-scale problem sizes.
- Even with a time limit much longer than 500 seconds, our DRL method still achieves competitive performance against SISR.

---

## Generalization Analysis of HCVRP

To verify the generalization of our method, we conduct experiments to apply the policy learned for a customer size to larger ones since generalizing to larger customer sizes is more meaningful in real-life situations, where we mainly focus on Sample12800 in our method.

### Results for MM-HCVRP (Three Vehicles)

- In Fig. 6, we record the results of the MM-HCVRP for the fleet of three vehicles, where the horizontal coordinate refers to the problems to be solved, and the vertical one refers to the average objective values of different methods.
- We observe that for each customer size, the corresponding policy achieves the smallest objective values in comparison with those learned for other sizes.
- However, they still outperform AM and those classical heuristic methods except for V3-C40 in solving problem sizes larger than or equal to 80, where V3-C40 is comparable with the best performed baseline (i.e., VNS), if we refer to Table II.
- Moreover, we also notice that most of the policies learned for proximal customer sizes tend to perform better than that of more different ones, e.g., the policies for V3-C80 and V3-C100 perform better than that of V3-C40 and V3-C60 in solving V3-C120.
- The rationales behind this observation might be that proximal customer sizes may lead to similar distributions of customer locations.

### Results for MS-HCVRP (Three Vehicles)

- In Fig. 7, we record the results of the MS-HCVRP for the fleet of three vehicles, where similar patterns could be found in comparison with that of the MM-HCVRP, and the policies learned for other customer sizes outperform all the classical heuristic methods and AM.

### Results for MM-HCVRP and MS-HCVRP (Five Vehicles)

- In Figs. 8 and 9, we record the generalization performance of the MM-HCVRP and MS-HCVRP for five vehicles, respectively.
- Similar patterns to the three vehicles could be observed, where for each customer size, the corresponding policy achieves a smaller objective value in comparison with those learned for other sizes.
- However, they still outperform most classical heuristic methods and AM in all cases, and are only slightly inferior to the corresponding policies.

---

## Discussion

To comprehensively evaluate the performance of our DRL method, we further apply our trained model to solve the instances randomly selected from the well-known CVRPLib benchmark, half of which follow uniform distribution regarding the customer locations, and the remaining half do not.

### Results on CVRPLib

- In Table IV, we record the comparison results on CVRPlib, where the Exact-solver is used to optimally solve the instances.
- **CVRPLib** (http://vrp.atd-lab.inf.puc-rio.br/index.php/en/) is a well-known online benchmark repository of VRP instances used for algorithm comparison in the VRP literature.
- More details of CVRPLib can be found in [55].
- In our experiment, we select ten instances and adapt them to our MM-HCVRP and MS-HCVRP settings by adopting their customer locations and demands.

---

## Figures

- **Fig. 6.** Generalization performance for MM-HCVRP (V3).
- **Fig. 7.** Generalization performance for MS-HCVRP (V3).
- **Fig. 8.** Generalization performance for MM-HCVRP (V5).
- **Fig. 9.** Generalization performance for MS-HCVRP (V5).

---

# IEEE Transactions on Cybernetics

## Abstract

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.

---

## Table IV: Our Method versus Baselines on CVRPLib

The MS-HCVRP. Regarding the DRL method, we directly exploit the trained models as in Tables II and III to solve the CVRPLib instances, where the model with the closest size to the instances is adopted. For example, we use the model trained for V3-C60 to solve B-n63-k10. We select SISR and VNS as baselines for both MM-HCVRP and MS-HCVRP, which perform better than other heuristics in previous experiments. Each reported objective value is averaged over ten independent runs with different random seeds. Note that it is prohibitively long for the Exact-solver to solve MS-HCVRP with more than 100 customers (i.e., CMT11).

From Table IV, we can observe that our DRL method tends to perform better than VNS on uniformly distributed instances, and is slightly inferior on the instances of nonuniform distribution for both MM-HCVRP and MS-HCVRP. Although inferior to SISR, our DRL method is able to engender solutions of comparable solution quality with much shorter computation time. For example, SISR consumes 1598 s to solve the CMT11, while our DRL method only needs 9.0 s.

We also notice that our DRL method tends to perform better on uniform distributed instances than that of nonuniform ones if we refer to the gap between our method and the exact method for MS-HCVRP and the SISR for MM-HCVRP.

This observation about different distributions indeed makes sense, especially given that as described in Section V-A, the customer locations in all training instances follow the uniform distributions, the setting of which is widely adopted in this line of research (e.g., [19], [21]–[23], and [51]). Since our DRL model is a learning method in nature, it does have favorable potential to deliver superior performance when both the training and testing instances are from the same (or similar) uniform distribution. It also explains why our DRL method outperforms most of the conventional heuristic methods in Tables II and III. When it comes to nonuniform distribution for testing, this superiority does not necessarily preserve, as indicated by the results in Table IV. However, it is a fundamental out-of-distribution challenge to all learning methods, including our DRL method. The purpose of Table IV is to reveal when our DRL method may perform inferior to others.

Considering that addressing the out-of-distribution challenge is not in the scope of this article, we will investigate it in the future.

---

## VI. Conclusion and Future Work

In this article, we cope with the HCVRP for both min–max and min–sum objectives. To solve this problem, we propose a learning-based constructive heuristic method, which integrates DRL and attention mechanism to learn a policy for the route construction. In specific, the policy network leverages an encoder, a vehicle selection decoder, and a node selection decoder to pick a vehicle and a node for this vehicle at each step.

Experimental results show that the overall performance of our method is superior to most of the conventional heuristics and competitive to the state-of-the-art heuristic method, that is, SISR with much shorter computation time. With comparable computation time, our method also significantly outperforms the other learning-based method. Moreover, the proposed method generalizes well to problems with a larger number of customers for both MM-HCVRP and MS-HCVRP.

One major purpose of our work is to nourish the development of DRL-based methods for solving the vehicle routing problems, which have emerged lately. Following the same setting adopted in this line of works [19], [21]–[23], [37], we randomly generate the locations within a square of [0,1] for training and testing. The proposed method works well for HCVRP with both min–max and min–sum objectives, but may perform inferior for other types of VRPs, such as VRP with time window constraint and dynamic customer requests.

Taking into account the above concerns and other potential limitations that our method may have, in the future, we will consider and study the following aspects:

1. Time window constraint, and dynamic customer requests or stochastic traffic conditions.
2. Generalization to a different number of vehicles.
3. Evaluation with other classical or realistic benchmark datasets with instances of different distributions (e.g., http://mistic.heig-vd.ch/taillard/problemes.dir/vrp.dir/vrp.html).
4. Improvement over SISR by integrating with active search [19] or other improvement approaches (e.g., [62]).

---

## References

[1] B. Golden, A. Assad, L. Levy, and F. Gheysens, “The fleet size and mix vehicle routing problem,” *Comput. Oper. Res.*, vol. 11, no. 1, pp. 49–66, 1984.

[2] Ç. Koc, T. Bekta¸s, O. Jabali, and G. Laporte, “A hybrid evolutionary algorithm for heterogeneous fleet vehicle routing problems with time windows,” *Comput. Oper. Res.*, vol. 64, pp. 11–27, Dec. 2015.

[3] X. Wang, B. Golden, E. Wasil, and R. Zhang, “The min–max split delivery multi-depot vehicle routing problem with minimum service time requirement,” *Comput. Oper. Res.*, vol. 71, pp. 110–126, Jul. 2016.

[4] S. Duran, M. A. Gutierrez, and P. Keskinocak, “Pre-positioning of emergency items for care international,” *Interfaces*, vol. 41, no. 3, pp. 223–237, 2011.

*Authorized licensed use limited to: Tsinghua University. Downloaded on September 02, 2022 at 01:31:15 UTC from IEEE Xplore. Restrictions apply.*

---

# This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.

## References

- [5] L. Bertazzi, B. Golden, and X. Wang, “Min–max vs. min–sum vehicle routing: A worst-case analysis,” *Eur. J. Oper. Res.*, vol. 240, no. 2, pp. 372–381, 2015.
- [6] X. Ma, Y. Song, and J. Huang, “Min–max robust optimization for the wounded transfer problem in large-scale emergencies,” in *Proc. Chin. Control Decis. Conf.*, 2010, pp. 901–904.
- [7] C. Alabas-Uslu, “A self-tuning heuristic for a multi-objective vehicle routing problem,” *J. Oper. Res. Soc.*, vol. 59, no. 7, pp. 988–996, 2008.
- [8] E. K. Hashi, M. R. Hasan, and M. S. U. Zaman, “GIS based heuristic solution of the vehicle routing problem to optimize the school bus routing and scheduling,” in *Proc. Int. Conf. Comput. Inf. Technol. (ICCIT)*, 2016, pp. 56–60.
- [9] X. Liu, H. Qi, and Y. Chen, “Optimization of special vehicle routing problem based on ant colony system,” in *Proc. Int. Conf. Intell. Comput.*, 2006, pp. 1228–1233.
- [10] M. Haimovich and A. H. R. Kan, “Bounds and heuristics for capacitated routing problems,” *Math. Oper. Res.*, vol. 10, no. 4, pp. 527–542, 1985.
- [11] J. Lysgaard, A. N. Letchford, and R. W. Eglese, “A new branch-and-cut algorithm for the capacitated vehicle routing problem,” *Math. Program.*, vol. 100, no. 2, pp. 423–445, 2004.
- [12] W. Y. Szeto, Y. Wu, and S. C. Ho, “An artificial bee colony algorithm for the capacitated vehicle routing problem,” *Eur. J. Oper. Res.*, vol. 215, no. 1, pp. 126–135, 2011.
- [13] X. Wang, S. Poikonen, and B. Golden, “The vehicle routing problem with drones: Several worst-case results,” *Optim. Lett.*, vol. 11, no. 4, pp. 679–697, 2017.
- [14] R. Baldacci and A. Mingozzi, “A unified exact method for solving different classes of vehicle routing problems,” *Math. Program.*, vol. 120, no. 2, pp. 347–380, 2009.
- [15] P. M. França, M. Gendreau, G. Laporte, and F. M. Müller, “The m-Traveling Salesman problem with minmax objective,” *Transp. Sci.*, vol. 29, no. 3, pp. 267–275, 1995.
- [16] N. Mostafa and A. Eltawil, “Solving the heterogeneous capacitated vehicle routing problem using k-means clustering and valid inequalities,” in *Proc. Int. Conf. Ind. Eng. Oper. Manag.*, 2017, pp. 2239–2249.
- [17] X. Li, P. Tian, and Y. P. Aneja, “An adaptive memory programming metaheuristic for the heterogeneous fixed fleet vehicle routing problem,” *Transp. Res. E, Logist. Transp. Rev.*, vol. 46, no. 6, pp. 1111–1127, 2010.
- [18] E. Yakıcı, “A heuristic approach for solving a rich min–max vehicle routing problem with mixed fleet and mixed demand,” *Comput. Ind. Eng.*, vol. 109, pp. 288–294, Jul. 2017.
- [19] I. Bello, H. Pham, Q. V. Le, M. Norouzi, and S. Bengio, “Neural combinatorial optimization with reinforcement learning,” in *Proc. Int. Conf. Learn. Represent.*, 2017, pp. 1–5.
- [20] L. Xin, W. Song, Z. Cao, and J. Zhang, “Multi-decoder attention model with embedding glimpse for solving vehicle routing problems,” in *Proc. 35th AAAI Conf. Artif. Intell.*, 2021, pp. 1–11.
- [21] M. Nazari, A. Oroojlooy, L. Snyder, and M. Takác, “Reinforcement learning for solving the vehicle routing problem,” in *Advances in Neural Information Processing Systems*. Red Hook, NY, USA: Curran Assoc., 2018, pp. 9839–9849.
- [22] W. Kool, H. van Hoof, and M. Welling, “Attention, learn to solve routing problems!” in *Proc. Int. Conf. Learn. Represent.*, 2018, pp. 1–25.
- [23] X. Chen and Y. Tian, “Learning to perform local rewriting for combinatorial optimization,” in *Advances in Neural Information Processing Systems*. Red Hook, NY, USA: Curran Assoc., 2019, pp. 6278–6289.
- [24] J. Li, L. Xin, Z. Cao, A. Lim, W. Song, and J. Zhang, “Heterogeneous attentions for solving pickup and delivery problem via deep reinforcement learning,” *IEEE Trans. Intell. Transp. Syst.*, early access, Feb. 10, 2021, doi: 10.1109/TITS.2021.3056120.
- [25] A. Vaswani et al., “Attention is all you need,” in *Advances in Neural Information Processing Systems*. Red Hook, NY, USA: Curran Assoc., 2017, pp. 5998–6008.
- [26] C. Prins, “Efficient heuristics for the heterogeneous fleet multitrip VRP with application to a large-scale real case,” *J. Math. Model. Algorithms*, vol. 1, no. 2, pp. 135–150, 2002.
- [27] L. Feng et al., “Solving generalized vehicle routing problem with occasional drivers via evolutionary multitasking,” *IEEE Trans. Cybern.*, vol. 51, no. 6, pp. 3171–3184, Jun. 2021.
- [28] G. B. Dantzig and J. H. Ramser, “The truck dispatching problem,” *Manag. Sci.*, vol. 6, no. 1, pp. 80–91, 1959.
- [29] Y. Tian, X. Zheng, X. Zhang, and Y. Jin, “Efficient large-scale multiobjective optimization based on a competitive swarm optimizer,” *IEEE Trans. Cybern.*, vol. 50, no. 8, pp. 3696–3708, Aug. 2020.
- [30] R. Cheng and Y. Jin, “A competitive swarm optimizer for large scale optimization,” *IEEE Trans. Cybern.*, vol. 45, no. 2, pp. 191–204, Feb. 2015.
- [31] R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff, “Test problems for large-scale multiobjective and many-objective optimization,” *IEEE Trans. Cybern.*, vol. 47, no. 12, pp. 4108–4121, Dec. 2017.
- [32] J. Xiao, T. Zhang, J. Du, and X. Zhang, “An evolutionary multiobjective route grouping-based heuristic algorithm for large-scale capacitated vehicle routing problems,” *IEEE Trans. Cybern.*, vol. 51, no. 8, pp. 417

---

# IEEE Transactions on Cybernetics

This article has been accepted for inclusion in a future issue of this journal. Content is final as presented, with the exception of pagination.

## References

1. [53] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in *Proc. Int. Conf. Mach. Learn.*, 2015, pp. 448–456.
2. [54] O. Vinyals, S. Bengio, and M. Kudlur, “Order matters: Sequence to sequence for sets,” 2015. [Online]. Available: [arXiv:1511.06391](https://arxiv.org/abs/1511.06391).
3. [55] E. Uchoa, D. Pecin, A. Pessoa, M. Poggi, T. Vidal, and A. Subramanian, “New benchmark instances for the capacitated vehicle routing problem,” *Eur. J. Oper. Res.*, vol. 257, no. 3, pp. 845–858, 2017.
4. [56] J. Christiaens and G. V. Berghe, “Slack induction by string removals for vehicle routing problems,” *Transp. Sci.*, vol. 54, no. 2, pp. 417–433, 2020.
5. [57] Z. Xu and Y. Cai, “Variable neighborhood search for consistent vehicle routing problem,” *Expert Syst. Appl.*, vol. 113, pp. 66–76, Dec. 2018.
6. [58] A. Palma-Blanco, E. R. González, and C. D. Paternina-Arboleda, “A two-pheromone trail ant colony system approach for the heterogeneous vehicle routing problem with time windows, multiple products and product incompatibility,” in *Proc. Int. Conf. Comput. Logist.*, 2019, pp. 248–264.
7. [59] P.-P. Matthopoulos and S. Sofianopoulou, “A firefly algorithm for the heterogeneous fixed fleet vehicle routing problem,” *Int. J. Ind. Syst. Eng.*, vol. 33, no. 2, pp. 204–224, 2019.
8. [60] J. A. Brito, F. E. McNeill, C. E. Webber, and D. R. Chettle, “Grid search: An innovative method for the estimation of the rates of lead exchange between body compartments,” *J. Environ. Monitor.*, vol. 7, no. 3, pp. 241–247, 2005.
9. [61] A. Pessoa, R. Sadykov, E. Uchoa, and F. Vanderbeck, “A generic exact solver for vehicle routing and related problems,” *Math. Program.*, vol. 183, no. 1, pp. 483–523, 2020.
10. [62] Y. Wu, W. Song, Z. Cao, J. Zhang, and A. Lim, “Learning improvement heuristics for solving routing problems,” *IEEE Trans. Neural Netw. Learn. Syst.*, early access, Apr. 1, 2021, doi: [10.1109/TNNLS.2021.3068828](https://doi.org/10.1109/TNNLS.2021.3068828).

## Authors

### Jingwen Li

- Received the B.E. degree in computer science from the University of Electronic Science and Technology of China, Chengdu, China, in 2018.
- Currently pursuing the Ph.D. degree with the Department of Industrial Systems Engineering and Management, National University of Singapore, Singapore.
- Research interests include:
  - Deep reinforcement learning for combinatorial optimization problems
  - Especially for vehicle routing problems

### Yining Ma

- Received the B.E. degree in computer science from the South China University of Technology, Guangzhou, China, in 2019.
- Currently pursuing the Ph.D. degree with the Department of Industrial Systems Engineering and Management, National University of Singapore, Singapore.
- Research interests include:
  - Deep reinforcement learning
  - Combinatorial optimization
  - Swarm intelligence

### Ruize Gao

- Received the B.E. degree from Shanghai Jiao Tong University, Shanghai, China, in 2020.
- Currently pursuing the Ph.D. degree with the Department of Computer Science and Engineering, Chinese University of Hong Kong, Hong Kong.
- Research interests include:
  - Trustworthy machine learning
  - Especially for adversarial robustness

### Zhiguang Cao

- Received the B.Eng. degree in automation from the Guangdong University of Technology, Guangzhou, China, in 2009.
- Received the M.Sc. degree in signal processing from Nanyang Technological University (NTU), Singapore, in 2012.
- Received the Ph.D. degree from the Interdisciplinary Graduate School, NTU, in 2017.
- Former positions:
  - Research Assistant Professor with the Department of Industrial Systems Engineering and Management, National University of Singapore, Singapore.
  - Research Fellow with the Future Mobility Research Lab, NTU.
- Current position:
  - Scientist with the Singapore Institute of Manufacturing Technology, Agency for Science Technology and Research, Singapore.
- Research interests focus on:
  - Neural combinatorial optimization

### Andrew Lim

- Received the Ph.D. degree in computer science from the University of Minnesota, Minneapolis, MN, USA, in 1992.
- Currently a Professor with the School of Computing and Artificial Intelligence, Southwest Jiaotong University, Chengdu, China.
- Recruited by NUS under The National Research Foundation’s Returning Singaporean Scientists Scheme in 2016.
- Previously held professorships in:
  - The Hong Kong University of Science and Technology, Hong Kong.
  - The City University of Hong Kong, Hong Kong.
- Published work in key journals such as:
  - Operations Research
  - Management Science
- Disseminated via international conferences and professional seminars.

### Wen Song

- Received the B.S. degree in automation and the M.S. degree in control science and engineering from Shandong University, Jinan, China, in 2011 and 2014, respectively.
- Received the Ph.D. degree in computer science from Nanyang Technological University (NTU), Singapore, in 2018.
- Former positions:
  - Research Fellow with the Singtel Cognitive and Artificial Intelligence Lab for Enterprises, NTU.
- Current position:
  - Associate Research Fellow with the Institute of Marine Science and Technology, Shandong University.
- Research interests include:
  - Artificial intelligence
  - Planning and scheduling
  - Multiagent systems
  - Operations research

### Jie Zhang

- Received the Ph.D. degree from the Cheriton School of Computer Science, University of Waterloo, Waterloo, ON, Canada, in 2009.
- Currently an Associate Professor with the School of Computer Science and Engineering, Nanyang Technological University, Singapore.
- Also an Associate Professor with the Singapore Institute of Manufacturing Technology, Singapore.
- During his Ph.D. study, held the prestigious NSERC Alexander Graham Bell Canada Graduate Scholarship.
- Recipient of the Alumni Gold Medal in 2009 Convocation Ceremony.
- Papers have been published by top journals and conferences and received several best paper awards.
- Active in serving research communities.

> **Authorized licensed use limited to: Tsinghua University. Downloaded on September 02, 2022 at 01:31:15 UTC from IEEE Xplore. Restrictions apply.**

---

