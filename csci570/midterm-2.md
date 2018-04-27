# CSCI 570: Analysis of Algorithms
## Midterm 2 Study Guide

### Major Topics:
  - Dynamic Programming
  - Network Flow

---

### 1. Dynamic Programming

#### Sources:
- Lecture - Week 7, 8
- Algorithm Design - Chapter 6

#### Overview:
    To solve a larger problem, we solve smaller overlapping subproblems and
    store their values in a table.
    DP is applicable when subproblems greatly overlap. Therefore, divide and
    conquer does not count as DP. Consider mergesort.
    DP is not greedy. DP is optimized brute force. DP tries every choice before
    solving the problem. It is much more expensive than Greedy.

#### Key Attributes:
      Optimal Substructure:
        Solution can be obtained by the combination of optimal solutions to its
        subproblems. Such optimal substructures are usually described
        recursively

      Overlapping Subproblems
        Space of subproblems must be small, so an algorithm solving the problem
        should solve all the subproblems over and over

#### Methods:
      Memoization:
        Optimization technique to speed up recursive programs by storing the
        intermediate results
        Filling up a table recursively in a top-down manner

      Tabulation:
        Filling up a table iteratively in a bottom up manner.
        This implementation is preferred since it takes less time and memory,
        However, nowadays compilers are good at optimizing recursive functions.

#### Definition:

##### I. Pseudo-Polynomial
      A numeric algorithm runs in pseudo-polynomial time if its running time is
      polynomial in the numeric value of input, but is exponential in length of
      input.
        T(n) = Θ(nW)
      For the knapsack problem, the input size capacity W is measured in bits,
      so the actual complexity is given by
        T(n) = Θ(n*2^(input size of W))
      Thus, if you change W by one bit, the amount of work (table size) will
      double.

#### Algorithms:

##### I. Bellman Ford

      Notes:
        - See Shortest Path (Bellman Ford)

##### II. Floyd Warshall

      Notes:
        - See All Pairs Shortest Path (Floyd Warshall)

#### Problems:

##### I. Fibonacci Numbers

      Statement:
        Fibonacci number F_n is defined as the sum of two previous Fibonacci
        numbers:
          F_n = F_n-1 + F_n-2

      Subproblem:
        Smaller n Fibonacci numbers

      Recurrence:
        OPT[n] = OPT[n-1] + OPT[n-2]
        OPT[0] = 1
        OPT[1] = 1

      Table:
        Array of size N
        Solution is in the last index
         0  1  2  3  4  5  6  7
        |1 |1 |2 |3 |5 |8 |13|21|

      Implementation:
        Memoization:
          int table[50]; // initialize to zero
          table[0] = table[1] = 1;
          int fib(int n) {
            if table[n] != 0
              return table[n]
            else
              table[n] = fib(n-1) + fib(n-2)
            return table[n] }
        Tabulation:
          int table[n]; // initialize to zero
          int fib(int n) {
            table[0] = table[1] = 1;
            for i in range 2, n:
              table[i] = table[i-1] + table[i-2]
            return table[n] }

      Complexity:
        Memoization: T(n) = T(n-1) + Θ(n) = Θ(n^2)
          - We only have to compute one recursive call and the second is a table
            look up
          - Each addition operation will take O(n) time. Assuming numbers are so
            big that they must be added bit by bit, we will need to add
            log(ψ^n) bits -> O(n) time.
        Tabulation: O(n^2)
          - n loops with an addition
          - Each addition operation will take O(n) time. Assuming numbers are so
            big that they must be added bit by bit, we will need to add
            log(ψ^n) bits -> O(n) time.
          - Faster than Memoization by a constant factor due to running from
            bottom up, rather top down and then bottom up

##### II. Money Changing Problem

      Statement:
        You are to compute the minimum number of coins needed to make change for
        a given amount m. Assume that we have an unlimited supply of coins.
        All denominations d_k are sorted in ascending order:
          1 = d_1 < d_2 < ... < d_n

      Subproblem:
        Let OPT[v,c] be the least number of coins to represent some amount
        v (0 <= v <= m), using the first c (1 <= c <= n) denominations.

      Recurrence:
        OPT[v,c] = MIN{ OPT[v,c-1],
                        OPT[v-c_val, c] + 1 }
        OPT[v,c] = OPT[v,c-1], if v < c_val
        OPT[0,c] = 0
        OPT[v,1] = v

      Table:
        NxM table
        Solution is at bottom right corner
          0  1  2  3  4  5  6  7  8
        1|0 |1 |2 |3 |4 |5 |6 |7 |8 |
        4|0 |1 |2 |3 |1 |2 |3 |4 |2 |
        6|0 |1 |2 |3 |1 |2 |1 |2 |2 |

      Backtrack:
        model = []
        for i = m, j = n:
          if 1 + OPT[i - j_val, j] < OPT[i, j-1]:
            model.append(j_val)
            i = i - j_val
          else:
            j = j-1
        return model

      Complexity:
        Solving: O(mn) * O(C) = O(mn)
          Table size * work at each cell
        Backtrack: O(m+n)
          Worst case is climbing stairs through cells

      Key Points:
        - Cannot be solved optimally by greedy algorithm for some non-US
          denominations
        - Can be solved with linear space complexity
        - Similar to knapsack problem, however since we allow repetition of
          items we replace v-1 with v in our recurrence formula

##### III. 0-1 Knapsack Problem

      Statement:
        Given a set of unique items, each with a weight and a value, determine
        the subset of items such that the total weight is less than or equal to
        a given capacity and the total value is as large as possible.

      Subproblem:
        Let OPT[i, w] be the largest value for a knapsack with capacity
        w (0<=w<=W), using the first i (0<=i<=N) items

      Recurrence:
        OPT[i, w] = MAX{  i_v + OPT[i-1, w - i_w],
                          OPT[i-1, w]}
        OPT[i, w] = OPT[i-1, w], if i_w > w
        OPT[0, w] = 0
        OPT[i, 0] = 0

      Table:
        NxW table
        Solution is at bottom right corner
              0  1  2  3  4  5
        w,v 0|0 |0 |0 |0 |0 |0 |
        2,3 1|0 |0 |3 |3 |3 |3 |
        3,4 1|0 |0 |3 |4 |4 |7 |
        4,5 1|0 |0 |3 |4 |5 |7 |
        5,6 1|0 |0 |3 |4 |5 |7 |

      Implementation:
        int knapsack(int cap, int w[], int v[], int n) {
          int OPT[n+1][cap+1];
          for (k = 0; k <= n; k++) {
            for (w = 0; w <= cap; w++) {
              if (k==0 || w==0) OPT[k][w] = 0;
              elif (w[k-1] > w) OPT[k][w] = OPT[k-1][w];
              else OPT[k][w] = max(v[k-1] + OPT[k-1][w-w[k-1]], OPT[k-1][w])
            }
          }
          return OPT[n][cap]
        }

      Backtrack:

      Complexity:
        Solve:
          O(N+1*W+1) * O(1) = O(N*W)
          Table size * work at cells
          Note: This is a pseudo-polynomial algorithm because it depends on the
            value of W, which could be much bigger than n.
            To show this we can compute the total input size:
              1. The number of items is n; input size O(log n) bits
              2. n weights; each cannot exceed W, so the total number of bits
                is O(nlogW)
              3. n values; let the largest be V, then the total number of bits
                is O(nlogV)
              Summing up: total input size is O(n(log W + log V))
              Compare this to running time: O(nW)
              O(nw) is not polynomial in input size O(nlogW)

      Key Points:
        - Fractional Knapsack problem can be solved with greedy algorithm
        - Can be solved with linear space complexity
        - Since we do not allow repetition of items we must use i-1 instead of i
          in our recurrence formula
        - Cannot be solved in polynomial time

##### IV. Longest Common Subsequence

      Statement:
        We are given two strings: string S of length n and string T of length m.
        Our goal is to produce their longest common subsequence.

      Subproblem:
        Let L[i, j] be the longest common subsequence of string S from [1..i]
        and string T from [1..j]

      Recurrence:
        LCS[i, j] = 1 + LCS[i-1, j-1], if S[i]=T[j]
        LCS[i, j] = max(LCS[i-1, j], LCS[i, j-1]) if S[i]≠T[i]
        LCS[i, 0] = 0
        LCS[0, j] = 0

      Table:
        N+1xM+1 table
        Answer in bottom left
          -  B  A  C  B  A  D
        -|0 |0 |0 |0 |0 |0 |0 |
        A|0 |0 |1 |1 |1 |1 |1 |
        B|0 |1 |1 |1 |2 |2 |2 |
        A|0 |1 |2 |2 |2 |3 |3 |
        Z|0 |1 |2 |2 |2 |3 |3 |
        D|0 |1 |2 |2 |2 |3 |4 |
        C|0 |1 |2 |3 |3 |3 |4 |

      Implementation:
        int LCS(char[] S, int n, char[] T, int m) {
          int table[n+1, m+1];
          table[0..n, 0] = table[0, 0..m] = 0

          for(int i; i <= n; i++) {
            for(int j; j<=m; j++) {
              if S[i] == T[j]
                table[i, j] = 1 + table[i-1, j-1];
              else
                table[i, j] = max(table[i-1, j], table[i, j-1]);
            }
          }
          return table[n, m]
        }

      Backtrack:
        Start from bottom right.
          If the cell above or to the left contains a value equal to the value
            in the cell, then move to that cell
          If both values are less than the value in the current cell, then move
            diagonally and save S[n] to the model

      Complexity:
        O(mn)

      Key Points:
        - We can solve this with O(n) space but we won't be able to reconstruct
          the solution

##### V. Shortest Path (Bellman Ford)

      Statement:
        Given a graph G with weighted edges (positive or negative), find the
        shortest path from s to any vertex v.

      Subproblem:
        D[v, k] denotes the length of the shortest path from s to v that uses
        at most k edges.

      Recurrence:
        Case 1: Path uses at most k-1 edges, D[v, k] = D[v, k-1]
        Case 2: Path uses at most k edges.
          - If w is adjacent to v, take (w, v) edge, and then select the best
            s-w path that uses at most k-1 edges (check all vertices adjacent
            to v), D[v, k] = min_(w,v)inE(D[w, k-1] + c_wv)

        D[v,k] = min(D[v, k-1], min_(w,v)inE(D[w, k-1] + c_wv))
        D[v,k] = 0, if k = 0 & v = s
        D[v,k] = ∞, if k = 0 & v ≠ s

      Table:
        VxK table

      Implementation:
        for k=1 to V-1:
          for each v in V:
            for each edge (w,v) in E:
              D[v,k] =  min(D[v, k-1], D[w, k-1] + c_wv))

      Complexity:
        O(VE)

      Key Points:
        - Can be done dynamically by finding distance from t for first neighbor
          on path to T
        - Bellman Ford can handle negative weights
        - Number of edges K should be less than the number of vertices V
        - Bellman Ford can be used to find a negative cycle if you perform one
          more cycle after reaching v-1 edges. If anything changes on the
          second cycle, we have a negative cycle

##### VI. All Pairs Shortest Path (Floyd Warshall)

      Statement:
        Given a graph G with weighted edges (positive or negative) find the
        shortest path between each pair of vertices

      Subproblem:
        Let D[i, j, k] be the shortest path from i to j for which all
        intermediate vertices can only be chosen from the set {1, 2,..., k}.
        A shortest path does not contain the same vertex more than once.

      Recurrence:
        D[i, j, k] = min_u(D[i, u, k-1] + D[u, j, k-1], D[i, j, k-1])
        D[i, j, 0] = c(i, j)

      Implementation:
        D[i, j, 0] = c[i, j] for all i and j
        for k=1..V:
          for i=1..V:
            for j=1..V:
              D[i, j, k] = min (D[i, j, k-1], D[i, k, k-1] + D[k, j, k-1])

      Table:
        V-1 VxV tables
        V-1th table is optimal path for each node

      Backtrack:
        To extract shortest path:
          Create a new matrix P[i,j], whenever we discover that the shortest
          path from i to j passes through an intermediate vertex k, we set
          P[i,j] = k.
          Recursively compute the shortest path from i to k and from k to j

      Key Points:
        - O(V^3) vs Bellman Ford O(EV^2)

##### VII. Weighted Interval

      Statement:
        Given N intervals where every job is given a start time, s, a finish
        time t, and a weight w. Maximize the weight.

      Subproblem:
        C[i] is the maximum weight that can be obtained from the first i jobs.
        p(i) uses binary search the interval with the latest finishing time from
         the set of intervals that does not overlap with i.

      Recurrence:
        C[i] = MAX(w_i + C[p(i)], C[i-1])
        C[0] = 0

      Table:
        Array of size N.
        Max value is at the last index.

      Complexity:
        O(n) * O(logn) = O(nlogn)
        Array Size & Binary Search runtime

      Key Points:
        - Must sort intervals by finish time

  Tips:
    - To generate recursion formula, try to start with a toy problem or try
      drawing the problem
    - To verify complexity, write out the implementation
    - Don't forget edge cases for recurrence

  //TODO Tree

---
### 2. Network Flow

#### Sources:
- Lecture - Week 9, 10
- Algorithm Design - Chapter 7

#### Definitions:

##### I. Flow Network
      Directed graph G = (V,E) with the following features:
        - Each edge e has a non-negative capacity c_e
        - Has a single source node s in V
        - Has a single sink node t in V

##### II. Steady State Flow
      Flow that does not change over time
      The value of flow v(f) is defined as follows:
        v(f) = Σ f(e), outOf(s)

##### III. Residual Graph
      G_f is the residual graph of G with the following definition:
        - G_f has the same set of nodes as G
        - for each edge e with f(e) < c_e, we include e in G_f with capacity
          c_e - f(e)
        - for each edge e with f(e) > 0, we include edge e' (opposite direction
          to e) in G_f with f(e) units of capacity

##### IV. Bottleneck
      If p is a simple path from s to t in G_f, then bottleneck(p) is the
      minimum residual capacity of any edge on P.

##### V. Strongly Polynomial
      An algorithm runs in strongly polynomial time if the number of operations
      is bounded by a polynomial in the number of integers in the input.
      This is relevant if input consists of integers.

##### VI. Bipartite Graph
      A bipartite graph G(V,E) is an undirected graph whose node set can be
      partitioned as V = X U Y with property that every edge e in E has one
      end in X and the other in Y.

##### VII. Matching
      A matching M in G is a subset of the edges M ≤ E such that each node
      appears in at most one edge in M.

##### VIII. Edge Disjoint
      A set of paths is edge-disjoint if their edge sets are disjoint

##### XI. Node Disjoint
      A set of paths is node-disjoint if their node sets (except for starting
      and ending) are disjoint

##### X. Circulation
      A circulation with demand {d_v} is a function f that assigns non-negative
      real numbers to each edge and satisfies:
        1. Capacity Conditions
          for each edge e in E: 0 <= f(e) <= c_e
        2. Demand Conditions
          for each node v in V: f_in(v) = f_out(v) = d_v

#### Algorithms:

##### I. Ford-Fulkerson:

      Assumptions:
        - no edges enter source (s) or leave sink (t)
        - at least one edge connected to each node
          Note: This simplifies complexity analysis O(m+n) => O(m)
        - all capacities are integers

      Notation:
        We call f(e) flow through edge e. f(e) has the following properties:
          1. Capacity Constraint:
            for each edge e in E, 0<=f(e)<=c(e)
          2. Conservation of Flow:
            Σ f(e), into(v) = Σ f(e), outOf(v), except for s & t

      Implementation:
        Max-Flow(G,s,t,c):
          Initially f(e)=0 for all e in G
          While there is an s-t path in the residual graph G_f
            Let P be a simple s-t path in G_f
            f' = augment(f, P)
            Update f to be f'
            Update G_f to G_f'
          Endwhile
          Return f

        Augment(f, P):
          Let b = bottleneck(P,f)
          For each edge (u, v) in P:
            If e = (u,v) is a forward edge:
              increase f(e) in G by b
            Else e = (u,v) is a backward edge:
              Let e = (v,u)
              decrease f(e) in G by b
            Endif
          Endfor
          Return f

      Complexity:
        O(Cm)

      Key Points:
        - The flow going through the network at every step is integer valued
        - Pseudo-polynomial

##### II. Scaled Ford-Fulkerson:

      Assumptions:
        See Ford-Fulkerson:Assumptions

      Notation:
        We call f(e) flow through edge e. f(e) has the following properties:
          1. Capacity Constraint:
            for each edge e in E, 0<=f(e)<=c(e)
          2. Conservation of Flow:
            Σ f(e), into(v) = Σ f(e), outOf(v), except for s & t
        Let ∆ be the largest power of 2 that is no larger than the max
          capacity out of s

      Implementation:
        Scaled-FF(G,s,t,c):
          Initially f(e)=0 for all e in G
          Set ∆ = largest power of 2 that is <= max capacity out of s
          While ∆ >= 1
            While there is an s-t path in the residual graph G_f
              Let P be a simple s-t path in G_f
              f' = augment(f, P)
              Update f to be f'
              Update G_f to G_f'
            Endwhile
            ∆ = ∆/2
          Endwhile
          Return f

      Complexity:
        O(logCm^2)
        Outer while = O(logC), Inner while = O(m), Inner function = O(m)

      Key Points:
        - During the ∆ scaling phase, each augmentation the flow increases by
          at least ∆
        - Weakly Polynomial

##### III. Edmunds-Karp (Short Pipes):

      Background:
        Same as Ford-Fulkerson, except that each augmenting path must be a
        shortest path with available capacity.

      Complexity:
        O(nm^2)

      Key Points:
        - Strongly polynomial

##### IV. Edmunds-Karp (Fat Pipes):

      Background:
        Same as Ford-Fulkerson, except choose the augmenting path with the
        largest bottleneck value.

      Complexity:
        O(E^2 logE logf)
        O(E log V) -> Primm's variant to find augmentation paths

      Key Points:
        - Weakly polynomial

#### Problems:

##### I. Max Flow

      Statement:
        Given a flow network G, find an s-t flow with max value.

      Strategy:
        1. Find a path from s to t
        2. Find the bottleneck value for this path
        3. Push flow through this path with value equal to bottleneck value
        4. Repeat

      Key Points:
        - Max Flow <= Capacity of any (A, B) cut

##### II. Min Cut

      Statement:
        Given a flow network G, find the min-cut of an s-t flow with max value.

      Strategy:
        1. Find max flow
        2. Construct residual graph G_f
        3. Run BFS to find reachable nodes from s. Let the set of these nodes
          be called A.
        4. Let B = V - A

      Key Points
        - Value of max flow = capacity of min-cut
        - Is min cut unique?
          - Find min cut closest to S
          - Reverse edge directions and find Max flow from s to t
          - Find min cut closest to T
          - Check if the cuts are the same

##### III. Bipartite Matching

      Statement:
        Find a matching M of largest possible size in G.

      Strategy:
        Design a flow network G' that will have a flow value v(f) = k iff there
        is a matching of size k in G. Moreover, flow f in G' should identify
        the matching M in G.

      Solution:
        - Construct G'
          - Connect super source to all nodes in X, set weights to 1
          - Connect all edges X & Y where edges are valid, set weights to 1
          - Connect super sink to all nodes in Y, set weights to 1
        - Run max flow on G', let f = max flow
        - Edges carrying flow between sets X & Y will correspond to our max
          size matching in G.

      Key Points:
        - In this problem, Ford Fulkerson is strongly polynomial

##### IV. Edge Disjoint

      Statement:
        Given a directed graph G with s & t in V. Find the max number of edge
        disjoint s-t paths in G.

      Strategy:
        Design a flow network G' that will have a flow value v(f) = k iff there
        are k edge-disjoint s-t paths in G. Moreover, flow f in G' should
        identify the set of edge disjoint paths in G.

      Solution:
        - Construct G'
          - Remove any edges going into s
          - Remove any edges going out of t
        - Run max flow in G'
        - v(f) will equal the max number of edge disjoint s-t paths
        - f will identify edges on these paths

      Key Points:
        - If we wind up with a cycle on our path, we can just remove the loop
        - We can solve this problem for an undirected graph by adding two edges
          in G' for every edge in G. If we find that we are using both edges in
          a pair, then reroute both paths by combining their paths to avoid the
          shared edge e.g. if path B(s,t) and path A(s,t) collide on the forward
          and backward edges of (u,v), then:
            B' = B(s, u) + A(u, t)
            A' = A(s, v) + B(v, t)

##### V. Node Disjoint

        Statement:
          Given a directed graph G with s & t in V. Find the max number of node
          disjoint s-t paths in G.

        Strategy:
          - Construct G'
            - For every node v, create a node v' for all edges into v, create a
              node v'' for all edges out of v, connect v' and v'' with an edge
              of capacity 1.

##### VI. Circulation with Lower Bounds

      Statement:
        Given a directed graph G(V,E) with capacities on the edges, and demands
        (positive - demand, negative - supply) on the nodes. Find if a feasible
        circulation exists.

      Strategy:
        Find a feasible circulation (if it exists in two passes):
          1. Find f_0 to satisfy all lower bounds
            - Push flow f_0 through G where f_0(e) = l_e
            - Construct G' where c_e' = c_e - l_e & d_v' = d_v - l_v
          2. Use remaining capacity of the network to find a feasible
            circulation f_1 (if it exists)
            - Find feasible circulation in G' (if no circulation in G' then no
              circulation in G)
          3. Combine the two flows: f = f_0 + f_1

      Key Points
        - When modeling a circulation problem as a max flow, remove demands from
          nodes
