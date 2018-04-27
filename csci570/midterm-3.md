# CSCI 570: Analysis of Algorithms
## Midterm 3 Study Guide

### Major Topics:
  - Randomization
  - NP Hardness
  - Approximation
  - Linear Programming

---
### 1. Randomization

#### Sources
- Lecture 10 (Week 12)
- Algorithm Design - Chapter 13

#### Overview:
  Randomized algorithms are algorithms that make random choices during
  execution. For example, you flip a coin when you have to make a decision.
  A randomized algorithm might give different outputs on the different inputs.
  Thus, its inevitable that a description of the properties of a randomized
  algorithm will involve probabilistic statements.

#### Key Attributes:
  - Analysis of Randomized Algorithms

    When we analyze randomized algorithms, we are interested in expected running
    time, which is the average amount of time computed over all possible
    outcomes.
      T(n) = max_|x|=n (E[T(x)])

  - Advantages of Randomized Algorithms

    May be faster than deterministic algorithms.
    May have smaller space complexity.
    May be simpler.
    Some problems can be solved with randomization that cannot be efficiently
      solved with deterministic algorithms.

#### Classifications:
  I. Las Vegas Algorithms

    Always returns a correct answer, but may run for longer than you expect.
    Runtime is subject to input randomness but they always succeed.
    For example: Quicksort

  II. Monte Carlo Algorithms

    May fail or return an incorrect answer, but its runtime is independent of
    input randomness.
    Typically succeed with high probability, therefore we have to run them
    several times.
    For example: Global Min Cut

#### Definition:
  I. Sample Space

    The sample space Ω consists of all possible outcomes of an experiment.
    For example, when tossing a coin, Ω = {H,T}

  II. Probability

    The probability of w where w is an element of Ω, denoted Pr(w) is the number
      such that:
        0 ≤ Pr(W) ≤ 1, and Pr(Ω) = 1

  III. Event

    A subset of a sample space is called an event.
    The probability of an event is equal to the sum of the probabilities of
      outcomes contained in that event.

  IV. Random Variable

    A discrete random variable, X, is a function from the sample space Ω into
      reals, R.
    The probability that the random variable X takes the value s is defined by:
      Pr(X = s) = Pr(w in Ω s.t.X(w) = s)

  V. Indicator Random Variable

    Let E be an event. The indicator of E is the random variable X which is 1
      when E occurs and 0 when E doesn't occur:
        X: Ω -> R
        X(k) = {1 if k in E, 0 if k not in E}

  VI. Expectation

    Let X be a random variable in experiment with sample space Ω. Its
      expectation is defined by:
        E[X] = Σ of k in Ω: Pr(X = k) * X(k)
    E[X] can be viewed as a sum of possible outcomes, each weighted by its
      probability

  VII. Expectation of an Indicator

    Let E be an event, let X be its indicator random variable.
      E[X] = Pr[E]

  VIII. Linearity of Expectation

    Given an experiment, let X and Y be random variables:
      E[X+Y] = E[X] + E[Y]

  IX. Union Bound

    The union bound for any random events A and B is:
      Pr(A V B) ≤ Pr(A) + Pr(B)

  X. Edge Contraction

    Contraction of an edge with endpoints u and v is the replacement of u and v
    with a single vertex such that edges incident to the new vertex are the
    edges other than e that were incident to either u or w. If there are
    multiple edges between u and v, collapsing any one of them deletes them all.

#### Algorithms:
  I. QuickSort:

    Overview:

      Input is a list L
      If length(L) = 0 or 1, return L
      Choose a pivot p
      Partition L - {p} by value into two sets
        L1 = {x in S | x <= p}
        L2 = {x in S | x >= p}
      Items with the same key as p can go into either list
      The pivot does not go into either list
      Sort L1 and L2 recursively        

    Runtime Analysis:

      I. Deterministic Pivot
        If partition produces two subproblems that are roughly of the same size
          then the recurrence of the running time is:
            T(n) = 2T(n/2) + O(n), so that:
            T(n) = O(nlogn)l
          This is merge sort and such a pivot is the median.

        If we fix a pivot to be an item at the particular index of the input
          array, then the runtime will degrade to quadratic. If p is the first
          item, then:
            T(n) = T(1) + T(n-1) + O(n), so that:
            T(n) = O(n^2)

      II. Random Pivot
        If on each iteration a pivot is chosen uniformly at random, then we can
          think of sorting as a binary search tree rooted at the pivot.
        Let X denote the random variable counting the total number of
          comparisons. For each pair of elements i < j, we define an indicator
          random variable Xij that equals 1 if two nodes i and j are compared
          during the course of the algorithm and 0 otherwise. Note that each
          pair will be compared when one is chosen as a pivot.

          Since each pair of elements is compared at most once, the total
          number of comparisons is:
            Σ i=1..n: Σ j=i+1..n: Xij
          The expected value is:
            E[T(n)] = Σ i=1..n: Σ j=i+1..n: E[Xij]
            E[Xij] = 0 * Pr[Xij=0] + 1 * Pr[Xij=1] = Pr[Xij=1]
          Finally,
            E[T(n)] = Σ i=1..n: Σ j=i+1..n: 2 / (j-1+1)
            E[T(n)] = 2(n+1)H_n - 4n, where H_n = Σ k=1..n: 1/k = O(logn)

        Worst Case:
          When a random pivot is the largest or the smallest element on each
            iteration. The probability that this happens is as follows:
            Let Xk be the event that we pick the largest or smallest element
              of the array of size k < n
            Let X be the worst-case for the array of size n
              X = Union(k=1..n): Xk
              Pr(X) = Pr(Union(k=1..n): Xk) = ∏(k=1..n) Pr(Xk)

    Key Points:

      Binary Search Trees and QuickSort
        We can view a binary search tree in two ways:
          1. As a tree resulting from insertions
          2. As a depiction of the successive splittings used in QuickSort
        This dual interpretation allows the transfer of the analysis of
          randomized quicksort to the expected insertion time of a random BST.

#### Data Structures:
  I. Treap

    Overview:
      Tree heap, randomized binary search tree
      Binary search tree with the heap ordering property
      Each item has two associated values, x.key and x.priority. Thus, it is
        simultaneously a binary search tree with respect to the key values and
        a heap with respect to the priority values.

    Methods:
      1. Insert:
        Algorithm:
          For the inserting item, we get its priority at random.
          Then we insert with respect to a key (BST) and then use rotations to
            fix the heap property with respect to priority

        Runtime Analysis:
          The cost of insertion and then rotations is proportional to the node
            depth.
          Let us define an indicator random variable Xij that equals 1 if node i
            is higher in the tree than node j, or 0 otherwise. Node i has a
            lower priority.
            The depth of the node k is the number of proper ancestors d(k)
            defined by:
              d(k) = ∑(i=1..n): Xik
            Expected value:
              E[d(k)] = ∑(i=1..n): E[Xik] = ∑(i=1..n): Pr[Xik=1]
            Pr[Xik=1] is the probability that node i is higher in the tree than
            node k, but both are in the same subtree
              Pr[Xik=1] = {1 / (k-i+1) if i<k, 0 if i=k, 1 / (i-k+1) if i<k}
            Thus, it follows that:
              E[d(k)] = ∑(i=1..n): Pr[Xik=1] =
                ∑(i=1..k):1/(k-i+1) + ∑(i=k+1..n):1/(i-k+1) < 2H_n
              E[d(k)] = O(logn)

  II. Skip Lists

    Sorted linked list with some random shortcuts
    Use shortcuts to make searches of sorted linked lists faster

    Method:
      1. Search
        Algorithm:
          Scan for x in a shortcut list L2
          If we find x, we're done
          Otherwise, we reach some value bigger than x and we know that x is
            not in the shortcut list.
          Search for x in the original list L1 starting from y (largest
            intermediate value in L2 less than x)

        Optimization:
          How many skip nodes should we put into the shortcut list?
          We need to optimize:
            Search Time = L2 + partOfL1 = L2 + L1/L2
            L2 = √(L1)
          Search speed:
            L2 + L1/L2 = √(L1) + L1/√(L2) = 2√(L1)

          We can optimize more by adding more lists on top of L2:
            Linked list of size N with two shortcuts:
              2√(N)
            Linked list of size N with three shortcuts:
              3 root3(N)
            Linked list of size N with k shortcuts:
              k rootk(N)
            To optimize k:
              k = logN

          Runtime Analysis:
            If we create k = logN shortcuts, then:
              k rootk(N) = logN * rootlogN(N) = logN * N^(log_n(2)) = 2logN
              O(log N)

    Structure:
      Deterministic Skip Lists with logn shortcuts are hard to implement.

      Randomized Skip Lists:
        Overview:
          When you insert an element, flip a fair coin. If its heads add that
          element to the next level up, and flip again otherwise move on to
          the next element.

        Analysis:
          Intuitively, since each level of the skip list has about half the
          number of nodes of the previous level, the total number of levels
          should be about O(logn).
          Each level of shortcuts decrease the search time roughly in half, so
          after O(logn) levels, we should have a search time of O(logn)

        Worst-Case:
          Let L(x) be the number of levels that contain a search key x, not
          counting the bottom level.

          In order for a search key x to appear on level k, it must have
          flipped k heads in a row. The list has k levels if at least one
          key is at k.
            Pr(max=k) = Pr(L(1) = k V L(2) = k V ... V L(n) = k),
          By union bound:
            Pr(L(1) = k V L(2) = k V ... V L(n) = k) ≤
             Pr(L(1) = k) + ... + Pr(L(n) = k) =
             1/2^k + ... + 1/2^k = n/2^k

          If we choose the number of levels k = clogn for any constant c > 1:
            Pr(max = c log n) ≤ n/n^c = 1/n^(c-1)
          Thus, we can conclude with a high probability that a skip list has
          O(logn) levels.

#### Problems:
  I. Global Min Cut

    Statement:
      Given a connected unweighted multigraph, we need to partition vertices
      into two disjoint sets with the minimum number fo edges between them.

    Strategy:
      1. Pick an edge at random and contract
      2. Remove self-loops a new vertex
      3. Repeat 1 & 2 until two nodes remain
      4. For the two remaining nodes, u1 and u2,
        set v1 = {nodes that went unto u1}
        set v2 = {nodes that wend into u2}

    Correctness:
      The probability that all contracted edges are not on our min cut is at
      least:
        (1 - 2/V) * (1 - (2/V-1)) * (1 - (2/V-2)) * ... * (1 - 2/3) =
          (V-2)/V * (V-3)/(V-1) * (V-4)/(V-2) * ... * 1/3 = 2/(V(V-1)) =
          O(1/V^2)
      Thus, we may not find the min cut on a single run.

      Let us assume that the algorithm fails N times to find a min cut. The
      probability of failing N times is:
        (1 - 1/V^2) ^ N
      Choosing N = 100V^2 we can make it close to 0
        (1 - 1/V^2)^N = (1 - 1/V^2)^100V^2 ≤ e^-100 by:
        lim(x->∞): (1-1/x) ^ ax = e^(-ax)

    Key Points:
      - The classical deterministic algorithm for this problem is based on
        network flow algorithms
      - Randomized algorithm runs in O(V^2 time), but we need to run the
        algorithm O(V^2) iterations, thus: O(V^4)

  II. First Success

    Statement:
      How many times should you expect to flip a coin before getting heads?

    Solution:
      If we repeatedly perform independent trials of an experiment, each of
      which succeeds with probability p > 0, then the expected number of trials
      we need to perform until the first success is 1/p.

  III. Memoryless Guessing

    Statement:
      TAs randomly permute midterms before handing them back to students in a
      class of N students. Let X be the number of students getting their own
      midterm back. What is E[X]?

    Solution:
      The expected number of correct predictions under the memoryless guessing
      strategy is 1, independent of n.

---
### 2. NP Hardness

#### Sources
- Lecture 11 (Week 13)
- Algorithm Design - Chapter 8

#### Definition:
  I. Decidable

    A problem is decidable or computable if it can be solved by a Turing machine
    that halts on every input.

  II. Complexity Class P

    The fundamental complexity class P or PTIME is a class of decision problems
    that can be solved by a deterministic Turing machine in polynomial time.

  III. Complexity Class NP

    The fundamental complexity class NP is a class of decision problems that
    can be solved by a nondeterministic Turing machine in polynomial time.

    The NP decision problem has a certificate that can be checked by a
    polynomial time deterministic Turing machine.

  IV. Deterministic Turing Machine

    The deterministic Turing machine means that there is only one valid
    computation starting from any given input. A computation path is like a
    linked list.

  V. Nondeterministic Turing Machine

    The nondeterministic Turing machine is defined in the same way as
    deterministic, except that a computation is like a tree, where at any state
    it's allowed to have a number of choices.

    The big advantage: it is able to try out many possible combinations in
    parallel and accept its input if any one of these computations accepts it.

  VI. Undecidable Problems

    Undecidable means that there is no computer program that always gives the
    correct answer: it may give the wrong answer or run forever without giving
    any answer.

  VII. The Halting Problem

    The halting problem is the problem of deciding whether a given Turing
    machine halts when presented with a given input.

    By Turing's Theorem: the halting problem is not decidable.

  VIII. Reduction

    If we can solve X then we can solve Y.
    Denoted by Y ≤p X or Y -> X.
    We use this to prove NP Completeness. Knowing that Y is hard, we prove that
      X is harder.

  IX. Polynomial Reduction

    To reduce problem Y to a problem X, we want a function f that maps Y to X
    such that:
      1. f is polynomial time computable
      2. Any instance of Y is solvable iff f(y) is an instance of X and is
        solvable.

  X. NP-Hard

    X is NP-Hard if Y is NP and Y ≤p X
    Do not have to be in NP

  XI. NP-Complete

    X is NP-Complete if X is NP-Hard and X is NP
    Most difficult NP Problems

  XII. Optimization vs Decision

    If one can solve an optimization problem in polynomial time, then one can
    answer the decision version in polynomial time.
    Conversely, by doing a binary search on the bound, b, one can transform a
    polynomial time answer to a decision version into a polynomial time
    algorithm for the corresponding optimization problem.
    In that sense, these are essentially equivalent. However, they belong to two
    different complexity classes.

#### Methods:
  I. NP Completeness Proof Method

    To show that X is NP-Complete:
      1. Show that X is in NP
      2. Pick a problem Y, known to be NP-Complete
      3. Prove Y ≤p X, (reduce Y to X)
      Note: you must prove the reduction both ways

#### Problems:
  I. Graph Isomorphism

    Structure:
      Two graphs G1 = (V1, E1) and G2 = (V2, E2) are isomorphic if there is a
      bijective function f: V1 -> V2 such that:
        Edge (v, w) in E1 <-> Edge {f(v), f(w)} in E2

    Key Facts:
      This problem is NP, but:
        - No NP-Completeness proof is known
        - No polynomial time algorithm is known

  II. Boolean Satisfiability Problem

    Structure:
      A propositional logic formula is built from variables, operators AND OR
      NOT, and parentheses. A formula is said to be satisfiable if it can be
      made True by assigning appropriate logical values (True, False) to its
      variables.

    Key Facts:
      CNF SAT is NP-Complete
      DNF SAT is Linear

### Taxonomy
  I. Packing Problems

    Overview:
      You're given a collection of objects and you want to choose at least k of
      them; making your life difficult is a set of conflicts among the objects,
      preventing you from choosing certain groups simultaneously.

    Examples:
      1. Independent Set:
        Given a graph G and a number k, does G contain an independent set of
        size at least k?

      2. Set Packing:
        Given a set U of n elements, a collection S1,...,Sm of subsets U, and a
        number k, does there exist a collection of at least k of these sets with
        the property that no two of them intersect?

  II. Covering Problems

    Overview:
      These form a natural contrast to packing problems.
      You're given a collection of objects and you want to choose a subset that
      collectively achieves a certain goal; the challenge is to achieve this
      while only choosing k of the objects.

    Examples:
      1. Vertex Cover:
        Given a graph G and a number k, does G contain a vertex cover of size
        at most k?

      2. Set Cover:
        Given a set U of n elements, a collection S1,..., Sm of subsets of U,
        and a number k, does there exist a collection of at most k of these
        sets whose unions is equal to all of U?

  III. Partitioning Problems

    Overview:
      Partitioning problems involve a search over all ways to divide up a
      collection of objects into subsets so that each object appears in
      exactly one of the subsets.

    Examples:
      1. 3 Dimensional Matching
        Arises naturally when you have a collection of sets and you want to
        solve a covering problem and a packing problem simultaneously.

        Given disjoint sets X, Y, and Z, each of size n, and given a set
        T ≤ X * Y * Z of ordered triples, does there exist a set of n triples in
        T so that each element of X v Y v Z is contained in exactly one of those
        triples?

      2. Graph Coloring
        This arises when you are seeking to partition objects in the presence of
        conflicts, and conflicting objects aren't allowed to go into the same
        set.

        Given a graph G and a bound k, does G have a k-coloring?

  IV. Sequencing Problems

    Overview:
      These problems involve searching over the set of all permutations of a
      collection of objects.

    Examples:
      1. Hamiltonian Cycle:
        Given a directed graph G, does it contain a Hamiltonian cycle?

      2. Hamiltonian Path:
        Given a directed graph G, does it contain a Hamiltonian path?

      3. Traveling Salesman:
        Given a set of distances on n cities, and a bound D, is there a tour of
        length at most D?

  V. Numerical Problems

    Overview:
      Natural to reduce problems where one has weighted objects and the goal is
      to select objects conditioned on a constraint on the total weight of
      objects selected.

    Examples:
      1. Subset Sum:
        Given natural numbers w1, ... wn, and a target number W, is there a
        subset of {w1, ... wn} that adds up precisely to W?

        Note this problem only becomes hard with large integers.

  VI. Constraint Satisfaction Problem

    Overview:
      These include all basic constraint satisfaction problems. These problems
      are very flexible and thus they are a good starting point for reducing any
      problem that does not fit naturally into the other five categories.

      Examples:
        1. 3-SAT:
          Given a set of clauses C1,...Ck, each of length 3, over a set of
          variables X = {x1,...xn}, does there exist a satisfying truth
          assignment?

          It helps to recall that there are two ways to view an instance of
          3-SAT:
            - As a search over the assignments to the variables, subject to the
              constraint that all clauses must be satisfied
            - As a search over ways to choose a single term from each clause,
              subject to the constraint that one mustn't choose conflicting
              terms from different clauses

---
### 3. Approximation

#### Sources
- Lecture 12 (Week 14)
- Algorithm Design - Chapter 11

#### Overview:
  For NP-Hard problems, our standard approach can be to use approximation
  algorithms to find a good solution. Our approximation algorithms should
    1. Run quickly (polynomial time for theory, low-order polynomial time for
      practice)
    2. Obtain solutions that are guaranteed to be close to optimal
  Approximation algorithms are usually based on greedy approaches or may be
  found by simplifying the constraints on the problem.

### Key Points:
  - Since we usually don't know what the optimal solution for NP-Hard problems
    is, it's hard to approximate how good our solutions are. For approximation
    algorithms, the general method we follow is:
      1. Find an estimation of how good the optimal solution can possibly be
      2. Compare our solution to this estimation

#### Definitions:
  I. 2-Approximation Algorithm

    An approximation algorithm that comes within a factor of two of the optimal
    solution.

#### Problems:

  I. Traveling Salesman

    Statement:
      Optimization:
        Given the set of distances, order n cities in a tour v_i1, v_i2,...v_in
        with i_1 = 1, so it minimizes the total distance:
          ∑ d(v_ij, v_ij+1) + d(v_in, v_i1)

      Decision:
        Given a set of distances on n cities and a bound d, is there a tour of
        length/cost at most D?

      Approximation Method:
        1. Generate a minimum spanning tree of our nodes in G using Primm's.
        2. Create a tour along the minimum spanning tree that goes around all
          the cities with repetitions allowed.
          The cost of this initial tour is 2 * cost of MST
        3. Modify the tour:
          When you reach a node that has already been visited, move back a step
          and then move to the next node that has not been visited on the tour
          yet. By replacing the two edges with a single direct edge, we reduce
          the cost of the tour by the triangle inequality.

      Analysis:
        Our approximate tour has a cost T:
          T ≤ 2 * cost of MST
        Because the MST is the lowest cost way to connect a graph, it holds that
        the cost of the optimal Traveling Salesman tour T*:
          T* > cost of MST
        Thus, if we combine these, we get:
          T ≤ 2T*

      Key Points:
        - Proof only holds for this specific type of Traveling Salesman problems,
          where we can leverage the triangle inequality.

  II. General Traveling Salesman Problem

    Theorem:
      If P ≠ NP, then for any constant p ≥ 1, there is no polynomial time
      approximation algorithm with approximation ratio ρ for the general TSP.

    Proof:
      Assume there is such an approximation algorithm. We will then use it to
      solve the Hamiltonian cycle problem in polynomial time.

      Let G = (V,E) be an instance of the Hamiltonian Cycle problem.
      Create G' = (V, E'), a complete graph and assign costs:
        c(u,v) = {1 if (u,v) in E, ρ|v| + 1 otherwise}

      We will show that G has a Hamiltonian cycle iff G' has a tour of
      cost ≤  ρ|v|:

        If there is a Hamiltonian cycle in G, we can find a tour of cost
        |v| ≤ ρ|v| in G'

        If there is a tour of cost ≤ ρ|v| in G', there is a hamiltonian cycle
        in G.

      Say G has a Hamiltonian cycle, then the blackbox has to find a tour of
      cost ≤ ρ|v| in G'.

      But if we have a tour of cost ≤ ρ|v|, we can use it to find a Hamiltonian
      cycle in G.

      So if there is a Hamiltonian Cycle in G, the approximation algorithm has
      to find it.

      Thus there are no approximation algorithms for the General Traveling
      Salesman problem that can guarantee to come within a constant factor of
      the optimal in polynomial time.6

  III. Load Balancing Problem

    Statement:
      Input:
        m resources with equal processing power, n jobs where job j takes t_j
        to process.

      Objective:
        Assignment of jobs to resources such that the maximum load on any
        machine is minimized.

      Notation:
        T_i: load on machine/resource i
        T*: value of the optimal solution

    Approximation Method:
      Greedy Balancing
        Always assign the next job to the machine with the lowest load at that
        time.

      Analysis:
        Optimal solution is greater than or equal to the average cost of all
        jobs:
          T* ≥ 1/m ∑j: t_j
        Optimal solution is greater than or equal to the cost of the largest job
          T* ≥ max_j t_j
        Before we assign the last job t_j to machine T_i, all the machines must
        have a cost of at least: T_i - t_j:
          m(T_i - t_j) ≤ ∑(k=1..m) T_k
          T_i - t_j ≤ 1/m ∑(k=1..m) T_k
        We can combine this with our previous formulas to find:
          T_i - t_j ≤ T*, and
          t_j ≤ T*
        Therefore:
          T_i ≤ 2T*

      Improved Greedy Balancing Approximation
        Initially sort jobs in decreasing order of length, then use the same
        greedy balancing.

      Analysis:
        If we have more than M jobs, then we know at least one machine must have
        more than two jobs on it. Thus, we can adjust our optimal solution to
        be:
          T* ≥ 2t_m+1
        Further, since we order our jobs, then we can assume that:
          t_j ≤ t_m+1
        Combining these, we get:
          T* ≥ 2 * t_j or t_j ≤ T*/2
        From our unimproved greedy balancing analysis, we know that:
          T_i - t_j ≤ T*, and now
          t_j ≤ T*/2
        Therefore:
          T_i ≤ 1.5T*

  IV. Vertex Cover

    Statement:
      Find the smallest vertex cover in graph G.

    2-Approximation Algorithm:
      Start with S = Null
      While S is not a vertex cover
        Select an edge e not covered by S
        Add both ends of e to S
      Endwhile

    Analysis:
      For every edge that we look at, the optimal solution must have at least
      one of these nodes in the vertex cover.
      Our algorithm selects both nodes. So our solution must be at most twice as
      big as the optimal.

    Key Points:
      - This avoids the problem of selecting a bad node every time that we can
        see if we select the nodes around the center of a * shape.
      - We cannot apply this algorithm to independent set, because this may
        result in us having a null set as we can see if we run this algorithm
        on a square

  V. Independent Set

    Statement:
      Find the largest set of independent nodes in a graph.

    Theorem:
      Unless P=NP, there is no 1/n^(1-ε) approximation for the maximum
      independent set problem for any ε > 0, where n is the number of nodes in
      the graph.

  VI. Set Cover Problem

    Statement:
      Given a set U of n elements, a collection s1, s2,...sm of subsets of U,
      and a number k, does there exist a collection of at most k of these sets
      whose union is equal to all of U?

    Key Points:
      Since this problem has a one to one correspondence with the vertex cover
      problem, we can use the vertex cover problem to generate a 2-approximation
      for this problem.

  VII. Max 3-SAT

    Statement:
      Given a set of clauses of length 3, find a truth assignment that satisfies
      the largest number of clauses.

    Approximation Method:
      Set all values to false
        If half or more clauses are satisfied
          Return set of all values set to false
        Else
          return set of all value set to true

      Analysis:
        This is a 0.5 approximation

### 4. Linear Programming

#### Sources
- Lecture 12 (Week 14)
- Algorithm Design - Chapter 11

#### Overview:
  Linear Programming is a powerful tool. We can take a lot of different types of
  problems reduce them to linear program and solve them. Basically a very
  powerful modeling too. Used very often in optimization problems.

#### Key Points:
  - For our purposes, we can simply model the problem and assume that we send it
    into an engine that can solve linear programming problems.
  - Generally our objective function be maximized subject to the constraints of
    our accepted values at a vertex of our bounding plane or object. In special
    cases, our objective function will be parallel to the bounding shape and in
    this case we will meet at multiple vertices. Thus, if we limit our search
    to the vertices of this region, we will be able to find the optimal solution
  - Linear Programming is in P
  - Most of the time we use the Simplex method, which does not run in P, but has
    good performance in practice.
  - Can be used to solve problems both approximately and exactly.
  - Remember zero contraints

#### Definition:

  I. System of Linear Equations

    [A][x] = [B]
    A - coefficient matrix
    x - vector of unknowns
    B - Right Hand Side Vector

  II. Linear Programming

    Continuous variables
    [A][x] ≥ [B]
    A - coefficient matrix MxN
    x - vector of unknowns
    B - Right Hand Side Vector

    Objective function: [C^T][x]

    Goal: Minimize the objective function subject to the above constraints.

  III. Integer Programming

    Deals with discrete variables

  IV. Mixed Integer Programming

    Deals with both types of variables

  V. Non-Linear Programming

    Deals with non linear constrains for objective functions

#### Methods:
  I. Simplex Method

    Overview:
      Start with some random vertex in the convex region.
      Check the value of the optimal solution.
      Move to the next vertex.
      If the value of the optimal solution is increasing, continue in that
      direction.
      Once it goes down, stops and retraces to the optimal solution.

    Key Points:
      - Theoretically the simplex method is not a polynomial-time solution
      - In practice, it runs in polynomial-time

#### Problems:
  I. Weighted Vertex Cover Problem

    Statement:
      For G = (V,E), S ≤ V is a vertex cover set such that each edge has at
      least one end in S
      Also, w_i ≥ 0 for each i in V.
      So the total weight of the set = w(s) = ∑ for all i in S: w_i
      Minimize w(s)

    Key Points:
      - This is an NP-Hard problem, because it is at least as hard as the
        optimization version of the Vertex Cover problem.

    Solution - ILP:
      x_i is a decision variable for each node i in V
      x_i = 0 if i not in S
      x_i = 1 if i in S
      x_i + x_j ≥ 1, if x_i and x_j share an edge

      objective function:
        Minimize ∑ for all i: w_i * x_i
        Subject to:
          x_i + x_j ≥ 1, for all (i,j) in E
          x_i in {0,1}

      Key Points:
        - This problem is Integer Linear Programming, because we have discrete
          values for x_i. Integer Linear Programming is NP-Hard

    Solution - LP:
      To find an approximate solution using LP, drop the requirement that
      x_i is in {0,1} and solve the LP in polynomial time to find
      {x_i*} between 0 and 1
      W_lp = ∑ for all i: w_i * x_i*

      Assume S' is the optimal vertex cover set and w(s') is the weight of the
      optimal solution:
        W_lp ≤ W(s'), because we relaxed the contraints and x_i can be anywhere
        between 0 and 1

      x_i* = 0, i not in S
      x_i* = 1, i in S

      Say S_≥1/2 = i in V: x_i* ≥ 1/2, because of this constraint:
        x_i + x_j ≥ 1, for all (i,j) in E, we know that this is a vertex cover
        solution.

      S is our approximate solution. We can compare it to our optimal solution,
      which is W_LP:
        W_LP ≤ W(s')
      And since we round up our x_i values at most a factor of 2 from 1/2 to 1
      in order to generate our set S, it holds that:
        W(S) ≤ 2 * W_LP
      And:
        W(S) ≤ 2 * W(S')

  II. Max Flow Problem

    Solution:
      Variables: flows going through edges
      Objective function: Maximize ∑ for e out of s: f(e)
      Constraints:
        0 ≤ f(e) ≤ c_e for each edge e in E
        ∑ for e into v: f(e) -  ∑ for e out of v: f(e) = 0, for v in V-{s,t}

    Key Points:
      - Handling lower bounds is very easy with linear programming, but this
        solution is slower than the max-flow algorithms.

  III. Multi-Commodity Flow

    Statement:
      f_i(e): flow of commodity i over edge e
      α_i: profit associated with one unit of flow for commodity i
      we have m commodities
      objective: maximize profit

    Solution:
      Maximize ∑i=1..m:∑e out of S: α_i * f_i(e)
      Subject to:
        0 ≤ ∑i=1..m: f_i(e) ≤ c_e
        f_i(e) ≥ 0 for all i & e
        ∑e into v: f_i(e) =  ∑e out of v: f_i(e),
          for each v in V and for each i=1 to m

  VI. Shortest Path using LP

    Statement:
      Find shortest path from s to t.

    Solution:
      Shortest distance from s to v is d(v) for each v in V
      d(v) ≤ d(u) + w(u,v) for each edge (u,v) in E
      d(s) = 0

      Objective Function:
        Maximize d(t) such that:
          d(t) ≤ d(neighbor) + cost of edge(t, neighbor)
