# CSCI 561: Foundations of Artificial Intelligence
## Midterm 3 Notes

### Quantifying Uncertainty
#### Sources
- Lecture 17-18 - Week 10-11
- Discussion - Week 10
- AIMA - Chapter 13

#### Key Points
  Uncertainty
    Outside scope of awareness
    Too complex to reason about in complete detail
    Too expensive or risky to ensure certainty
    Problem/Information has inherent randomness

  Sources of Uncertainty
    Theoretical Ignorance
    Practical Ignorance
    Laziness

  Probability Disjunction: P(A v B) = P(A) + P(B) - P(A ^ B)

  Binomial Coefficient: (x choose y) = x! / y! * (x - y)!

  P(a) = Probability that a occurs
  P(A) = Probability Vector of A

  P(A|B) = P(A ^ B) / P(B)

  A and B are independent iff:
    P(A|B) = P(A) or
    P(B|A) = P(B) or
    P(A,B) = P(A)P(B)

  P(x^y) = 0 if x and y are mutually exclusive

  P(!x) = 1 - P(x)

  Bayes' Rule:
    P(a|b) = P(b|a) P(a) / P(b) = α P(b|a)P(a)
    P(a|b) / P(!a|b) = (P(b|a) P(a)) / (P(b|!a) P(!a))
    Shifts from causal to diagnostic direction

  Normalization:
    P(X|e) = P(X,e)/P(e) = αP(X,e) = αΣ for all y: P(X, e, y)
    Sum out the unwanted variables

  Conditional Independence
    Two dependent variables may be come conditionally independent when conditioned
      on a third variable.
    P(X,Y|Z) = P(X|Z) P(Y|Z) or
    P(X|Y,Z) = P(X|Z) or
    P(Y|X,Z) = P(Y|Z)
    Can decompose individual large tables into sets of smaller ones
    Can reduce the size of representation from exponential to linear

  Probability Distributions
    Prior/Unconditional/Marginal Distribution
      Probability of a variable without any evidence, P(X)
    Joint Distribution
      Probability of combinations of variables, P(X,Y)
    Posterior/Conditional Distribution
      Probability of variables given knowledge of other variables, P(X|Y)

### Probabilistic Reasoning
#### Sources
- Lecture 19 - Week 11
- Discussion - Week 11
- AIMA - Chapter 14

#### Key Points
  Absolute Independence
    Two variables who don't affect each other:
      P(A|B) = P(A), P(A,B) = P(A)P(B)

  Conditional Independence
    Two variables who don't affect each other given a third variable:
      P(A ^ B|C) = P(A|C)P(B|C)

    A Node is conditionally independent of its non-descendents given its parents
    A Node is conditionally independent of all others given its markov blanket

  Probability Formulas
    Conditional Probability: P(A|B) = P(A ^ B) / P(B)
    Product Rule: P(A ^ B) = P(A|B) P(B) = P(B|A) P(A)
    Bayes' Rule: P(A|B) = P(B|A) P(A) / P(B)

  Bayesian Networks
    Graphical representation of probability distribution
      Nodes represent variables
      Directed links from parent node to child node
      Each node has a probability distribution given its parents
    Contain CPTs -> Conditional Probability tables

     +--------+     +----------+    CPTs:
     |Burglary|     |Earthquake|    P(B)|0.001    P(E)|0.002
     +---+----+     +------+---+          P(A)
         |    +-----+      |        B | E|0.95
         +--> |Alarm| <----+        B |!E|0.94
              +-----+               !B| E|0.29
              |    |                !B|!E|0.001
    +---------v+   +v---------+     ...
    |John Calls|   |Mary Calls|
    +----------+   +----------+

    John and Mary's Calls are directly influenced by the alarm
    Calls are conditionally independent of burglaries or earthquakes given alarm
    Calls are not absolutely independent of burglaries or earthquakes

  Bayesian Network Inference
    Inference by Enumeration
      John & Mary Call, was there a burglary?
          P(B|J,M) = αP(B,J,M)
        Using marginalization, introduce hidden variables
          αΣAΣE P(B,J,M,E,A)
        Using product rule, break into conditionals
          αΣAΣE P(B) P(E|B) P(A|B,E) P(J|B,E,A) P(M|B,E,A,J)
        Independence Rules
          αΣAΣE P(B) P(E) P(A|B,E) P(J|A) P(M|A)

    Inference can be optimized with Variable Elimination
      Use dynamic programming to save intermediate vars

    Approximate Inference - Monte Carlo Method
      Treat network as a simulator
      Run one simulation to get a sample of distribution
      As we get more samples, answer becomes more accurate

      Direct Sampling
        Sampling with no evidence
        Reset and generate another sample
        P(Event) = # Samples with Event / Total # of samples

      Rejection Sampling
        Generate samples as in direct sampling
        Reject samples that are inconsistent with evidence

      Likelihood Weighting
        Don't sample observed variables
          Weight sample by likelihood of observed values
          Normalize weights to get final probability

      Markov Chain Monte Carlo
        Local search through sample states
          Gibbs Sampling
            Evidence vars fixed to observed states,
              all others assigned to random vars
            Generate a new state by sampling one non-evidence var
              Conditioned on values of markov blanket (parents, children, children's parents)
            Over time the % of time in a state approaches its probability

      Dempster-Shafer Theory
        Belief functions
          90% sure the alarm goes off with 50% chance
            Bel(Alarm) = 0.9 * 0.5 = 0.45

      Fuzzy Logic
        Set membership is no longer true/false
        Uses truth values from 0 to 1
        Logical operators defined over fuzzy predicates

  Building a Bayes Net
    Pick a set of vars and order them from parents -> children
    Loop through variables in order
      Add node to network
      Determine its parents from nodes already in network
      Write down CPT for the node given its parents

    Put causes before effects
      Links should flow causal not diagnostic
      Otherwise you will add compensatory links
    Finding an optimal ordering is NP-Hard
    Finding an optimal variable elimination is NP-Hard

### Making Simple Decisions
#### Sources
- Lecture 20 - Week 12
- Discussion - Week 12
- AIMA - Chapter 16

#### Key Points
  Properties of Preferences
    Orderable
      A>B or A<B or A~B is true
      Violated when: A>B sometimes and A<B others
    Transitive
      A>B>C -> A>C
      Violated when: A>B>C>A
    Continuous
      A>B>C -> There exists some p [p: A, 1-p: C] ~ B
      Violated when: There is not some p [p: A, 1-p: C] ~ B for A>B>C
    Substitutable
      A~B -> For all p [p: A, 1-p: C] ~ [p: B, 1-p: C]
      A>B -> For all p [p: A, 1-p: C] > [p: B, 1-p: C]
    Monotonic
      A>B -> (p>q <-> [p: A, 1-p: B] > [q: A, 1-q: B]
      Violated: Allais Paradox
    Decompostable
      [p: A, 1-p,[q: B, 1-q: C]] ~ [p: A, (1-p)q: B, (1-p)(1-q): C]
      Violated when: you prefer probabilities in a series rather than at once

  Decision Networks - Influence Diagrams
    Similar to Bayesian Networks
      Action Choices: Rectangular Decision Nodes
      Bayesian Network: Circular Chance Nodes
      Outcome Preferences: Diamond Utility Nodes

    Solving Decision Networks
      Consider each possible set of decisions
      Compute posterior probability of utility node parents
      Compute expected utility (EU)
      Choose decisions with Maximum EU (MEU)

### Making Complex Decisions
#### Sources
- Lecture 21 - Week 12
- Discussion - Week 12
- AIMA - Chapter 17

#### Key Points
  Markov Decision Process
    Consists of:
      S: States
      A: Actions
      P(s'|s,a): Transition Model
      R(s): Reward
    Goal:
      Find optimal policy π

    Finite Horizon
      Robot will operate for N moves
      Utility and policy are not stationary
    Infinite Horizon
      Robot will keep operating
      Utility and policy are stationary
      Potentially Unbounded -> Use a discount factor γ ≤ 1
        Reward now is better than reward later

    Value Iteration
      Bellman Equation
        U(s) = R(s) + γ * Max_Action(Σ(P(s'|s,a)U(s')))
        Repeating this formula will converge to an optimal policy
      Q Values
        Utilities of each possible action

  POMDPs
    Partially Observable MDPs
      Observations instead of direct knowledge of state
      States -> Belief States
        Update b = P(S|observations) each turn
      Policies are defined over belief states π(b)

    Value Iteration still works
      Compute U(b) for belief states

    Expressive but Expensive
      Space of beliefs is the space of distributions over states

  Game Theory
    Other agents introduce added uncertainty
      But we can assume they are also optimizing

    Single Move Games
      All agents reveal their moves simultaneously

    Repeated Games
      Multipe rounds of a game

    Sequential Games
      Chess, backgammon
      Introduce randomness as a third player

    Imperfect Information
      Use belief states

    Pure Strategy
      Deterministic Policy
        Each agent chooses a best response to others strategies
        Each agent continues updating
        If we reach a point where no one updates -> Nash Equilibrium

    Mixed Strategy
      Choose actions with some probability
        Best response strategy is exactly the same
        If any player switches strategies they will do worse
          No incentive to change: Nash Equilibrium -> Local Optimum

### Learning From Examples
#### Sources
- Lecture 22 - Week 13
- Discussion - Week 13
- AIMA - Chapters 18, 20

#### Key Points
  Machine Learning
    An agent is learning well if it improves its performance on future tasks
    after making observations about the world

    Reasons
      Agents rarely know everything when they start
        Theoretical Ignorance
        Practical Ignorance
        Laziness
      Agents should use experience to fill in gaps
        Deductive learning leads to correct knowledge
          Logical inference generates entailed statements
          Probabilistic Reasoning can lead to updated belief states
          Often have insufficient knowledge for inference
        Inductive Learning can arrive at incorrect conclusions
          But can be better than not trying to learn at all

    Rote Learning
      Memorization

    Inductive Learning
      Supervised Learning
        Someone has labeled the data with the "correct" value
      Unsupervised Learning
        We don't know the "correct" value, but we can detect patterns
      Reinforcement Learning
        We don't know the "correct" value, but we get reward/punishment

    Supervised Learning
      Given a training set of data (input -> output)
      Choose an input -> output function

      Decision Trees - C4.5
        Learned function is set of if-then-else rules
        Branches: test of input variables
        Leaves: value for output variable

        Learning a Decision Tree
          If all of our examples have the same output value:
            Create a leaf node with that value
          Else, choose the best input var to branch on
            Divide examples according to their values for this variable
            Start back at top with each separate subset of examples
          If no more examples, this is a sequence of branches we've never seen
            Create a leaf node with the most common output value overall
          If we have no more input variables, the rest is noise
            Create a leaf node with the most common output value overall

          To choose an attribute test we can calculate entropy
            Entropy (H): A precise measure of information
              H(V) = -Σ_vi P(V = vi) log_2 P(V=vi)
              For a 50-50 binary variable, V:
                H(V) = -0.5 * log_2 0.5 - 0.5 * log_2 0.5 = 1
              For a 100-0 binary variable, V:
                H(V) = -1 * log_2 1 - 0 * log_2 0 = 0
            Information Gain
              IG(A) = I(p/p+n, n/p+n) - remainder(A)
                I(p/p+n, n/p+n) = Entropy
                Remainder(A) =  Σ(i..V) (p_i + n_i)/(p + n)  * I(p_i/p_i+n_i, n_i/p_i+n_i)
              IG(A) = Entropy of Total - Sum(Ratio part/whole * Entropy Part)

        Extending Decision Trees
          Missing Data
            Consider all possible values but weight by frequency
          Input Vars with infinite domains
            Branch on split points
          Continuous-Valued Output
            Regression Tree: Leaf nodes are functions of inputs
          Multivalued Attributes
            Increasing number of values often increases information
            We run the risk of overfitting
              We fit the data well but we are too specialized
              Cannot handle new data

      Evaluating Learning
        Test on new data
        Cross-Validation
          Holdout Cross-Validation
            Partition our existing data into training and testing sets
            Our learning never uses the examples in the test set
          K-Fold Cross-Validation
            Partition our existing data into k subsets
            k rounds of learning, using a different subset as test set
          Leave-One-Out Cross Validation
            k = # of examples

      Naive Bayes
        Probabilistic Classification if we don't want just a T/F classification
        Output variable is parent of all the input variables
        "Naively" assumes conditional independence of input
        Classification is posterior probability
          Missing data is easy to handle with Maximum Likelihood

      K-Nearest Neighbors
        Nonparametric Method
        Don't summarize data into parameters, use your data directly

        For a new input, find k examples that are closest to it
        Choose the output that appears most often in those examples

        Pro: no loss of information
        Con: too much data

      Support Vector Machines
        Nonparametric Method
        Keep only examples that are "support vectors" (mark border)
          Often a constant number per dimension
        Linear separators may not always exist, use kernal functions to
          translate space into an alternate one with more dimensions

      Supervised Learning Summary
        Pros
          - no prior knowledge
          - hugely successful in practice
        Cons
          - hard to exploit prior knowledge
          - not guaranteed to be correct
          - may not have anyone to supervise the learning system

  Neurons
    Connectionisum
      Activation is passed along links from neuron to neuron
      Neuron become a computational unit

    Activation, A
      Represents how strongly true or false a varaible is
    Input activation, in
      Each link from node i to node j has a weight w_ij
        Strength of influence that node i has on node j
        Node 0 is typically a dummy input
      in_j = Σi: w_ij a_i
    Activation function, g
      Translate input activation into output activation
      a_j = g(in_j) = g(Σi: w_ij a_i)

    Perceptrons
      Single-layer feed-forward with threshold activation
        output nodes have activation of 0 or 1
        output node is 1 iff weighted sum of inputs ≥ 0
      Similar to Noisy Or, but different
        Input nodes can be any real value
        Weights on links do not necessarily have probabilistic meaning

      Can represent some logical functions and any linear inequality
        AND, OR, NOT, weighted sum and threshold, n-dimesional hyper plane

      Majority Function
        1 iff > 1/2 of n binary variables are 1
        representable within a perceptron
          w_i(n+1) = 1
          w_0(n+1) = -(n/2)
          a_n+1 = 1 iff a_1 + a_2 + ... + a_n -(n/2) ≥ 0
        decision tree needs an exponential number of branches

      Learning
        Parametric Learning
          find weights for all of the links in the network
          don't need to keep any data or support vectors
        Iterative refinement of weights
          start with random initial weights, w
          for each data point with input x and output y
            compare perceptron output on x against real output y
            if they do not match, then update w to correct error
          repeat until convergence

      Linear Inseparability
        Linear separability is rare
        Perceptrons cannot represent inseparable functions
          support vector machines solve by transforming dimensions
          neural networks solve by adding more layers

    Network Structures
      Recurrent Networks
        Directed cycles are allowed
        Activation flows and hopefully settles into a stable state

      Feed-Forward networks
        links flow in one direction
          no directed cycles
        units are sorted into layers
          input (evidence) variables are in the first layer
          output (query) variables are in the last layer
          all other variables are hidden units

      Multilayer Neural Networks
        Introduce hidden nodes
          Expand the possible functions the network can capture
          Although the hidden nodes have no real-world meaning
          Therefore it is hard to know how many layers and nodes
        Our data does not include values for hidden nodes
          Otherwise they would be input or output nodes
          But we can "blame" hidden nodes for errors in the next layer
          Assign blame based on the strength of their influence

        Back-Propagation
          Start from output nodes
            Perceptron: w_i <- w_i + α(y-h_w(x)) * x_i
            Multilayer: w_i,j <- w_i,j + α ∆_ j * a_i
          Assign blane to hidden nodes
            based on error in next layer
            modified error becomes ∆_ j = g(in_j)Σ_k w_j,k ∆_ k
              increases with error in next layer
              increases with weight on link to next layer

  Deep Learning
    Multilayer neural networks are one form
      train each layer as if it were a standalone neural network
      a hierarchy of loosely connected learning systems
    Basic principles are the same as what you have learned
      newer methods for dealing with larger data sets
      finding representations and structures similar to the brains
      exploiting steady increases in computer power

  Reinforcement Learning
    Utility function is unknown at the beginning
      when agent visits each state, it receives a reward
        possibly negative
    What function to learn?
      R(s): utility based agend
          if it already knows transition model
          use MDP to solve for MEU actions
      U(s,a): Q-Learning agent
        if it doesn't already know the transition model
        pick action that has highest U in current state
      π*(s): reflex agent
        learn a policy directly, then pick the action that the policy says

    Q-Learning
      Modify bellman equation to learn Q values of actions
        U(s) = R(s) + γ max_action Σ_s1 (P(s1|s,a)U(s1))
      Update Q after each step
        if we get a good reward now, increase Q
        if we get a good reward later, increase Q
        Q(s,a) <- Q(s,a) +  α(R(s) + γ max_action(Q(s'-a')-Q(s,a)))
          α is the learning rate
          γ is the discount factor
      Converges to correct values if α decays over time
        Similar to simulated annealing

    SARSA
      Modify Q learning to choose action a', not max
      On policy instead of off policy
        SARSA learns based on real behavior, not optimal behavior
          Good if real world provides obstacles to optimal behavior
        Q learning learns optimal behavior, beyond real behavior
          Good if training phase has obstacles that won't persist

    Q-Learning and SARSA
      converge to correct values
      avoids complexity of solving an MDP

    Data Mining
      C4.5, Decision Trees
      K-Means
      SVMs
      Apriori
      EM
      PageRank
      AdaBoost
      k-Nearest Neighbors
      Naive Bayes
      CART
