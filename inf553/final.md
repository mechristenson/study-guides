# INF 553: Foundations and Applications of Data Mining
## Final Study Guide

### Test Details
  - Monday April 27th, 2020
  - 6:30 - 8:30 EST
  - https://blackboard.usc.edu/ -> Quizzes
  - Calculator Allowed

### Major Topics
  - Introduction to Data Mining
  - MapReduce
  - Frequent Item Sets and Association Rules
  - Shingling, Minhashing, Locality Sensitive Hashing
  - Recommendation Systems
  - Analysis of Massive Graphs
  - Mining Data Streams
  - Clustering Massive Data
  - Link Analysis
  - Web Advertising

### Sources of Information
  - INF 553 Lecture Notes, Wei-Min Shen [Lec]
  - Mining of Massive Datasets, Leskovec, Ullman [Book]

---
### Introduction to Data Mining

#### Sources
  - [Lec] 1/13/20
  - [Book] Chapter 1: Data Mining

#### Concepts
1. Data Mining
  - Knowledge Discovery from Data
  - Find patterns that are
    - Valid: hold on new data with some certainty
    - Useful: should be possible to act on the item
    - Unexpected: non-obvious to the system
    - Understandable: humans should be able to interpret the pattern
2. Big Data Lifecycle
  - Data is Stored
  - Data is Managed
  - Data is Analyzed
3. Challenges of Big Data
  - Too big to store in one place
  - Unexpected Network Failures
  - Unpredictable Diversity in Data
  - Messy, noisy and errors are inevitable
  - Dynamic
4. Data Mining Tasks
  - Descriptive Tasks
    - Find human interpretable patterns that describe the data, e.g. clustering
  - Predictive Methods
    - Use some variables to predict unknown or future values of other variables, e.g. Recommender systems
5. Bonferroni's Principle
  - If you look in more places for interesting patterns than your amount of data will support, you are bound to find crap.

#### Questions
---
### Map Reduce

#### Sources
  - [Lec] 1/13/20
  - [Lec] 1/20/20
  - [Book] Chapter 2: Large-Scale File Systems and Map-Reduce

#### Concepts
1. Cluster Computing
  - Processing massive amounts of data requires cluster computing
  - Cluster architecture consists of:
    - Server "Nodes" with their own CPU, GPU, Memory, Disk
    - Rack of (16-64) Nodes connected by Switch with ~ 1GB/s bandwidth between nodes
    - System of Racks connected by Switch to a Backbone Switch with ~2-10GB/s bandwidth between Switches
2. Common Challenges of Cluster Computing
  - Node Failures
    - Even if a node can stay up for 3 years, with 1000 servers we will have one failure per day.
    - Need to store data persistently and keep it available when nodes fail
    - Need to cleanly handle node failures during a long computation
  - Network Bottleneck
    - Need a framework that does not move data too much during computation
  - Distributed/Parallel programming is hard
    - Need a simple model that hides complexity
3. Map Reduce
  - Programming model that addresses challenges of cluster computing
  - Node Failure
    - Store data redundantly on multiple nodes
  - Network Bottleneck
    - Move computation close to data to minimize data movement
  - Distributed Programming
    - Map and Reduce functions provide simple model
4. Distributed File System
  - Stores data multiple times across a cluster
  - Provides global file namespace
  - Ex. Google GFS, Hadoop HDFS
  - Used for huge files
  - Data is rarely updated in place
  - Reads and appends are common
5. DFS Implementation
  - Data is kept in chunks spread across machines (chunk servers)
  - Each chunk is replicated on different machines
  - Chunk servers also serve as compute servers -> bring computation to data
6. DFS Components
  - Chunk Servers
    - Files are split into contiguous chunks (usually 16-64 mb)
    - Each chunk is replicated (usually 2x or 3x)
    - Replicas exist in different racks if possible
  - Master Node
    - Stores meta data about where files are stored
    - Should be replicated, otherwise it is a single point of failure
  - Client Library for File Access
    - Talks to master to find chunk servers
    - Connects directly to chunk servers to access data without going through master
7. Word Count Task
  - Case 1: File is too large for memory but all <word, count> pairs fit in memory
    - Use hash table (word:count) to store number of times that word appears
    - Make simple sweep through file and have word count pairs for every unique word
  - Case 2: <word, count> pairs do not fit in memory
    - Map-Reduce
8. Map Reduce
  - Map
    - Divide files into many records
    - Extract something from each record as a key
    - Output one or multiple things for each record
  - Shuffle
    - Group by Key
    - Ensure all k,v pairs with same key go to the same reduce task
    - Hash, merge, shuffle, sort parition
  - Reduce
    - Aggregate, summarize, filter or transform
    - Output the result
9. Combiners
  - Often a map task will produce many pairs with the same key
  - We can use combiners to aggregate all values in some way so we reduce computation cost on the reducers
  - We can save network time by pre-aggregating values inside the mapper
  - Run after mappers and before reducers
  - Receive all data emitted by a mapper as input
  - Output is sent to reducers
  - If a combiner is needed, instances of combiner are run on every node that runs map tasks
  - Combiner is a mini-reduce process (usually the same as the reduce function)
  - When to use?
    - Can be used if a reduce function is commutative and associative
    - If the reducer cannot be used directly as combiner, we can still write a third function to use like average
10. Map Reduce Environment
  - Partitions input data
  - Schedules programs execution across a set of nodes
  - Performs groupByKey step
  - Handles machine failures
  - Manages inter-machine communication
11. Map Reduce Data Flow
  - Input and final output are stored on a DFS
    - Scheduler tries to scheduler map tasks "close" to the physical storage location of the data
  - Intermediate results are stored on the local filesystem of map workers
  - Output can be an input to another MR task
12. Coordination: Master
  - Master node handles coordination
    - Task status: idle, in-progress, completed
    - Idle tasks: get scheduled as workers become available
    - When a map task completes, it sends the master the location and sizes of its R intermediate files, one for each reducer
    - Master pushes this info to reducers
    - Pings workers periodically to detect failures
13. Dealing with Failures
  - Map Worker Failure
    - Map tasks completed or in-progress at worker are reset to idle
    - Idle tasks are eventually rescheduled to another worker
  - Reduce worker failure
    - Only in-process tasks are reset to idle
    - Idle reduce tasks are restarted on other workers
  - Master failure
    - Map-reduce task is aborted and client is notified
14. How Many Map and Reduce Jobs
  - Rule of thumb:
    - Make M much larger than the number of nodes in the cluster
    - One DFS cluster per map task is common
    - Improves dynamic load balancing and speeds up recovery from worker failures
  - R is usually smaller than M
    - Output is spread across R files
  - Google Example:
    - 200,000 map tasks and 5000 reduce tasks on 2000 machines
15. Map Reduce Refinements:
  - Combiners (See Above)
  - Partition Function
    - Control how keys get partitioned
      - Reduce needs to ensure that records with the same intermediate key end up at the same worker
    - System uses a default partition function hash(key) % R, sometimes it is useful to override this.
16. Good Problems for Map Reduce
  - Big data (Terabytes not Gigabytes)
  - Don't need fast response time
    - Good pre-computation engine
  - Applications that work in batch mode
  - Runs over entire data set
  - Does not provide good support for random access to dataset
  - Data should be expressible as k,v pairs without losing context or dependencies
    - Graphs are hard
17. Map Reduce Examples

  1. Word Count

    ```input: big document
    output: word counts
    map(key, value):
      // key: document name, value: text of document
      for each word in value:
        emit(word, 1)
    shuffle(key)
      // sort by key
      return hash(key) % buckets
    reduce(key, value):
      // key: word, value: iterator over counts
      result = 0
      for each count v in values:
        result += v
      emit(key, result)
      ```

  2. Inverted Index

    ```input: list of documents
    output: words and the documents they are located in
    map(key, value):
      // key: document name, value: text of document
      for each word in value:
        emit(word, document_name)
    shuffle(key)
      // sort by key
      return hash(key) % buckets
    reduce(key, value):
      // key: word, value: iterator over document_names
      result = []
      for each doc in document_names
        result.append(doc)
      emit(word, result)
      ```

  3. Integers divisible by 7

    ```input: large file of integers
    output: all unique integers that are evenly divisible by 7
    map(k, v):
      // key: chunk, value: list of values in chunk
      // note: we check for uniqueness AND divisiblity in map to reduce communication cost!
      for v in set(value_list):
        if (v % 7) == 0:
          emit(v, 1)
    shuffle:
      // group together all values for same integer:
        emit (integer, (1, 1, 1 ...))
    reduce(k, v):
      // eliminate duplicates
      emit (key, 1)
      ```

  4. Find Largest Integer

      ```input: large file of integers
      ouput: largest integer from file
      map:
        - params: (chunk_id, [ints])
        - return: (1, max([ints]))
      shuffle:
        - return: (1, [max_ints])
      reduce:
        - params: (1, [max_ints])
        - return: max([max_ints])
        ```

  5. Count the number of unique integers

      ```input: large file of integers
      output: number of distinct integers
      map-1:
        emit(int, 1) for unique integers
      shuffle-1:
        combine (int, (1, 1, 1...)) results from map-1
      reduce-1:
        eliminate duplicates, return (int, 1)
      map-2:
        input: chunk of unique integers from reduce-1
        count number of unique ints: (1, 3, 5) => (1, 3)
      reduce-2:
        sum all counts from map-2
        ```

  6. Map Reduce with Combiner - Compute Average

      ```map: produces (key, (number of ints, sum of ints) for each chunk
      reduce: sums the sum of integers and the number of integers, calculates average
      ```

  7. Relational Join

      ```map:
          key: key used for join, value: tuple with all fields from the table
          reduce: emit joined values
          emit (key, value) pair
      combine:
        group together all values with each key
      reduce:
        emit joined values
      ```

  8. Matrix Multiplication - One Phase

    ```input:
      - A: LxM matrix
      - B: MxN matrix
    output:
      - C: LxN Matrix
    map:
      for each element (i,j) of A, emit ((i,k), A[i,j]) for k 1..N
      for each element (j,k) of B, emit ((i,k), B[j,k]) for k 1..N
    reduce:
      C[i,k] = sum_j(A[i,j] x B[j,k])
    ```

  9. Matrix Multiplication - Two Phase

    ```phase 1: multiply appropriate values
    map 1:
      for each matrix element A[i, j], emit(j, ('A', i, A[i,j]))
      for each matrix element B[j, k], emit(j, ('B', k, A[i,k]))
    reduce 1:
      for each key j, produce all possible products
      for each value of (i,k) emit ((i,k), (A[i,j] * B[j,k]))
    phase 2: add up values
    map 2:
      Let the pair of ((i,k), (A[i,j] * B[j,k])) pass through
    reduce 2:
      For each (i,k), add up the values, emit ((i,k), SUM(values))
    ```

#### Questions
---
### Frequent Item Sets and Association Rules

#### Sources
  - [Lec] 1/27/20
  - [Lec] 2/3/20
  - [Book] Chapter 6: Frequent Itemsets

#### Concepts

#### Questions
---
### Shingling, Minhashing, Locality Sensitive Hashing

#### Sources
  - [Lec] 2/10/20
  - [Lec] 2/18/20
  - [Book] Chapter 3: Finding Similar Items

#### Concepts

#### Questions
---
### Recommendation Systems

#### Sources
  - [Lec] 2/24/20
  - [Lec] 3/2/20
  - [Book] Chapter 9: Recommendation Systems

#### Concepts

#### Questions
---
### Analysis of Massive Graphs

#### Sources
  - [Lec] 3/9/20
  - [Lec] 3/23/20
  - [Book] Chapter 10: Analysis of Social Networks

#### Concepts

#### Questions
---
### Mining Data Streams

#### Sources
  - [Lec] 3/30/20
  - [Book] Chapter 4: Mining Data Streams

#### Concepts

#### Questions
---
### Clustering Massive Data

#### Sources
  - [Lec] 4/6/20
  - [Book] Chapter 7: Clustering

#### Concepts

#### Questions
---
### Link Analysis

#### Sources
  - [Lec] 4/6/20
  - [Lec] 4/13/20
  - [Book] Chapter 5: Link Analysis

#### Concepts

#### Questions
---
### Web Advertising

#### Sources
  - [Lec] 4/20/20
  - [Book] Chapter 8: Advertising on the Web

#### Concepts

#### Questions
