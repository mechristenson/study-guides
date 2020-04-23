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

#### Map Reduce Examples

  1. Word Count
```
input: big document
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
```
input: list of documents
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
```
input: large file of integers
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
```
input: large file of integers
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
```
input: large file of integers
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
```
map: produces (key, (number of ints, sum of ints) for each chunk
reduce: sums the sum of integers and the number of integers, calculates average
```

  7. Relational Join
```
map:
  key: key used for join, value: tuple with all fields from the table
  reduce: emit joined values
  emit (key, value) pair
combine:
  group together all values with each key
reduce:
  emit joined values
```

  8. Matrix Multiplication - One Phase
```
input:
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
```
phase 1: multiply appropriate values
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
1. Market Basket Model
  - Identify items that are bought together by sufficiently many customers
  - Components
    - A large set of items, e.g. things sold in a supermarket
    - A large set of baskets
    - Each basket is a small subset of items
  - Want to discover association rules between items
  - Want to focus on common events
2. Applications of the Market Basket Model
  - Identify items bought together
    - items: products
    - baskets: set of products someone bought in a trip to the store
  - Plagarism detection
    - baskets: sentences
    - items: documents containing those sentences
    - look for items that appear together in several baskets
      - documents that share sentences
  - Identify related concepts in web documents
    - items: words
    - baskets: web pages
  - Drug Interactions
    - baskets: patients
    - items: drugs and side effects
3. Support
  - The support for itemset i is the number of baskets containing all items in I
  - Given a support threshold s, sets of items that appear in at least s baskets are called "Frequent Itemsets"
4. Association Rules
  - If-then rules about the contents of baskets
  - {i1,... ik} -> j means if a basket contains all of items i1...ik, then its likely to contain j
5. Confidence
  - Probability of j given i1..ik
  - Ratio of support: (support for I AND j) / (support for I)
  - We want to identify association rules with high confidence
6. Interest
  - Not all high confidence rules are interesting
  - Interest of association rule I->j: difference between its confidence and the fraction of baskets that contain j
  - Interest(I->j) = conf(I->j) - Pr[j]
  - Interesting rules are those with high positive or negative values
  - High positive/negative interest means the presence of I encourages or discourages the presence of j
7. Finding Useful Association rules
  - Problems usually state: find all association rules with support >= s and confidence >= c
  - Most work is in calculating frequent itemsets
  - Note: if {i} -> j has high support and confidence, then both {i} and {i,j} are frequent
  - Assume that there are not too many frequent itemsets or candidates for high support, high confidence rules
  - We should adjust the support threshold to avoid having so many frequent itemsets that we can't act on them
8. Naive Frequent Item Algorithm
  - Read file once, counting into main memory, the occurrences of each pair
  - Number of pairs in a basket of n items: n choose 2, (n! / (k!(n-k)!))
  - This builds quickly: number of pairs in all baskets ~#items^2
  - If you can't store all pairs, you can't use this algorithm
  - Two approaches:
    - Count all pairs in triangular matrix
      - requires only 4bytes/pair, but requires a count for each pair
    - Keep a table of triples [i,j,c] = count of the pair of items
      - requires 12 bytes but only for pairs with count >0
9. Naive Triangle Matrix Approach
  - N = total number of items
  - Order each pair so i<j
  - Keep pair counts in lexicographical order
  - Every time you see a pair, increment that position
  - Total number of pairs: n(n-1)/2, total bytes = 2n^2
10. Triples Approach
  - Beats approach 1 if fewer than 1/3 of possible pairs actually occur in the market basket data
11. A-Priori Algorithm
  - Simple solution when we cannot fit all item pairs into memory
  - Key idea: monotonicity, if a set of items i appears at least s times, so does every subset J of I
  - Key idea: contrapositive, If item i does not appear in s baskets, then no pair including i can appear in s baskets
  - Algorithm:
```
pass1: read baskets and count in main memory the occurrences of each item
identify frequent itemsets of size 1

pass2: find all candidate pairs of frequent items of size 1 and count them
```
  - Main Memory
    - pass1: item counts
    - pass2: frequent items, counts of pairs of frequent items
  - Can use triangular matrix method if we re-number frequent items 1,2... and keep a map
12. General A-Priori Algorithm
  - For each k we construct two sets of k-tuples:
    - Candidate K-tuples
    - Frequent k-tuples
```
- Steps:
  - Count support of candidate itemsets
  - Prune non-frequent itemsets
  - Generate itemsets for next iteration
    - Repeat if candidate itemsets exist
    - Return frequent itemsets if no more candidates exist
```
13. PCY Algorithm
  - Main Idea: In Apriori pass 1, most memory is idle. Use this memory to reduce memory required in phase 2
```
Pass 1:
  - Count items
  - Maintain a hash table with as many buckets as fit in memory

  For each basket:
    For each item in basket:
      add 1 to item's count
    For each pair of items:
      hash pair to a bucket
      add 1 to the count of the bucket

Pass 1.5:
  - Replace buckets by a bit-vector: 1 means bucket count exceeded support s, 0 means it did not

Pass 2:
  - Count all pairs {i,j} that meet the conditions for being a candidate pair:
    - Both i and j are frequent
    - The pair {i,j} has been hashed into a frequent bucket
```
  - Main Memory
    - pass1: item counts, hash table for pairs
    - pass2: frequent items, bitmap, counts of candidate pairs
      - must use triples method for pass 2 since we don't count all pairs
      - must eliminate 2/3 of candidate pairs for PCY to beat Apriori
14. Multistage Algorithm
  - Main Idea: Memory is the bottleneck -> limit the number of candidates to be counted
  - Main Idea: After pass 1 of PCY, rehash only those pairs that qualify for Pass 2 of PCY
```
Pass 1:
  - Count items
  - Hash pairs {i,j} and increment bucket count

Pass 2:
  - Hash pairs {i,j} into Hash2 iff:
    - i,j are frequent
    - {i,j} hashes to a frequent bucket in B1

Pass 3:
  - Count all pairs {i,j} iff:
    - Both i and j are frequent
    - {i,j} hashes to a frequent bucket in B1
    - {i,j} hashes to a frequent bucket in B2
```
  - Main Memory
    - pass1: item counts, first hash table for pairs
    - pass2: frequent items, bitmap 1, second hash table for pairs
    - pass3: frequent items, bitmap 1, bitmap 2, counts of candidate pairs
  - Note: the two hash functions must be independent
  - Note: we need to check both hashes on the third pass
    - It is possible for a pair to hash to an infrequent bucket in B1, but a freq bucket in B2
  - We can put any number of hash passes between first and last stage to reduce false positives
    - Eventually all memory would be consumed by bitmaps with no memory left for counts
    - The truly frequent pairs will always hash to a frequent bucket
15. Multihash Algorithm
  - Key Idea: Use several independent hash tables on the first pass
  - Potentially get a benefit like multistage but in only two passes
  - Risk: halving the number of buckets doubles the average count
  - Main Memory
    - pass1: item counts, first hash table, second hash table
    - pass2: frequent items, bitmap 1, bitmap 2, counts of candidate pairs
16. Limited Pass Algorithms
  - There are many applications where it is sufficient to find most but not all frequent itemsets
  - Find all or most frequent itemsets in at most two passes:
    - Random Sampling
    - SON
    - Toivonen's Algorithm
17. Random Sampling
  - Take a random sample of the market baskets that fit in main memory
```
  Read entire dataset
  For each basket select that basket for the sample with probability p
    for input data with m baskets
  Run apriori or one of its improvements in main memory
    set support threshold to (p*s) or even (.9*p*s)
  Make a second pass through the full dataset
    Count all itemsets that were identified as frequent by the sample
    Verify that the candidate pairs are truly frequent
```
  - Main Memory
    - Copy of sample baskets, space for counts
  - This algorithm may produce:
    - false negatives: itemset is frequent in the whole but not in the sample
    - false positives: itemset is frequent in the sample but not in the whole
  - Making a second pass of the data will eliminate false positives but not false negatives
  - We can reduce false negatives by reducing our support threshold
18. SON Algorithm
  - Avoids false negatives and false positives
  - Requires two full passes over the data
```
Pass 1:
  Repeatedly read small subsets of the baskets into main memory
  Run an in-memory algorithm to find all frequent itemsets
  An itemset becomes a candidate if it is found to be frequent in any one or more subsets of the baskets

Pass 2:
  Count all candidate itemsets and determine which are frequent in the entire set
```
  - Key Idea: Monotonicity - an itemset cannot be frequent in the entire set of baskets unless it is frequent in at least one subset
  - SON lends itself to distributed data mining - MapReduce
```
Phase 1: Find local candidate itemsets
Map:
  - Input is chunk of all baskets, fraction p of total input file
  - Find itemsets that are frequent in that subset
  - Output is set of k,v pairs (F,1) where F is a frequent itemset from that sample
Reduce:
  - Sums and returns keys that are frequent as candidate items sets

Phase 2: Find true frequent itemsets
Map:
  - Each map task takes output from the first reduce and a chunk of total input data
  - All candidate itemsets go to every map task
  - Count occurrences of each candidate among the chunk of baskets
  - Output (C, v): C - candidate itemset, v - support
Reduce:
  - Sums associated values for each key: total support for itemset
  - Emit truly frequent itemsets
```
19. Toivonen's Algorithm
  - Given sufficient main memory, uses one pass over a small sample and one full pass over data
  - Gives no false positives or false negatives
  - Small but finite probability that it will fail to produce an answer
  - Must be repeated with a different sample until it gives an answer
```
Find candidate frequent itemsets from a sample
  - Start like random sampling but lower the threshold slightly for the sample
  - Goal is to avoid missing any itemset that is frequent in the full set of baskets
  - Smaller the threshold -> more memory consumed but more likely to find an answer
Process whole file and construct the negative border from frequent itemsets
  - Negative border: collection of itemsets that are infrequent in sample but all their immediate subsets are frequent
  - Count all candidate frequent itemsets from first pass
  - Count all itemsets on the negative border
  - Case 1: no itemset from negative border turns out to be frequent in the whole dataset
    - We found the exact correct set of frequent itemsets
  - Case 2: some member of the negative border is frequent in the whole dataset
    - Must repeat algorithm with new sample
```
  - Goal: save time by looking at a sample on first pass
  - If some member of the negative border is frequent in the whole dataset, can't be sure that there are not larger data sets that are also frequent, must restart
  - Choose low support threshold so probability of failure is low while number of itemsets still fit in main memory.

#### Questions
---
### Shingling, Minhashing, Locality Sensitive Hashing
1. Finding Similar Sets
  - Find near-neighbors in high-dimensional space
  - Common Problems:
    - Pages with similar words
    - Customers who purchased similar products
    - Images with similar features
  - Naive Solution
    - Given high dimensional data points and a distance function
    - Find all pairs of points that are within some distance threshold
    - O(N^2)
2. Three Steps for Finding Similar Docs
  - Shingling - Convert documents to sets
  - Min-hashing - Convert large sets to short signatures, while preserving similarity
  - Locality-Sensitive Hashing - Focus on pairs of signatures likely to be from similar documents
3. Shingles
  - A k-shingle (or k-gram) for a document is a sequence of k tokens that appears in the doc
    - Tokens can be characters, words or something else
  - Working assumption: documents that have lots of shingles in common have similar text, even if the text appears in a different order
  - Must pick k large enough or else most documents will have most shingles
  - We can hash long shingles to numbers to save space
  - We can represent a document by its set of k-shingles (bag of words)
4. Jaccard Similarity
  - The size of the intersection of two sets divided by the size of their union
  - Sim(A,B) = |AnB|/|AuB|
  - Jaccard Distance = 1 - Jaccard Similarity
5. Minhashing
  - If we use k-shingles to create signatures, we quickly have too many data points to compare sets efficiently
  - We can convert large sets of shingles to short signatures while preserving similarity
  - Vectors representing membership of shingles are sparse, so we can condense them with a minhash
  - Minhashing:
    - Pick a random permutation of rows
    - Define a hash function h(C) = the number of the first (in permuted order) row in which column c has 1
    - Use several (e.g. 100) independent hash functions to create a signature
    - Construct a signature matrix for columns
    - The similarity of signatures is the fraction of the hash functions in which they agree
  - Note: probability that minhash func for random permutation of rows produces same value for 2 sets = jaccard similarity of those sets!
  - Can simulate the effect of a random permutation by a random hash function
    - Map row numbers to as many buckets as there are rows
    - May have collisions on buckets but this is not important as long as number of buckets is large
  - Computing Minhash Signatures:
```
// i = hash function, c = document
Initialize M(i,c) with infinity in each entry

// iterate through original table to build signature matrix
For each row r
  For each column c
    if c has 1 in row r
      for each hash function hi do
        if hi(r) is smaller value than M(i,c) then
          M(i,c) := hi(r)
```
6. Locality Sensitive Hashing
  - Key idea: focus on pairs of signatures likely to be from similar documents
  - Even if we use minhashes to summarize docs with signatures, comparing signatures may take a long time
  - LSH Overview
    - Hash items several times
    - Candidate pair: any pair that hashes to the same bucket for any of the hashings
    - Check only candidate pairs for similarity
    - False positives: dissimilar pairs that hash to the same bucket
    - False negatives: truly similar pairs that do not hash to the same bucket for at least one of the hash functions


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
