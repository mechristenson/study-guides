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

#### Sources
  - [Lec] 2/10/20
  - [Lec] 2/18/20
  - [Book] Chapter 3: Finding Similar Items

#### Concepts
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
```
Partition signature matrix M into b bands with r rows per band
For each band, hash its portion of each column to a hash table with k buckets
  - Make k as large as possible
  - Use a separate bucket array for each band so columns with same vector in diff bands don't hash to same bucket
  - Candidate pairs are those that hash to the same bucket for at least one band
Tune M b and r to catch most similar pairs but few non-similar pairs
Verify in main memory if candidate pairs really have similar signatures
Optional: verify in data if candidate pairs really represent similar documents
```
7. Analysis of Banding Technique
  - With b bands of r rows each
  - Pair of documents have jaccard similarity t
  - Probability that all rows in a band are equal = t^r
  - Probability that some row in a band is unequal = 1 - t^r
  - Probability that no band has rows that are all equal = (1 - t^r)^b
  - Probability of being a candidate pair = 1 - ((1 - t^r)^b)
  - To avoid false negatives -> more bands less rows -> more opportunities to match
  - To avoid false positives -> less bands more rows -> harder to match and less opportunities
8. Banding Trade off Graphs
  - General Graph
    - Probability of becoming a candidate pair on the y axis
    - Jaccard Similarity of two sets on the x-axis
  - Ideal Graph
    - Vertical line representing similarity threshold s
      - If t < s (left of the line) no chance of sharing a bucket
      - If t > 1 (right of the line) 100% chance of sharing a bucket
  - 1 Band of 1 Row
    - Diagonal line x=y
  - B bands of R rows
    - S-curve
      - Less than support and s-curve: true negative (not similar, not cand pair)
      - Less than support and greater than s-curve: false positive (cand pair but not similar)
      - Greater than support and less than s-curve: false negative (not cand pair but similar)
      - Greater than support and s-curve: true positive (cand pair, similar)
9. Families of Functions for LSH
  - Families of functions (including minhash functions) that can serve to produce candidate pairs efficiently
  - Three conditions for family of functions
    - More likely to make close pairs be candidate pairs than distant pairs
    - Statistically independent
    - Efficient in two ways
      - Able to identify candidate pairs in much less time than to look at all pairs
      - Combinable to build functions better at avoiding false positives and negatives
  - A family H of hash functions is said to be (d1, d2, p1, p2)-sensitive if for any x and y in S
    - If d(x,y) <= d1, then prob over all h in H, that h(x) = h(y) is at least p1
    - If d(x,y) >= d2, then prob over all h in H, that h(x) = h(y) is at most p2
    - Note: we say nothing about what happens when the distance between items is between d1 and d2
      - We can make d1 and d2 as close as we wish
- Applications of LSH
  - LSH for Fingerprints
    - Place grid over fingerprint and normalize
    - Represent finger prints by set of grid points where minutiae are found
      - ridges merging or ridge ending
    - Make a bit vector for each fingerprint's set of grid points with minutiae
    - Optional: minhash bit vectors to obtain signatures
    - Many-to-many problem: take db of prints and check if there are any pairs that represent the same individual
    - Many-to-one problem: compare a fingerprint at a crime scene to a db of finger prints
  - Finding same/similar news articles
    - define a shingle to be a stop word plus the next two words -> avoids ads

#### Questions
---
### Recommendation Systems

#### Sources
  - [Lec] 2/24/20
  - [Lec] 3/2/20
  - [Book] Chapter 9: Recommendation Systems

#### Concepts
1. The Long Tail
  - Shelf space is scarce for traditional retailers
  - Web enables near-zero cost dissemination of information about products
  - More choice necessitates better filters
    - Recommendation engines
2. Key Problems for Recommendation Systems
  - Gathering known ratings for utility matrix
    - Explicit: surveys
    - Implicit: learn ratings from user actions
  - Extrapolate unknown ratings from known ones
    - Key problem: utility matrix is sparse
      - Most people have not rated most items
    - Cold Start
      - New items have no ratings
      - New users have no history
  - Evaluating extrapolation methods
2. Three Approaches to Recommendation Systems
  - Content Based
    - Use characteristics of an item
    - Recommend items that have similar content to items user liked in the past
    - Recommend items that match pre-defined attributes of the user
  - Collaborative Filtering
    - Build a model from a user's past behavior and similar decisions made by other users
    - Use the model to predict items that the user may like
    - Collaborative: suggestions made to a user utilize information across the entire user base
  - Hybrid Approaches
3. Content-Based Recommendations Systems
  - Main Idea: Recommend items to customer x that are similar to previous items rated highly by x
    - Requires characterizing the content of items in some way
  - General Strategy
    - Construct item profiles (item-based)
      - Create vectors representing items
        - Boolean vectors may indicate occurrence of high TF.IDF words, genres, tags
        - Numerical vectors may contain ratings
    - Construct user profiles (user-based)
      - Create vectors with same components that describe users's preferences
    - Recommend items to users based on content
      - Calculate cosine distance between item and user vectors
      - Classification algorithms
4. Cosine Distance
  - Cosine Distance = 1 - Cosine Similarity
  - `Cosine Similarity = (A.B) / |A||B|`
5. TF.IDF
  - Term Frequency * Inverse Document Frequency
  - Term Frequency, `TFij = fij / (max_k fkj)`, where fij is the frequency of term i in document j
    - Most frequent term has TFij = 1
  - Inverse Document Frequency `IDFi = log(N/ni)`, where N is num of docs and ni is number of docs that contain i
6. Recommender Systems for Documents
  - Make Recommendations based on features of documents
  - Hard to classify by topic
  - Use largest TF.IDF scores
  - Use source, author
7. User Profiles
  - Boolean utility matrix: average components of vectors representing item profiles for the items in which utility matrix has 1 for the user
  - Non-boolean utility matrix: weight the vectors representing profiles of items by utility (rating) value
    - normalize ratings for user by subtracting the users average rating
8. Summary of Content-Based Approach
  + No need for data on other users
  + No cold-start or sparsity problems (i.e new items can receive recommendations)
  + Able to recommend to users with unique tastes
  + Able to recommend new & unpopular items
  + Able to provide explanations
  - Finding the appropriate features is hard
  - Recommendations for new users is hard
  - Overspecialization
    - Never recommends items outside user's content profile
9. Collaborative Filtering Recommendation Systems
  - Main Idea: Collect user feedback and exploit similarities in rating behavior among users
  - Two classes of CF Algorithms:
    - Neighborhood-based or Memory-based approaches
      - User-based CF
      - Item-based CF
    - Model-based approaches
      - Estimate parameters of statistical models for user ratings
      - Latent factor and matrix factorization models
  - Works for any kind of item with no feature selection needed
10. User Based Collaborative Filtering
  - Weighted combination of ratings for a subset of similar users is used to make predictions for active user
  - Steps:
    - Assign a weight to all users wrt similarity with the active user
    - Select k users that have the highest similarity with the active user
    - Compute a prediction from a weighted combination of the selected neighbor's ratings
11. Similarity Metrics for Users
  - Jaccard Similarity
    - Ignore values in matrix, only look at which items are rated
    - Slight improvement but still not great:
      - Ignore values below 2/5, only look at items are rated 3 or above
  - Cosine Similarity
    - Slightly better than Jaccard
    - Improve ratings with normalization by subtracting average rating of user from each rating
  - Pearson Correlation
    - Most commonly used measure of similarity
12. Pearson Correlation between users
  - Measure extent to which two variables linearly relate
  - For users u, v: Pearson correlation is:
    `Wuv = (sum((Rui - Ru_avg)(Rvi - Rv_avg))) / (sqrt(sum((Rui - Ru_avg)^2)) * sqrt(sum((Rvi - Rv_avg)^2)))`
    - Wuv = pearson correlation coefficient between u and v
    - Rui = Rating of user u on item i
    - Ru_avg = Average rating of items for user u EXCEPT target item
    - sum = sum over corated items
13. Making User-based CF predictions with Pearson
  - Weighted average of ratings is used to generate predictions:
    `Pai = Ra_avg + (sum(Rui - Ru_avg) * Wau) / (sum(abs(Wau)))`
    - Pai = predicted rating for user a of item i
    - users = users who rated item i
    - Ru_avg = average of ALL rated items for users
    - sum = sum over all users who rated item i
14. Item Based Collaborative Filtering
  - Neighborhood-based CF algorithms do not scale well when applied to millions of users and items
  - Matching user's rated items to similar items leads to faster online systems and better recommendations
  - Similarities between pairs of items i and j are computed offline
15. Pearson Correlation between items
  - For items i, j: Pearson correlation is:
    `Wij = (sum((Rui - Ri_avg)(Ruj - Rj_avg))) / (sqrt(sum((Rui - Ri_avg)^2)) * sqrt(sum((Rvi - Rj_avg)^2)))`
    - Wij = pearson correlation coefficient between items i and j
    - Rui = Rating of user u on item i
    - Ri_avg = Average rating of item i by these users
    - sum = sum over set of users who rated both i,j
16. Making Item-based CF predictions with Pearson
  - Weighted average of ratings is used to generate predictions:
    `Pui = (sum(Run * Win)) / (sum(abs(Win)))`
    - Win = weight between items i and n
    - Run = rating for user u on item n
    - sum = sum over neighborhood set N of items rated by u that are most similar to i
17. Extensions to Memory-Based Algorithms
  - Default Voting
    - Pairwise similarity is not reliable when there are too few votes
    - Focusing on co-rated items also neglects global rating behavior in user's rating history
    - Assume some default voting values for missing ratings
    - Approaches
      - Reduce weights of users that have fewer than 50 items in common
      - Use average of a clique (small group of co-rated items) as a default voting to extend a user's ratings
      - Use a neutral or somewhat negative preference for unobserved ratings
  - Inverse User Frequency
    - Universally liked items are not as useful in capturing similarity as less common items
    - Multiply original rating by inverse frequency, `log(N/nj)`
  - Case Amplification
    - Transform applied to weights used in CF predictions
    - `wij' = wij * |wij|^(p-1), where p >= 1`
    - Reduces noise in data, favors high weights
  - Imputation Boosted CF
    - When rating for CF tasks are extremely sparse, its hard to produce accurate predictions using Pearson corr
    - Use imputation technique to fill in missing data
      - mean imputation, linear regression imputation, predictive mean matching imputation...
18. Model Based Collaborative Filtering
  - Provide recommendations by estimating parameters of statistical models for user ratings
  - Supervised learning based on training set
  - Example models:
    - Bayesian Models
    - Clustering Models
    - Dependency Networks
    - Classification algorithms
    - Regression models
19. Clustering Collaborative Filtering Algorithms
  - Clustering is an intermediate step
  - Resulting clusters used for futher analysis or processing
  - Clustering algorithms have better scalability than typical CF methods because they make predictions on smaller clusters rather than whole customer base
  - Complex and expensive clustering computation must be run offline
  - Recommendation quality is generally low
  - Optimal clustering over large data sets is impractical
  - Examples
    - K-Means
    - Hierarchical Clustering
20. Regression Based CF Algorithms
  - Good at making predictions for numerical values
  - Uses an approximation of the ratings to make predictions based on a regression model
  - Examples
    - Linear Regression
21. Challenges of Collaborative Filtering
  - Data Sparsity: User item matrix is extremely sparse
    - Solutions:
      - Dimensionality reduction techniques
        - Singular Value Decomposition: remove unrepresentative or insignificant users or items
        - Latent Semantic Indexing: similarity between users is determined by representation in reduced space
        - Principle Component Analysis
  - Cold Start problem: what to do when a new user or item enters the system
    - Solutions:
      - Content-based systems: do not rely on ratings from other users
      - Hybrid CF: external content info can be used to produce predictions for new users and items
  - Synonyms: same or very similar items that have different names or entries
    - Solutions:
      - Automatic term expansion - construction of thesaurus
      - SVD Techniques: construct a semantic space where terms and docs that are closely associated are placed close to eachother
  - Scalability: O(n) algorithms are even too large with millions of customers and users
    - Solutions:
      - Dimensionality Reduction
      - Memory-based CF algorithms have good scalability
        - Instead of calc similarities between all pairs of items, only calc between pairs of corated items by a user
  - Grey Sheep: users whose opinions do not consistently agree or disagree with any group of people
    - Do not benefit from CF
    - Solutions:
      - Hybrid methods
  - Black Sheep: idiosyncratic tastes make recommendations nearly impossible
  - Shilling attacks: users may provide ratings to benefit themselves (yelp)
    - Solutions:
      - Item-based CF
      - Hybrid CF
22. Hybrid Recommender Systems
  - Combination of multiple recommendation techniques together for producing output
23. Weighted Hybrid
  - Score of different recommendation components are combined numerically using a linear formula
  - Weights are static
  - Appropriate when component recommenders have consistent relative accuracy across the product space
  - Training: Each individual recommender processes the training data
  - Two approaches:
    - Intersection of candidate sets from each recommender
    - Union of candidate sets from each recommender
  - Scoring: Candidates -> Recommenders -> Weighted Combination -> Combined Score
24. Mixed Hybrid
  - Recommendations from different recommenders are presented together, no attempt to combine evidence
  - Candidate Generation: User Profile -> Recommenders -> Candidates
  - Scoring: Candidates -> Recommenders -> Ranked List -> Combined Display
25. Switching Hybrid
  - Selects a single recommender from its constituents based on the recommendation situation
  - Components may not have consistent performance for all types of users
  - Switching Decision: User Profile -> Recommenders -> Selection Criteria -> Selected Recommendation
  - Candidate Generation: User Profile -> Selected Recommender -> Candidates
  - Scoring: Candidate -> Selected Recommender -> Score
26. Feature Combination
  - Inject features of one source into an algorithm designed to process data from a different source
  - Training: Data -> Contributing Recommender; Data, Contributing Recommender -> Actual Recommender
27. Feature Augmentation
  - Generate a new feature for each item by using the recommendation logic of the contributing domain
  - Used when there is a strong primary recommendation component and a desire to add additional knowledge sources
  - Candidate Generation: User Profile -> Contributing Recommender -> Actual Recommender -> Candidates
  - Content boosted CF - use naive bayes to fill in missing values of rating matrices
28. Cascade Hybrid
  - Create a strictly hierarchical hybrid
  - Scoring: Candidate -> Primary Recommender; Candidate, Primary Recommender -> Secondary Recommender -> Score
29. Meta Level Hybrid
  - Uses a model learned by one recommender as input for another recommender
  - Similar to feature augmentation in that the contributing recommender is providing input to the actual recommender
  - Contributing recommender completely replaces the original knowledge source with a learned model that the actual recommender uses
  - Candidate Generation: User Profile -> Contributing Recommender -> Actual Recommender -> Candidates
  - Scoring: Candidate -> Actual Recommender -> Score

#### Questions
---
### Analysis of Massive Graphs

#### Sources
  - [Lec] 3/9/20
  - [Lec] 3/23/20
  - [Book] Chapter 10: Analysis of Social Networks

#### Concepts
1. Betweenness
  - The number of shortest paths which pass over an edge
  - Edges with high betweenness are the least likely to be inside a community
2. Girvan Newman Algorithm
  - Discover communities using divisive hierarchical clustering
  - Start with one cluster and recursively split using notion of edge betweenness
  - Undirected weighted networks
```
Visit each node
  Perform a BFS of the graph starting at X
    - Depth level of each node is the length of the shortest path from X to that node
    - Edges between depth levels are called Directed Acyclic Graph edges
    - Each DAG edge is part of at least one shortest path from root X
  Label each node by the number of shortest paths that reach it from the node
  Calculate Credits starting from the bottom of the graph
    - Each leaf node gets a credit of 1
    - Each node that is not a leaf or a root gets 1 + sum of credits of DAG edges below
    - Each edge gets a share of credit of the node (Z) below it proportional to the frac of shortest paths thru the edge from Z
Sum contributions from each edge
Divide by two to get true betweenness
```
3. Modularity
  - A measure of how well a network is partitioned into communities
  - Given a partition of the network into groups s in S:
    `Q = Sum_s[(# edges in group s) - (expected # of edges in the group s)]`
    - Expected edges is derived from a null model - a graph which matches one specific graph in its structural features, but is otherwise an instance of a random graph.
  - Null Model
    - Same degree distribution but random connections
    `Expected number of edges between nodes = (ki * kj)/2m`
  - Modularity of partitioning S of graph G
    `Q(G,S) = 1/2m sum_s(sum_i(sum_j(Aij - (ki * kj)/2m)))`
  - Modularity takes range [-1, 1]
    - Positive if number of edges exceeds expected number
    - 0.3-0.7<Q means significant community structure
4. Spectral Clustering
  - Key Idea: Find the balanced min cut of a graph
  - Cut: weight of edges with only one vertex in a group
    `cut(A,B) = sum_i,j(Wij), where i in A and j in B`
  - Min cut only considers external cluster connections so we need to add more criteria
  - Normalized cut: connectivity between groups relative to density of each group
    `ncut(A,B) = cut(A,B)/vol(A) + cut(A,B)/vol(B)`
    - vol(A): total weight of edges with at least one endpoint in A
    - lower is better
  - Finding optimal cut is NP-hard
5. Partitioning Graphs with Matrix Algebra
  - Eigenvalues and Eigenvectors of Laplacian Matrix of a graph give us a good idea how to partition
  - Adjacency Matrix
    - NxN matrix
    - Aij = 1 if edge between i and j
    - Symmetric
    - Eigenvectors are real and orthogonal
  - Degree Matrix
    - NxN diagonal matrix
    - Dii = degree of node i
  - Laplacian Matrix
    - NxN symmetric matrix
    - L = D - A
    - Eignenvalues are non-negative real numbers
    - Eigenvectors are real and orthogonal
    - Second smallest eigenvector provides optimal cut for graph
      - Ncut as an optimization problem
6. Spectral Clustering Algorithms
  - Three basic stages:
    - Preprocessing
      - Build a matrix representation of the graph -> Laplacian Matrix
    - Decomposition
      - Compute eigenvalues and eigenvectors of the matrix
      - Map each point to a lower dimensional representation based on one or more eigenvectors
    - Grouping
      - Assign points to two or more clusters based on new representation
      - Naive approach: split at 0 or median value
      - Expensive approach: minimize ncut in 1-dimension (iterate over ordering of nodes by value in eigenvector)
7. K-Way Spectral Clustering
  - Two Basic Approaches:
    - Recursive Bi-paritioning
      - Inefficient, unstable
    - Cluster multiple eigenvectors
      - Build a reduced space from multiple eigenvectors
      - Prevent instability due to information loss
8. Trawling
  - Searching for small communities in the Web Graph
  - Types of web communities
    - Explicit: communities that manifest themselves as newsgroups or resource collections
    - Implicit: communities that result from the nature of content creation of web
  - Find complete bipartite graphs from explicit communities (topics) to implicit communities (articles)
    - Similar to finding frequent itemsets
9. Overlapping Communities
  - Previous algorithms were designed to discover non-overlapping communities
  - Often this model is incorrect, and overlapping communities are preferable
  - Edge density can be higher inside community overlaps
10. Maximum Likelihood Estimation
  - Assume a generative model (for creating instances of some graph)
    - Model has parameters for determining the probability of generating any particular instance of the graph
    - We want to find the value of the parameters that gives the largest value of likelihood for the graph
  - Simple MLE
    - Each edge is present with probability p and presence or absence of edge is chosen independently
      - Ex. graph has 15 nodes and 23 edges, and 105 pairs of nodes
        - Probablity of having any such graph is `p^23(1-p)^82`
    - Given a graph, G
      - Write up a probability function f(p) of G
        - To find best p:
          - Set derivative of f(p) to 0
          - Solve the equation for 0
11. Affiliation Graph Model
  - Given
    - A number of communities
    - A number of individuals
  - Two sets of parameters
    - membership parameter: each community can have any set of individuals as members
    - probability Pc for each community: probability that two members are connected by an edge
  - If u and v are both in communities C and D but not in other communities
    - Prob of no edge Euv = (1-Pc)(1-Pd)
    - Prob of edge Euv = 1 - (1-Pc)(1-Pd)
  - General Case: if u and v are members of a set of M communities, prob of an edge is:
    `Puv = 1 - Π_c_in_M(1-Pc)`
  - If u and v share no community then we let
    `Puv = ε`
    - ε is a very tiny number
  - AGM can express a variety of community structures: non-overlapping, overlapping, nested
12. Detecting Communities with AGM
  - Use MLE to find the model M, C, Pc when given graph G(V,E)
  - Likelihood of AGM generating graph G:
    `P(G|Θ) = Π_edges P(u,v) Π_nonedge_pairs (1-P(u,v))`
  - Hard to find model parameters in discrete graph
13. BigCLAM
  - Key Idea: Memberships have strengths rather than discrete values -> simplifies optimization problem
  - Each community, a, links nodes independently:
    `Pa(u,v) = 1 - exp(-Fua * Fva)`
    - Fua is the membership strength of node u to community a
    - If u and v are not in the same community:
      `Pa(u,v) = 1 - exp(0) = 0`
    - If Fua and Fva are both large numbers:
      `Pa(u,v) = 1 - exp(-LargeNumber) = 1 - SmallNumber`
    - If Fua and Fva are both Small numbers:
      `Pa(u,v) = 1 - exp(-SmallNumber) = 1 - LargeNumber`
  - Factor Matrix - community membership strength matrix F, Nodes x Communities
    - Probability that at least one common C links nodes u, v:
      `P(u,v) = 1 - Πc(1-Pc(u,v)) = 1 - exp(-Fu * FvT)`
    - Find F that maximizes l(F):
      `l(F) = Σ_edges log(1 - exp(-Fu * FvT)) - Σ_nonedge_pairs Fu * FvT`
      - Computing this gradient takes linear time
    -BigCLAM 2.0
      - If we cache `Σ_v Fv`, we can significantly speed up BigCLAM
14. Directed AGM
  - Everything is mostly the same except now we have 2 matricies, F and H:
    - F: outgoing community memberships
    - H: incoming community memberships
    - Edge Prob
      `P(u,v) = 1 - exp(-Fu HvT)`
    - Network Log Likelihood
      `l(F,H) = Σ_edges log(1 - exp(-Fu * HvT)) - Σ_nonedge_pairs Fu * HvT`

#### Questions
---
### Mining Data Streams

#### Sources
  - [Lec] 3/30/20
  - [Book] Chapter 4: Mining Data Streams

#### Concepts
1. Stream Data Processing
  - Stream of tuples arriving at rapid rate
    - Contrast with traditional DBMS where all tuples are stored in secondary storage
  - Infeasible to use all tuples to answer queries
    - Cannot store them all in main memory
    - Too much computation
    - Query Response time is critical
2. Query Types
  - Standing Queries
    - Executed whenever new data tuple arrives
    - Keep only one value
  - Ad Hoc Queries
    - Normal queries asked one time
    - Need entire stream to have an exact answer
3. Fixed Proportion Sampling
  - Use hash to choose whether or not to save tuple
    - e.g. hash user into 10 buckets, only save tuples with user in bucket 0
    - careful how you decide to sample: position vs user
3. Fixed Size Sampling
  - Sample size may grow too big if the stream runs for a while even with fixed proportion
  - Put an upperbound on the sample size
    - Hash queries to large number of buckets, like 100
    - Put them into sample if they hash < 10
    - When sample grows too big throw away 9 and all associated users, and so on
  - Alternative:
```
// s - exact number of tuples to store, n - total number of tuples seen
Store first s elements of the stream to S
For all other elements:
  With probability s/n keep nth element
    If we keep nth element, replace one element in S uniformly at random
```
4. Bloom Filter
  - Check if an object o is in a set S without comparing to all objects in S
  - No false negatives, but may have false positives
  - Components:
    - Array A of n bits, initially all 0's
    - A set of hash functions that takes an object and returns a position in the array
    - A set of S objects
  - Construction:
    - For each object o in S, apply each hash function to o and set resulting indices to 1
  - Application:
    - For each new object o', apply each hash function to o and see if resulting indices are 1
  - False positive rate
    `(number of items not in S but identified as in S) / (all items seen)`
5. Flajolet Martin Algorithm
  - Estimate count of distinct elements
```
Hash every element to a sufficiently long bit string (try to get unique string per element)
Maintain R = length of longest length of trailing zeros among all bit strings
Estimate count: 2^R
```
  - We can hash multiple times and take average of 2^R values but we may get very large 2^R values
  - We can take median values of multiple runs but we get all values with power of 2
  - Optimal: partition hashes into small groups, take average of a group and take the median of the averages
6. Moments
  - Kth moment of S:
    `Σi=1..n (mi)^k`
    - mi is the number of occurrences of an element vi in S
    - 0th moment - number of distinct elements
    - 1st moment - length of s
    - 2nd moment - sum of squared frequencies (measures evenness of distribution of elements in S)
7. AMS Algorithm
  - Estimating 2nd moment of a stream of length n
```
Pick k random numbers between 1 and n
For each number:
  construct a variable X at position t
  record its count from position t onwards to n as X.value
Estimate:
  1/k Σi=1..k n(2xk.value - 1)
```
  - Estimation for kth moment:
    `n(c^k - (c-1)^k), where c = X.val`
8. Sliding Window
  - Useful model of stream processing is that queries are about a window of length N
    - Save N most recent elements received
    - Interesting case where N is so large that the data cannot be stored in memory or on disk
9. Simple Approximate Solution to Counting Bits
  - Assuming uniformity:
    - Maintain 2 counters:
      - S: number of 1s from beginning of the stream
      - Z: number of 0s from beginning of the stream
    - 1's in last N bits = `N * (s/(s+z))`
10. DGIM Method
  - Does not assume uniformity
  - Store O(log^2N) bits per stream
  - Solution gives an approximate answer never off by more than 50%
  - Key Idea
    - Partition N into a small set of buckets
    - Remember count and timestamp for each bucket
    - Use counts to approximate answers to queries
  - Creating Buckets
    - Right most bit of each bucket = 1
    - Every 1 position in the window is in some bucket
    - A position can only be in a single bucket
    - At most two buckets can have the same size
    - Size of older buckets >= size of new ones
    - Either one or two buckets with same power of 2 number of 1s
    - Buckets do not overlap in timestamps
    - Buckets are sorted by size
    - Buckets disappear when their end time is >N time units in the past
  - Updating buckets
    - when a new bit comes in:
      - drop the oldest bucket if its end time is prior to N time units before current time
      - if current bit is 0, do nothing
      - if current bit is 1:
        - create a new bucket of size 1 for this bit
        - if there are now 3 buckets of size 1
          - combine oldest two buckets into a bucket of size 2
        - repeat for all bucket sizes until you don't combine
  - Querying
    - sum sizes of all buckets except the last, half the size of the last bucket
      - case 1: estimate < actual: estimate is at least 50% of c
      - case 2: estimate > actual: estimate is at most 50% greater than c
  - Extensions we can allow N buckets of each size to reduce amount of uncertainty but raise storage required

#### Questions

---
### Clustering Massive Data

#### Sources
  - [Lec] 4/6/20
  - [Book] Chapter 7: Clustering

#### Concepts
1. Clustering
  - Given set of points and notion of distance between points, group points into some number of clusters where:
    - Members of a cluster are close/similar to each other
    - Members of different clusters are dissimilar
  - Points are usually in high dimensional space
  - Similarity often defined using distance measure: Euclidean (points), Cosine (vectors), Jaccard (sets), Edit, etc.
  - In high dimensional spaces, almost all pairs of points are about the same distance apart
  - In high dimensional spaces, data vectors become orthogonal
2. Methods of Clustering
  - Hierarchical
    - Agglomerative (bottom up)
      - initially each point is a cluster
      - repeatedly combine two nearest clusters into one
    - Divisive (top down)
      - start with one cluster and recursively split it
  - Point Assignment
    - maintain a set of clusters
    - points belong to the nearest cluster
3. Hierarchical Clustering
  - Key Operation: repeatedly combine two nearest points
  - Represent a cluster of many points by a centroid (average of points), clustroid (representative point)
  - Measure cluster distances by distances of centroids or clustroids
  - Closest Clustroid Options:
    - Smallest maximum distance to other points
    - Smallest average distance to other points
    - Smallest sum of squares of distances to other points
  - Nearness of clusters
    - intercluster distance: minimum of the distances between any two points, one from each cluster
    - pick a notion of cohesion of clusters, e.g. maximum distance from the clustriod
  - Cohesion
    - use the diameter of the merged cluster (maximum distance between points in the cluster)
    - use the average distance between points in the cluster
    - density-based approach: take the diameter or avg distance and divide by the number of points in the cluster
4. Naive Implementation of Hierarchical clustering
  - Initially O(n^2) for creating matrix and finding pair with minimum distance
  - Overall complexity O(n^3)
  - At each step, compute pairwise distances between all pairs of clusters, then merge
  - Careful implementation using priority queue can reduce time to O(N^2 logN)
    - Still too expensive for really big datasets that dont fit into memory
5. K-Means Clustering
  - Assumes Euclidean space
  - Start by picking k, the number of clusters
  - Initialize clusters by picking one point per cluster
    - Ex. Pick one point at random, then k-1 other points, each as far away as possible from the other points
  - Populating clusters:
```
For each point, place it in the cluster whose current centroid is nearest
After all points have been assigned, update the locations of the centroids of the k clusters
Reassign all points to their closest centroid
  - Sometimes points move between clusters
Repeat 2 and 3 until convergence
  - Convergence: points don't move between clusters and centroids stabilize
```
  - How to select k
    - Trial and error
    - Average distance to centroid will fall rapidly until the optimal k, then k will increase quickly as distance changes little, find the middle of that trend
6. BFR Algorithm
  - Extension of k-means to large data
  - Assumes that clusters are normally distributed around a centroid in euclidean space
  - Efficient way to summarize clusters O(clusters), not O(data)
  - Points are read from disk one main-memory full at a time
  - Most points from previous memory loads are summarized by simple statistics
```
Beginning with initial load, select initial k centroids by some sensible approach:
1. take k random points
2. take a small random sample and cluster optimally
3. take a sample, pick a random point, then k-1 more points each as far as possible from previous points

Three classes of points:
- Discard set: points close enough to a centroid to be summarized
- Compression set: groups of points that are close together but not close to any existing centroid
  - Point are summarized but not assigned to a cluster
- Retained set: isolated points waiting to be assigned to a compression set

For each cluster, the discard set and compression set is summarized by:
- Number of points, N
- Vector SUM, whose ith component is the sum of the coordinates of the points in the ith dimension
- Vector SUMSQ, ith component = sum of squares of coordinates in ith dimension
- 2d + 1 values represent any size cluster, d = dimensions
- Average in each dimension (centroid) can be calculated as SUMi/N
- Variance of a cluster's discard set in dimension i is (SUMSQi/N) - (SUMi/N)^2

Processing points:
  - Find points that are sufficiently close to a cluster centroid and add those points to that cluster and the DS
  - Use any main-memory algorithm to cluster the remaining points and the old RS
    - Clusters go to CS, outliers go to RS
  - DS Set: adjust statistics of the clusters to account for the new points
    - Add Ns, SUMs, and SUMSQs
  - Consider merging compressed sets in CS
  - If this is the last round, merge all compressed sets in teh CS and all RS points into their nearest cluster
```
  - Sufficiently close is defined in BFR two ways:
    - Mahalanobis distance is less than a threshold
    - High likelihood of the point belonging to currently nearest centroid (within N standard deviations of mean)
7. Mahalanobis Distance
  - Normalized euclidean distance from centroid
```
For a point (x1...xd) and centroid (c1...cd)
  1. Normalize in each dimension yi = (xi - ci) / σi
  2. Take the sum of squares of yi
  3. Take the square root:
    d(x,c) = sqrt(Σi=1..d ((xi - ci) / σi)^2)
```
  - If clusters are normally distributed in d dimensions, then after transformation, one standard deviation = sqrt(d)
8. CURE Algorithm
  - Extension of k-means to clusters of arbitrary shapes
  - Problems with BFR and K-Means
    - Assumes clusters are normally distributed in each dimension
    - Assumes axes are fixed - ellipses at an angle are not OK
  - Allows clusters to assume any shape
  - Uses a collection of representative points to represent clusters
```
Pass 1:
0. Pick a random sample of points that fit in main memory
1. Initial clusters:
  - cluster these points hierarchically - group nearest points/clusters
2. Pick representative points:
  - For each cluster, pick a sample of points as dispersed as possible
  - From the sample pick representative points by moving them 20% toward the center of the cluster
Pass 2:
Rescan the whole data set and visit each point p in the data set
Place each point in the closet cluster (closest representative point)
```
9. CURE vs BFR
  - Distribution of Data:
    - CURE: do not assume any particular distribution
    - BFR: data should be normally distributed
  - Representation of Cluster
    - CURE: a set of representatives
    - BFR: centroid
  - Common: both assume data in euclidean space

#### Questions
---
### Link Analysis

#### Sources
  - [Lec] 4/6/20
  - [Lec] 4/13/20
  - [Book] Chapter 5: Link Analysis

#### Concepts
1. Early Web Search
  - Keywords extracted from web pages to build inverted index
  - Queries matched with web pages via look up in inverted index
  - Pages ranked by occurrences of kewords -> highly susceptible to spam
2. Term Spam
  - Disguise a page as something it is not about
    - add thousands of keywords
    - fade spam words into background
    - copy top ranked pages
3. Page Rank
  - Key Idea: rank pages by linkage
    - How many pages point to a page
    - How important are those pages
  - Based on limiting distribution: probability distribution will converge eventually
  - Probability that a random surfer lands on the page
    - Naive model assumes a strongly connected web graph
  - Flow Model
    - A vote from an important page is worth more than a vote from an unimportant page
    - The rank of a page is:
      `rj = Σi->j ri/di`
      - di = out-degree of node i
  - Matrix Forumlation
    - Stochastic Adjacency Matrix:
      - If j had n outlinks and outlink j->i then Mij = 1/n, else Mij = 0
      - Columns sum to 1
    - Rank vector r is a vector with one entry per web page
      - ri is the importance score of page i
    - The flow equation is:
      `ri = Mri+1`
4. Solving Mv = v by Power Iteration:
```    
Suppose there are N web pages
Initialize r0 = [1/N...1/N]T
Iterate rk+1 = Mrk
Stop when |rk+1 - rk| < ε, where ε is some small constant
```
  - Complexity = O(kn^2), where k = # of iterations
5. Solving Mv=v by Finding Eigenvector
  - rank vector r is an eigenvector of the stochastic web matrix M
  - r is M's first or principal eigenvector with corresponding eigenvalue 1
  - largest eigenvalue of M is 1 since M is column stochastic and non-negative
```
Calculating Eigenvalues
Mv = λv
(M - λI)v = 0, and v is not a null vector, so
det(M - λI) = 0
Solve and find minimum λ
```
  - This becomes solving Mv=v by Gaussian Elimination
    - Make upper triangle matrix using pivot points
    - Multiply upper triangle M by v, and solve for v's
  - Complexity = O(n^3)
6. Dead Ends
  - Node has no outlinks
  - PageRank importance "leaks out"
  - Non stochastic matrix
  - Solutions:
    - Always teleport
    - Taxation
      - reduces loss but does not eliminate loss
      - At some point net loss and gain will balance out
    - Prune and propagate
      - Preprocess graph to recursively eliminate dead ends
      - Compute page rank on reduced graph
      - Approximate values for dead ends by propagating values from reduced graph
        - Results in non-stochastic estimation for page rank
7. Spider Traps
  - All out-links are within a group
  - Random walks get stuck in a trap
  - Spider traps absorb all pagerank importance
  - Solutions:
    - Taxation/teleportation
8. Teleports/Taxation
  - At each time step a surfer has two options: follow a link at random or jump to some random page
  - Surfer will teleport out of a spider trap within a few time steps
  - Matrix formulation
    - Random teleport is equivalent to taxing each page a fraction (1-B) of its score and redistributing evenly
  - Formula
    `v' = BMv + (1-B)e/n`
    - B = damping factor usually .8-.9
    - e = n dimensional vector with all 1s
    - n = number of nodes in the graph
  - Makes graph strongly connected
9. PageRank iteration with MapReduce
  - If n is small enough that each map task can store the full vector v and v' in main memory, we can break up our M matrix into strips
  - Given the size of the web, v is much too large to fit in main memory
    - Break M into vertical strips and break v into corresponding horizontal strips to execute mapreduce
      `vi' = Σj=1..n mij * vj`
    - Combiner gets more complicated
      - k^2 map tasks
      - each task gets one square of the matrix M and one stripe of the vector v
        - each stripe is sent to k different map tasks
        - vj is sent to the task handling mij for each of the k possible values of i
      - Advantage: keep both the jth stripe of v and ith stripe of v' in main memory as we process mij
        - Mij only transmitted once
10. Topic-Specific Page Rank
  - Normal page rank measures generic popularity of a page
  - Goal: evaluate pages by how close they are to a particular topic
  - Allows search queries to be answered based on interests of a user
  - Key Idea: Teleports go to a topic-specific set of relevant pages, "teleport set"
    - Bias the random walk
    - Discover Topic PageRank Vectors Rs for a topic set S
  - Integrating Topic-Sensitive PageRank into a search engine
    - Decide on topics for which to create specialized pagerank vectors
    - Pick a teleport set for each of these topics
    - Determine the topic or set of topics most relevant to a particular search query
    - Use pagerank vector for those topics to order teh responses to the search query
  - How to select the topic set that is most relevant to the search query?
    - User can pick from a topic menu
    - Infer the topics from the context of the query
    - Infer the topics from information about the user
11. WebSpam
  - Deliberate action to boost a webpages position in search results
  - Not commensurate with page's real value
12. LinkSpam
  - Create link structures that boost PageRank of a particular page
  - SpamFarm
    - TargetPage is linked to by comments section of accessible pages
    - Supporting pages are linked to by TargetPage to hold PageRank value and create a large teleport set
    - Supporting pages link to TargetPage to send it teleport value
13. TrustRank
  - Teleport set consists of "Trusted" and "Authoratative" pages
    - Authoritative: receive a lot of links on a topic
    - Trusted: are inaccessible to spammers (no comments sections)
14. Hubs and Authorities (HITS)
  - Variant of topic-sensitive PageRank
  - Identify trustworthy pages
    - Manually examine pages with high pageranks
    - Pick pages within controlled domains: edu, gov, mil
    - Avoid blogs or forums or any comment sections
  - Each page has two scores
    - Hub: does it provide many pointers to a topic?
      - Sum of authority scores pointing to the page
    - Authority: how important is a page about the topic?
      - Sum of hub scores pointing to the page
  - Compute via mutual recursion
```
Start with h0 = vector of all 1's
Compute a1 = LTh0 then scale so largest component is 1
Compute h1 = La1, and scale
Repeat until h and a don't change much
```
  - LT = un-normalized M in PageRank
  - L: L[i,j] = 1 if there is a link from node i to node j

#### Questions
---
### Web Advertising

#### Sources
  - [Lec] 4/20/20
  - [Book] Chapter 8: Advertising on the Web

#### Concepts
1. Web Advertisements
  - Direct Ads
    - Craigslist, Autotrading Sites, Social Network feeds
  - Featured Ads
    - Featured on websites
    - Pay per impression
  - Sponsored Products in stores
    - Collaborative filtering
    - Similar Items
  - Search Ads
    - Top results
    - Most lucrative type of online advertisements
    - Charges user per impression of their ads
2. Adwords Model
  - Matches search queries to advertisements
  - Require algorithms for optimizing this assignment
3. Online Algorithms
  - See input one piece at a time and make irrevocable decisions along the way
  - Only know past queries and current queries
  - Similar to handling data streams
  - Cannot do as well as an offline algorithm
4. Online Bipartite Matching
  - Simplified version of matching ads to search queries
  - Look for maximal matching in a bipartite graph
    - Maximal match: largest number of possible matches
    - Perfect match: all vertices of graph are matched
5. Greedy Bipartite Matching Algorithm
  - Pair new query with any eligible partner, if none then do not pair query
  - Competitive Ratio:
    `min_inputs(|Mgreedy| / |Moptimal|)`
    - For input I suppose greedy produces matching Mgreedy while optimal is Moptimal
    - Represents greedy's worst performance over all possible inputs I
6. Adwords Problem
  - Given:
    - A set of bids by advertisers for search queries
    - A click through rate for each advertiser query pair
    - A budget for each advertiser
    - A limit on the number of ads to be displayed with each search query
  - Respond to each search query with a set of advertisers such that:
    - The size of the set is no larger than the limit on the number of ads per query
    - Each advertiser has big on the search query
    - Each advertiser has enough budget left to pay for the ad if it is clicked upon
  - Goal: Maximize search engines revenues
  - Expected Revenue = Bid * Click-through-rate
  - Complications:
    - Budget
    - CTR of an ad is unknown
      - We can measure this historically, but its still a very hard problem
        - exploitation: keep showing an ad for which we have good estimates of CTR
        - exploration: show a brand new ad to get a better sense of its CTR
7. Greedy Algorithm for Adwords Problem
  - Assign the query to the highest bidder who still has budget left
    - Problematic if highest bidder has a more flexible query
  - General case: competitive ratio is 1/2
8. BALANCE Algorithm
  - For each query pick the advertiser with the largest unspent budget
    - Break ties arbitrarily but in a deterministic way
  - General case: competitive ratio is 1-1/e = 0.63
  - Balance works well when bids are 1,0
    - In practice bids and budgets can be arbitrary
    - In general setting balance can perform poorly
```
Consider query q, bidder i
  - bid: xi
  - budget: bi
  - amount spent so far: mi
  - fraction of budget left: fi = 1-(mi/bi)

Ψi(q) = xi (1-exp(-fi)) -> bid ranking

Allocate query q to bidder with largest value of Ψ
```
  - Improvements needed
    - Bias choice of an ad in favor of higher bids
    - Consider the fraction of budget remaining
    - Dont leave too much of any advertisers budget unused
    - Algorithm does not account for possiblity that CTR differs for different ads
      - Multiply bid by CTR when computing Psi
    - Consider historical frequency of queries
9. Real Adwords Aspects
  - Bid on subset, superset, synonyms instead of set of keywords
  - Charge advertizers based on complicated formulas
  - Search engines use second-price auction
    - Each advertiser pays amount of second place bidder
    - Less susceptible to being gamed
    - Lead to higher revenues for search engines

#### Questions
