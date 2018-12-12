# CSCI 585: Database Systems
## Midterm Study Guide

### Test Details:
  - Wednesday December 12th, 2018
  - 14:00 - 15:00
  - SGM 101

### Major Topics:
  - Spatial DBs
  - NoSQL
  - Big Data
  - MapReduce
  - Data Science
  - Data Mining
  - Machine Learning
  - TensorFlow
  - R
  - WEKA
  - Data Visualization
  - Extras (General Databases/Data Science Knowledge)
  - Midterm Review (SQL - See Midterm Study Guide)

### Sources of Information
  - CSCI 585 Lecture Notes, Saty Raghavachary [Lecture]
  - Course Homepage <http://bytes.usc.edu/cs585/f18_DS0agI4Me/home/index.html> [Web]

---
### Spatial Databases

#### Sources:
  - Spatial DBs Lecture Notes
  - Lecture 10/18 - Starts @ 40:20

#### Definitions
1. Spatial Database
  - A database that is optimized to store and query data related to objects in space, including points, lines and polygons.

#### Concepts
1. Characteristics of Geographic Data
  - Has location
  - Has size
  - Is auto-correlated
  - Scale dependent
  - Maybe temporally dependent

2. Entity View vs Field View
  - Two conceptions of space:
    - Entity View: space as an area filled with a set of discrete objects
    - Field View: space as an area covered with essentially continuous surfaces

#### Questions
  - Fall 2016 Question 1
---
### NoSQL

#### Sources:
  - NoSQL Intro Lecture Notes
  - NoSQL DBs Lecture Notes
  - Lecture

#### Definitions
1. NoSQL Databases (History of Name)
  - Non-Relational, non SQL
  - NO SQL
  - NotOnly SQL -> Current Meaning

2. NoSQL Databases
  - Used to handle BigUsers, BigData, BigVariety, and BigFlux, where RDBMSs are unsuited
  - Flexible, efficient, avaiable and scalable db design
  - Schema-less: no tables, no relations; NoSQL doesnt care what its storing; schema is implicit - it lies in application code that reads and writes data
  - Flexible: easy to add new types of data
  - Scalable: ability to horizontally scale (adding more nodes)
  - Fast: easy to processe large volumes of data

3. JSON
  - JavaScript Object Notation
  - File-format that uses human-readable text to transmit data objects
  - DBs are represented as an [] array of {} object, where each object is a set of key-value pairs
  - Very straight forward to represent table data as JSON

4. XML
  - Extensible Markup Language
  - More verbose but popular alternative format to JSON for creating DBs.

5. Polyglot Persistence
  - Use of different storage technologies to store different parts of a single application.
  - Individual data stores are tied together by the application using DB APIs

6. Nodes and Clusters
  - Node: a single NoSQL DB instance - holds part of a db
  - Cluster: a collection of nodes - holds an entire db

7. Key-Value DBs
  - Database is constructed of keys which represent a value in the db
  - Querying occurs on keys only
  - Entire value for matching key is returned when queried
  - Entire value for matching key must be updated when its necessary to make changes

8. Column Family DBs
  - Rather than dealing with rows of data, we deal with columns of them
  - Good for aggregate queries and queries involving a subset of all columns

9. Document DBs
  - Collection of documents with an arbitrary number of fields in each document
  - Documents can be JSON, XML, etc

10. GraphDBs
  - Data structure comprised of verticies and edges

#### Concepts
1. Drivers of NoSQL Movement
  - Avoidance of complexity
  - High throughput
  - Easy, quick scalability
  - Ability to run on commodity hardware
  - Avoidance of need for O-R mapping
  - Avoidance of complexity associated with cluster setup
  - Realization that one-size-fits-all was wrong
  - Ability to use DB APIs for various programming languages

2. BASE vs ACID
  - NoSQL DBs are characterized by their BASE properties rather than the RDBMS ACID properties
  - BASE: Basic Availability, Soft State (relating to consistency), Eventual Consistency
  - ACID: Atomic, Consistent, Isolation, Durability
  - NoSQL DBs focus on availability first, where approximate answers are okay, this results in a simpler and faster DB

3. Types of NoSQL Databases
  - Key-Value Store: DynamoDB
  - Column-Family Store: Cassandra
  - Document Store: MongoDB
  - Graph Store: Neo4j

4. NoSQL DB Comparison Metrics
  - Architecture
    - Topology
    - Consistency
    - Fault Tolerance
    - Structure and format
  - Administration
    - Logging and Statistics
    - Configuration Management
    - Backup
    - Disaster Recovery
    - Maintenance
    - Monitoring
    - Security
  - Deployment
    - Availability
    - Scalability
  - Development
    - Documentation
    - Integration
    - Support
    - Usability
  - Performance
  - Scalability

3. NoSQL Pros & Cons
<table>
    <tr>
        <td>Pros</td>
        <td>Cons</td>
    </tr>
    <tr>
        <td>High Scalability</td>
        <td>Too many options</td>
    </tr>
    <tr>
        <td>Schema Flexibility</td>
        <td>Limited Query Capabilities</td>
    </tr>
    <tr>
        <td>Distributed Computing</td>
        <td>Eventual Consistency is not intuitive for strict scenarios</td>
    </tr>
    <tr>
        <td>No Complicated Relationships</td>
        <td>Lacks Joins, Group by, Order by facilities</td>
    </tr>
    <tr>
        <td>Lower (Hardware) Costs</td>
        <td>ACID Transactions</td>
    </tr>
    <tr>
        <td>Open Source</td>
        <td>Limited guarantee of support</td>
    </tr>
</table>

4. NoSQL vs RDBMS
<table>
    <tr>
        <td>Feature</td>
        <td>NoSQL</td>
        <td>RDBMS</td>
    </tr>
    <tr>
        <td>Data Volume</td>
        <td>Handles Huge Data Volumes</td>
        <td>Handles Limited Data Volumes</td>
    </tr><tr>
        <td>Data Validity</td>
        <td>Highly Guaranteed</td>
        <td>Less Guaranteed</td>
    </tr><tr>
        <td>Scalability</td>
        <td>Horizontally</td>
        <td>Horizontally and Vertically</td>
    </tr><tr>
        <td>Query Language</td>
        <td>No declarative query language</td>
        <td>SQL</td>
    </tr><tr>
        <td>Schema</td>
        <td>No predefined schema</td>
        <td>Predefined Schema</td>
    </tr><tr>
        <td>Data Type</td>
        <td>Supports unstructured and unpredictable data</td>
        <td>Supports relational data and its relationships</td>
    </tr><tr>
        <td>ACID/BASE</td>
        <td>BASE</td>
        <td>ACID</td>
    </tr><tr>
        <td>Transaction Management</td>
        <td>Weaker transactional guarantee</td>
        <td>Strong transactional guarantees</td>
    </tr><tr>
        <td>Data Storage</td>
        <td>Schema-free collections utilized to store different types</td>
        <td>No collections are used for data storage, instead used DML</td>
    </tr>
</table>

#### Questions
  - Spring 2017 Question 6
  - Spring 2017 Question 8
---
### Big Data
---
### MapReduce

#### Sources
  - MapReduce Lecture Notes
  - Lecture

#### Definitions
1. Map(Shuffle)Reduce
  - A programming paradigm developed by Google which is popular for processing Big Data in NoSQL Databases
  - Parallelizable data analysis and disk usage, which allows dramatic processing gains
  - Programmer only needs to supply a mapper task and a reducer task and the rest is automatically handled
  - Summary: MapReduce is a programming model and an associated implementation for processing and generating large data sets. Users specify a map function that processes a key/value pair to generate a set of intermediate key/value pairs, and a reduce function that merges all intermediate values associated with the same intermediate key.

2. GFS - Google File System
  - A high-performance distributed filesystem created by Google to run MapReduce
  - Abstracts details of network file access so that remote reads/writes and local reads/writes are handled in code identically
  - File system is implemented as a process in each machine's OS, striping is used to split each file and store the resulting chunks on several chunk servers, details of which are handled by a single master

3. Hadoop
  - An open source tool that is modeled after the MapReduce paradigm and filesystem and is used identically

4. Hadoop: Hive
  - Provides SQL-like scripting language called HQL.
  - Translates most queries to MapReduce jobs, exploiting the scalability of Hadoop while providing a familiar SQL abstraction

5. Hadoop: Pig
  - Provides an engine for executing dataflows in parallel on hadoop
  - Includes a language, Pig Latin for expressing these dataflows.
  - Pig Latin includes operators for many of the traditional data operations (join, sort, filter) as well as the ability for users to develop their own functions for reading, writing and processing data.
  - Pig Latin scripts are compiled into MR jobs that are then run on the cluster
  - Pig Latin scripts describe a directed acyclic graph where the edges are data flows and the nodes are operators that process the data.

6. Hadoop: Oozie
  - Workflow system that coordinates execution of multiple MR jobs.
  - Capable of managing a significant amount of complexity, basing execution on external events that include timing and presence of required data.

7. Hadoop: Musketeer
  - A tool used to decouple front end workflows (Hive, Giraph, Spark, GraphX) from backend workflows (Hadoop, Spark)
  - Allows users to write workflows once, and execute on alternative systems
  - Multiple components of a single workflow can be executed on different backend systems
  - Existing workflows can be ported to new systems

8. YARN: MapReduceV2
  - Yet Another Resource Negotiator, improves on MRv1's static batch-oriented processing
  - Makes it possible to create (near) real-time applications
  - Offers better (component-wise) scalability and cluster utilization
  - Makes non-MR applications possible

9. Apache Spark
  - Open source in-memory data processing engine (fast) built for iterative processing of data.
  - Alternative to map-reduce, runs on top of HDFS
  - Better efficiency: general execution graphs, in-memory data storage
  - Leverages SparkSQL query language
  - Addresses MR's inability to handle complex (multi-pass) processing, interactive queries or real-time (stream) processing.
  - Uses Resilient Distributed Data Sets
    - Distributed collections of objects that can be cached in memory across cluster
    - Manipulated through parallel operators
    - Automatically recomputed on failure
  - Modules:
    - Spark Core
    - Spark Streaming - stream processing
    - Spark SQL - data frame processing
    - MLib - algorithms
    - GraphX - graph views

10. Apache Flink
  - Parallel data processing platform similar to MR
  - Generalizes MapReduce by allowing Map, Reduce, Join, CoGroup, Filter and Iteration transformations
  - Transformations can be assembled in arbitrary data flows including multiple sources, sinks, branching and merging flows
  - Allow use of any Java or Scala data types rather than the rigid key-value pair model
  - Super set of MapReduce programming model

11. Apache Storm
  - Open source distributed realtime computation system
  - Makes it easy to reliably process unbounded streams of data, doing for realtime processing what Hadoop did for batch processing
  - Many use cases: realtime analytics, online machine learning, continuous computation, etc.
  - Fast, scalable, fault tolerant

12. Apache Kafka
  - Open source pub-sub real-time messaging system
  - Provides strong durability and fault tolerance

13. Bulk Synchronous Parallel Model
  - Alternative to MR
  - BSP executions are executed on a set of processors which are connected in a communication network, but work independently.
  - Each step of the BSP process contains: Local computation, Communication, and Barrier Synchronization
  - Google's implementatiuon of BSP is called Pregel
  - Giraph and Hama are open source versions of Pregel

#### Concepts
1. MapReduce Steps
  0. Data is split into file segments, held in a compute cluster made up of nodes
  1. Mapper task is run in parallel on all the segments (i.e. each segment of each node); each mapper produces output in the form of multiple (key, value) pairs
  2. (Optional) Combiner operation performs a per-key reduction on each node
  3. Key/value output pairs from all mappers are forwarded to a shuffler, which consolidates each key's value into a list and associates it with that key
  4. Shuffler forwards keys and their value lists to multiple reducer tasks; each reducer processes incoming key-value lists and emits a single value for each key

2. Similarities between MapReduce and TensorFlow
  Both MapReduce and TensorFlow help carry out dataflow computation, where data processing nodes are connected in the form of an acyclic graph, with data flowing through the nodes. The systems track the dependencies between the nodes and schedule parallel node execution where ever possible.

3. Example MR Projects
  - WordCount for a Book

4. DBs used with Hadoop
  - MongoDB, Cassandra, HBase, Hive, Spark, Blue, Solr, Memcached, Solr

#### Questions
  - Spring 2017 Question 4
  - Spring 2017 Question 5
---
### Data Science
---
### Data Mining

#### Sources:
  - Data Mining Lecture Notes
  - Lecture

#### Definitions

#### Concepts
1. Data Mining vs Statistics
  - Statistics is about summarization of data.
  - In statistics, we collect and analyze numerical data of a smaller representative sample for the purpose of inferring proportions in a whole.
  - In Data Mining, we don't summarize or make inferences about a larger population, we analyze all available data and look for patterns in it.

2. Data Mining vs Machine Learning
  - Data Mining is a subset of Machine Learning.
  - Data Mining stops with the discovery of patterns in data.
  - In Machine Learning, we publish the model that we mine, and continue processing new incoming data, using that model.

#### Questions
  - Spring 2017 Question 1

---
### Machine Learning
---
### TensorFlow

#### Sources:
  - TensorFlow Lecture Notes
  - Lecture

#### Definitions
1. TensorFlow
  - Open Source offering from Google
  - Data flow programming system where a user defines a data flow graph that represents computations on various tensors in the form of a dependency graph. The TensorFlow session then parallelizes the computations across a set of local or remote nodes.
  - TensorFlow makes it possible to express rigid neural networks as flexible graphs in order to allow the computations to be parallelized across many different nodes (and types of nodes).

2. Tensor
  - Tensors are simply matrices of numbers.
  - A 0-tensor is a scalar
  - A 1-tensor is a vector
  - A 2-tensor is a matrix, and so on.

3. TensorFlow Graph
  - The TensorFlow graph model represents each individual computation and the inputs/outputs in order to visually display the entire system and create a dependency model for parallelizing the computations.

#### Concepts
1. Drawing TF Graphs
  - When drawing a TF graph, separate each tensor-operation into its own labelled circular node and attach any labelled square inputs (tensors) to that node.

#### Questions
  - Spring 2017 Question 2

---
### R

#### Sources:
  - R Lecture Notes
  - Lecture

#### Definitions
1. R
  - A special purpose functional programming language for any data-related programming. Handles I/O, statistics, mathematical calculations, data mining, machine learning, visualization. More specialized than Python
  - Cross-platform, open source, extensible
  - Built on top of S, a statistics programming language

#### Concepts
1. Five Non-Atomic Datatypes
  - Vector - a sequence with identical types of elements
  - Matrix - a vector shaped as a rectangle
  - Array - a vector of >2D
  - List - a sequence with different types of elements
  - DataFrame - list of column vectors, columns can be of different types; relational table

2. Functions are Objects

3. Built-In Distribution Functions
  - Beta, Binomial, Exponential, Gamma, Geometric, Poisson, Logistic, etc.

4. Most Important Datatypes for Data Processing
  - Vector - used to create an array of values that can be processes as a single entity
  - Data Frame - used to create tabular objects with column names that are assigned column data as vectors.

#### Syntax
1. Creating a Vector
```
x <- c(10.4, 5.6, 3.1, 6.4, 21.7)
```
```
assign("x", c(10.4, 5.6, 3.1, 6.4, 21.7))
```

2. Listing Objects
```
objects()
```

3. Detecting Type
```
typeof(x)
```

4. Vector Operations
```
v <- 2*x + y + 1
# generates a new vector v of length 10 constructed by adding together,
# element by element, 2*x repeated 2 times, y repeated just once,
# and 1 repeated 10 times - in other words, this is a 'vector' op!
```

5. Combining Vectors
```
y <- c(x,x)
```

6. Vector Functions
```
range()
min()
max()
sum()
prod()
mean()
sort()
```

7. Creating Sequences
```
m <- 1:30
```
```
n <- seq(2,5)
```

8. Applying Functions
```
incmeans <- tapply(incomes, statef, mean)
# The function tapply() is used to apply a function.
# In this case we apply the function mean() to each group of components of 'incomes'
#  defined by the levels of 'statef', similar to python's map function
```

9. Creating Arrays from Vectors
```
z = 1:1500
dim(z) <- c(3,5,100)
```

10. Listing Sample Data Sets
```
data()
```

11. Editing Data in spreadsheet-fashion
```
xnew <- edit(xold)
xnew <- edit(data.frame())
```

#### Packages
  - ggplot2 - high-level package for creating statistical graphs
  - plotly - plotting package
  - plyr - dataflow package
  - rCharts - creating interactive JavaScript-based visualizations

#### Questions
  - Spring 2017 Question 3
---
### WEKA
---
### Data Visualization

#### Sources
 - Data Visualization Lecture Notes
 - Lecture

#### Definitions
1. Data Visualization
  - Involves the study of tools and techniques for turning data into image/graphics to obtain better insight into data.
  - We use graphical depictions of data to understand, communicate, act and decide

#### Concepts
1. Single Variable Visualizations
  - Pie-charts: express relative fractions of a quantity
  - Histogram/Bar Charts/Density Plot
  - Bubble Plots

2. Spatial Data
  - Plotting spatial data on a map reveals patterns and trends in a direct way
  - Useful for planning purposes (travel, product placement, group detection)

3. Spatio-temporal Data
  - Superimposing time varying data on a map reveals course, trends, etc.

4. Interactivity
  - Being able to interact with data provides more understanding
  - Can turn items on/off, drill down or roll up, explore the time dimension

5. Animation
  - Watching data being animated provides us with fresh perspectives

6. Real-Time
  - Real-time visualizations provides a level of immediacy/freshness/relevance/interest that is simply absent in non-real-time data.

7. Networks
  - Shows relationships between entities
  - Node attributes can add detail in form of labels
  - Edge attributes can quantify data

8. Tools for Generating Data Visualizations
  - Data Science Software
    - Weka
    - KNIME
    - Rapid Miner
  - Code
    - R, Shiny
    - matplotlib
  - Online Tools
    - datavisual
    - infogram
  - Math, analysis and plotting packages
    - Mathematica
    - MATLAB
  - 3rd Party Software
    - Tableau
    - Qlik

#### Questions
  - Spring 2017 Question 7
---
### Extras
---
### Questions for Saty:
  - Spring 2017 Question 1: In Machine Learning we do make inferences about a larger population, like Statistics.