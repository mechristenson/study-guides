# CSCI 585: Database Systems
## Final Study Guide

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
  - Comprised of Spatial Data Types, Spatial Operators and Functions, and Spatial Indices
  - Uses include: crime data, land use, traffic, space, genes, military strategy

2. Types of Spatial Data
  - Points, vertices, nodes
  - Polylines, arcs, linestrings
  - Polygons, regions
  - Pixels, raster

3. GIS - Geographic Information System
  - Specific application architecture built on top of a general SDBMS
  - Used for: search, location analysis, terrain analysis, flow analysis, distribution, spatial analysis, measurements
  - Platforms
    - OpenLayers
    - ESRI: ArcGIS
    - QGis
    - MapBox
    - GIS Cloud

4. Spatial Relationships
  - In 1D (and higher), spatial relationships can be expressed with the following terms:
    - Touch
    - Inside
    - Contains
    - Disjoint
    - Equals
    - On
    - Covers
    - Overlaps

5. Minimum Bounding Rectangles (MBRs)
  - Bounding boxes used to estimate and compute spatial relationships between objects
  - Used in the Filter and Refine step of Query Processing

6. Oracle Spatial
  - Oracle offers a 'Spatial' library for spatial queries
  - Handles Spatial Data Types, Spatial Analysis and Spatial Indexing in Oracle DB, accessible through SQL
  - Oracle Spatial Data Types: Networks (lines), Locations (points), Parcels (polygons), Imagery (raster, grids), Topological Relations (persistent topology), Addresses (geocoded points)
  - Oracle Spatial Operators:
    - Filter (find objects by primary filter)
    - Relate (find all objects by primary and secondary (relational i.e. touch) filter)
    - NN (find n nearest neighbors)
    - WithinDistance (find all objects within a distance of target)
  - Spatially Indexed with R-Trees

7. Postgres PostGIS Application
  - Spatial DB functionality add-on for Postgres
  - Supports Queries: distance, equals, disjoint, intersect, touches, crosses, overlaps, contains, length, area, centroid

8. Google KML
  - Google's format for encoding spatial data

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

3. Types of Spatial Relationships
  - Topology-Based - using definitions of boundary, interior, exterior
  - Metric-Based - distance/euclidean, angle measurements
  - Direction-Based
  - Network-Based - shortest path

4. Spatial Indexing
  - Vastly speeds up processing of spatial data
  - B-Trees (self-balancing generalized binary search tree (nodes can have more than 2 leaves))
    - Issues
      - Sorting is not naturally defined on spatial data
      - Many efficient search method are based on sorting data sets
    - Solution
      - Space filling curves impose an ordering on the locations in a multi-dimensional space
  - R-Trees: Use MBRs to create a hierarchy of bounds
  - KD-Trees: Space-partitioning data structure for organizing points in a k-dimensional space
  - Quad Trees: Each node is a leaf node with indexed points or null, an internal node has exactly 4 children

5. Query Processing
  - Since MBRs are not exact, query processing usually takes on two steps: Filter and Refine
    - Filter: detect all objects in filter region
    - Refine: remove all objects whose MBRs are in region but actual geometry is not
  - This process allows for more efficient spatial computations

6. Spatial Data Visualizations
  - Dot Map: Map with points shown for events
  - Proportional Symbol Map: Map with different sized symbols for higher probability regions
  - Diagram Map: Map with diagrams displaying information about a region
  - Choropleth Map: Map with different regions colored differently based on a metric

7. DBs with Spatial Extenstions
  - Oracle: Locator, Spatial, SDO
  - Postgres: PostGIS
  - DB2: Spatial Datablade
  - SQL Server: Geometric and Geodetic types
  - MySQL: Built-in Spatial Library
  - SQLite: SpatiaLite

#### Syntax
1. Creating Spatial Entities
```SQL
CREATE TABLE County(
  Name varchar(30),
  State varchar (30),
  Shape Polygon
);
```

2. Example County Border Query
```SQL
SELECT C1.Name
FROM County C1, County C2
WHERE Touch(C1.Shape, C2.Shape) = 1
      AND C2.Name = 'Los Angeles';
```

3. Spatial Functions
  - Returns a Geometry
    - Union
    - Difference
    - Intersect
    - XOR
    - Buffer
    - CenterPoint
    - ConvexHull
  - Returns a number
    - Length
    - Area
    - Distance

4. Spatial Operators
  - Implemented as functional extensions in SQL
  - Topological Operators
    - Inside
    - Touch
    - Covers
    - Equal
    - Contains
    - Disjoint
    - Covered By
    - Overlap Boundary
  - Distance Operators
    - Within Distance
    - Nearest Neighbor

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

#### Sources:
  - Machine Learning Lecture Notes
  - Lecture

#### Definitions
1. Machine Learning
  - One subset of AI
  - Focuses on the construction and study of systems that can learn from data to optimize a performance function, such as optimizing reward or minimizing loss.
  - Goal is to develop deep insights from data faster, extract knowledge from data with greater precision, improve the bottom line and reduce risk

2. Supervised Learning
  - Algorithms that are trained using examples where both features (inputs) and labels (outputs) are known.
  - Linear Regression, Perceptron, Logistic Regression, Multiclass Classification, Neural Networks, SVM, Decision Trees

3. Unsupervised Learning
  - Algorithms where a system operates on unlabeled examples and try to find a hidden structure
  - Goal is to explore data to find intrinsic structures within it using methods like clustering or dimensional reduction
  - Euclidean Space: K-Means Clustering, Gaussian Mixtures, Principal Component Analysis
  - Non-Euclidean Space: ISOMAP, Local Linear Embedding, Laplacian Eigenmaps, Kernel PCA

4. Semisupervised Learning
  - Algorithms operate on both labeled and unlabeled data for training
  - Goal is unsupervised learning where labels are viewed as side information to help the algorithm find the right intrinsic structure
  - Use cases include image analysis, textual analysis, disease detection

5. Reinforcement Learning
  - Algorithm discovers which actions yield the greatest rewards through trial and error
  - Three primary components
    - Agent - learner or decision maker
    - Environment - everything the agent interacts with
    - Actions - what the agent can do
  - Goal is for agent to choose actions that maximise the expected reward over a given period of time
  - Agent will reach the goal much quicker by following a good policy, so the goal in reinforcement learning is learn the best policy
  - Use cases include robotics and navigation
  - Markov Decision Processes, Partially Observable Markov Decision Processes

6. Neural Networks
  - A form of AI that uses neuron-like connected units to learn patterns in training data that has known outcomes, and uses the learning to be able to gracefully respond to new data
  - Comprised of an interconnected set of weighted non-linear functions
  - Use Cases: recognize/classify features, detect anomalies, predict exchange rates, calculate numerical values
  - The bigger the training set, the better the learning
  - Revolutionizing AI due to advances in:
    - Big data to use for training
    - Better algorithms from academic and industry research
    - Faster cloud computing platforms and libraries

7. NN: Types of Neurons (Activation Functions)
  - Linear Neurons:
    - input values get passed through verbatim
  - Binary Threshold Neurons:
    - neuron outputs a 1 only when a threshold is exceeded
  - Rectified Linear Neurons (ReLU):
    - outputs a positive value, or 0 otherwise
  - Sigmoid Neurons:
   - give real valued output that is a smooth and bounded function of total input
   - Nice derivatives which help learning
   - Can act as probability distribution for the output

8. NN: Backpropagation
  - Method learning weights
  - Iteratively adjust weights starting from last hidden layer and moving to first hidden layer
  - Aims to reduce the error between expected and actual output by finding the minimum of the quadratic loss for a given training input

9. Deep Learning
  - Specialized neural networks where we use large numbers of hidden layers, each of which learning/processing a single feature
  - Widely used in image recognition, speech recognition, object recognition, and NLP

10. Convolution
  - A blending operation between two functions, where one function is convolved (pointwise-multiplied) with another and the results are summed
  - Used heavily in image processing filters for blurring, sharpening, edge-detection, etc.

11. Convolutional Neural Networks
  - Biologically Inspired convolutional filters are used across a whole layer to enable to entire layer to detect a feature. Detection regions are overlapped like cells in the eye.
  - CNN is where we represent a neurons weights as a matrix (kernel) and slide it (IP-stype) over an input to produce a convolved output
  - Each neuron is convolved over the entire input and an output is generated from all the convolutions.
  - The CNN process typically looks like: input -> convolution -> normalize (ReLU) -> reduce (pool) -> convolution ... -> output
  - Example: shape detection by pieces, image detection by pieces
  - CNN is not good for data that is not spatially laid out

12. Ensemble Methods
  - Use of multiple learning algorithms to obtain a better predictive performance than could be obtained from any of the constituent learning algorithms alone
  - Minimize or eliminate any variances or biases between the individual learners

#### Concepts
1. Types of AI
  - Type I:
    - Reactive Machines - make optimal moves, no memory, no past experience
    - Application of rules/logic
    - Game trees
  - Type II:
    - Limited Memory - human-compiled/provided or one-shot past experience that are stored for lookup
    - Neural networks, expert systems
    - Current level of progress
    - Progress to this area ended AI Winter
  - Type III:
    - Theory of Mind
    - Understanding that people, createures and objects in the world can have thoughts and emotions that affect the AI program's own behavior
  - Type IV:
    - Self-Awareness
    - Machines that have consciousness, can form representations about themselves

2. Types of ML
  - Supervised Learning
  - Unsupervised Learning
  - Semisupervised Learning
  - Reinforcement Learning

3. Building Neural Networks
  - Create layers of neurons, where each layer is a set of neurons that feed their outputs downstream to the next layer
  - Each layer is responsible for learning some aspect of our target, usually operate in a hierarchical fashion
  - Layers learn by adjusting input weights so that neurons only fire when they are given "good" inputs

4. Machine Learning Libraries
  - Keras
  - Torch
  - CAFFE
  - Deeplearning4j
  - TensorFlow
  - Spark MLib
  - Scikit-learn

5. Hardware for ML
  - GPUs and other forms of hardware are used to accelerate deep learning
    - Provide advantages in massively parallel processing, arbitrary speed increases from upgrading hardware
    - work great with DNNs
  - NVIDIA has made a framework called DIGITS (Deep Learning GPU Training System) for leveraging GPUs for DNNs
  - Microsoft has created GPU-based network for doing face and speech recognition
  - TPU (Tensor Flow Processing Unit) DNN chip developed by Google.

6. Cloud ML
  - NN/ML platforms are starting to emerge that package cloud storage, cloud computing, and algorithms for developing and deploying ML apps
  - Google CoLab
  - Google Cloud Vision API
  - Amazon Rekognition API
  - Amazon ML Solutions Lab
  - FloydHub

7. CapsNet
  - Developed by Geoff Hinton to improve upon idea of CNNs
  - CNNs can get confused by distorted images because they are not learning the correct way
  - CapsNet takes Neural network idea and seperates neural layers into capsules where each capsule handles one type of visual stimuli, which is then designed in a unified micro-network architecture.

#### Questions
  - Fall 2016 Question 2
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

#### Sources
  - WEKA Lecture Notes
  - Lecture

#### Definitions
1. WEKA
  - Waikato Environment for Knowledge Analysis
  - Toolbox of algorithms and workbench to create new ones
  - Allows users to test and compare learning algorithms on the same data in one platform

2. WEKA Workbench
  - Collection of ML Algorithms and data pre-processing tools, accessible by GUI
  - Modular and Extensible via Java API plugin scheme
  - Can be used standalone or as a library

3. UI
  - Explorer Panel provides data pre-processing, algorithms, visualization tools
  - Experimenter Panel allows comparison of algorithms
  - Knowledge Flow Panel allows incremental updates of ML algorithms

4. ARFF
  - Attribute Relation File Format
  - Standard format for expressing WEKA data

#### Concepts
1. WEKA Projects
  - Linguistics: GATE, Bali
  - Biology: BioWEKA
  - Distributed Systems: GridWeka
  - Data Mining: KNIME, RapidMiner
  - Scientific Workflows: Kepler

#### Syntax
1. ARFF File Example

```
@relation weather

@attribute outlook (sunny, overcast, rainy)
@attribute waveHeight real
@attribute surf (yes, no)

@data
sunny, 3, yes
rainy, 10, no
sunny, 6, yes
```
#### Questions
  - Fall 2016 Question 3
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

2. Mulitvariate Visualizations
  - Visualizations that display multiple variables

3. Spatial Data
  - Plotting spatial data on a map reveals patterns and trends in a direct way
  - Useful for planning purposes (travel, product placement, group detection)

4. Spatio-temporal Data
  - Superimposing time varying data on a map reveals course, trends, etc.

5. Interactivity
  - Being able to interact with data provides more understanding
  - Can turn items on/off, drill down or roll up, explore the time dimension

6. Animation
  - Watching data being animated provides us with fresh perspectives

7. Real-Time
  - Real-time visualizations provides a level of immediacy/freshness/relevance/interest that is simply absent in non-real-time data.

8. Networks
  - Shows relationships between entities
  - Node attributes can add detail in form of labels
  - Edge attributes can quantify data

9. Tools for Generating Data Visualizations
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
