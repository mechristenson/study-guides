# CSCI 572: Information Retrieval and Web Search Engines
## Final Study Guide

### Test Details:
  - Monday March 2nd, 2020
  - 8:00 - 8:50
  - SGM 123

### Major Topics:
  - Search Engine Basics
  - Characterizing the Web
  - Search Engine Evaluation
  - Web Crawling
  - Identifying Duplicates
  - Information Retrieval
  - Processing Text: Lexicon & Text Normalization
  - Building an Inverted Index
  - Video Search Engines
  - Google Query Formulation
  - Page Rank

### Sources of Information
  - CSCI 572 Lecture Notes, Ellis Horowitz [Lecture]
  - Course Homepage <http://csci572.com/> [Web]

---
### Search Engine Basics

#### Sources:
  - Lecture 1/13/20

#### Concepts
1. Web Search Engine Phases
  - Crawler
    - Collects web pages recursively and builds corpus
    - For each known URL, fetch the page, parse it and extract new URLs, Repeat
  - Indexers
    - Creates inverted indexes
    - Various policies wrt which words are indexed, capitalization, support for
      unicode, stemming, support for phrases, etc.
  - Query Processer
    - Serves query results
    - Front end: query reformulation, word stemming, capitalization,
      optimization of Booleans, etc.
    - Back end: finds matching documents and ranks them

#### Questions
  - Spring 2012 Question 1

---
### Characterizing the Web

#### Sources
  - Lecture 1/15/20

#### Concepts
1. Apache Tika
  - Toolkit that detects and extracts metadata and text content from various documents
    using existing parser libraries
  - Tika unifies these parsers under a single interface
  - Useful for search engine indexing
  - Features
    - Document Type Detection
    - Language Detection
    - Context Extraction
    - Metadata Extraction
  - Represents content as N-grams

2. Technology Cycles
  - Mainframe Computing (1960s)
  - Mini Computing (1970s)
  - Personal Computing (1980s)
  - Desktop Computing (1990s)
  - Mobile Internet Computing (2000s)
  - Wearables, Iot Computing (2010s)

3. Measuring the Web
  - 1.7 Billion Websites
  - 72% belong to .com
  - 30 trillion unique URLs
  - Thousands of different content types
  - 1 trillion web pages at about 100k bytes/page requires 100 petabytes
  - Google processes 24 petabytes per day
  - Internet Archive has more than 10 petabytes
  - 55% of documents are in english
  - 100 of millions of pages of spam
  - 8 links/page average
  - 30-40% of websites are (near) duplicates

4. Deep Web vs Dark Web
  - Deep Web is 90% of the information on the internet, but is not accessible by
    surface web crawlers
  - Dark Web is part of the deep web only accessible through certain browsers such
    as Tor, designed to ensure anonymity

#### Questions
  - Spring 2012 Question 39, 40
  - Spring 2013 Question 37
  - Spring 2017 Question 1, 2, 3

---
### Search Engine Evaluation

#### Sources
  - Lecture 1/22/20
  - Homework 1

#### Concepts
1. Recall and Precision
  - Measure of quality of search engines
  - Precision: #(relevant items retrieved) / #(all retrieved items)
  - Recall: #(relevant items retrieved) / #(all relevant items)
  - You can get high recall (but low precision) by retrieving all docs for all
    queries
  - In a good system, precision decreases as the number of docs retrieved
    increases

2. Spearman's Footrule
  - Distance between two lists of items is the sum of the total displacement of
    all items.

3. Harmonic Mean
  - Tends strongly toward the least element of the list, making it useful in analyzing
    search engine results
    - Algorithm:
      - Add the reciprocals of the numbers in the set
      - Divide the sum by n
      - Take the reciprocal of the result

4. Discounted Cumulative Gain
  - The premise of DCG is that highly relevant documents appearing lower in a search
    result should be penalized as the graded relevance value is reduced logarithmically
    proportional to the position of the result
  - DCG = REL(1) + Sum(i=2, p) ( REL(i) / log2(i))
    - REL(i) is the relevance of the ith result for p total results

5. Mean Average Precision
  - The mean of the average precision scores for each query in a set of queries
  - Most commonly used measure in research papers

6. F Score
  - Harmonic mean of recall and precision
  - F = 2RP / (R+P)

#### Questions
  - Spring 2012 Question 6
  - Spring 2013 Question 12
  - Spring 2014 Question 2
  - Fall 2017 Question 7, 21, 24, 25

---
### Web Crawling

#### Sources
  - Lecture 1/27/20

#### Concepts
1. Robots.txt
  - Protocol for announcing limitations for a web crawler as it visits a website
  - Placed at root directory of website
  - Should be the first file that a crawler looks at
  - Example:
    `User-agent: *
     Disallow: /yoursite/temp/`
  - Example:
    `<meta name=robots content="noindex,nofollow">`
    - Prevents page from being included in GoogleBots index
    - Prevents GoogleBot from following any links on the page

2. Cho and Garcia-Molina's Optimal Web Crawler
  - Researched development of web crawler optimized for freshness
    - Freshness: measure of whether a crawled page is up-to-date
  - Tested two revisiting policies
    - Uniform Policy
      - Involves revisiting all pages in the collection with the same frequency
        regardless of their rates of change
    - Proportional Policy
      - Involves revisiting more often the pages that change more frequently
    - Uniform performs better, because when a page changes too often, the
      crawler will waste time by trying to re-crawl it too fast, and still will
      not be able to keep its copy of the page fresh
    - To improve freshness, we should penalize elements that change too often

3. Coordination of Distributed Crawling
  - Independent
    - No coordination, every process follows its extracted links
  - Dynamic Assignment
    - A central coordinator dynamically divides the web into small partitions and
      assigns each partition to a process
  - Static Assignment
    - Web is partitioned and assigned without a central coordinator before the
      crawl starts

4. Common Webcrawlers
  - Google
    - Googlebot
  - Yahoo
    - Slurp

5. Cloaking
  - A website that returns altered webpages to search engines crawling the site
  - Often used to improve search engine ranking by misleading the search engine
    robot that into thinking the content on a page is different than it is

6. Page Jacking
  - Illegally copying legitimate website content (usually in the form of source
    code) to another website designed to replicate the original website.

7. Normalizing URLs
  - Convert the scheme and host to lowercase
  - Capitalize letters in escape sequences
  - Decode percent-encoded octets of unreserved characters
  - Remove the default port

8. Mercator Crawler
  - Two queue system
    - Front queue
      - controls freshness
      - list
    - Back queue
      - controls politeness
      - min heap

9. Crawler4j Details
  - Number of crawlers is set in Controller class
    `controller.start(MyCrawler.class, numberOfCrawlers);`
  - To visit binary encoded files, we need to set in the controller class
    `CrawlConfig config = new CrawlConfig();
     config.setIncludeBinaryContentInCrawling(true);`
  - To filter out unwanted filetypes we can set in the Crawler class:
    `private final static Pattern FILTERS = Pattern.compile(".*(\\.(css|js|mp3|zip|gz))$");`
  - Status Codes
    - 200 OK
    - 201 Created
    - 301 Moved
    - 401 Unauthorized
    - 501 Not Implemented

10. Avoiding Spider Traps
  - Remove any ID query strings from indexed urls to avoid re-visiting pages based
    on session id
  - Trim url length if it becomes too long

#### Questions
  - Spring 2012 Question 2, 3, 7, 22, 23, 30
  - Spring 2013 Question 11, 14, 28, 40
  - Spring 2014 Question 15, 16, 25
  - Fall 2017 Question 12
---

### Identifying Duplicates

#### Sources
  - Lecture 1/29/20

#### Concepts
1. Jaccard Similarity
  - Similarity measure for sets in the range [0,1]
  - JS = |A intersect B| / |A union B|

2. Edit Distance (Levenshtein Distance)
  - Distance between two words is the minimum number of single-character edits
    (insertions, deletions, substitutions)

3. De-Duplication
  - Process of identifying and avoiding essentially identical web pages
  - For web crawling we try to identify identical and near-identical web pages
    and only index a single version to return as a search result
  - Results in:
    - Smarter Crawling
      - Avoid returning many duplicate results to a query
      - Allow fetching from the fastest or freshest server
    - Better Connectivity Analysis
      - Avoid double counting for Page Rank
    - Add Redundancy in result listings
    - Reduce Crawl time

4. Solving the Duplicate Problem: Exact Match
  - Compute cryptographic finger prints for indexed pages

5. Properties of Distance Measures
  - No negative distances
  - Distance is 0 if x = y
  - Distance is symmetric
  - Triangle Inequality

6. Cosine Similarity
  - A . B / (|A||B|) = cos (theta)

7. Properties of Cryptographic Hash Functions
  - Easy to calculate hash for given data
  - Computationally difficult to calculate an alphanumeric text that has a given
    hash
  - A small change to text yields a totally different hash value
  - Unlikely that two slightly different messages will have the same hash
  - Example: MD5, SHA1

#### Questions
  - Spring 2012 Question 5, 10, 20, 21
  - Spring 2013 Question 29
  - Spring 2014 Question 20
  - Fall 2017 Question 10, 14, 17
---

### Information Retrieval

#### Sources
  - Lecture 2/3/20

#### Concepts
1. TF.IDF
    - Term Frequency x Inverse Document Frequency
      - The measure of importance for a word in a document.
      - A term occurring frequently in the document but rarely in the rest of
        the collection is given high weight
      - IDF = log(N/df(i)), where N is the number of documents, df(i) is the
        number of documents containing the specific term i

2. Inverse Document Frequency Observations
  - IDF has no effect on ranking for one term queries. IDF is constant for a term
    across a collection, so since each document's relevance is determined only by
    a single product (tf * idf) for that term t, IDF can be ignored.

3. Document Frequency
  - In the range of 0-N

4. Collection Frequency
  - Total number of times a term t appears across all documents in a collection

5. Ranked Retrieval of Documents
  - We can leverage a heap to quickly select the top k documents related to a query

#### Questions
  - Spring 2012 Question 4
  - Spring 2013 Question 13
  - Spring 2014 Question 1
  - Spring 2017 Question 6
---

### Processing Text: Lexicon and Text Normalization

#### Sources
  - Lecture 2/10/20

#### Concepts
1. Operations for Refining Vocabulary
  - Tokenization
    - Task of chopping a document unit into pieces called tokens
  - Stop Words
    - Remove most commonly used words which are not very helpful in determining
      the relevance between documents and queries
  - Capitalization/Case folding
    - Reducing all letters to lowercase
    - Sometimes case folding causes us to lose meaning for words, e.g. Bush v bush
  - Synonyms
    - Similar meaning words
  - Similar sounding words
    - Soundex algorithm encodes words to pronouncation codes
  - Stemming and Lemmatization
    - Reduce multiple forms of a word to a common base form
    - Stemming usually refers to heuristic process of chopping of ends of words
    - Lemmatization refers to doing things properly with the use of a vocabulary
      and morphological analysis of words

2. Zipf's Law
  - Power law, of form y = kx^c,
    - where c = -1
    - y is term frequency
    - x is the term's rank in the frequency table
  - Frequency of any word is inversely proportional to its rank in the frequency
    table.
  - Most frequent word will appear approximately twice as often as the second most
    word, three times as often as the third most frequent word, etc.
  - Displayed on a log-log plot so that it appears as a straight line with slope c

3. Soundex Algorithm
  - Phonetic algorithm for indexing names by their sound when pronounced in English
  - Basic aim is for names with the same pronunciation to be encoded to the same
    string so that matching can occur despite minor differences in spelling.
  - Consists of a Letter and three numbers

4. Porter's Algorithm
  - Standard algorithm used for english stemming
  - Created by Martin Porter
  - Reduces multiple forms of a word to a common base
  - Consists of some suffix stripping rules

5. Heap's Law
   - Describes the number of distinct words in a set of documents as a function
     of the document length
   - Power law, of form y = kx^c,
     - where k ~= [10, 100], c ~= [0.4 - 0.6]
     - y is size of the vocabulary
     - x is document length

6. Lexicon
  - The entire database of unique words for a domain, including syntactic, semantic,
    and morphological information

7. Hypernym
  - A word with a broad meaning that more specific words fall under
  - e.g. Cat, Bombay

8. Encoding Standards
  - Unicode, ASCII, UTF-8

#### Questions
  - Spring 2012 Question 11, 12, 13, 17, 18, 19
  - Spring 2013 Question 9, 18, 32, 36
  - Spring 2014 Question 6, 7, 17
  - Fall 2017 Question 8, 9, 16, 23
---
### Building an Inverted Index

#### Sources
  - Lecture 2/12/20

#### Concepts
1. Inverted Index
  - Composed of a vector containing all distinct words of the text collection in
    lexicographical order and for each word in the vocabulary, a list of all documents
    in which that word occurs.
  - Composed of a document to index dictionary and a postings list

2. Term-Document Incidence Matrix
  - Matrix representation of inverted index
  - Sparse matrix representation of terms and documents
  - Term, document intersection is 1 if word appears in document

3. Phrase Queries
  - We want to answer queries such as Santa Monica as a phrase
  - If we only store terms and documents in our inverted index we can't answer a
    phrase query
  - If we store bi-grams or tri-grams, our vocabulary will explode
  - Positional Indexes
    - Alternative solution where we store term and the indexes where it appears
    - Expands postings storage rather than vocabulary

4. Query Processing Across the Postings List
  - For queries that join two documents, we must merge the inverted indices of both
    documents
  - We can do this by walking through the two postings simultaneously in time linear
    to the total number of postings entries

5. Skip Pointers
  - To speed up the merging of postings, we can use Skip Pointers so that we don't
    need to walk over every element of the postings
  - Only helpful for AND queries
  - Useful when corpus is relatively static
  - Added at indexing time

#### Questions
  - Spring 2012 Question 14, 15, 16
  - Spring 2013 Question 4
  - Spring 2014 Question 19, 22
  - Fall 2017 Question 13
---
### Video Search Engines

#### Sources
  - Lecture 2/19/20

#### Concepts
1. ContentID
  - The system that Youtube uses to identify content that is uploaded by someone
    who does not own the copyright
  - Creates digital fingerprint of a video and audio which a company can use to
    prevent others from stealing their copyrighted content

2. Ranking of Videos
  - Relevance: using meta data and user preferences
  - Ordered by date of upload
  - Ordered by number of views
  - Ordered by duration
  - Ordered by user rating

3. Indexing of Videos
  - Handled by acquiring meta-data associated with the video
    - Author, title, creation date, duration, coding quality, tags, description

4. Covisitation Count
  - A tool that Youtube uses to come up with related videos
  - Total number of times a video y was watched after x across all user sessions

5. Uploading a video
  - Name, description, tags, language, license, age restrictions, categories

#### Questions
  - Fall 2017 Question 23
---
### Google Query Formulation

#### Sources
  - Lecture 2/24/20

#### Concepts
1. Google Search Operators
  - allinanchor:
    - pages must contain all query terms in the anchor text on links to the
      page
  - allintext:
    - pages must contain all query terms in the text of the page
  - allintitle:
    - pages must contain all query in the title of the page
  - allinurl:
    - pages must contain all query in the url of the page
  - filetype:
    - pages must end in specific file suffix
  - inanchor:
    - pages must contain some of the query terms in the anchor text on links
      to the page
  - intext:
    - pages must contain some of query terms in the text of the page
  - site:
    - pages must be in the specified domain
  - cache:
    - shows the version of a web page that google has in its cache
  - link:
    - restricts results to those web pages that have links to the specified
      url
  - related:
    - lists web pages that are similar to a specified web page
  - info:
    - presents some information that Google has about a particular web page
  - stocks:
    - treats query as stock ticker symbols

2. Auto Completion
  - Process of predicting a word or phrase that the user wants to type in
    without the user actually typing it in.
  - Google uses past history, soundex algorithms and spelling correction
    algorithms to assist in making guesses
  - Challenge is to search a large index or a long list of popular queries in
    a very short amount of time so the results pop up while the user is typing
  - Google does auto completion after the user enters the first character
  - Google does autocompletion and spell checking at the same time
  - Bing posts suggestions before any characters are entered

3. Relevance Feedback
  - After initial retrieval results are presented, Google allows the user to provide
    feedback on the relevance of one or more retrieved documents (related searches)

4. Google Search Behavior
  - Collects all user queries, uses it for recommendations and ads
  - Joins multiple keywords with ANDs by default
    - e.g. President OR "Abe Lincoln" died -> (President OR "Abe Lincoln") AND died
  - Google offers full-word wildcard queries

5. Mean Reciprocal Rank
  - Statistical measure for evaluating any process that produces a list of possible
    responses to a sample of queries, ordered by probability of correctness.
  - Used to rank quality of auto-complete answers
    - MRR = Sum(i=1,N) (1/rank(Qi))/N
      - Given N queries, Q
      - rank(Qi) is the rank of the ith query

#### Questions
  - Spring 2012 Question 8, 9
  - Spring 2013 Question 17, 33
  - Spring 2014 Question 3, 24
  - Fall 2017 Question 4, 16
---

### Page Rank

#### Sources
  - Lecture 2/26/20

#### Concepts
1. Page Rank
  - Web link analysis algorithm used by Google
  - Pages "vote" on how important other pages are by linking to them
  - PR(A) = (1-d) + d (PR(T1)/C(T1) + ... PR(Tn)/C(Tn))
      - T1..n are pages that point to A
      - d is a damping factor [0,1],
      - C(T1) is the number of links out of T1
      - PR(T1) is the page rank of site T1

2. PageRank Observations
  - Loop or Fully Connected Structure will result in all pages approaching 1
  - Hierarchy Structure to Website will result in highly ranked homepage
  - Increasing internal links can minimize damage to your PR when you give away
    votes by linking to external sites
  - If a particular page is highly important, use a hierarchical structure
  - Where a group of pages may contain outward links, increase the number of internal
    links to retain as much PR as possible
  - Where a group of pages do not contain outward links, the number of internal
    links in the site has no effect on the site's average PR. You may as well use
    a link structure that gives the user the best navigational experience
  - Use Site Maps: linking to a site map on each page increases the number of internal
    links in the site, spreading PR out and protecting you against your vote "donations"
    to other sites
  - Fully Meshed vs Looping with 1 in 1 out
    - Looping with 1 in 1 out causes everything to rise except the the link to home
      which drops due to split from outlink
    - Fully Meshed with 1 in 1 out causes everything to rise

3. HITS Algorithm
  - Hyperlink-Induced Topic Search
  - Link analysis algorithm that preceded Page Rank
  - Based on observations
    - Hubs: serve as large directories of information on a given topic
    - Authorites: serve as pages that best describe the information on a given topic
  - Authority score is sum of scaled hub values that point to a page
  - Hub score is the sum of the scaled authority values of the pages it points to

#### Questions
  - Spring 2012 Question 24, 25, 26
  - Spring 2013 Question 19, 20, 35
  - Spring 2014 Question 8, 9, 10, 18, 23
  - Fall 2017 Question 18, 19, 20
