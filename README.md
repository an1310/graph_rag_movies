### Beyond Vector Search: A Simpler, Open-Source Path to Graph RAG

Retrieval-Augmented Generation (RAG) promises to make AI systems smarter by allowing them to answer questions with up-to-date, domain-specific information without costly retraining. However, most RAG pipelines treat information as a flat, disconnected list of documents. They retrieve isolated text chunks based on vector similarity, missing the obvious relationships between them.

To overcome this, developers have explored Graph RAG. The goal is to teach the RAG system to understand how different pieces of information are connected. While powerful, these approaches have often been too complex to be practical.

This article introduces a simpler approach that enhances our existing vector search with lightweight, metadata-based graph traversal. It requires no complex graph construction or separate database. The connections are defined at query time by specifying which document metadata fields to use as "edges." As demonstrated in an open-source notebook, this method allows for powerful, context-aware retrieval using our own hardware and models for maximum privacy and efficiency.

### The Challenge: Putting Movie Reviews in Context

A common goal for an AI-powered recommendation system is to provide meaningful answers to open-ended questions. Using a dataset of movie reviews, we want to support prompts like:

* “What are some classic Mel Brooks comedies?”
* “Which of his movies are parodies?”
* “What did critics say about Blazing Saddles vs. Spaceballs?”

A great answer requires combining subjective review content with structured data like genre, director, and release year. A traditional RAG system might retrieve relevant review snippets but would fail to connect them to the broader context. The system needs to:

1.  Retrieve relevant reviews using vector-based semantic search.
2.  Enrich each review with its corresponding movie details (title, year, director, etc.).
3.  Connect this information to other relevant data points, such as other reviews for the same movie or other films by the same director.

### How Graph RAG Addresses the Challenge

A plain RAG system might recommend a movie based on a few positive reviews. With Graph RAG, the system can pull in a richer set of context—like negative reviews for the same film or details about other movies in the same genre—to provide a more balanced and comprehensive recommendation.

This implementation uses a clean, two-step solution:

1.  **Build a Standard RAG System:** First, we embed our documents (movie reviews and movie details) using an open-source model like BGE-M3 and store them in a vector database. Each document contains its text content and structured metadata (e.g., `movie_id`, `genre`, `director`).
2.  **Add Graph Traversal with `GraphRetriever`:** After retrieving an initial set of documents via vector search, `GraphRetriever` follows connections defined in the metadata. For instance, it can link a review to its movie using the `reviewed_movie_id` and `movie_id` fields. This step merges related content into a single context window for the Large Language Model (LLM) to use.

Crucially, **no pre-built knowledge graph is needed**. The graph is defined dynamically at query time using metadata. If we want to add new connections based on actors or genres, we simply update the retriever’s configuration without reprocessing the data.

### Graph RAG in Action: A Mel Brooks Example

To demonstrate how this works, we’ll walk through a setup using a sample dataset of Mel Brooks movies and reviews. This process involves creating a vector store, structuring the data as LangChain documents, and configuring the graph traversal strategy.

#### The Dataset: Mel Brooks Classics

For this example, we use two datasets: one with metadata for several Mel Brooks films and another with reviews for those films. Each review is linked to a movie via a shared ID, creating a natural relationship perfect for `GraphRetriever`.

Here are the data snippets that will be loaded into our system.

**Movie Data:**
```python
movies_data_string = """
id,title,audienceScore,tomatoMeter,rating,releaseDateTheaters,runtimeMinutes,genre,director,writer
blazing_saddles,Blazing Saddles,91,91,R,1974-02-07,93,"Comedy,Western",Mel Brooks,"Mel Brooks,Norman Steinberg"
spaceballs,Spaceballs,83,52,PG,1987-06-24,96,"Comedy,Sci-Fi",Mel Brooks,"Mel Brooks,Thomas Meehan"
young_frankenstein,Young Frankenstein,92,92,PG,1974-12-15,106,"Comedy,Horror",Mel Brooks,"Gene Wilder,Mel Brooks"
the_producers,The Producers,81,91,NR,1967-11-22,88,Comedy,Mel Brooks,Mel Brooks
"""
```

**Review Data:**
```python
reviews_data_string = """
id,reviewId,criticName,originalScore,reviewText,scoreSentiment
blazing_saddles,101,Gene Siskel,4/4,"A hilarious, biting satire of Westerns that could never be made today. It's brilliant.",POSITIVE
blazing_saddles,102,Roger Ebert,3/4,"While sometimes juvenile, the film's relentless barrage of gags produces some genuine comedic gold.",POSITIVE
spaceballs,201,Janet Maslin,,,"A weak parody that offers more merchandising opportunities than laughs. The jokes are telegraphed and tired.",NEGATIVE
spaceballs,202,Dave Kehr,2/4,"Brooks' scattershot approach misses more than it hits, but fans of his style will find something to enjoy.",POSITIVE
young_frankenstein,301,Vincent Canby,5/5,"A masterful parody that lovingly recreates the look and feel of 1930s horror films. Gene Wilder is superb.",POSITIVE
the_producers,401,Bosley Crowther,,,"An audacious, brilliantly offensive comedy that pushes the boundaries of taste in the most wonderful way.",POSITIVE
"""
```

#### Step 1: Set Up Open-Source Models and Vector Store

We begin by setting up our open-source embedding and language models. This example uses BGE-M3 for embeddings and Qwen2.5-72B-Instruct (run via vLLM) for generation, ensuring all data processing happens locally. For the vector store, we'll use an in-memory option for simplicity, but production systems could use persistent stores like Chroma or Milvus.

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import VLLMOpenAI
from langchain_core.vectorstores import InMemoryVectorStore

# Initialize BGE-M3 embeddings for retrieval
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# Initialize the LLM client, pointing to a local vLLM server
llm = VLLMOpenAI(
    openai_api_base="http://localhost:8000/v1",
    model="Qwen/Qwen2.5-72B-Instruct",
    temperature=0,
)

# Use an in-memory vector store for the demo
vectorstore = InMemoryVectorStore(embeddings)
```
After loading the movie and review data into `Document` objects, we add them to the vector store.

#### Step 2: Configure and Run GraphRetriever

Next, we configure `GraphRetriever` to traverse the connection between a review and its movie. We define a single directional edge: from a review's `reviewed_movie_id` to a movie's `movie_id`. We use an "eager" traversal strategy, which retrieves an initial set of documents and then pulls in adjacent, connected documents.

```python
from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever

retriever = GraphRetriever(
    store=vectorstore,
    edges=[("reviewed_movie_id", "movie_id")],
    strategy=Eager(start_k=10, adjacent_k=10, max_depth=1),
)
```
With this configuration, a query first performs a semantic search for up to 10 initial documents. Then, for each of those documents, it follows the defined edge to retrieve up to 10 connected documents.

#### Step 3: Invoke a Query and Generate a Response

Now we can ask a natural language question. The retriever will find relevant reviews and automatically pull in the associated movie metadata.

```python
INITIAL_PROMPT_TEXT = "What are some good family movies?"
query_results = retriever.invoke(INITIAL_PROMPT_TEXT)
```

After compiling the retrieved movies and reviews into a formatted text block, we pass this rich context to our LLM to generate a final, synthesized answer.

**Final prompt to the LLM:**
```
A list of Movie Reviews appears below. Please answer the Initial Prompt text
(below) using only the listed Movie Reviews. Please include all movies that
might be helpful to someone looking for movie recommendations.

Initial Prompt:
What are some classic Mel Brooks comedies?

Movie Reviews:
 Movie Title: The Producers
 Movie ID: the_producers
 Review: An audacious, brilliantly offensive comedy that pushes the boundaries of taste in the most wonderful way.

 Movie Title: Young Frankenstein
 Movie ID: young_frankenstein
 Review: A masterful parody that lovingly recreates the look and feel of 1930s horror films. Gene Wilder is superb.

 Movie Title: Blazing Saddles
 Movie ID: blazing_saddles
 Review: While sometimes juvenile, the film's relentless barrage of gags produces some genuine comedic gold.
 Review: A hilarious, biting satire of Westerns that could never be made today. It's brilliant.
```

**Final generated response:**
> Based on the provided reviews, here are some classic Mel Brooks comedies:
>
> * **The Producers**: This is described as an "audacious, brilliantly offensive comedy."
> * **Young Frankenstein**: Called a "masterful parody" of 1930s horror films.
> * **Blazing Saddles**: This film is noted as a "hilarious, biting satire of Westerns" with a "relentless barrage of gags."

By expanding the context window beyond simple semantic search, the graph-aware system provides a more complete and helpful response grounded in both reviews and metadata.

### Why an Open-Source Approach Matters

Using a fully open-source, on-premises stack for Graph RAG offers significant advantages over relying on external APIs.

| Metric | OpenAI | Open-Source (On-Premises) |
| :--- | :--- | :--- |
| **Privacy** | External API | Fully On-Premises |
| **Embedding Cost** | $0.13/1M tokens | ~$0.0001/1M tokens |
| **LLM Cost** | $5-15/1M tokens | ~$0.01/1M tokens |
| **Customization** | Limited | Full Fine-tuning |
| **Latency** | Network-dependent | Consistent <1s |

This approach not only reduces costs and enhances data privacy but also provides the freedom to fine-tune models on specific domains for even better performance.

### Conclusion

Graph RAG doesn't have to be a complex undertaking. By leveraging metadata we already have, `GraphRetriever` provides a powerful yet simple way to build context-aware AI systems. When combined with the speed and privacy of modern open-source models, this technique bridges the gap between unstructured opinions and structured facts, producing query responses that are more intelligent, trustworthy, and complete.
