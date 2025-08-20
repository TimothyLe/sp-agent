# sp-agent
A simple information delivery agent using a RAG system  

Chunking strategy by [Levels of Text Splitting by Greg Kamradt](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)  

## Prequisites

- python3  
- jupyter notebook  
- `pip install -r requirements.txt`

## Key Features 

**🚀 High Performance:**
- Parallel processing of multiple files
- Efficient token estimation (4 chars/token rule)
- Embedding caching to avoid recomputation
- Batch processing for embeddings

**🧠 Semantic Intelligence:**
- Respects paragraph and sentence boundaries
- Adaptive chunking based on content structure
- Keyword extraction for each chunk
- Cosine similarity ranking for queries

**🔧 Your Model Integration:**
- Designed to work with `CompendiumLabs/bge-base-en-v1.5-gguf`
- Ready for integration with `Llama-3.2-1B-Instruct-GGUF`
- Proper embedding dimension handling

**⚡ Fast Processing:**
- Compiled regex patterns for speed
- ThreadPoolExecutor for parallel file processing
- In-memory caching with disk persistence
- Minimal overhead chunking algorithm

## Advanced Features

1. **Adaptive Chunking**: Respects semantic boundaries while maintaining size constraints
2. **Rich Metadata**: Each chunk includes keywords, token counts, source info, and embeddings  
3. **Caching System**: Persistent embedding cache for faster subsequent runs
4. **Statistics**: Built-in analytics for your chunk collection
5. **Batch Processing**: Optimized for large document collections

## Quick Start 

1. Install required packages  
2. Add your data to `./datasets` or use default data
```
cp <YOUR_PDF> ./datasets 
./process_training.sh
```  
3. Configure your RAG
```python
jupyter notebook
...
# open in browser
# scroll all the way down to last box
# edit the following
def start_rag_chunker():
    chunker = FastRAGChunker(
        embedding_model_path="sentence-transformers/all-MiniLM-L6-v2", # change model to custom
        chunk_size=512,
        chunk_overlap=50,
        max_workers=2,
        cache_embeddings=True
    )
# when finished press CTRL+ENTER and follow prompt
```
4. Ask questions and receive answers or unanswered context. Enter quit or exit to end the conversation  
