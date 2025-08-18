# sp-agent
A simple information delivery agent using a RAG system  

Chunking strategy by [Levels of Text Splitting by Greg Kamradt](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)  

# Prequisites

- python3  
- jupyter notebook  
- `pip install -r requirements.txt`

# How to Run

1. Install required packages  
2. Add your data to `datasets/` or use default data
```
cp <YOUR_PDF> .
python3 src/utils/pdf_convert_llm.py <YOUR_PDF> <DESIRED_NAME>
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
...
    chunks = chunker.process_files_parallel([
            "datasets/docker.txt", 
            "datasets/k8.txt", 
            "datasets/terraform.txt"
        ]) # add your custom files here
# when finished press CTRL+ENTER and follow prompt
```
