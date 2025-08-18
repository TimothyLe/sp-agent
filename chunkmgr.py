import os
import re
import asyncio
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
from sentence_transformers import SentenceTransformer
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

# # similarity between vectors
# def cosine_similarity(a, b):
#   dot_product = sum([x * y for x, y in zip(a, b)])
#   norm_a = sum([x ** 2 for x in a]) ** 0.5
#   norm_b = sum([x ** 2 for x in b]) ** 0.5
#   return dot_product / (norm_a * norm_b)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    chunk_id: str
    source_file: str
    start_pos: int
    end_pos: int
    token_count: int
    sentence_count: int
    keywords: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    semantic_score: float = 0.0

@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata
    
    def __len__(self):
        return len(self.text)
    
    def __hash__(self):
        return hash(self.metadata.chunk_id)

# HP chunker supports semantic chunking and ranking
class FastRAGChunker:
    
    def __init__(
        self,
        embedding_model_path: str = "CompendiumLabs/bge-base-en-v1.5-gguf",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        max_workers: int = 4,
        cache_embeddings: bool = True,
        cache_dir: str = "./embedding_cache"
    ):
        """
        Initialize the RAG chunking agent.
        
        Args:
            embedding_model_path: HuggingFace model path for embeddings
            chunk_size: Target size for chunks (in tokens)
            chunk_overlap: Overlap between chunks (in tokens)
            min_chunk_size: Minimum chunk size to avoid tiny chunks
            max_workers: Number of parallel workers for processing
            cache_embeddings: Whether to cache embeddings to disk
            cache_dir: Directory for caching embeddings
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_workers = max_workers
        self.cache_embeddings = cache_embeddings
        self.cache_dir = Path(cache_dir)
        
        if self.cache_embeddings:
            self.cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"Loading embedding model: {embedding_model_path}")
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        self.stop_words = set(stopwords.words('english'))
        
        self.sentence_endings = re.compile(r'[.!?]+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
        self.whitespace_normalize = re.compile(r'\s+')
        
        logger.info("RAG Chunker initialized successfully")
    
    def _generate_chunk_id(self, text: str, file_path: str, start_pos: int) -> str:
        """Generate unique chunk ID based on content and position"""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        file_name = Path(file_path).stem
        return f"{file_name}_{start_pos}_{content_hash}"
    
    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract keywords from text using simple frequency analysis"""
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha() and w not in self.stop_words and len(w) > 2]
        
        # Simple frequency-based keyword extraction
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top_k
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [kw[0] for kw in keywords]
    
    def _estimate_tokens(self, text: str) -> int:
        """Fast token count estimation (roughly 4 chars per token)"""
        return len(text) // 4
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = self.whitespace_normalize.sub(' ', text)
        # Remove control characters but keep newlines
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        return text.strip()
    
    def _semantic_split(self, text: str) -> List[str]:
        """
        Split text using semantic boundaries (paragraphs, sentences).
        Prioritizes natural breakpoints over rigid character limits.
        """
        # First try paragraph splits
        paragraphs = self.paragraph_breaks.split(text)
        if len(paragraphs) > 1:
            return [p.strip() for p in paragraphs if p.strip()]
        
        # Fall back to sentence splits
        sentences = sent_tokenize(text)
        if len(sentences) > 1:
            return sentences
        
        # If no natural breaks, return as single chunk
        return [text]
    
    def _chunk_text_adaptive(self, text: str, file_path: str) -> List[Chunk]:
        """
        Adaptive chunking that respects semantic boundaries while maintaining size limits.
        """
        chunks = []
        text = self._clean_text(text)
        
        if len(text) < self.min_chunk_size:
            return chunks  # Skip very small texts
        
        # Split into semantic units
        semantic_units = self._semantic_split(text)
        
        current_chunk = ""
        current_pos = 0
        
        for unit in semantic_units:
            unit_tokens = self._estimate_tokens(unit)
            current_tokens = self._estimate_tokens(current_chunk)
            
            # If adding this unit would exceed chunk size, finalize current chunk
            if current_tokens > 0 and (current_tokens + unit_tokens) > self.chunk_size:
                if current_tokens >= self.min_chunk_size:
                    chunks.append(self._create_chunk(current_chunk, file_path, current_pos))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and chunks:
                    overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + " " + unit
                else:
                    current_chunk = unit
                
                current_pos += len(current_chunk) - len(unit)
            else:
                # Add unit to current chunk
                current_chunk = (current_chunk + " " + unit).strip()
        
        # Add final chunk
        if len(current_chunk) >= self.min_chunk_size:
            chunks.append(self._create_chunk(current_chunk, file_path, current_pos))
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk"""
        tokens = text.split()
        if len(tokens) <= overlap_size:
            return text
        return " ".join(tokens[-overlap_size:])
    
    def _create_chunk(self, text: str, file_path: str, start_pos: int) -> Chunk:
        """Create a chunk object with metadata"""
        chunk_id = self._generate_chunk_id(text, file_path, start_pos)
        keywords = self._extract_keywords(text)
        
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            source_file=file_path,
            start_pos=start_pos,
            end_pos=start_pos + len(text),
            token_count=self._estimate_tokens(text),
            sentence_count=len(sent_tokenize(text)),
            keywords=keywords
        )
        
        return Chunk(text=text, metadata=metadata)
    
    def _load_cached_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Load cached embedding if available"""
        if not self.cache_embeddings:
            return None
        
        cache_file = self.cache_dir / f"{chunk_id}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached embedding for {chunk_id}: {e}")
        return None
    
    def _save_cached_embedding(self, chunk_id: str, embedding: np.ndarray):
        """Save embedding to cache"""
        if not self.cache_embeddings:
            return
        
        cache_file = self.cache_dir / f"{chunk_id}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding for {chunk_id}: {e}")
    
    def _compute_embeddings_batch(self, chunks: List[Chunk]) -> List[Chunk]:
        """Compute embeddings for a batch of chunks"""
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache first
        for i, chunk in enumerate(chunks):
            cached_embedding = self._load_cached_embedding(chunk.metadata.chunk_id)
            if cached_embedding is not None:
                chunk.metadata.embedding = cached_embedding
            else:
                texts_to_embed.append(chunk.text)
                indices_to_embed.append(i)
        
        # Compute embeddings for uncached chunks
        if texts_to_embed:
            logger.info(f"Computing embeddings for {len(texts_to_embed)} chunks")
            embeddings = self.embedding_model.encode(
                texts_to_embed,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Assign embeddings and cache them
            for idx, embedding in zip(indices_to_embed, embeddings):
                chunk = chunks[idx]
                chunk.metadata.embedding = embedding
                self._save_cached_embedding(chunk.metadata.chunk_id, embedding)
        
        return chunks
    
    def process_file(self, file_path: Union[str, Path]) -> List[Chunk]:
        """
        Process a single text file and return chunks.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of Chunk objects
        """
        file_path = Path(file_path)
        logger.info(f"Processing file: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []
        
        # Create chunks
        chunks = self._chunk_text_adaptive(content, str(file_path))
        
        # Compute embeddings
        chunks = self._compute_embeddings_batch(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {file_path}")
        return chunks
    
    def process_files_parallel(self, file_paths: List[Union[str, Path]]) -> List[Chunk]:
        """
        Process multiple files in parallel.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Combined list of all chunks
        """
        all_chunks = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_file, fp): fp 
                for fp in file_paths
            }
            
            for future in future_to_file:
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    file_path = future_to_file[future]
                    logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Processed {len(file_paths)} files, created {len(all_chunks)} total chunks")
        return all_chunks
    
    def rank_chunks_by_query(
        self,
        chunks: List[Chunk],
        query: str,
        top_k: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """
        Rank chunks by semantic similarity to a query.
        
        Args:
            chunks: List of chunks to rank
            query: Query text
            top_k: Number of top chunks to return
            
        Returns:
            List of (chunk, similarity_score) tuples, sorted by relevance
        """
        if not chunks:
            return []
        
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        
        chunk_embeddings = np.array([chunk.metadata.embedding for chunk in chunks])
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
        ranked_chunks = [
            (chunk, float(sim)) 
            for chunk, sim in zip(chunks, similarities)
        ]
        
        ranked_chunks.sort(key=lambda x: x[1], reverse=True)
        
        for chunk, score in ranked_chunks:
            chunk.metadata.semantic_score = score
        
        return ranked_chunks[:top_k]
    
    def save_chunks(self, chunks: List[Chunk], output_path: Union[str, Path]):
        """Save chunks to disk for later use"""
        output_path = Path(output_path)
        with open(output_path, 'wb') as f:
            pickle.dump(chunks, f)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    
    def load_chunks(self, input_path: Union[str, Path]) -> List[Chunk]:
        """Load chunks from disk"""
        input_path = Path(input_path)
        with open(input_path, 'rb') as f:
            chunks = pickle.load(f)
        logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
        return chunks
    
    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict:
        """Get statistics about the chunks"""
        if not chunks:
            return {}
        
        token_counts = [chunk.metadata.token_count for chunk in chunks]
        text_lengths = [len(chunk.text) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_tokens_per_chunk": np.mean(token_counts),
            "avg_chars_per_chunk": np.mean(text_lengths),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "total_tokens": sum(token_counts),
            "total_characters": sum(text_lengths),
            "unique_source_files": len(set(chunk.metadata.source_file for chunk in chunks))
        }

def start_rag_chunker():
    chunker = FastRAGChunker(
        embedding_model_path="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=512,
        chunk_overlap=50,
        max_workers=2,
        cache_embeddings=True
    )
    
    # chunker = FastRAGChunker(
    #     embedding_model_path="CompendiumLabs/bge-base-en-v1.5-gguf",
    #     chunk_size=512,
    #     chunk_overlap=50,
    #     max_workers=4,
    #     cache_embeddings=True
    # )
    
    try:
        print("Processing files...")
        # chunks = chunker.process_file(fname)
        chunks = chunker.process_files_parallel([
            "datasets/docker.txt", 
            "datasets/k8.txt", 
            "datasets/terraform.txt"
        ])
        
        # Display statistics
        stats = chunker.get_chunk_stats(chunks)
        print(f"\nChunk Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Demo semantic search
        input_query = input('Ask me a question: ')
        # retrieved_knowledge = retrieve(input_query)
        print(f"\nSearching for: '{input_query}'")
        ranked_chunks = chunker.rank_chunks_by_query(chunks, input_query, top_k=3)
        
        print("\nTop 3 relevant chunks:")
        for i, (chunk, score) in enumerate(ranked_chunks, 1):
            print(f"\n{i}. Similarity: {score:.3f}")
            print(f"   Text: {chunk.text[:200]}...")
            print(f"   Keywords: {chunk.metadata.keywords}")
            print(f"   Tokens: {chunk.metadata.token_count}")        

        # Reuse chat code from 1st example
        import ollama
        LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
        instruction_prompt = f'''You are a helpful chatbot.
        Use only the following pieces of context to answer the question. Don't make up any new information:
        '''
        
        # ollama chatbot
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
            ],
            stream=True,
        )
        
        # print the response from the chatbot in real-time
        print('Chatbot response:')
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    start_rag_chunker()
