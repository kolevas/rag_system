import os
from pathlib import Path
from llama_index.core.schema import Document
import sys

# Add the parent directory to the path to import preprocessing functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_pdf_files import preprocess_pdf

class PDFTransformer:
    def __init__(self, chunk_size=1500, overlap=200, blob_metadata=None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.blob_metadata = blob_metadata or {}

    def __call__(self, documents):
        processed_docs = []
        for doc in documents:
            try:
                # Get file path from document metadata
                file_path = doc.metadata.get('file_path', doc.text)
                
                if not os.path.exists(file_path):
                    print(f"Warning: File not found: {file_path}")
                    # Fall back to original document
                    processed_docs.append(doc)
                    continue
                
                print(f"Processing PDF: {file_path}")
                
                with open(file_path, "rb") as f:
                    file_content = f.read()
                
                file_dict = {
                    "name": os.path.basename(file_path),
                    "content": file_content
                }
                
                result = preprocess_pdf(
                    file_dict,
                    chunk_size=self.chunk_size,
                    overlap=self.overlap,
                    blob_metadata=self.blob_metadata
                )
                
                chunks = result["result"]["chunks"]
                base_metadata = result["result"]["metadata"]
                
                # Merge with original metadata
                enhanced_metadata = {
                    **doc.metadata,
                    **base_metadata,
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_type": "pdf",
                    "transformer": "PDFTransformer"
                }
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        **enhanced_metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                    processed_docs.append(Document(text=chunk, metadata=chunk_metadata))
                
                print(f"✅ Processed PDF into {len(chunks)} chunks: {file_path}")
                
            except Exception as e:
                print(f"❌ Error processing PDF {file_path}: {e}")
                # Fall back to original document
                processed_docs.append(doc)
        
        return processed_docs