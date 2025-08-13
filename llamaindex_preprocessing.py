import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import chromadb
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Add preprocessing path
sys.path.append('./preprocessing')
from preprocessing.document_transformers.pdf_transformer import LlamaIndexPDFTransformer as PDFTransformer
from preprocessing.document_transformers.doc_transformer import LlamaIndexDOCTransformer as DOCTransformer
from preprocessing.document_transformers.ppt_transformer import LlamaIndexPPTTransformer as PPTTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Config:
    """Centralized configuration for Azure OpenAI."""
    azure_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    api_version: str = "2025-01-01-preview"
    deployment_name: str = "gpt-4o"

    def __post_init__(self):
        load_dotenv()
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", self.api_version)
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", self.deployment_name)

    @property
    def azure_available(self) -> bool:
        return bool(self.api_key and self.azure_endpoint)


class LlamaIndexPreprocessor:
    """Handles preprocessing, indexing, and vector store management for LlamaIndex."""

    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.txt', '.md'}

    def __init__(self, persist_directory="./chroma_db",
                 collection_name="multimodal_downloaded_data_with_embedding"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.index = None

        self._setup_embeddings()
        self._setup_vector_store()
        self._setup_ingestion_pipeline()

    # ---------- Setup ----------
    def _setup_embeddings(self):
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            max_length=512, device="cpu", trust_remote_code=False, normalize=True
        )
        self._log_setup("HuggingFace embeddings", True)

    def _setup_vector_store(self):
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        self.chroma_collection = self.chroma_client.get_or_create_collection(self.collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self._log_setup("ChromaDB vector store", True)

    def _setup_ingestion_pipeline(self):
        try:
            transformers = [
                PDFTransformer(chunk_size=1500, overlap=200),
                DOCTransformer(chunk_size=1500, overlap=200),
                PPTTransformer(chunk_size=1500, overlap=200),
                SentenceSplitter(chunk_size=1500, chunk_overlap=200,
                                 paragraph_separator="\n\n", secondary_chunking_regex=r"[.!?]\s+"),
                Settings.embed_model
            ]
            self.ingestion_pipeline = IngestionPipeline(transformations=transformers, vector_store=self.vector_store)
            self._log_setup("Ingestion pipeline", True)
        except Exception as e:
            self._log_setup("Ingestion pipeline", False, e)
            self.ingestion_pipeline = None

    # ---------- Logging ----------
    @staticmethod
    def _log_setup(component: str, success: bool, error: Exception = None):
        status = "✅" if success else "⚠️"
        message = f"{status} {component} configured" if success else f"{status} {component} setup failed: {error}"
        print(message)

    # ---------- Index Building ----------
    def build_index(self, doc_directory: Optional[str] = None) -> bool:
        if self._try_load_existing_index():
            return True

        if not doc_directory or not os.path.exists(doc_directory):
            print("No existing index found and no document directory provided.")
            return False

        docs = self._load_documents_with_paths(doc_directory)
        if not docs:
            print(f"❌ No documents found in directory: {doc_directory}")
            return False

        print(f"📝 Creating new index from {len(docs)} documents...")
        self._create_index(docs)
        return True

    def _try_load_existing_index(self) -> bool:
        if self.chroma_collection.count() <= 0:
            return False
        try:
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store, storage_context=self.storage_context
            )
            print(f"✅ Loaded existing index from ChromaDB (collection: {self.collection_name})")
            return True
        except Exception as e:
            print(f"⚠️ Could not load existing index: {e}")
            return False

    def _create_index(self, docs: List[Document]):
        if self.ingestion_pipeline:
            nodes = self.ingestion_pipeline.run(documents=docs)
            self.index = VectorStoreIndex(nodes, storage_context=self.storage_context)
            print(f"✅ Created index with ingestion pipeline from {len(nodes)} processed chunks")
        else:
            self.index = VectorStoreIndex.from_documents(docs, storage_context=self.storage_context)
            print("✅ Created index with fallback processing")

    # ---------- Document Loading ----------
    def _load_documents_with_paths(self, doc_directory: str) -> List[Document]:
        documents = []
        try:
            for file_path in Path(doc_directory).rglob('*'):
                if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    continue

                file_type = self._detect_file_type(file_path)
                metadata = {
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'file_type': file_type,
                    'original_extension': file_path.suffix[1:].lower()
                }

                if file_type in {'pdf', 'docx', 'doc', 'pptx', 'ppt'}:
                    documents.append(Document(text=str(file_path), metadata=metadata))
                else:
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        documents.append(Document(text=content, metadata=metadata))
                    except Exception as e:
                        print(f"⚠️ Could not read text file {file_path}: {e}")

            print(f"📁 Loaded {len(documents)} document file references")
            return documents
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            return SimpleDirectoryReader(doc_directory).load_data()

    @staticmethod
    def _detect_file_type(file_path: Path) -> str:
        try:
            header = file_path.read_bytes()[:100]
            if header.startswith(b'%PDF'):
                return 'pdf'
            if header.startswith(b'PK\x03\x04'):
                return 'docx' if file_path.suffix.lower() in ['.docx', '.doc'] else 'pptx'
            if header.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
                return 'doc' if file_path.suffix.lower() == '.doc' else 'ppt'
            return 'txt'
        except Exception:
            return file_path.suffix[1:].lower() or 'txt'

    # ---------- Pipeline Processing ----------
    def process_documents_with_pipeline(self, doc_directory: str, update_existing: bool = False) -> bool:
        if not self.ingestion_pipeline or not os.path.exists(doc_directory):
            print("❌ Ingestion pipeline not available or invalid directory")
            return False

        docs = self._load_documents_with_paths(doc_directory)
        if not docs:
            return False

        print(f"🔧 Processing {len(docs)} documents with custom transformers...")
        try:
            nodes = self.ingestion_pipeline.run(documents=docs)
            if update_existing and self.index:
                for node in nodes:
                    self.index.insert(node)
                print(f"✅ Added {len(nodes)} chunks to existing index")
            else:
                self.index = VectorStoreIndex(nodes, storage_context=self.storage_context)
                print(f"✅ Created new index from {len(nodes)} chunks")
            return True
        except Exception as e:
            print(f"❌ Error processing documents: {e}")
            return False

    def configure_pipeline(self, chunk_size=1500, overlap=200, enable_transformers=None) -> bool:
        enable_transformers = enable_transformers or ['pdf', 'doc', 'ppt', 'text']
        transformer_map = {
            'pdf': lambda: PDFTransformer(chunk_size=chunk_size, overlap=overlap),
            'doc': lambda: DOCTransformer(chunk_size=chunk_size, overlap=overlap),
            'ppt': lambda: PPTTransformer(chunk_size=chunk_size, overlap=overlap),
            'text': lambda: SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap,
                                             paragraph_separator="\n\n", secondary_chunking_regex=r"[.!?]\s+")
        }
        try:
            transformers = [transformer_map[t]() for t in enable_transformers if t in transformer_map]
            for t in enable_transformers:
                if t in transformer_map:
                    print(f"✅ {t.upper()} transformer enabled")

            transformers.append(Settings.embed_model)
            self.ingestion_pipeline = IngestionPipeline(transformations=transformers, vector_store=self.vector_store)
            print(f"✅ Pipeline reconfigured (chunk_size={chunk_size}, overlap={overlap})")
            return True
        except Exception as e:
            print(f"❌ Error configuring pipeline: {e}")
            return False

    # ---------- Accessors ----------
    def get_index(self): return self.index
    def get_vector_store(self): return self.vector_store
    def get_storage_context(self): return self.storage_context


if __name__ == "__main__":
    preprocessor = LlamaIndexPreprocessor()
    print("LlamaIndex Preprocessor initialized!")
    if preprocessor.build_index():
        print("Index built successfully!")
    else:
        print("No existing index found. Provide a document directory to build a new one.")
