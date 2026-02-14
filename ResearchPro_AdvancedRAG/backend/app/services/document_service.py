from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from config.config import hf_embeddings

from backend.app.services.vision_service import MultimodalProcessor

class DocumentProcessor:
    def __init__(self):
        self.vectorstore = None
        self.multimodal_processor = MultimodalProcessor()
        self.processed_docs = []

    def load_and_process_pdf(self, filepath: str):
        self.processed_docs = self.multimodal_processor.load_and_process(filepath)

        return self.processed_docs
    
    #Removed extract tables and images 
    
    

    def create_retriever(self, docs):
        """
        Creates Semantic retrievers
        """
        # Semantic Retriever (Vector Search)
        print("Creating vector store...")
        self.vectorstore = FAISS.from_documents(docs, hf_embeddings)
        semantic_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        return semantic_retriever



    
    def get_statistics(self) -> dict:
        return {
            "processed_documents": len(self.processed_docs),
            "extracted_tables": len(self.extracted_tables),
            "extracted_images": len(self.extracted_images) if hasattr(self, 'extracted_images') else 0,
            "vectorstore_ready": self.vectorstore is not None
        }
