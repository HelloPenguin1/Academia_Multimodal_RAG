from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from config.config import hf_embeddings

from backend.app.services.vision_service import MultimodalProcessor

class DocumentProcessor:
    def __init__(self):
        self.vectorstore = None
        self.multimodal_processor = MultimodalProcessor()
        self.processed_docs = []
        self.extracted_tables = []

    def load_and_process_pdf(self, filepath: str):
        self.processed_docs = self.multimodal_processor.load_and_process(filepath)
        self.extracted_tables = self._extract_tables_from_docs(self.processed_docs)
        self.extracted_images = self._extract_images_from_docs(self.processed_docs)
        print(f"Generated {len(self.processed_docs)} enriched documents with {len(self.extracted_tables)} tables.")
        print("Extracted tables:")
        for i, t in enumerate(self.extracted_tables):
            print(i, "Page:", "Preview:", t['content'][:100])
            
        

        return self.processed_docs
    
    def _extract_tables_from_docs(self, docs):
        """Extract tables from processed documents for separate indexing"""
        extracted_tables = []
        for doc in docs:
            if doc.metadata.get('has_tables', False):
                tables = doc.metadata.get('original_tables', [])
                for table_html in tables:
                    extracted_tables.append({
                        'content': table_html,
                        'html': table_html,
                        'page_number': doc.metadata.get('page_number', 0),
                        'source': 'pdf'
                    })
        
        return extracted_tables
    
    def _extract_images_from_docs(self, docs):
        """Extract images from processed documents for separate indexing"""
        extracted_images = []
        for doc in docs:
            if doc.metadata.get("has_images", False):
                imgs = doc.metadata.get("original_images", [])
                for img in imgs:
                    extracted_images.append({
                        "content": "[IMAGE BASE64]",
                        "base64":  img.get("base64"),
                        "description": img.get("description"),
                        "page_number": doc.metadata.get("page_number", 0),
                        "source": "image"
                    })
        return extracted_images
    
    
    

    def create_retrievers(self, docs):
        """
        Creates Semantic (FAISS) and Syntactic (BM25) retrievers
        """
        # 1. Semantic Retriever (Vector Search)
        print("Creating vector store...")
        self.vectorstore = FAISS.from_documents(docs, hf_embeddings)
        semantic_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # 2. Syntactic Retriever (Keyword Search)
        print("Creating BM25 retriever...")
        syntactic_retriever = BM25Retriever.from_documents(
            documents=docs,
            preprocess_func=lambda text: text.lower().split()
        )
        syntactic_retriever.k = 5

        return semantic_retriever, syntactic_retriever




    def get_table_context(self, query: str) -> str:
        """Extract table context relevant to the user query"""
        if not self.extracted_tables:
            return ""
        
        query_lower = query.lower()
        relevant_tables = []
        
        # Find tables matching query keywords
        query_words = [word for word in query_lower.split() if len(word) > 3]
        
        for table in self.extracted_tables:
            table_content = table.get('content', '').lower()
            if any(word in table_content for word in query_words):
                relevant_tables.append(table)
        
        if relevant_tables:
            context = "\n\n=== RELEVANT TABLES FROM DOCUMENT ===\n"
            for i, table in enumerate(relevant_tables[:3], 1):
                page = table.get('page_number', 'unknown')
                context += f"\n[Table {i} - Page {page}]\n"
                content = table.get('content', '')
                content = content.replace("\n", " ")
                content = content[:400]  
                context += f"{content}...\n"
            return context
        
        return ""
    
    def get_image_context(self, query: str) -> str:
        if not self.extracted_images:
            return ""

        visual_keywords = ["figure", "image", "chart", "graph", "diagram", "visual"]

        if not any(w in query.lower() for w in visual_keywords):
            return ""

        context = "\n\n=== IMAGE ANALYSIS ===\n"

        for i, img in enumerate(self.extracted_images[:3], 1):
            page = img.get("page_number", "?")
            desc = img.get("description", "No analysis.")
            desc = img.get("description", "No analysis.")
            if len(desc) > 400:
                desc = desc[:400] + "..."

            context += f"\n[Image {i} — Page {page}]\n{desc}\n"

        return context

    
    def get_statistics(self) -> dict:
        return {
            "processed_documents": len(self.processed_docs),
            "extracted_tables": len(self.extracted_tables),
            "extracted_images": len(self.extracted_images) if hasattr(self, 'extracted_images') else 0,
            "vectorstore_ready": self.vectorstore is not None
        }
