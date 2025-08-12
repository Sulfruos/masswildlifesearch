import PyPDF2
import re
from typing import List, Dict, Any
import json
from pathlib import Path
import hashlib

class PDFToCorpusConverter:
    def __init__(self): 
        # Basic section patterns (simplified)
        self.section_patterns = {
            'description': r'(?i)(description|physical\s+description|appearance)',
            'life_history': r'(?i)(life\s+history|reproduction|breeding|lifecycle)',
            'food_habits_habitat': r'(?i)(food|habits?|habitat|diet|feeding)',
            'tips_for_residents': r'(?i)(tips\s+for\s+residents|management|control|solutions|advice\s+for\s+residents)',
            'general': r'(?i)(general|overview|introduction)'
        }
    
    def extract_text_from_pdf(self, pdf_path : str) -> str:
        """Extract text content from PDF file."""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def split_into_sections(self, text: str) -> Dict[str, str]:
        """Split text into logical sections based on headers"""
        sections = {}
        current_section = "general"
        current_content = []

        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # check if line is section header
            section_found = None
            for section_key, pattern in self.section_patterns.items():
                if re.match(pattern, line):
                    section_found = section_key
                    break

            if section_found:
                # save previous section
                if current_content:
                    sections[current_section] = ' '.join(current_content).strip()

                # start new section
                current_section = section_found
                current_content = []
            else:
                current_content.append(line)
        
        # save final section
        if current_content:
            sections[current_section] = ' '.join(current_content).strip()
        
        return sections
    
    def create_chunks(self, sections: Dict[str, str], min_chunk_size: int = 100) -> List[Dict[str, str]]:
        """Create text sections combining sections if needed"""
        chunks = []

        for section, content in sections.items():
            sentences = re.split(r'(?<=[.!?])\s+', content)
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                if current_length + len(sentence) > 400 and current_chunk:
                    chunks.append({
                        'section': section,
                        'content': ' '.join(current_chunk).strip()
                    })
                    current_chunk = [sentence]
                    current_length = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence)

            if current_chunk:
                chunks.append({
                        'section': section,
                        'content': ' '.join(current_chunk).strip()
                    })
        
        return chunks
    
    def generate_id(self, section: str, index: int, filename: str = "") -> str:
        """Generate unique ID for each chunk."""
        base_name = Path(filename).stem.lower().replace('-', '_').replace(' ', '_') if filename else "doc"
        return f"{base_name}_{section}_{index:03d}"
    
    def convert_pdf_to_corpus(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Main conversion function."""
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Split into sections
        sections = self.split_into_sections(text)
        
        # Create chunks
        chunks = self.create_chunks(sections)
        
        # Build corpus entries
        corpus_entries = []
        for i, chunk in enumerate(chunks):
            if len(chunk['content'].strip()) < 50:  # Skip very short chunks
                continue
                
            entry = {
                "id": self.generate_id(chunk['section'], i + 1, pdf_path),
                "content": chunk['content'].strip(),
                "section": chunk['section'],
                "source": pdf_path
            }
            corpus_entries.append(entry)
        
        return corpus_entries

filename = "copperheads.pdf"

converter = PDFToCorpusConverter()
res = converter.convert_pdf_to_corpus(filename)

for entry in res:
        print(f"ID: {entry['id']}")
        print(f"Section: {entry['section']}")
        print(f"Content: {entry['content']}")
        print(f"Source: {entry['source']}")
        print("---")

base_name = Path(filename).stem.lower().replace('-', '_').replace(' ', '_') if filename else "doc"
with open(f'{base_name}.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)