import glob
import json
import os
import re

import pdfplumber

from config import GEMINI_API_KEY
from google import genai
from google.genai import types



# =========================================================
# 1. GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)




CORPUS_FOLDER = (
    "/home/karam-ali-jan/Documents/Research_work/"
    "system/final_corpus_package"
)

OUTPUT_FILE = (
    "/home/karam-ali-jan/Documents/Research_work/"
    "system/final_corpus_package/"
    "final_corpus_chunks.json"
)




def generate_embedding(text):


    response = client.models.embed_content(
        model="gemini-embedding-001",

        contents=text,

        config=types.EmbedContentConfig(
            output_dimensionality=1536
        )
    )

    
    embedding = response.embeddings[0].values

    return embedding




def extract_text_from_pdf(pdf_path):
  

    all_text = []

    
    with pdfplumber.open(pdf_path) as pdf:

        
        for page in pdf.pages:

           
            page_text = page.extract_text()

          
            if page_text:
                all_text.append(page_text)

 
    return "\n".join(all_text)




def split_into_chunks(text, document_id):
    
    heading_pattern = re.compile(
        r"(\d+\.\d+\s+[A-Z][^\n]*)"
    )

   
    parts = heading_pattern.split(text)

    chunks = []

    
    for i in range(1, len(parts), 2):

        
        heading = parts[i].strip()

        
        if i + 1 >= len(parts):
            break

       
        body = parts[i + 1].strip()

        
        section_number = heading.split()[0]

       
        section_id = section_number.replace(".", "_")

        
        chunk_id = (
            f"{document_id}_sec{section_id}"
        )

        
        chunk_text = (
            f"{heading} {body}"
        )

        
        print(
            f"Generating embedding for {chunk_id}..."
        )

        embedding = generate_embedding(
            chunk_text
        )

     

        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": document_id,
                "text": chunk_text,
                "embedding": embedding
            }
        )

    return chunks




if __name__ == "__main__":

    
    all_chunks = []

    
    pdf_files = sorted(
        glob.glob(
            os.path.join(
                CORPUS_FOLDER,
                "*.pdf"
            )
        )
    )

    print(
        f"Found {len(pdf_files)} PDF documents.\n"
    )

    

    for pdf_path in pdf_files:

        
        document_id = os.path.splitext(
            os.path.basename(pdf_path)
        )[0]

        print("=" * 60)

        print(
            f"Processing: {document_id}"
        )

        print("=" * 60)

        

        text = extract_text_from_pdf(
            pdf_path
        )

        print(
            f"Extracted {len(text)} characters."
        )

        
        chunks = split_into_chunks(
            text,
            document_id
        )

        
        all_chunks.extend(chunks)

        print(
            f"Created {len(chunks)} chunks.\n"
        )

        

        for chunk in chunks:

            
            word_count = len(
                chunk["text"].split()
            )

            warning = ""

            
            if word_count < 15:

                warning = (
                    " <-- SHORT: please review"
                )

            
            elif word_count > 80:

                warning = (
                    " <-- LONG: consider splitting"
                )

            print(
                f"[{chunk['chunk_id']}] "
                f"{word_count} words"
                f"{warning}"
            )

        print()


    

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


    

    print("=" * 60)

    print(
        f"TOTAL DOCUMENTS: {len(pdf_files)}"
    )

    print(
        f"TOTAL CHUNKS: {len(all_chunks)}"
    )

    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )

    print("=" * 60)