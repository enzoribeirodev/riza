
import re
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_and_convert_to_markdown(file_path):
    print(f"Iniciando conversão do arquivo: {file_path}")
    pipeline_options = PdfPipelineOptions()

    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    })

    result = converter.convert(file_path)
    markdown_text = result.document.export_to_markdown()
    print("Conversão para Markdown concluída.")
    return markdown_text


def chunk_text(markdown_text, chunk_size = 600, chunk_overlap = 100):
    print("Iniciando o processo de chunking híbrido...")
    headers_to_split_on = [("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_text)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = []
    for split in md_header_splits:
        h2 = split.metadata.get('Header 2', '')
        h3 = split.metadata.get('Header 3', '')
        prefix = ""
        if h2 and h2.lower() not in split.page_content[:100].lower():
            prefix = f"[{h2}]"
            if h3:
                prefix += f" [{h3}]"
            prefix += "\n\n"
        
        content_with_context = prefix + split.page_content
        split.page_content = content_with_context

        if len(split.page_content) > chunk_size:
            sub_chunks = text_splitter.split_documents([split])
            chunks.extend(sub_chunks)
        else:
            chunks.append(split)
    print(f"Chunking concluído. {len(chunks)} chunks intermediários gerados.")
    return chunks


def clean_and_filter_chunks(chunks, min_length = 50):
    print("Iniciando limpeza e filtragem dos chunks...")
    for chunk in chunks:
        content = chunk.page_content
        content = re.sub(r'', '', content)
        content = re.sub(r'/C\d+', '', content)
        content = re.sub(r'\(\s*\)', '', content)
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        chunk.page_content = content.strip()
    
    filtered_chunks = [c for c in chunks if len(c.page_content) > min_length]
    print(f"Limpeza concluída. {len(filtered_chunks)} chunks finais restantes.")
    return filtered_chunks


def process_pdf_to_chunks(file_path):
    markdown_text = load_and_convert_to_markdown(file_path)
    intermediate_chunks = chunk_text(markdown_text)
    final_chunks = clean_and_filter_chunks(intermediate_chunks)
    return final_chunks