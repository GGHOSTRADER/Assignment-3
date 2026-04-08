import os
import re
import argparse
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from termcolor import colored
from config import get_embeddings, DATA_FOLDER, DB_FOLDER, FILES
from typing import List, Tuple
import shutil

# Disable Chroma telemetry to avoid warning messages
os.environ["CHROMA_TELEMETRY"] = "False"


def has_tables(page_text: str) -> bool:
    """
    Enhanced table detection for ALL types of tables including:
    - Markdown tables (with | pipes)
    - Space-aligned financial statements
    - Indented line items with aligned numbers
    - Multi-row financial data
    """
    lines = page_text.split("\n")

    # Method 1: Markdown-style tables (| col | col |)
    markdown_pattern = r"(\|.*\|[\s]*\n[\s]*\|[-:\s|]+\|[\s]*\n(\|.*\|[\s]*\n)+)"
    if re.search(markdown_pattern, page_text, re.MULTILINE):
        return True

    # Method 2: Financial statement indicators (MUST be early in page)
    financial_headers = [
        r"CONDENSED CONSOLIDATED STATEMENTS? OF",
        r"STATEMENTS? OF OPERATIONS",
        r"STATEMENTS? OF CASH FLOWS",
        r"BALANCE SHEET",
        r"\(in millions[,.]?\)",
        r"\(in thousands[,.]?\)",
        r"\(Unaudited\)",
    ]

    for header in financial_headers:
        if re.search(header, page_text, re.IGNORECASE):
            # Found financial statement header - likely has tables
            return True

    # Method 3: Look for aligned columns with numbers (MOST IMPORTANT for your case)
    # Check for lines with multiple number patterns separated by spaces
    lines_with_multiple_numbers = 0
    number_dense_lines = 0

    for line in lines[:50]:  # Check first 50 lines
        # Find all numbers in this line
        numbers = re.findall(r"\$?\s*\(?\d{1,3}(?:,\d{3})*\)?", line)
        numbers = [n for n in numbers if n.strip() and not n.strip() == "$"]

        if len(numbers) >= 2:
            lines_with_multiple_numbers += 1

        # Check if line is number-dense (numbers make up >30% of line)
        if numbers:
            digit_count = sum(len(re.findall(r"\d", n)) for n in numbers)
            line_length = len(line)
            if line_length > 0 and digit_count / line_length > 0.15:
                number_dense_lines += 1

    # Method 4: Look for indented line items (like "Products" under "Net sales:")
    indented_items = 0
    for i, line in enumerate(lines[:40]):
        if line.strip() and line.startswith(" "):  # Indented line
            # Check if next few lines have numbers
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.search(r"\d", next_line):
                    indented_items += 1

    # Method 5: Look for column headers with units
    column_headers = [
        r"(?:Three|Six|Nine|Twelve)\s+Months?\s+Ended",
        r"202\d\s+202\d",  # Years like "2024 2023"
        r"Net sales:",
        r"Cost of sales:",
        r"Operating expenses:",
    ]

    has_column_headers = any(
        re.search(pattern, page_text, re.IGNORECASE) for pattern in column_headers
    )

    # Method 6: Check for consistent number of numbers across rows
    # This is a strong indicator of a table
    number_counts = []
    for line in lines[:40]:
        numbers = re.findall(r"\$?\s*\(?\d{1,3}(?:,\d{3})*\)?", line)
        numbers = [n for n in numbers if n.strip() and not n.strip() == "$"]
        if len(numbers) >= 2:
            number_counts.append(len(numbers))

    # If most rows have the same number of numbers, it's likely a table
    consistent_columns = False
    if len(number_counts) >= 3:
        most_common = max(set(number_counts), key=number_counts.count)
        consistent_columns = number_counts.count(most_common) / len(number_counts) > 0.6

    # Decision logic - COMBINE multiple signals
    table_signals = [
        lines_with_multiple_numbers >= 3,  # At least 3 lines with 2+ numbers
        number_dense_lines >= 3,  # At least 3 number-dense lines
        indented_items >= 2,  # At least 2 indented items
        has_column_headers,  # Has financial column headers
        consistent_columns,  # Consistent number columns
    ]

    # If 2 or more signals are true, it's a table
    if sum(table_signals) >= 2:
        return True

    # Special case: Financial statements with "Total" rows
    if (
        re.search(r"Total\s+net sales|\*\*Total\*\*", page_text)
        and lines_with_multiple_numbers >= 2
    ):
        return True

    return False


def extract_tables_from_page(page_text: str) -> List[Tuple[str, str]]:
    """
    Extract tables from a page and convert to natural language.
    Returns list of (original_table, natural_language_version)
    """
    tables = []

    # Method 1: Extract markdown-style tables
    markdown_pattern = r"(\|.*\|[\s]*\n[\s]*\|[-:\s|]+\|[\s]*\n(\|.*\|[\s]*\n)+)"
    markdown_matches = list(re.finditer(markdown_pattern, page_text, re.MULTILINE))

    for match in markdown_matches:
        table_content = match.group(0)
        nl_table = convert_table_to_natural_language(table_content)
        tables.append((table_content, nl_table))

    # Method 2: Try to detect and extract space-aligned tables
    if not tables:
        space_table = extract_space_aligned_table(page_text)
        if space_table:
            nl_table = convert_table_to_natural_language(space_table)
            tables.append((space_table, nl_table))

    # Method 3: If still no tables but page was flagged as having tables, extract raw
    if not tables:
        # Just use the whole page as potential table content
        nl_table = convert_table_to_natural_language(page_text)
        tables.append((page_text[:500], nl_table))

    return tables


def extract_space_aligned_table(page_text: str) -> str:
    """
    Enhanced extraction for space-aligned financial tables.
    """
    lines = page_text.split("\n")
    table_lines = []
    in_table = False
    line_item = ""

    for line in lines:
        # Skip empty lines
        if not line.strip():
            if in_table and table_lines:
                break
            continue

        # Check if line has multiple numbers (table data)
        numbers = re.findall(r"\$?\s*\(?\d{1,3}(?:,\d{3})*\)?", line)
        numbers = [n for n in numbers if n.strip() and not n.strip() == "$"]

        # Check if line is a header (has words but also numbers OR years)
        has_years = re.search(r"202\d", line)
        has_words = re.search(r"[A-Za-z]{3,}", line)

        if len(numbers) >= 2 or has_years:
            if not in_table:
                in_table = True
            # Combine with previous line if it was a label (like "Research and development")
            if line_item and not re.search(r"\d", line) and has_words:
                line = f"{line_item} {line}"
                line_item = ""
            table_lines.append(line)
        elif has_words and in_table and not numbers:
            # This might be a label for the next line (e.g., "Research and development")
            line_item = line.strip()
        elif in_table and not has_words and not numbers:
            # End of table
            break

    if len(table_lines) >= 2:
        # Convert to markdown-like format
        markdown_lines = []
        for line in table_lines:
            # Split by multiple spaces or number boundaries
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) < 2:
                # Try splitting by number patterns
                parts = re.findall(
                    r"([A-Za-z][A-Za-z\s,]+?|\$?\s*\(?\d{1,3}(?:,\d{3})*\)?)", line
                )
                parts = [p.strip() for p in parts if p.strip()]

            if len(parts) >= 2:
                markdown_lines.append("| " + " | ".join(parts) + " |")

        if markdown_lines:
            return "\n".join(markdown_lines)

    return None


def convert_table_to_natural_language(table_text: str) -> str:
    """
    Convert a table to natural language sentences for better embedding.
    Handles financial statements, comparative data, and standard tables.
    """
    lines = [line.strip() for line in table_text.split("\n") if line.strip()]
    if len(lines) < 2:
        return table_text

    # Parse headers (first row)
    headers = []
    for line in lines:
        if "|" in line:
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if parts and any(not re.search(r"\d", p) for p in parts):
                headers = parts
                break

    if not headers:
        # No clear headers, treat as key-value pairs
        descriptions = []
        for line in lines:
            if "|" in line:
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) == 2:
                    descriptions.append(f"{parts[0]} is {parts[1]}")
                elif len(parts) > 2:
                    descriptions.append(" | ".join(parts))
        return "\n".join(descriptions) if descriptions else table_text

    # Parse data rows
    data_rows = []
    for line in lines:
        if "|" in line:
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) == len(headers) and any(re.search(r"\d", p) for p in parts):
                data_rows.append(parts)

    # Convert to natural language
    descriptions = []
    for row_idx, row in enumerate(data_rows):
        row_desc = []
        for header, cell in zip(headers, row):
            if cell and cell != "-" and cell != "—":
                # Handle parentheses for negative numbers
                if cell.startswith("(") and cell.endswith(")"):
                    cell = f"-{cell[1:-1]}"
                # Remove $ and commas for cleaner text
                cell = cell.replace("$", "").replace(",", "")
                row_desc.append(f"{header} is {cell}")

        if row_desc:
            descriptions.append(f"Record {row_idx + 1}: " + ", ".join(row_desc))

    # If no rows found, try alternative format
    if not descriptions:
        for line in lines:
            if "|" in line:
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 2:
                    descriptions.append(" | ".join(parts))

    return "\n".join(descriptions) if descriptions else table_text


def clean_regular_text(text: str) -> str:
    """
    Standard cleaning for non-table text.
    """
    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace single newlines with spaces (but keep paragraph breaks)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" +", " ", text)
    return text.strip()


def process_document_with_branching(docs: List[Document]) -> List[Document]:
    """
    Main branching logic: Process tables one way, regular text another way.
    Handles mixed pages intelligently.
    """
    processed_documents = []
    table_count = 0
    regular_count = 0
    mixed_pages = 0

    for doc in docs:
        has_table = has_tables(doc.page_content)

        if has_table:
            # Branch 1: This page HAS tables
            tables = extract_tables_from_page(doc.page_content)

            if tables:
                # Remove table content from original text
                text_without_tables = doc.page_content
                for table_content, _ in tables:
                    text_without_tables = text_without_tables.replace(table_content, "")

                # Process regular text that remains (if any)
                if text_without_tables.strip():
                    regular_doc = Document(
                        page_content=clean_regular_text(text_without_tables),
                        metadata={
                            **doc.metadata,
                            "content_type": "regular_text",
                            "has_tables_on_page": True,
                        },
                    )
                    processed_documents.append(regular_doc)
                    regular_count += 1
                    if tables:
                        mixed_pages += 1

                # Process each table
                for table_content, nl_table in tables:
                    table_doc = Document(
                        page_content=nl_table,
                        metadata={
                            **doc.metadata,
                            "content_type": "table",
                            "original_table_preview": (
                                table_content[:200] + "..."
                                if len(table_content) > 200
                                else table_content
                            ),
                        },
                    )
                    processed_documents.append(table_doc)
                    table_count += 1
            else:
                # Table detection was false positive, treat as regular text
                doc.page_content = clean_regular_text(doc.page_content)
                doc.metadata["content_type"] = "regular_text"
                processed_documents.append(doc)
                regular_count += 1
        else:
            # Branch 2: No tables, process normally
            doc.page_content = clean_regular_text(doc.page_content)
            doc.metadata["content_type"] = "regular_text"
            processed_documents.append(doc)
            regular_count += 1

    # Print branching statistics
    if mixed_pages > 0:
        print(
            colored(
                f"   - 📊 Mixed pages detected: {mixed_pages} (contain both text and tables)",
                "cyan",
            )
        )
    print(
        colored(
            f"   - 📦 Branching results: {table_count} table chunks, {regular_count} text chunks",
            "green",
        )
    )

    return processed_documents


def build_vector_dbs(update_doc: bool = False):
    """
    ETL Pipeline with branching for tables vs. regular text.

    Args:
        update_doc: If True, overwrite existing DBs. If False, skip existing ones.
    """
    print(colored("🚀 Starting RAG Pipeline with Table Branching...", "cyan"))

    if update_doc:
        print(
            colored("⚠️  Update mode: ON - Will overwrite existing databases", "yellow")
        )
    else:
        print(colored("ℹ️  Update mode: OFF - Will skip existing databases", "cyan"))

    embeddings = get_embeddings()

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(
            colored(
                f"⚠️ Warning: {DATA_FOLDER} directory was empty. Creating it...",
                "yellow",
            )
        )

    # Dynamic file discovery
    all_files = FILES.copy()
    for f in os.listdir(DATA_FOLDER):
        if f.endswith(".pdf"):
            key = f.split(".")[0].lower()
            if key not in all_files:
                all_files[key] = f
                print(
                    colored(
                        f"✨ Found new document: {f} (Setting key to '{key}')", "green"
                    )
                )

    for key, filename in all_files.items():
        persist_dir = os.path.join(DB_FOLDER, key)
        file_path = os.path.join(DATA_FOLDER, filename)

        # Check if DB already exists
        if os.path.exists(persist_dir):
            if update_doc:
                print(
                    colored(
                        f"🔄 Update mode: Removing existing DB for '{key}' at {persist_dir}...",
                        "yellow",
                    )
                )
                shutil.rmtree(persist_dir)
                print(colored(f"   - Old DB removed.", "yellow"))
            else:
                print(
                    colored(
                        f"✅ DB for '{key}' already exists at {persist_dir}. Skipping...",
                        "yellow",
                    )
                )
                continue

        if not os.path.exists(file_path):
            print(colored(f"❌ Missing source file: {filename}", "red"))
            continue

        print(colored(f"\n{'='*60}", "cyan"))
        print(colored(f"🔨 Building Vector Index for: {key}", "cyan"))
        print(colored(f"{'='*60}", "cyan"))

        # 1. Load PDF
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        print(f"   - Loaded {len(docs)} pages.")

        # 2. Analyze document composition
        pages_with_tables = sum(1 for doc in docs if has_tables(doc.page_content))
        print(
            f"   - Pages with tables: {pages_with_tables}/{len(docs)} ({pages_with_tables/len(docs)*100:.1f}%)"
        )

        # 3. Process with branching logic
        processed_docs = process_document_with_branching(docs)

        # 4. Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )

        splits = splitter.split_documents(processed_docs)

        # 5. Show final distribution
        table_chunks = len(
            [s for s in splits if s.metadata.get("content_type") == "table"]
        )
        text_chunks = len(
            [s for s in splits if s.metadata.get("content_type") == "regular_text"]
        )
        print(
            f"   - Final chunks: {table_chunks} tables, {text_chunks} text (total: {len(splits)})"
        )

        # 6. Embed & Store
        print("   - Embedding and storing... (This may take a while)")
        Chroma.from_documents(splits, embeddings, persist_directory=persist_dir)
        print(colored(f"🎉 Successfully built DB for {key}!\n", "green"))


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Build vector databases from PDF documents"
    )
    parser.add_argument(
        "--update_doc",
        type=str,
        choices=["yes", "no"],
        default="no",
        help="Whether to overwrite existing databases (yes=overwrite, no=skip)",
    )

    args = parser.parse_args()

    # Convert 'yes'/'no' to boolean
    update_mode = args.update_doc.lower() == "yes"

    # Run the pipeline
    build_vector_dbs(update_doc=update_mode)
