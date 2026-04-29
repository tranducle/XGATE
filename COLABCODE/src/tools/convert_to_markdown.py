"""
Convert various file formats to Markdown using Microsoft MarkItDown.

Supported formats:
- PDF, PowerPoint, Word, Excel
- Images (OCR + EXIF metadata)
- Audio (speech transcription)
- HTML, CSV, JSON, XML
- ZIP files (iterates over contents)
- YouTube URLs
- EPubs

No AI API required - pure file conversion (except for optional image descriptions).
"""

from pathlib import Path
from typing import Optional, Union
import os
import logging

logger = logging.getLogger(__name__)

try:
    from markitdown import MarkItDown

    MARKITDOWN_AVAILABLE = True
except ImportError:
    MarkItDown = None
    MARKITDOWN_AVAILABLE = False
    logger.warning("markitdown not installed. Run: pip install 'markitdown[all]'")


def convert_file_to_markdown(
    file_path: str, output_path: Optional[str] = None, enable_plugins: bool = False
) -> str:
    """
    Convert a file to Markdown format.

    Args:
        file_path: Path to the input file (PDF, DOCX, PPTX, XLSX, images, etc.)
        output_path: Optional path to save the output markdown file
        enable_plugins: Whether to enable MarkItDown plugins (default False)

    Returns:
        Markdown text content

    Supported formats:
        - PDF (.pdf)
        - Word (.docx)
        - PowerPoint (.pptx)
        - Excel (.xlsx, .xls)
        - Images (.png, .jpg, .jpeg, .gif, .bmp, .tiff) - OCR + EXIF
        - Audio (.wav, .mp3) - Speech transcription
        - HTML (.html, .htm)
        - Text-based (.csv, .json, .xml, .txt)
        - Archives (.zip) - iterates contents
        - EPub (.epub)
    """
    if not MARKITDOWN_AVAILABLE:
        return "ERROR: markitdown not installed. Run: pip install 'markitdown[all]'"

    if not os.path.exists(file_path):
        return f"ERROR: File not found: {file_path}"

    try:
        md = MarkItDown(enable_plugins=enable_plugins)
        result = md.convert(file_path)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(result.text_content, encoding="utf-8")
            logger.info("Saved markdown to: %s", output_path)

        return result.text_content

    except Exception as e:
        return f"ERROR converting file: {str(e)}"


def convert_url_to_markdown(url: str) -> str:
    """
    Convert a URL to Markdown format.

    Supports:
        - YouTube URLs (fetches transcript)
        - Web pages (converts HTML to Markdown)

    Args:
        url: URL to convert (YouTube or web page)

    Returns:
        Markdown text content
    """
    if not MARKITDOWN_AVAILABLE:
        return "ERROR: markitdown not installed. Run: pip install 'markitdown[all]'"

    try:
        md = MarkItDown(enable_plugins=False)
        result = md.convert(url)
        return result.text_content

    except Exception as e:
        return f"ERROR converting URL: {str(e)}"


def convert_multiple_files(file_paths: list, output_dir: Optional[str] = None) -> dict:
    """
    Convert multiple files to Markdown.

    Args:
        file_paths: List of file paths to convert
        output_dir: Optional directory to save output files

    Returns:
        Dict mapping file paths to markdown content or error messages
    """
    results = {}

    for file_path in file_paths:
        if output_dir:
            output_path = os.path.join(output_dir, Path(file_path).stem + ".md")
        else:
            output_path = None

        results[file_path] = convert_file_to_markdown(file_path, output_path)

    return results


def get_supported_extensions() -> list:
    """Return list of supported file extensions."""
    return [
        # Documents
        ".pdf",
        ".docx",
        ".doc",
        ".pptx",
        ".ppt",
        ".xlsx",
        ".xls",
        # Images
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        # Audio
        ".wav",
        ".mp3",
        # Web/Text
        ".html",
        ".htm",
        ".xml",
        ".csv",
        ".json",
        ".txt",
        # Archives
        ".zip",
        # Ebooks
        ".epub",
    ]


# Tool registry constant (consistent with LITERATURE_TOOLS, WRITING_TOOLS, etc.)
CONVERT_TOOLS = [
    {
        "name": "convert_file_to_markdown",
        "description": "Convert a file (PDF, DOCX, PPTX, XLSX, images, audio) to Markdown",
        "parameters": [
            {
                "name": "file_path",
                "type": "string",
                "required": True,
                "description": "Path to the input file",
            },
            {
                "name": "output_path",
                "type": "string",
                "required": False,
                "description": "Optional path to save output",
            },
        ],
    },
    {
        "name": "convert_url_to_markdown",
        "description": "Convert a URL (web page or YouTube) to Markdown",
        "parameters": [
            {
                "name": "url",
                "type": "string",
                "required": True,
                "description": "URL to convert",
            },
        ],
    },
    {
        "name": "convert_multiple_files",
        "description": "Convert multiple files to Markdown in batch",
        "parameters": [
            {
                "name": "file_paths",
                "type": "array",
                "required": True,
                "description": "List of file paths",
            },
            {
                "name": "output_dir",
                "type": "string",
                "required": False,
                "description": "Output directory",
            },
        ],
    },
    {
        "name": "get_supported_extensions",
        "description": "Get list of supported file extensions for conversion",
        "parameters": [],
    },
]


# CLI interface for direct usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python convert_to_markdown.py <file_path> [output_path]")
        print("\nSupported formats:")
        print("  " + ", ".join(get_supported_extensions()))
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    result = convert_file_to_markdown(input_file, output_file)

    if not output_file:
        print(result)
