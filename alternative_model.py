import json
import re
from pathlib import Path
from typing import List, Set, Any
import numpy as np
import instructor
from pydantic import BaseModel, Field, model_validator
from openai import OpenAI


# ==========================================
# 1. DEFINE PYDANTIC SCHEMA WITH CONSTRAINTS
# ==========================================

class BoundingBoxField(BaseModel):
    value: str = Field(
        description="The exact text value extracted for this entity from the document content."
    )
    bbox: List[int] = Field(
        description="The 4-integer array [x1, y1, x2, y2] copied EXACTLY from the matching <block> tag's bbox attribute."
    )


class DocumentExtractionSchema(BaseModel):
    compound_name: BoundingBoxField = Field(description="The name of the compound.")
    source_name: BoundingBoxField = Field(description="The source or origin of the sample.")
    investigator_name: BoundingBoxField = Field(description="The name of the investigator(s).")

    @model_validator(mode="before")
    @classmethod
    def verify_coordinates_exist_in_context(cls, data: Any, handler: Any) -> Any:
        # Access the valid set of stringified bounding boxes passed via validation context
        allowed_boxes: Set[str] = handler.context.get("allowed_boxes", set())

        for field_name, field_data in data.items():
            if isinstance(field_data, dict) and "bbox" in field_data:
                # Format generated list back into "x1,y1,x2,y2" string match
                generated_box_str = ",".join(map(str, field_data["bbox"]))

                # Enforce strict spatial context grounding constraint
                if generated_box_str not in allowed_boxes:
                    raise ValueError(
                        f"Hallucination detected in '{field_name}'! "
                        f"The bbox [{generated_box_str}] does not exist in the source document tokens. "
                        f"You must strictly choose from the available prompt tags."
                    )
        return data


# ==========================================
# 2. GEOMETRIC XY-CUT SORTING ALGORITHM
# ==========================================

def recursive_xy_cut(elements: list, thresh_x: int = 15, thresh_y: int = 10) -> list:
    """
    Recursively segments and orders document elements into human reading order
    (top-to-bottom, left-to-right columns).
    Each element must have a 'box' key containing [x1, y1, x2, y2].
    """
    if len(elements) <= 1:
        return elements

    # Convert coordinates to numpy array for vector operations
    boxes = np.array([el["box"] for el in elements])

    # Check for vertical blank columns (horizontal projections)
    x1s, x2s = boxes[:, 0], boxes[:, 2]
    sorted_x_indices = np.argsort(x1s)

    split_idx = -1
    for i in range(len(sorted_x_indices) - 1):
        idx_curr = sorted_x_indices[i]
        idx_next = sorted_x_indices[i + 1]
        # If gap between elements exceeds the threshold, split into columns
        if x1s[idx_next] - max(x2s[sorted_x_indices[:i + 1]]) > thresh_x:
            split_idx = i + 1
            break

    if split_idx != -1:
        # Sort left column cluster and right column cluster recursively
        left_side = [elements[idx] for idx in sorted_x_indices[:split_idx]]
        right_side = [elements[idx] for idx in sorted_x_indices[split_idx:]]
        return recursive_xy_cut(left_side, thresh_x, thresh_y) + recursive_xy_cut(right_side, thresh_x, thresh_y)

    # If no vertical columns found, check for horizontal paragraph blocks (vertical projection)
    y1s, y2s = boxes[:, 1], boxes[:, 3]
    sorted_y_indices = np.argsort(y1s)

    for i in range(len(sorted_y_indices) - 1):
        idx_curr = sorted_y_indices[i]
        idx_next = sorted_y_indices[i + 1]
        # If vertical gap exceeds threshold, split lines
        if y1s[idx_next] - max(y2s[sorted_y_indices[:i + 1]]) > thresh_y:
            split_idx = i + 1
            break

    if split_idx != -1:
        top_side = [elements[idx] for idx in sorted_y_indices[:split_idx]]
        bottom_side = [elements[idx] for idx in sorted_y_indices[split_idx:]]
        return recursive_xy_cut(top_side, thresh_x, thresh_y) + recursive_xy_cut(bottom_side, thresh_x, thresh_y)

    # Flatten out remaining items sorted cleanly top to bottom
    return [elements[idx] for idx in sorted_y_indices]


# ==========================================
# 3. PREPROCESSING ENGINE (PATHLIB PATH INPUT)
# ==========================================

def process_ocr_to_xml(json_path: Path, target_width: int = 1000, target_height: int = 1000) -> tuple[str, set]:
    """
    Reads an OCR JSON file using pathlib, normalizes coordinates to a 0-1000 grid,
    sorts fields via XY-Cut, and yields an XML string for the LLM.
    """
    with json_path.open("r", encoding="utf-8") as file:
        raw_json_data = json.load(file)

    elements = raw_json_data.get("form", [])
    if not elements:
        return "", set()

    # Determine bounding canvas wrapper dimensions dynamically
    boxes = np.array([el["box"] for el in elements])
    max_orig_x = int(np.max(boxes[:, 2]))
    max_orig_y = int(np.max(boxes[:, 3]))

    # Step 1: Normalize and Scale layout onto an integer 0-1000 canvas grid
    for el in elements:
        x1, y1, x2, y2 = el["box"]
        nx1 = int(round((x1 / max_orig_x) * target_width))
        ny1 = int(round((y1 / max_orig_y) * target_height))
        nx2 = int(round((x2 / max_orig_x) * target_width))
        ny2 = int(round((y2 / max_orig_y) * target_height))
        el["box"] = [nx1, ny1, nx2, ny2]

    # Step 2: Apply the XY-Cut Layout Parser for clean reading paths
    ordered_elements = recursive_xy_cut(elements)

    # Step 3: Interleave sorted words/blocks into XML format string
    xml_output_lines = []
    allowed_boxes = set()

    for el in ordered_elements:
        box_str = ",".join(map(str, el["box"]))
        text_val = el["text"]

        # Track valid structural context keys to match against hallucinations later
        allowed_boxes.add(box_str)

        # Create structural XML tag block string wrapper
        xml_line = f'<block bbox="{box_str}">{text_val}</block>'
        xml_output_lines.append(xml_line)

    final_xml_prompt = "\n".join(xml_output_lines)
    return final_xml_prompt, allowed_boxes


# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    data_path = Path.cwd() / "dataset"
    data_files = [item for item in data_path.rglob("*.json") if item.is_file()]
    target_json_path = data_files[0]

    if not target_json_path.exists():
        print(f"Error: Target file not found at {target_json_path}")
        exit(1)

    # Execute preprocessing pass directly from the file path
    xml_prompt_context, valid_bboxes = process_ocr_to_xml(target_json_path)

    print("--- GENERATED COMPACT XML PROMPT FROM FILE ---")
    print(xml_prompt_context)
    print("\nTracking Valid Coordinate Keys:", valid_bboxes)
    print("------------------------------------\n")

    # Connect Instructor library interface wrapper to your running Llama engine instance
    client = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:8000/v1",  # pointing to local vLLM / Llama instance
            api_key="local-llama-token"
        ),
        mode=instructor.Mode.JSON
    )

    try:
        # Run extraction task incorporating self-correcting validation cycles (max_retries)
        extraction_result = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            response_model=DocumentExtractionSchema,
            validation_context={"allowed_boxes": valid_bboxes},
            max_retries=3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI document extractor. Read the provided text fields wrapped in <block> tags. "
                        "Identify the requested values and assign their EXACT coordinate array [x1, y1, x2, y2] "
                        "from the bbox attribute of the matching wrapper tag. Do not approximate or invent numbers."
                    )
                },
                {"role": "user", "content": xml_prompt_context}
            ]
        )

        print("--- EXTRACTED TARGET STRUCTURAL DATA ---")
        print(json.dumps(extraction_result.model_dump(), indent=2))

    except Exception as error:
        print(f"Extraction execution terminated: Validation mapping constraints failed: {error}")
