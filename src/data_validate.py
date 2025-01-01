from pydantic import BaseModel, validator, HttpUrl, ValidationError
from typing import Literal
import re
import os 

# -------------------------------------------------------------------------
# ENUM-Like Literal for allowed data types
# -------------------------------------------------------------------------
DataType = Literal["url", "text", "github_repo_url", "github_repo_text"]


# -------------------------------------------------------------------------
# Pydantic Model for Data Validation
# -------------------------------------------------------------------------
class DataPayload(BaseModel):
    """
    DataPayload validates different kinds of inputs:
      1) A URL (general web URL)
      2) A text snippet
      3) A GitHub repository URL (e.g., 'https://github.com/org/repo')
      4) A GitHub repo text spec (e.g., 'ollama_ai/llama3.1')
    """
    data_type: DataType
    data: str

    # ---------------------------------------------------------------------
    # 1. Validate the combination of `data_type` and `data`
    # ---------------------------------------------------------------------
    @validator("data")
    def validate_data(cls, v, values):
        """
        Main logic to ensure the `data` field matches the declared `data_type`.
        """
        dt = values.get("data_type")
        if dt == "url":
            # We rely on a secondary pydantic model or a custom check to ensure v is a valid URL.
            # Instead of a secondary model, we can do a quick check here:
            try:
                _ = HttpUrl(v)  # Attempt to parse as a valid URL
            except ValidationError as e:
                raise ValueError(f"Invalid URL: {v}. Error: {e}")
        
        elif dt == "text":
            # Accept raw text; optionally impose length or format constraints
            if v.startswith("http://") or v.startswith("https://"):
                raise ValueError("Text data cannot contain URLs.")
            if not v or len(v.strip()) == 0:
                raise ValueError("Text data is empty.")
            # You could also check for maximum length, etc.
        
        elif dt == "github_repo_url":
            # Check if it starts with "https://github.com/" or "http://github.com/"
            if not v.startswith("https://github.com/") and not v.startswith("http://github.com/"):
                raise ValueError("Invalid GitHub repo URL. Must start with 'https://github.com/'.")
            # Optionally, parse further to ensure a valid path structure: e.g., https://github.com/<user>/<repo>
           
        elif dt == "github_repo_text":
            # Expect a string like "owner/repo" (or optionally "owner/repo@tag")
            # We explicitly disallow any URL-like pattern (e.g. "https://github.com/...").
            if v.startswith("http://") or v.startswith("https://"):
                raise ValueError("Invalid GitHub repo text. Cannot be a URL.")
            # Minimal check: must contain at least one slash
            if "." in v or "@" in v or " " in v or len(v) < 3 or "/" not in v:
                raise ValueError("Invalid GitHub repo text. Must be in 'owner/repo' form.")
            pattern = r'^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$'

            if not re.match(pattern, v):
                raise ValueError(
                    "Invalid GitHub repo text. Must be in 'owner/repo' form, "
                    "using only letters, digits, underscores, hyphens, or periods."
                )
                
        elif dt == "file":
            if isinstance(v, list):
                invalid_files = [file for file in v if not os.path.exists(file)]
                if invalid_files:
                    raise ValueError(f"The following files do not exist: {', '.join(invalid_files)}")
            elif isinstance(v, str):
                if not os.path.exists(v):
                    raise ValueError(f"File {v} does not exist.")
            else:
                raise ValueError("Invalid file data. Must be a string (path) or a list of paths.")
        
        else:
            return v  # No validation for other types
        
        return v


# -------------------------------------------------------------------------
# Validation Class / Utility
# -------------------------------------------------------------------------
class DataValidator:
    """
    Provides a convenient interface for validating data inputs using the
    DataPayload Pydantic model. If valid, returns the validated data model.
    """

    def __init__(self):
        """
        You can store additional settings, loggers, or external references
        here if needed.
        """
        pass

    def validate_input(self, data_type: DataType, data: str) -> DataPayload:
        """
        Validate an incoming piece of data against the Pydantic model.
        
        :param data_type: One of ('url', 'text', 'github_repo_url', 'github_repo_text').
        :param data:      The actual data string to validate.
        :return:          A validated DataPayload object.
        :raises ValidationError: If data doesn't meet the criteria.
        """
        payload = DataPayload(data_type=data_type, data=data)
        return payload


# -------------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    validator = DataValidator()

    # EXAMPLE 1: Valid URL
    try:
        validated_1 = validator.validate_input("url", "https://example.com/docs")
        print("Validated 1:", validated_1)
    except ValidationError as e:
        print("Validation Error 1:", e)

    # EXAMPLE 2: Invalid GitHub Repo URL
    try:
        validated_2 = validator.validate_input("github_repo_url", "https://mywebsite.com/owner/repo")
        print("Validated 2:", validated_2)
    except ValidationError as e:
        print("Validation Error 2:", e)

    # EXAMPLE 3: Valid GitHub Repo Text
    try:
        validated_3 = validator.validate_input("github_repo_text", "ollama_ai/llama3.1")
        print("Validated 3:", validated_3)
    except ValidationError as e:
        print("Validation Error 3:", e)

    # EXAMPLE 4: Simple text
    try:
        validated_4 = validator.validate_input("text", "Here is some raw text input.")
        print("Validated 4:", validated_4)
    except ValidationError as e:
        print("Validation Error 4:", e)
