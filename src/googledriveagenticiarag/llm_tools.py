# cache and tools ::  https://python.langchain.com/docs/integrations/chat/anthropic/
from langchain_core.tools import tool

# For demonstration purposes, we artificially expand the
# tool description.
description = (
    f"Get file complete url to accessto it"
)

@tool(description=description)
def get_url(document_id: str) -> str:
    # TODO
    return f"https://drive.google.com/file/d/{document_id}/view"
