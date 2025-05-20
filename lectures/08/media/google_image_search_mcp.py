#!/usr/bin/env python3
"""
Google Image Search MCP Server

This script creates an MCP server that exposes Google Image Search functionality
as tools that can be used by LLM agents.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional

# Import MCP SDK
from modelcontextprotocol.sdk.server import Server
from modelcontextprotocol.sdk.server.stdio import StdioServerTransport
from modelcontextprotocol.sdk.types import (
    CallToolRequestSchema,
    ErrorCode,
    ListResourcesRequestSchema,
    ListResourceTemplatesRequestSchema,
    ListToolsRequestSchema,
    McpError,
    ReadResourceRequestSchema,
)

# Import our Google Image Search functionality
from google_image_search import (
    GoogleImageSearch,
    CREDENTIALS_PATH,
    SEARCH_ENGINE_ID,
    DOWNLOAD_DIR,
    MEDIA_DIR,
)

# Check if search engine ID is set
if SEARCH_ENGINE_ID == "YOUR_SEARCH_ENGINE_ID":
    print(
        "Error: You need to set the SEARCH_ENGINE_ID in google_image_search.py"
    )
    sys.exit(1)


class GoogleImageSearchMcpServer:
    """MCP server for Google Image Search functionality."""

    def __init__(self):
        """Initialize the MCP server."""
        self.server = Server(
            {
                "name": "google-image-search-server",
                "version": "0.1.0",
            },
            {
                "capabilities": {
                    "resources": {},
                    "tools": {},
                },
            },
        )

        # Initialize the Google Image Search client
        self.search_client = GoogleImageSearch(
            CREDENTIALS_PATH, SEARCH_ENGINE_ID
        )

        # Set up request handlers
        self.setup_tool_handlers()
        self.setup_resource_handlers()

        # Error handling
        self.server.onerror = lambda error: print(
            f"[MCP Error] {error}", file=sys.stderr
        )

        # Handle graceful shutdown
        import signal

        signal.signal(signal.SIGINT, self.handle_sigint)

    def handle_sigint(self, sig, frame):
        """Handle SIGINT (Ctrl+C) to gracefully shut down the server."""
        print("\nShutting down MCP server...", file=sys.stderr)
        self.server.close()
        sys.exit(0)

    def setup_tool_handlers(self):
        """Set up handlers for MCP tools."""
        # List available tools
        self.server.setRequestHandler(
            ListToolsRequestSchema, self.handle_list_tools
        )

        # Handle tool calls
        self.server.setRequestHandler(
            CallToolRequestSchema, self.handle_call_tool
        )

    def setup_resource_handlers(self):
        """Set up handlers for MCP resources."""
        # List available resources
        self.server.setRequestHandler(
            ListResourcesRequestSchema, self.handle_list_resources
        )

        # List resource templates
        self.server.setRequestHandler(
            ListResourceTemplatesRequestSchema,
            self.handle_list_resource_templates,
        )

        # Handle resource reads
        self.server.setRequestHandler(
            ReadResourceRequestSchema, self.handle_read_resource
        )

    async def handle_list_tools(self, request):
        """Handle listing available tools."""
        return {
            "tools": [
                {
                    "name": "search_images",
                    "description": "Search for images using Google Image Search",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for finding images",
                            },
                            "limit": {
                                "type": "number",
                                "description": "Maximum number of results to return",
                                "default": 10,
                            },
                            "image_type": {
                                "type": "string",
                                "description": "Type of image to search for",
                                "enum": [
                                    "clipart",
                                    "face",
                                    "lineart",
                                    "stock",
                                    "photo",
                                    "animated",
                                ],
                            },
                            "image_size": {
                                "type": "string",
                                "description": "Size of image to search for",
                                "enum": [
                                    "huge",
                                    "icon",
                                    "large",
                                    "medium",
                                    "small",
                                    "xlarge",
                                    "xxlarge",
                                ],
                            },
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "download_image",
                    "description": "Download an image to a local directory",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "image_url": {
                                "type": "string",
                                "description": "URL of the image to download",
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Directory path where the image should be saved",
                                "default": DOWNLOAD_DIR,
                            },
                            "filename": {
                                "type": "string",
                                "description": "Filename for the downloaded image (including extension)",
                            },
                        },
                        "required": ["image_url", "filename"],
                    },
                },
                {
                    "name": "analyze_images",
                    "description": "Analyze image search results to find the most relevant ones",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "search_results": {
                                "type": "array",
                                "description": "Array of image search results to analyze",
                                "items": {
                                    "type": "object",
                                },
                            },
                            "criteria": {
                                "type": "string",
                                "description": "Criteria for selecting the best images (e.g., 'professional', 'colorful', etc.)",
                            },
                        },
                        "required": ["search_results", "criteria"],
                    },
                },
                {
                    "name": "search_and_download",
                    "description": "Search for images and download the most relevant ones",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for finding images",
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Directory path where the images should be saved",
                                "default": DOWNLOAD_DIR,
                            },
                            "num_results": {
                                "type": "number",
                                "description": "Number of images to download",
                                "default": 5,
                            },
                            "image_type": {
                                "type": "string",
                                "description": "Type of image to search for",
                                "enum": [
                                    "clipart",
                                    "face",
                                    "lineart",
                                    "stock",
                                    "photo",
                                    "animated",
                                ],
                            },
                            "image_size": {
                                "type": "string",
                                "description": "Size of image to search for",
                                "enum": [
                                    "huge",
                                    "icon",
                                    "large",
                                    "medium",
                                    "small",
                                    "xlarge",
                                    "xxlarge",
                                ],
                            },
                        },
                        "required": ["query"],
                    },
                },
            ]
        }

    async def handle_call_tool(self, request):
        """Handle tool calls."""
        tool_name = request.params.name
        args = request.params.arguments

        try:
            if tool_name == "search_images":
                return await self.tool_search_images(args)
            elif tool_name == "download_image":
                return await self.tool_download_image(args)
            elif tool_name == "analyze_images":
                return await self.tool_analyze_images(args)
            elif tool_name == "search_and_download":
                return await self.tool_search_and_download(args)
            else:
                raise McpError(
                    ErrorCode.MethodNotFound,
                    f"Unknown tool: {tool_name}",
                )
        except Exception as e:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error executing tool {tool_name}: {str(e)}",
                    }
                ],
                "isError": True,
            }

    async def tool_search_images(self, args):
        """Tool implementation for search_images."""
        query = args.get("query")
        limit = args.get("limit", 10)
        image_type = args.get("image_type")
        image_size = args.get("image_size")

        if not query:
            raise McpError(
                ErrorCode.InvalidParams,
                "Missing required parameter: query",
            )

        results = self.search_client.search_images(
            query,
            num_results=limit,
            image_type=image_type,
            image_size=image_size,
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "success": True,
                            "results": results,
                            "count": len(results),
                            "query": query,
                        },
                        indent=2,
                    ),
                }
            ]
        }

    async def tool_download_image(self, args):
        """Tool implementation for download_image."""
        image_url = args.get("image_url")
        output_path = args.get("output_path", DOWNLOAD_DIR)
        filename = args.get("filename")

        if not image_url:
            raise McpError(
                ErrorCode.InvalidParams,
                "Missing required parameter: image_url",
            )
        if not filename:
            raise McpError(
                ErrorCode.InvalidParams,
                "Missing required parameter: filename",
            )

        path = self.search_client.download_image(
            image_url, output_path, filename
        )

        if path:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "success": True,
                                "path": path,
                                "url": image_url,
                            },
                            indent=2,
                        ),
                    }
                ]
            }
        else:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "success": False,
                                "error": "Failed to download or validate image",
                                "url": image_url,
                            },
                            indent=2,
                        ),
                    }
                ],
                "isError": True,
            }

    async def tool_analyze_images(self, args):
        """Tool implementation for analyze_images."""
        search_results = args.get("search_results", [])
        criteria = args.get("criteria", "")

        if not search_results:
            raise McpError(
                ErrorCode.InvalidParams,
                "Missing required parameter: search_results",
            )
        if not criteria:
            raise McpError(
                ErrorCode.InvalidParams,
                "Missing required parameter: criteria",
            )

        # Simple filtering based on title and snippet matching criteria
        filtered_results = []
        criteria_terms = criteria.lower().split()

        for result in search_results:
            score = 0
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()

            # Score based on criteria terms appearing in title and snippet
            for term in criteria_terms:
                if term in title:
                    score += 2
                if term in snippet:
                    score += 1

            # Add score to result
            result["relevance_score"] = score
            filtered_results.append(result)

        # Sort by relevance score
        sorted_results = sorted(
            filtered_results,
            key=lambda x: x.get("relevance_score", 0),
            reverse=True,
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "success": True,
                            "analyzed_results": sorted_results,
                            "criteria": criteria,
                            "count": len(sorted_results),
                        },
                        indent=2,
                    ),
                }
            ]
        }

    async def tool_search_and_download(self, args):
        """Tool implementation for search_and_download."""
        query = args.get("query")
        output_path = args.get("output_path", DOWNLOAD_DIR)
        num_results = args.get("num_results", 5)
        image_type = args.get("image_type")
        image_size = args.get("image_size")

        if not query:
            raise McpError(
                ErrorCode.InvalidParams,
                "Missing required parameter: query",
            )

        downloaded_paths = self.search_client.search_and_download(
            query, output_path, num_results, image_type, image_size
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "success": True,
                            "downloaded_paths": downloaded_paths,
                            "count": len(downloaded_paths),
                            "query": query,
                        },
                        indent=2,
                    ),
                }
            ]
        }

    async def handle_list_resources(self, request):
        """Handle listing available resources."""
        # We don't have any static resources for now
        return {"resources": []}

    async def handle_list_resource_templates(self, request):
        """Handle listing resource templates."""
        # We don't have any resource templates for now
        return {"resourceTemplates": []}

    async def handle_read_resource(self, request):
        """Handle reading resources."""
        # We don't have any resources for now
        raise McpError(
            ErrorCode.InvalidRequest,
            f"Invalid URI format: {request.params.uri}",
        )

    async def run(self):
        """Run the MCP server."""
        transport = StdioServerTransport()
        await self.server.connect(transport)
        print(
            "Google Image Search MCP server running on stdio", file=sys.stderr
        )


if __name__ == "__main__":
    import asyncio

    server = GoogleImageSearchMcpServer()
    asyncio.run(server.run())
