#!/bin/bash

# Setup script for MCP Tool Generator environment

echo "Setting up MCP Tool Generator environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Create .env file from template
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file. Please update with your configuration."
fi

# Create necessary directories (in case they don't exist)
mkdir -p tools/{draft,staged,active,sandbox/{temp_code,temp_data,logs}}
mkdir -p logs
mkdir -p data/sample_datasets

echo "Setup complete!"
echo "Activate the environment with: source venv/bin/activate"
echo "Update .env file with your LLM endpoint configuration"
