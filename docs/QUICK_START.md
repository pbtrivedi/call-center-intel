# Quick Start

## Prerequisites
- Python 3.11+
- ffmpeg (`brew install ffmpeg` on macOS)

## Setup

```bash
# Clone and enter repo
git clone https://github.com/pbtrivedi/call-center-intel.git
cd call-center-intel

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
make install

# Configure environment
cp .env.example .env
# Edit .env — set LLM_PROVIDER and the corresponding API key
```

## Run

```bash
make run
# Open http://localhost:7860
```

## Test

```bash
make test          # unit tests only
make test-all      # unit + integration + security
```

## Docker

```bash
docker compose up --build
```
