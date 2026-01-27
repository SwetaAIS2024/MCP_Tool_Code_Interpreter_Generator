# Docker Sandbox Setup

## Overview
This directory contains Docker configuration for running untrusted code in an isolated sandbox environment.

## Files
- `Dockerfile.sandbox`: Base image for sandbox execution
- `docker-compose.sandbox.yml`: Resource limits and security policies

## Security Features
- No network access (`network_mode: none`)
- Memory limit: 512MB
- CPU limit: 0.5 cores
- Read-only filesystem (except /tmp)
- Non-root user execution
- Dropped capabilities

## Usage

### Build the image:
```bash
docker-compose -f docker/docker-compose.sandbox.yml build
```

### Run sandbox:
```bash
docker-compose -f docker/docker-compose.sandbox.yml up
```

### Clean up:
```bash
docker-compose -f docker/docker-compose.sandbox.yml down
```

## Configuration
Edit `config/sandbox_policy.yaml` to adjust security policies.
