#!/bin/bash

# Stop all running containers
echo "Stopping all containers..."
docker stop $(docker ps -a -q)

# Remove all containers
echo "Removing all containers..."
docker rm -f $(docker ps -a -q)

# Remove all volumes
echo "Removing all volumes..."
docker volume rm -f $(docker volume ls -q)

# Remove all images
echo "Removing all images..."
docker rmi -f $(docker images -q)

# Clear build cache
echo "Clearing build cache..."
docker builder prune -f

echo "Cleanup completed."