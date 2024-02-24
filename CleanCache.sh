#!/bin/bash
# CleanCache.sh

GREEN='\033[0;32m'
RED='\033[0;31m'
RESET='\033[0m'

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYCACHE_PATTERN="__pycache__"

read -p "Are you sure you want to clean the project cache? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo -e "${RED}Cache cleaning aborted.${RESET}"
    exit 1
fi

clean_cache() {
  for item in $(find "$PROJECT_DIR" -name "$PYCACHE_PATTERN" -o -name "*.pyc")
  do
    echo -e "${RED}Removing: $item${RESET}"
    rm -rf "$item"
  done
}

clean_cache

echo -e "${GREEN}Cache cleaned successfully!${RESET}"