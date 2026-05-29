#!/bin/bash

################################################################################
# Standalone Comparison Plots Generator
# Run this script to generate comparison plots from existing report files
################################################################################

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Base directory
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}                   COMPARISON PLOTS GENERATION${NC}"
echo -e "${BLUE}================================================================================${NC}\n"

cd "${EVAL_DIR}"

# Generate plots for output_1
echo -e "${GREEN}Creating comparison plots for output_1...${NC}"
python create_comparison_plots.py \
    --results_dir "${EVAL_DIR}/output1/results" \
    --output_key output_1

echo ""

# Generate plots for output_2
echo -e "${GREEN}Creating comparison plots for output_2...${NC}"
python create_comparison_plots.py \
    --results_dir "${EVAL_DIR}/output2/results" \
    --output_key output_2

echo -e "\n${BLUE}================================================================================${NC}"
echo -e "${GREEN}✓ Plots generation complete!${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo -e "Plots saved to:"
echo -e "  - ${EVAL_DIR}/output1/results/plots/"
echo -e "  - ${EVAL_DIR}/output2/results/plots/"
echo -e "${BLUE}================================================================================${NC}\n"
