# GNNs

This subdirectory is dedicated to Graph Neural Network (GNN) research, experiments, examples, and utilities.

## Structure

- `data/` (optional): input graphs, preprocess outputs.
- `notebooks/`: Jupyter notebooks for prototyping and visualization.
- `src/`: implementation code (models, layers, training loops).
- `docs/` (optional): notes, design docs.
- `tests/`: unit/integration tests.

## Contents

- `README.md` (this file)
- `example`: sample graph, pipeline, or tutorial.
- `models`: GCN, GAT, GraphSAGE, etc.
- `utils`: graph loading, transforms, metrics.

## Goals

- collect and organize GNN experiments
- reproduce papers on node classification, graph classification, link prediction
- provide clear training / evaluation scripts

## Quick start

1. clone repo
2. install dependencies (`requirements.txt`)
3. run notebook or script in `src/`
4. adapt dataset and model

## Notes

- keep graph data separate from code
- follow consistent API for model/dataset
- document key experiments

> Graph Neural Networks are about learning on graph-structured data and making results reproducible.