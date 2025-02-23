# Code Architecture

```mermaid
graph TD
    subgraph CLI
        cli[cli.py] --> train[Train Command]
        cli --> generate[Generate Command]
    end

    subgraph Core
        model[model.py<br/>MinimalTransformer] --> dataset[dataset.py<br/>Base64Dataset]
        model --> utils[utils.py<br/>Generation Utils]
    end

    subgraph Examples
        test_cycling[test_cycling.py] --> model
        inference[inference.py] --> model
        train_example[train.py] --> model
    end

    subgraph Testing
        test_model[test_model.py] --> model
        test_dataset[test_dataset.py] --> dataset
        test_utils[test_utils.py] --> utils
    end

    train --> model
    generate --> model
    dataset --> utils
```

The codebase is organized into several key components:

1. **Core Implementation** (`fluctlight/`)
   - `model.py`: MinimalTransformer implementation with RoPE
   - `dataset.py`: Data handling and Base64 dataset
   - `utils.py`: Text generation and encoding utilities
   - `cli.py`: Command-line interface

2. **Examples** (`examples/`)
   - Interactive text generation demos
   - Training scripts
   - Inference examples

3. **Tests** (`tests/`)
   - Comprehensive test suite for all components
   - Integration tests for the full pipeline

The architecture emphasizes modularity and clear separation of concerns, making it easy to extend or modify individual components.
