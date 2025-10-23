# Distill - Competition ML Framework

An Elixir umbrella project for organizing and managing machine learning competitions (such as Kaggle) using the Nx ecosystem (Nx, Axon, Explorer, Scholar).

## Project Structure

This is an umbrella application with the following structure:

```
distill/
├── apps/
│   └── kcore/          # Core library used by all competition apps
│       ├── lib/
│       │   ├── kcore/
│       │   │   ├── data.ex      # Data loading and transformation utilities
│       │   │   ├── metrics.ex   # ML metrics (accuracy, etc.)
│       │   │   └── kcore.ex     # Main module
│       │   └── mix/
│       │       └── tasks/
│       │           └── comp.new.ex  # Mix task to scaffold new competitions
│       └── test/        # Comprehensive test suite
├── config/
│   ├── config.exs      # Shared configuration
│   ├── dev.exs         # Development config
│   ├── test.exs        # Test config (uses Nx.BinaryBackend)
│   └── prod.exs        # Production config
└── README.md
```

## Core Library (`kcore`)

The `kcore` app provides shared functionality for all competition solutions:

### `Kcore.Data`

Utilities for loading and transforming data:

- **`load_csv/2`** - Load train and test CSV files
- **`to_xy/3`** - Convert DataFrame to `{x, y}` tensor pairs for training
- **`drop_to_nx/2`** - Drop columns and convert DataFrame to tensor (for test data)
- **`write_submission!/4`** - Write predictions to CSV in Kaggle submission format

### `Kcore.Metrics`

Machine learning metrics:

- **`accuracy/2`** - Calculate classification accuracy

## Getting Started

### Prerequisites

- Elixir 1.18+
- Erlang/OTP 27+

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd distill
```

2. Install dependencies:
```bash
mix deps.get
```

3. Run tests:
```bash
mix test
```

## Creating a New Competition

Use the `mix comp.new` task to scaffold a new competition app:

```bash
mix comp.new <slug>
```

For example, to create a Titanic competition app:

```bash
mix comp.new titanic
```

This will:
- Create a new app at `apps/ktitanic/`
- Add `kcore` as a dependency
- Generate a model template in `lib/ktitanic/model.ex`
- Generate a feature engineering template in `lib/ktitanic/fe.ex`
- Create directories for data (`priv/data/`) and results (`priv/runs/`)
- Create a mix task `mix comp.titanic.train` to run your model

### Competition App Structure

```
apps/ktitanic/
├── lib/
│   ├── ktitanic/
│   │   ├── fe.ex          # Feature engineering pipeline
│   │   └── model.ex       # Model definition and training
│   └── mix/
│       └── tasks/
│           └── comp.titanic.train.ex  # Training task
└── priv/
    ├── data/              # Place train.csv and test.csv here
    └── runs/              # Model outputs (submission.csv)
```

## Workflow

1. **Create competition app**:
   ```bash
   mix comp.new titanic
   ```

2. **Add your data**:
   ```bash
   # Download competition data from Kaggle
   cp ~/Downloads/train.csv apps/ktitanic/priv/data/
   cp ~/Downloads/test.csv apps/ktitanic/priv/data/
   ```

3. **Implement feature engineering** in `apps/ktitanic/lib/ktitanic/fe.ex`:
   ```elixir
   defmodule Ktitanic.FE do
     alias Explorer.DataFrame, as: DF
     
     def pipeline(df) do
       df
       |> DF.mutate(Age: fill_missing(Age, :mean))
       |> DF.mutate(Sex: cast(Sex == "male", :integer))
       # ... more transformations
     end
   end
   ```

4. **Implement your model** in `apps/ktitanic/lib/ktitanic/model.ex`:
   ```elixir
   defmodule Ktitanic.Model do
     alias Kcore.Data
     
     @train "apps/ktitanic/priv/data/train.csv"
     @test  "apps/ktitanic/priv/data/test.csv"
     @out   "apps/ktitanic/priv/runs/submission.csv"
     
     def run do
       {train, test} = Data.load_csv!(@train, @test)
       
       # Transform data
       train = Ktitanic.FE.pipeline(train)
       test = Ktitanic.FE.pipeline(test)
       
       # Prepare tensors
       {x_train, y_train} = Data.to_xy(train, "Survived", drop: ["PassengerId"])
       x_test = Data.drop_to_nx(test, ["PassengerId"])
       ids = Explorer.DataFrame.pull(test, "PassengerId")
       
       # Train model
       model = build_model()
       trained_model = Axon.Loop.trainer(model, :binary_cross_entropy, :adam)
         |> Axon.Loop.run(x_train, y_train, epochs: 10)
       
       # Make predictions
       predictions = Axon.predict(trained_model, x_test)
       
       # Write submission
       Data.write_submission!(@out, ["PassengerId", "Survived"], ids, predictions)
     end
     
     defp build_model() do
       Axon.input("input")
       |> Axon.dense(64, activation: :relu)
       |> Axon.dense(32, activation: :relu)
       |> Axon.dense(1, activation: :sigmoid)
     end
   end
   ```

5. **Train and generate submission**:
   ```bash
   mix comp.titanic.train
   ```

6. **Submit to Kaggle**:
   ```bash
   6. **Submit to Kaggle**:
   ```bash
   # Your submission file is at apps/ktitanic/priv/runs/submission.csv
   ```

## Dependencies
   ```

## Dependencies

The framework uses the Nx ecosystem:

- **[Nx](https://github.com/elixir-nx/nx)** (~> 0.10.0) - Numerical computing
- **[Explorer](https://github.com/elixir-nx/explorer)** (~> 0.10.1) - DataFrames for data manipulation
- **[Axon](https://github.com/elixir-nx/axon)** (~> 0.7.0) - Neural networks
- **[Scholar](https://github.com/elixir-nx/scholar)** (~> 0.4.0) - Machine learning algorithms
- **[EXLA](https://github.com/elixir-nx/nx/tree/main/exla)** (optional) - XLA compiler backend
- **[Torchx](https://github.com/elixir-nx/nx/tree/main/torchx)** (optional) - LibTorch backend

## Configuration

The Nx backend is configured in `config/config.exs`:

```elixir
# Choose your backend (default: Torchx.Backend)
config :nx, default_backend: Torchx.Backend

# Alternative backends:
# config :nx, default_backend: EXLA.Backend
# config :nx, default_backend: Nx.BinaryBackend
```

Tests use `Nx.BinaryBackend` for simplicity (configured in `config/test.exs`).

## Running Tests

```bash
# Run all tests
mix test

# Run tests for a specific app
cd apps/kcore && mix test

# Run with detailed output
mix test --trace
```

## Development

### Adding New Utilities

Add shared utilities to the `kcore` app:

1. Create your module in `apps/kcore/lib/kcore/`
2. Add tests in `apps/kcore/test/kcore/`
3. Run tests: `cd apps/kcore && mix test`

### Adding New Metrics

Add metrics to `apps/kcore/lib/kcore/metrics.ex`:

```elixir
defmodule Kcore.Metrics do
  def accuracy(y_true, y_pred) do
    # ... existing implementation
  end
  
  def f1_score(y_true, y_pred) do
    # Your new metric
  end
end
```

## Tips

- Keep feature engineering in the `FE` module for reusability
- Use `Explorer.DataFrame` for data manipulation before converting to tensors
- Test your pipeline with small data samples first
- Use `mix comp.new` for consistent project structure
- Add custom metrics to `Kcore.Metrics` for reuse across competitions

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
