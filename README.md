# Distill - Competition ML Framework

An Elixir umbrella project for organizing and managing machine learning competitions (such as Kaggle) using the Nx ecosystem (Nx, Axon, Explorer, Scholar).

## Project Structure

This is an umbrella application with the following structure:

```
distill/
├── apps/
│   ├── dcore/          # Core library used by all competition apps
│   │   ├── lib/
│   │   │   ├── dcore/
│   │   │   │   ├── data.ex      # Data loading and transformation utilities
│   │   │   │   ├── metrics.ex   # ML metrics (accuracy, etc.)
│   │   │   │   └── dcore.ex     # Main module
│   │   │   └── mix/
│   │   │       └── tasks/
│   │   │           └── comp.new.ex  # Mix task to scaffold new competitions
│   │   └── test/        # Comprehensive test suite
│   └── d<competition>/  # Generated competition apps (e.g., dtitanic)
│       ├── lib/
│       │   ├── d<competition>/
│       │   │   ├── fe.ex        # Feature engineering pipeline
│       │   │   └── model.ex     # Model definition and training
│       │   └── mix/
│       │       └── tasks/
│       │           └── comp.<competition>.train.ex
│       └── priv/
│           ├── data/            # Place train.csv and test.csv here
│           └── runs/            # Model outputs (submission.csv)
├── config/
│   ├── config.exs      # Shared configuration
│   ├── dev.exs         # Development config
│   ├── test.exs        # Test config (uses Nx.BinaryBackend)
│   └── prod.exs        # Production config
└── README.md
```

## Core Library (`dcore`)

The `dcore` app provides shared functionality for all competition solutions:

### `Dcore.Data`

Utilities for loading and transforming data:

- **`load_csv/2`** - Load train and test CSV files
- **`to_xy/3`** - Convert DataFrame to `{x, y}` tensor pairs for training
- **`drop_to_nx/2`** - Drop columns and convert DataFrame to tensor (for test data)
- **`write_submission!/4`** - Write predictions to CSV in Kaggle submission format

### `Dcore.Metrics`

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
- Create a new app at `apps/dtitanic/`
- Add `dcore` as a dependency
- Generate a model template in `lib/dtitanic/model.ex`
- Generate a feature engineering template in `lib/dtitanic/fe.ex`
- Create directories for data (`priv/data/`) and results (`priv/runs/`)
- Create mix tasks:
  - `mix comp.titanic.validate` - Validate model with train/val split
  - `mix comp.titanic.train` - Train and generate submission.csv

### Competition App Structure

```
apps/dtitanic/
├── lib/
│   ├── dtitanic/
│   │   ├── fe.ex          # Feature engineering pipeline
│   │   └── model.ex       # Model definition and training
│   └── mix/
│       └── tasks/
│           └── comp.titanic.train.ex  # Training task
│           └── comp.titanic.validate.ex  # Validation task
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
   cp ~/Downloads/train.csv apps/dtitanic/priv/data/
   cp ~/Downloads/test.csv apps/dtitanic/priv/data/
   ```

3. **Implement feature engineering** in `apps/dtitanic/lib/dtitanic/fe.ex`:
   ```elixir
   defmodule Dtitanic.FE do
     require Explorer.DataFrame
     alias Explorer.DataFrame, as: DF
     alias Explorer.Series
     
     def pipeline(df) do
       df
       # Convert Sex to binary (male = 1, female = 0)
       |> DF.mutate(Sex: cast(col("Sex") == "male", :integer))
       # Fill missing Age with median
       |> fill_age()
       # Fill missing Fare with median (for test set)
       |> fill_fare()
       # Fill missing Embarked with mode (most common)
       |> fill_embarked()
       # Convert Embarked to numeric (one-hot encoding)
       |> DF.mutate(Embarked_C: cast(col("Embarked") == "C", :integer))
       |> DF.mutate(Embarked_Q: cast(col("Embarked") == "Q", :integer))
       |> DF.mutate(Embarked_S: cast(col("Embarked") == "S", :integer))
       # Drop columns we won't use
       |> DF.discard(["Name", "Ticket", "Cabin", "Embarked"])
     end
     
     defp fill_age(df) do
       age_series = DF.pull(df, "Age")
       median_age = age_series |> Series.median() |> then(fn x -> x || 28.0 end)
       filled_age = Series.fill_missing(age_series, median_age)
       DF.put(df, "Age", filled_age)
     end
     
     defp fill_fare(df) do
       fare_series = DF.pull(df, "Fare")
       median_fare = fare_series |> Series.median() |> then(fn x -> x || 14.5 end)
       filled_fare = Series.fill_missing(fare_series, median_fare)
       DF.put(df, "Fare", filled_fare)
     end
     
     defp fill_embarked(df) do
       embarked_series = DF.pull(df, "Embarked")
       filled_embarked = Series.fill_missing(embarked_series, "S")
       DF.put(df, "Embarked", filled_embarked)
     end
   end
   ```

4. **Implement your model** in `apps/dtitanic/lib/dtitanic/model.ex`:
   ```elixir
   defmodule Dtitanic.Model do
     alias Dcore.Data
     
     @train "apps/dtitanic/priv/data/train.csv"
     @test  "apps/dtitanic/priv/data/test.csv"
     @out   "apps/dtitanic/priv/runs/submission.csv"
     
     def run do
       IO.puts("Loading data...")
       {train, test} = Data.load_csv(@train, @test)
       
       IO.puts("Applying feature engineering...")
       train = Dtitanic.FE.pipeline(train)
       test = Dtitanic.FE.pipeline(test)
       
       IO.puts("Preparing tensors...")
       {x_train, y_train} = Data.to_xy(train, "Survived", drop: ["PassengerId"])
       x_test = Data.drop_to_nx(test, ["PassengerId"])
       test_ids = Explorer.DataFrame.pull(test, "PassengerId")
       
       IO.puts("Building model...")
       model = build_model(x_train)
       
       IO.puts("Training model...")
       trained_state = train_model(model, x_train, y_train)
       
       IO.puts("Making predictions...")
       predictions = Axon.predict(model, trained_state, x_test)
         |> Nx.squeeze()
         |> Nx.round()
       
       IO.puts("Writing submission...")
       Data.write_submission!(@out, ["PassengerId", "Survived"], test_ids, predictions)
       
       IO.puts("Done! Submission written to #{@out}")
     end
     
     defp build_model(x_train) do
       {_rows, n_features} = Nx.shape(x_train)
       
       Axon.input("input", shape: {nil, n_features})
       |> Axon.dense(64, activation: :relu)
       |> Axon.dropout(rate: 0.3)
       |> Axon.dense(32, activation: :relu)
       |> Axon.dropout(rate: 0.3)
       |> Axon.dense(1, activation: :sigmoid)
     end
     
     defp train_model(model, x_train, y_train) do
       y_train = Nx.reshape(y_train, {:auto, 1})
       train_data = [{x_train, y_train}]
       
       model
       |> Axon.Loop.trainer(:binary_cross_entropy, :adam)
       |> Axon.Loop.metric(:accuracy)
       |> Axon.Loop.run(train_data, %{}, epochs: 20, compiler: EXLA)
     end
   end
   ```

5. **Validate your model locally** (optional but recommended):
   ```bash
   mix comp.titanic.validate
   ```
   
   This will:
   - Split your training data into 80% train / 20% validation
   - Train the model on the training portion
   - Evaluate accuracy on the validation portion
   - Give you a local estimate before submitting to Kaggle
   
   Example output:
   ```
   Train size: 712, Val size: 179
   =================================
   Validation Accuracy: 73.03%
   =================================
   ```

6. **Train and generate submission**:
   ```bash
   mix comp.titanic.train
   ```

7. **Submit to Kaggle**:
   ```bash
   6. **Submit to Kaggle**:
   ```bash
   # Your submission file is at apps/dtitanic/priv/runs/submission.csv
   ```

## Dependencies
   ```

## Dependencies

The framework uses the Nx ecosystem:

- **[Nx](https://github.com/elixir-nx/nx)** (~> 0.10.0) - Numerical computing
- **[Explorer](https://github.com/elixir-nx/explorer)** (~> 0.10.1) - DataFrames for data manipulation
- **[Axon](https://github.com/elixir-nx/axon)** (~> 0.7.0) - Neural networks
- **[Scholar](https://github.com/elixir-nx/scholar)** (~> 0.4.0) - Machine learning algorithms
- **[EXLA](https://github.com/elixir-nx/nx/tree/main/exla)** (~> 0.10.0) - XLA compiler backend (recommended)
- **[Torchx](https://github.com/elixir-nx/nx/tree/main/torchx)** (optional) - LibTorch backend

## Configuration

The Nx backend is configured in `config/config.exs`:

```elixir
# Default backend: EXLA (recommended for performance)
config :nx, default_backend: EXLA.Backend

# Alternative backends:
# config :nx, default_backend: Torchx.Backend
# config :nx, default_backend: Nx.BinaryBackend
```

Tests use `Nx.BinaryBackend` for simplicity (configured in `config/test.exs`).

## Running Tests

```bash
# Run all tests
mix test

# Run tests for a specific app
cd apps/dcore && mix test

# Run with detailed output
mix test --trace
```

## Development

### Adding New Utilities

Add shared utilities to the `dcore` app:

1. Create your module in `apps/dcore/lib/dcore/`
2. Add tests in `apps/dcore/test/dcore/`
3. Run tests: `cd apps/dcore && mix test`

### Adding New Metrics

Add metrics to `apps/dcore/lib/dcore/metrics.ex`:

```elixir
defmodule Dcore.Metrics do
  def accuracy(y_true, y_pred) do
    # ... existing implementation
  end
  
  def f1_score(y_true, y_pred) do
    # Your new metric
  end
end
```

## Tips

- **Use validation before submission**: Run `mix comp.<slug>.validate` to get a local accuracy estimate before submitting to Kaggle
- Keep feature engineering in the `FE` module for reusability
- Use `Explorer.DataFrame` for data manipulation before converting to tensors
- Use `col("ColumnName")` in `DF.mutate` to reference columns with uppercase names
- Pull series out and use `Series.fill_missing` to handle missing values, then `DF.put` back
- Test your pipeline with small data samples first
- Use `mix comp.new` for consistent project structure
- Add custom metrics to `Dcore.Metrics` for reuse across competitions
- Use `%{}` as initial state in `Axon.Loop.run` for proper model initialization
- Set `compiler: EXLA` in training loop for best performance
- Validation accuracy is typically optimistic - expect test accuracy to be 5-10% lower due to distribution shift

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
