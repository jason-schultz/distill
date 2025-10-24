defmodule Dtitanic.Model do
  alias Dcore.Data
  @train "apps/dtitanic/priv/data/train.csv"
  @test "apps/dtitanic/priv/data/test.csv"
  @out "apps/dtitanic/priv/runs/submission.csv"

  def run do
    IO.puts("Loading data ...")
    {train, test} = Data.load_csv(@train, @test)

    IO.puts("Applying feature engineering ...")
    train = Dtitanic.FE.pipeline(train)
    test = Dtitanic.FE.pipeline(test)

    IO.puts("Preparing tensors ...")
    {x_train, y_train} = Data.to_xy(train, "Survived", drop: ["PassengerId"])
    x_test = Data.drop_to_nx(test, ["PassengerId"])
    test_ids = Explorer.DataFrame.pull(test, "PassengerId")

    IO.puts("Building Model ...")
    model = build_model(x_train)

    IO.puts("Training Model ...")
    trained_state = train_model(model, x_train, y_train)

    IO.puts("Making predictions ...")

    predictions =
      Axon.predict(model, trained_state, x_test)
      |> Nx.squeeze()
      |> Nx.round()

    IO.puts("Writing submission ...")
    Data.write_submission!(@out, ["PassengerId", "Survived"], test_ids, predictions)

    IO.puts("Done! Submission written to #{@out}")
  end

  def validate(val_split \\ 0.2) do
    IO.puts("Loading data ...")
    {train, _test} = Data.load_csv(@train, @test)

    IO.puts("Applying feature engineering ...")
    train = Dtitanic.FE.pipeline(train)

    IO.puts("Preparing tensors ...")
    {x, y} = Data.to_xy(train, "Survived", drop: ["PassengerId"])

    IO.puts(
      "Splitting into train/validation (#{(1 - val_split) * 100}% / #{val_split * 100}%) ..."
    )

    {{x_train, y_train}, {x_val, y_val}} = split_train_val(x, y, val_split)

    IO.puts("Train size: #{elem(Nx.shape(x_train), 0)}, Val size: #{elem(Nx.shape(x_val), 0)}")

    IO.puts("Building Model ...")
    model = build_model(x_train)

    IO.puts("Training Model ...")
    trained_state = train_model(model, x_train, y_train)

    IO.puts("Evaluating on validation set ...")

    # Get predictions
    predictions =
      Axon.predict(model, trained_state, x_val)
      |> Nx.squeeze()
      |> Nx.round()

    # Calculate accuracy
    accuracy = Dcore.Metrics.accuracy(y_val, predictions)

    IO.puts("\n=================================")
    IO.puts("Validation Accuracy: #{Float.round(accuracy * 100, 2)}%")
    IO.puts("=================================\n")

    # Show some sample predictions vs actual
    y_val_list = Nx.to_flat_list(y_val) |> Enum.take(10)
    pred_list = Nx.to_flat_list(predictions) |> Enum.take(10)

    IO.puts("Sample predictions (first 10):")
    IO.puts("Actual:    #{inspect(y_val_list)}")
    IO.puts("Predicted: #{inspect(pred_list)}")

    accuracy
  end

  defp split_train_val(x, y, val_split) do
    n = elem(Nx.shape(x), 0)
    val_size = floor(n * val_split)
    train_size = n - val_size

    # Split features
    x_train = Nx.slice_along_axis(x, 0, train_size, axis: 0)
    x_val = Nx.slice_along_axis(x, train_size, val_size, axis: 0)

    # Split labels
    y_train = Nx.slice_along_axis(y, 0, train_size, axis: 0)
    y_val = Nx.slice_along_axis(y, train_size, val_size, axis: 0)

    {{x_train, y_train}, {x_val, y_val}}
  end

  defp build_model(x_train) do
    # Get number of reatures from training data
    {_rows, n_features} = Nx.shape(x_train)

    Axon.input("input", shape: {nil, n_features})
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(rate: 0.3)
    |> Axon.dense(32, activation: :relu)
    |> Axon.dropout(rate: 0.3)
    |> Axon.dense(1, activation: :sigmoid)
  end

  defp train_model(model, x_train, y_train) do
    # Reshape y to match output
    y_train = Nx.reshape(y_train, {:auto, 1})

    train_data = [{x_train, y_train}]

    model
    |> Axon.Loop.trainer(:binary_cross_entropy, :adam)
    |> Axon.Loop.metric(:accuracy)
    |> Axon.Loop.run(train_data, %{}, epochs: 20, compiler: EXLA)
  end
end
