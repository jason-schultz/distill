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
