defmodule Mix.Tasks.Comp.New do
  use Mix.Task
  @shortdoc "Scaffold a new competition app under apps/"
  @moduledoc "mix comp.new <slug> (eg. mix comp.new titanic)"

  def run([slug]) do
    app = "d" <> slug
    app_path = Path.join(["apps", app])

    unless File.exists?(app_path) do
      {_, 0} = System.cmd("mix", ["new", app], cd: "apps", into: IO.stream(:stdio, :line))
    end

    mix_path = Path.join([app_path, "mix.exs"])
    mix = File.read!(mix_path)

    mix =
      String.replace(
        mix,
        "      # {:sibling_app_in_umbrella, in_umbrella: true}\n    ]",
        "      # {:sibling_app_in_umbrella, in_umbrella: true}\n      {:dcore, in_umbrella: true}\n    ]"
      )

    File.write!(mix_path, mix)

    # dirs
    File.mkdir_p!(Path.join([app_path, "priv/data"]))
    File.mkdir_p!(Path.join([app_path, "priv/runs"]))
    File.mkdir_p!(Path.join([app_path, "lib", app]))
    File.mkdir_p!(Path.join([app_path, "lib/mix/tasks"]))

    # FE file
    fe_path = Path.join([app_path, "lib/#{app}/fe.ex"])

    File.write!(fe_path, """
    defmodule #{Macro.camelize(app)}.FE do
      alias Explorer.DataFrame, as: DF
      def pipeline(df), do: df
    end
    """)

    # Model file
    model_path = Path.join([app_path, "lib/#{app}/model.ex"])

    model_code = """
    defmodule #{Macro.camelize(app)}.Model do
      alias Dcore.Data
      @train "apps/#{app}/priv/data/train.csv"
      @test  "apps/#{app}/priv/data/test.csv"
      @out   "apps/#{app}/priv/runs/submission.csv"

      def run do
        {train, test} = Data.load_csv(@train, @test)
        train = #{Macro.camelize(app)}.FE.pipeline(train)
        test = #{Macro.camelize(app)}.FE.pipeline(test)

        {x_train, y_train} = Data.to_xy(train, "Target", drop: ["Id"])
        x_test = Data.drop_to_nx(test, ["Id"])
        test_ids = Explorer.DataFrame.pull(test, "Id")

        model = build_model(x_train)
        trained_state = train_model(model, x_train, y_train)

        predictions = Axon.predict(model, trained_state, x_test)
          |> Nx.squeeze()
          |> Nx.round()

        Data.write_submission!(@out, ["Id", "Pred"], test_ids, predictions)
        IO.puts("Submission written to \#{@out}")
      end

      def validate(val_split \\\\ 0.2) do
        {train, _test} = Data.load_csv(@train, @test)
        train = #{Macro.camelize(app)}.FE.pipeline(train)

        {x, y} = Data.to_xy(train, "Target", drop: ["Id"])
        {{x_train, y_train}, {x_val, y_val}} = split_train_val(x, y, val_split)

        model = build_model(x_train)
        trained_state = train_model(model, x_train, y_train)

        predictions = Axon.predict(model, trained_state, x_val)
          |> Nx.squeeze()
          |> Nx.round()

        acc = Dcore.Metrics.accuracy(y_val, predictions)
        IO.puts("Validation Accuracy: \#{Float.round(acc * 100, 2)}%")
        acc
      end

      defp split_train_val(x, y, val_split) do
        n = elem(Nx.shape(x), 0)
        val_size = floor(n * val_split)
        train_size = n - val_size

        x_train = Nx.slice_along_axis(x, 0, train_size, axis: 0)
        x_val = Nx.slice_along_axis(x, train_size, val_size, axis: 0)
        y_train = Nx.slice_along_axis(y, 0, train_size, axis: 0)
        y_val = Nx.slice_along_axis(y, train_size, val_size, axis: 0)

        {{x_train, y_train}, {x_val, y_val}}
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
    """

    File.write!(model_path, model_code)

    # Task file - Train
    task_path = Path.join([app_path, "lib/mix/tasks/comp.#{slug}.train.ex"])

    File.write!(task_path, """
    defmodule Mix.Tasks.Comp.#{Macro.camelize(slug)}.Train do
      use Mix.Task
      @shortdoc "Train #{slug} and write submission.csv"
      def run(_args) do
        Mix.Task.run("app.start")
        #{Macro.camelize("d" <> slug)}.Model.run()
      end
    end
    """)

    # Task file - Validate
    validate_path = Path.join([app_path, "lib/mix/tasks/comp.#{slug}.validate.ex"])

    File.write!(validate_path, """
    defmodule Mix.Tasks.Comp.#{Macro.camelize(slug)}.Validate do
      use Mix.Task
      @shortdoc "Validate #{slug} model with train/val split"
      def run(_args) do
        Mix.Task.run("app.start")
        #{Macro.camelize("d" <> slug)}.Model.validate()
      end
    end
    """)

    Mix.shell().info("Scaffolded #{app} under apps/#{app}")

    Mix.shell().info(
      "Add FE + model logic, drop CSVs into priv/data, then run: mix comp.#{slug}.validate or mix comp.#{slug}.train"
    )
  end

  def run(_), do: Mix.raise("Usage: mix comp.new <slug>")
end
