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

    File.write!(model_path, """
    defmodule #{Macro.camelize(app)}.Model do
      alias Dcore.Data
      @train "apps/#{app}/priv/data/train.csv"
      @test  "apps/#{app}/priv/data/test.csv"
      @out   "apps/#{app}/priv/runs/submission.csv"
      def run do
        {train, test} = Data.load_csv(@train, @test)
        # TODO: transform with #{Macro.camelize(app)}.FE.pipeline/1 and train a model
        Data.write_submission!(@out, ["Id","Pred"], Explorer.Series.from_list([]), Nx.tensor([]))
      end
    end
    """)

    # Task file
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

    Mix.shell().info("Scaffolded #{app} under apps/#{app}")

    Mix.shell().info(
      "Add FE + model logic, drop CSVs into priv/data, then run: mix comp.#{slug}.train"
    )
  end

  def run(_), do: Mix.raise("Usage: mix comp.new <slug>")
end
