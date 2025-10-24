defmodule Mix.Tasks.Comp.Titanic.Train do
  use Mix.Task
  @shortdoc "Train titanic and write submission.csv"
  def run(_args) do
    Mix.Task.run("app.start")
    Dtitanic.Model.run()
  end
end
