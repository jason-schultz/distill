defmodule Mix.Tasks.Comp.Titanic.Validate do
  use Mix.Task
  @shortdoc "Validate titanic model with train/val split"

  def run(_args) do
    Mix.Task.run("app.start")
    Dtitanic.Model.validate()
  end
end
