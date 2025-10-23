defmodule Dcore.MixProject do
  use Mix.Project

  def project do
    [
      app: :dcore,
      version: "0.1.0",
      build_path: "../../_build",
      config_path: "../../config/config.exs",
      deps_path: "../../deps",
      lockfile: "../../mix.lock",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"},
      # {:sibling_app_in_umbrella, in_umbrella: true}
      {:nx, "~> 0.10.0"},
      {:explorer, "~> 0.10.1"},
      {:scholar, "~> 0.4.0"},
      {:axon, "~> 0.7.0"},
      {:exla, "~> 0.10.0", optional: true},
      {:torchx, "~> 0.10.1", optional: true},
      {:vega_lite, "~> 0.1.11"},
      {:nimble_csv, "~> 1.3.0"},
      {:req, "~> 0.5.15"}
    ]
  end
end
