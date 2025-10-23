defmodule Dcore.Data do
  alias Explorer.DataFrame, as: DF
  alias Explorer.Series

  NimbleCSV.define(KCSV, separator: ",", escape: "\"")

  def load_csv(train_path, test_path) do
    {DF.from_csv!(train_path), DF.from_csv!(test_path)}
  end

  # Convert a DF to {x, y} Nx tensors for label column `label`,
  # dropping any columns listed in :drop
  def to_xy(df, label, opts \\ []) do
    drop = Keyword.get(opts, :drop, [])

    x =
      df
      |> DF.discard(Enum.uniq([label | drop]))
      |> df_to_tensor()

    y =
      df
      |> DF.pull(label)
      |> Series.to_tensor()
      |> Nx.as_type(:f32)

    {x, y}
  end

  # Drop columns and produce Nx
  def drop_to_nx(df, drops) do
    df
    |> DF.discard(drops)
    |> df_to_tensor()
  end

  # Helper to convert DataFrame to Nx tensor
  defp df_to_tensor(df) do
    df
    |> DF.names()
    |> Enum.map(fn name ->
      df
      |> DF.pull(name)
      |> Series.to_tensor()
    end)
    |> Nx.stack(axis: 1)
    |> Nx.as_type(:f32)
  end

  def write_submission!(out_path, headers, ids_series, preds_tensor) do
    ids = ids_series |> Series.to_list()
    preds = preds_tensor |> Nx.to_flat_list() |> Enum.map(&round/1)

    rows = Enum.zip(ids, preds) |> Enum.map(fn {a, b} -> [a, b] end)
    csv = [headers | rows] |> KCSV.dump_to_iodata() |> IO.iodata_to_binary()
    File.mkdir_p!(Path.dirname(out_path))
    File.write!(out_path, csv)
    out_path
  end
end
