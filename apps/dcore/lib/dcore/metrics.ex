defmodule Dcore.Metrics do
  def accuracy(y_true, y_pred) do
    y_true = Nx.as_type(y_true, :u8)
    y_pred = Nx.as_type(y_pred, :u8)
    correct = Nx.equal(y_true, y_pred) |> Nx.sum()
    total = Nx.size(y_true)
    Nx.to_number(correct) / total
  end
end
