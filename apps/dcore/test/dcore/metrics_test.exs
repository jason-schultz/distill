defmodule Dcore.MetricsTest do
  use ExUnit.Case, async: true
  alias Dcore.Metrics

  describe "accuracy/2" do
    test "calculates perfect accuracy" do
      y_true = Nx.tensor([0, 1, 0, 1, 0])
      y_pred = Nx.tensor([0, 1, 0, 1, 0])

      assert Metrics.accuracy(y_true, y_pred) == 1.0
    end

    test "calculates zero accuracy" do
      y_true = Nx.tensor([0, 1, 0, 1, 0])
      y_pred = Nx.tensor([1, 0, 1, 0, 1])

      assert Metrics.accuracy(y_true, y_pred) == 0.0
    end

    test "calculates partial accuracy" do
      y_true = Nx.tensor([0, 1, 0, 1, 0])
      y_pred = Nx.tensor([0, 1, 1, 1, 0])

      # 4 out of 5 correct
      assert Metrics.accuracy(y_true, y_pred) == 0.8
    end

    test "works with float inputs" do
      y_true = Nx.tensor([0.0, 1.0, 0.0, 1.0])
      y_pred = Nx.tensor([0.0, 1.0, 1.0, 1.0])

      # 3 out of 4 correct
      assert Metrics.accuracy(y_true, y_pred) == 0.75
    end

    test "works with single prediction" do
      y_true = Nx.tensor([1])
      y_pred = Nx.tensor([1])

      assert Metrics.accuracy(y_true, y_pred) == 1.0
    end

    test "handles large tensors" do
      y_true = Nx.tensor(List.duplicate(1, 1000) ++ List.duplicate(0, 1000))
      y_pred = Nx.tensor(List.duplicate(1, 1000) ++ List.duplicate(0, 1000))

      assert Metrics.accuracy(y_true, y_pred) == 1.0
    end

    test "calculates accuracy for multi-class" do
      y_true = Nx.tensor([0, 1, 2, 3, 4, 0, 1, 2])
      y_pred = Nx.tensor([0, 1, 2, 3, 0, 0, 1, 1])

      # 6 out of 8 correct (indices 0,1,2,3,5,6 match)
      assert Metrics.accuracy(y_true, y_pred) == 0.75
    end
  end
end
