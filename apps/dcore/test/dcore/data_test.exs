defmodule Dcore.DataTest do
  use ExUnit.Case, async: true
  alias Dcore.Data
  alias Explorer.DataFrame, as: DF
  alias Explorer.Series

  @moduletag :tmp_dir

  describe "load_csv/2" do
    test "loads train and test CSV files", %{tmp_dir: tmp_dir} do
      train_path = Path.join(tmp_dir, "train.csv")
      test_path = Path.join(tmp_dir, "test.csv")

      # Create sample CSV files
      File.write!(train_path, """
      Id,Feature1,Feature2,Label
      1,1.5,2.5,0
      2,3.5,4.5,1
      3,5.5,6.5,0
      """)

      File.write!(test_path, """
      Id,Feature1,Feature2
      4,7.5,8.5
      5,9.5,10.5
      """)

      {train, test} = Data.load_csv(train_path, test_path)

      assert DF.names(train) == ["Id", "Feature1", "Feature2", "Label"]
      assert DF.names(test) == ["Id", "Feature1", "Feature2"]
      assert DF.n_rows(train) == 3
      assert DF.n_rows(test) == 2
    end
  end

  describe "to_xy/3" do
    setup %{tmp_dir: tmp_dir} do
      csv_path = Path.join(tmp_dir, "data.csv")

      File.write!(csv_path, """
      Id,Feature1,Feature2,Feature3,Label
      1,1.0,2.0,3.0,0
      2,4.0,5.0,6.0,1
      3,7.0,8.0,9.0,0
      4,10.0,11.0,12.0,1
      """)

      df = DF.from_csv!(csv_path)
      %{df: df}
    end

    test "converts DataFrame to {x, y} tensors", %{df: df} do
      {x, y} = Data.to_xy(df, "Label", drop: ["Id"])

      # x should be shape {4, 3} (4 rows, 3 features)
      assert Nx.shape(x) == {4, 3}
      assert Nx.type(x) == {:f, 32}

      # y should be shape {4} (4 labels)
      assert Nx.shape(y) == {4}
      assert Nx.type(y) == {:f, 32}

      # Check some values
      assert Nx.to_flat_list(y) == [0.0, 1.0, 0.0, 1.0]
    end

    test "drops specified columns from features", %{df: df} do
      {x, _y} = Data.to_xy(df, "Label", drop: ["Id", "Feature3"])

      # Should only have Feature1 and Feature2
      assert Nx.shape(x) == {4, 2}

      # First row should be [1.0, 2.0]
      first_row = Nx.slice_along_axis(x, 0, 1, axis: 0) |> Nx.to_flat_list()
      assert first_row == [1.0, 2.0]
    end

    test "works without drop option", %{df: df} do
      {x, y} = Data.to_xy(df, "Label")

      # Should have all columns except Label
      assert Nx.shape(x) == {4, 4}
      assert Nx.shape(y) == {4}
    end
  end

  describe "drop_to_nx/2" do
    setup %{tmp_dir: tmp_dir} do
      csv_path = Path.join(tmp_dir, "test.csv")

      File.write!(csv_path, """
      Id,Feature1,Feature2,Feature3
      1,1.0,2.0,3.0
      2,4.0,5.0,6.0
      3,7.0,8.0,9.0
      """)

      df = DF.from_csv!(csv_path)
      %{df: df}
    end

    test "drops columns and converts to tensor", %{df: df} do
      x = Data.drop_to_nx(df, ["Id"])

      # Should have 3 rows, 3 features
      assert Nx.shape(x) == {3, 3}
      assert Nx.type(x) == {:f, 32}

      # First row should be [1.0, 2.0, 3.0]
      first_row = Nx.slice_along_axis(x, 0, 1, axis: 0) |> Nx.to_flat_list()
      assert first_row == [1.0, 2.0, 3.0]
    end

    test "drops multiple columns", %{df: df} do
      x = Data.drop_to_nx(df, ["Id", "Feature3"])

      # Should only have Feature1 and Feature2
      assert Nx.shape(x) == {3, 2}
    end

    test "converts empty drop list", %{df: df} do
      x = Data.drop_to_nx(df, [])

      # Should have all 4 columns
      assert Nx.shape(x) == {3, 4}
    end
  end

  describe "write_submission!/4" do
    test "writes submission CSV file", %{tmp_dir: tmp_dir} do
      out_path = Path.join(tmp_dir, "submission.csv")
      headers = ["Id", "Prediction"]

      ids_series = Series.from_list([1, 2, 3, 4])
      preds_tensor = Nx.tensor([0.2, 0.7, 0.3, 0.9])

      result_path = Data.write_submission!(out_path, headers, ids_series, preds_tensor)

      assert result_path == out_path
      assert File.exists?(out_path)

      # Read and verify contents
      content = File.read!(out_path)
      lines = String.split(content, "\n", trim: true)

      # header + 4 rows
      assert length(lines) == 5
      assert hd(lines) == "Id,Prediction"
      assert Enum.at(lines, 1) == "1,0"
      assert Enum.at(lines, 2) == "2,1"
      assert Enum.at(lines, 3) == "3,0"
      assert Enum.at(lines, 4) == "4,1"
    end

    test "creates directory if it doesn't exist", %{tmp_dir: tmp_dir} do
      out_path = Path.join([tmp_dir, "nested", "dir", "submission.csv"])
      headers = ["Id", "Pred"]

      ids_series = Series.from_list([1, 2])
      preds_tensor = Nx.tensor([0.4, 0.6])

      Data.write_submission!(out_path, headers, ids_series, preds_tensor)

      assert File.exists?(out_path)
    end

    test "rounds predictions to integers", %{tmp_dir: tmp_dir} do
      out_path = Path.join(tmp_dir, "submission.csv")
      headers = ["Id", "Label"]

      ids_series = Series.from_list([1, 2, 3])
      preds_tensor = Nx.tensor([0.4, 0.6, 1.5])

      Data.write_submission!(out_path, headers, ids_series, preds_tensor)

      content = File.read!(out_path)
      lines = String.split(content, "\n", trim: true)

      # Check rounding behavior
      # 0.4 rounds to 0
      assert Enum.at(lines, 1) == "1,0"
      # 0.6 rounds to 1
      assert Enum.at(lines, 2) == "2,1"
      # 1.5 rounds to 2
      assert Enum.at(lines, 3) == "3,2"
    end

    test "handles different ID types", %{tmp_dir: tmp_dir} do
      out_path = Path.join(tmp_dir, "submission.csv")
      headers = ["PassengerId", "Survived"]

      # Test with string IDs
      ids_series = Series.from_list(["A1", "B2", "C3"])
      preds_tensor = Nx.tensor([1.0, 0.0, 1.0])

      Data.write_submission!(out_path, headers, ids_series, preds_tensor)

      content = File.read!(out_path)
      lines = String.split(content, "\n", trim: true)

      assert Enum.at(lines, 1) == "A1,1"
      assert Enum.at(lines, 2) == "B2,0"
      assert Enum.at(lines, 3) == "C3,1"
    end
  end
end
