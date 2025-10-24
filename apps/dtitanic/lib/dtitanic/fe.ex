defmodule Dtitanic.FE do
  require Explorer.DataFrame
  alias Explorer.DataFrame, as: DF
  alias Explorer.Series

  def pipeline(df) do
    df
    # Convert Sex to binary (male = 1, female = 0)
    |> DF.mutate(Sex: cast(col("Sex") == "male", :integer))
    # Fill missing Age with median
    |> fill_age()
    # Fill missing Fare with median
    |> fill_fare()
    # Fill missing Embarked with mode (most common)
    |> fill_embarked()
    # Convert Embarked to numeric (one-hot style)
    |> DF.mutate(Embarked_C: cast(col("Embarked") == "C", :integer))
    |> DF.mutate(Embarked_Q: cast(col("Embarked") == "Q", :integer))
    |> DF.mutate(Embarked_S: cast(col("Embarked") == "S", :integer))
    # Drop columns we won't use
    |> DF.discard(["Name", "Ticket", "Cabin", "Embarked"])
  end

  defp fill_fare(df) do
    fare_series = DF.pull(df, "Fare")
    median_fare = fare_series |> Series.median() |> then(fn x -> x || 14.4542 end)

    filled_fare = Series.fill_missing(fare_series, median_fare)
    DF.put(df, "Fare", filled_fare)
  end

  defp fill_age(df) do
    age_series = DF.pull(df, "Age")
    median_age = age_series |> Series.median() |> then(fn x -> x || 28.0 end)

    filled_age = Series.fill_missing(age_series, median_age)
    DF.put(df, "Age", filled_age)
  end

  defp fill_embarked(df) do
    embarked_series = DF.pull(df, "Embarked")

    # Fill missing values with "S" (most common port)
    filled_embarked = Series.fill_missing(embarked_series, "S")

    # Replace the Embarked column with the filled version
    DF.put(df, "Embarked", filled_embarked)
  end
end
