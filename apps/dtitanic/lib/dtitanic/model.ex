defmodule Dtitanic.Model do
  alias Dcore.Data
  @train "apps/dtitanic/priv/data/train.csv"
  @test  "apps/dtitanic/priv/data/test.csv"
  @out   "apps/dtitanic/priv/runs/submission.csv"
  def run do
    {train, test} = Data.load_csv(@train, @test)
    # TODO: transform with Dtitanic.FE.pipeline/1 and train a model
    Data.write_submission!(@out, ["Id","Pred"], Explorer.Series.from_list([]), Nx.tensor([]))
  end
end
