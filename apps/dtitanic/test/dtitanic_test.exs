defmodule DtitanicTest do
  use ExUnit.Case
  doctest Dtitanic

  test "greets the world" do
    assert Dtitanic.hello() == :world
  end
end
