defmodule DcoreTest do
  use ExUnit.Case
  doctest Dcore

  test "greets the world" do
    assert Dcore.hello() == :world
  end
end
