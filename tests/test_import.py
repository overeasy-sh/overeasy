def test_import_overeasy(capsys):
    with capsys.disabled():
        import overeasy
    captured = capsys.readouterr()
    assert captured.out == ""